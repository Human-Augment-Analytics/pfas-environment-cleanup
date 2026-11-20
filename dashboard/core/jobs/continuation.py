"""Conservative output decisions and opt-in successors in the existing task queue."""

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from dashboard.core import config as configuration
from dashboard.core import execution, models
from dashboard.core.chemistry import optimization
from dashboard.core.chemistry.preparation import digest

if TYPE_CHECKING:
    from dashboard.core.jobs.tasks import Manager

MAX_CONTINUATIONS = 10
PIPELINE_DIRECTORIES = ("dashboard/core", "dashboard/app", "qespresso_pipeline")


class Outcome(StrEnum):
    CONVERGED = "converged"
    INCOMPLETE = "incomplete"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


class Inspection(TypedDict):
    outcome: Outcome
    reason: str
    complete_coordinates: bool
    completed_scf_cycles: int
    job_done: bool


def pipeline_hash() -> str:
    """Invalidate prior automatic authorization when pipeline code changes."""
    fingerprint = hashlib.sha256()
    for directory in PIPELINE_DIRECTORIES:
        for path in sorted((configuration.ROOT / directory).rglob("*.py")):
            fingerprint.update(str(path.relative_to(configuration.ROOT)).encode())
            fingerprint.update(path.read_bytes())
    return fingerprint.hexdigest()


def inspect(directory: Path) -> Inspection:
    text, output = optimization.inspect(directory)
    outcome, reason = classify(text, output)
    return Inspection(
        outcome=outcome,
        reason=reason,
        complete_coordinates=bool(output["positions"])
        and not output["partial_positions"],
        completed_scf_cycles=len(output["steps"]),
        job_done=output["job_done"],
    )


def classify(text: str, output: optimization.Output) -> tuple[Outcome, str]:
    if output["qe_error"] or output["failed_scf_cycles"]:
        return Outcome.FAILED, "QE error or failed SCF cycle requires review"
    if output["partial_positions"] or not output["steps"]:
        return Outcome.UNCERTAIN, "Partial coordinates or no completed SCF sample"
    if (
        output["job_done"]
        and output["steps"]
        and (
            output["relaxation_completed"]
            or optimization.calculation(text) == execution.QE_DEFAULT_CALCULATION
        )
    ):
        return Outcome.CONVERGED, "QE completed the requested calculation"
    try:
        optimization.replace_positions(text, output["positions"])
    except ValueError as error:
        return Outcome.UNCERTAIN, str(error)
    return Outcome.INCOMPLETE, "Complete fixed-cell coordinates; relaxation incomplete"


def configure(manager: "Manager", task_id: str, limit: int) -> models.TaskRecord:
    """Authorize a bounded sequence under the current pipeline and saved settings."""
    if type(limit) is not int or not 0 <= limit <= MAX_CONTINUATIONS:
        raise ValueError(
            f"Continuation limit must be between 0 and {MAX_CONTINUATIONS}"
        )
    with manager.guard:
        task = manager.store.get(task_id)
        validate_policy_task(manager, task, limit)
        return manager.store.update(
            task_id,
            continuation_policy=models.ContinuationPolicy(
                remaining=limit,
                pipeline_hash=pipeline_hash(),
            ),
            continuation_error="",
        )


def validate_policy_task(
    manager: "Manager", task: models.TaskRecord, limit: int
) -> None:
    if (
        task["kind"] != execution.TaskKind.QE
        or task.get("target") != execution.ExecutionMode.SLURM
    ):
        raise ValueError("Automatic continuation requires a Slurm QE task")
    if limit:
        text = (
            manager.config.artifacts / task["id"] / execution.INPUT_FILE
        ).read_text()
        if optimization.calculation(text) != optimization.FIXED_CELL_CALCULATION:
            raise ValueError(
                "Automatic continuation requires a fixed-cell relax calculation"
            )


def eligible(task: models.TaskRecord, inspection: Inspection) -> bool:
    if task.get("stop_requested") or task["status"] == execution.TaskStatus.CANCELED:
        return False
    state = task.get("scheduler", {}).get("state")
    known_end = state == execution.SlurmState.TIMEOUT or (
        state == execution.SlurmState.COMPLETED
        and task.get("exit_code") == 0
        and inspection["job_done"]
    )
    return known_end and inspection["outcome"] == Outcome.INCOMPLETE


def existing_successor(manager: "Manager", parent: str) -> models.TaskRecord | None:
    identifier = optimization.CONTINUATION_PREFIX + parent
    return next(
        (
            task
            for task in manager.store.tasks()
            if task["kind"] == execution.TaskKind.QE
            and task.get("input_id") == identifier
        ),
        None,
    )


def advance(manager: "Manager", task: models.TaskRecord) -> None:
    """Run under the chain mutation lock; accepted tasks are the durable intent."""
    policy = task.get("continuation_policy")
    if not policy or not policy["remaining"]:
        return
    if existing_successor(manager, task["id"]):
        manager.store.update(task["id"], continuation_error="")
        return
    try:
        prepare_successor(manager, task, policy)
        manager.store.update(task["id"], continuation_error="")
    except (OSError, ValueError) as error:
        manager.store.update(task["id"], continuation_error=str(error))
        raise ValueError(
            f"Continuation of {task['id']} requires review: {error}"
        ) from error


def prepare_successor(
    manager: "Manager", task: models.TaskRecord, policy: models.ContinuationPolicy
) -> None:
    inspection = inspect(manager.config.artifacts / task["id"])
    if inspection["outcome"] == Outcome.CONVERGED:
        return
    if not eligible(task, inspection):
        raise ValueError(
            inspection["reason"]
            + "; automatic continuation requires a known timeout or clean exit"
        )
    enqueue(manager, task, policy)


def preview(manager: "Manager", task: models.TaskRecord) -> dict:
    return manager.preview(
        task["candidate"],
        task["system"],
        task["processes"],
        task["target"],
        task["runtime"],
        optimization.CONTINUATION_PREFIX + task["id"],
        scheduler=task.get("scheduler_settings"),
        memory_gib=task["resources"]["memory_bytes"] / execution.GIB,
        timeout=task["timeout"],
    )


def validate_snapshot(
    manager: "Manager", task: models.TaskRecord, policy: models.ContinuationPolicy
) -> None:
    if pipeline_hash() != policy["pipeline_hash"]:
        raise ValueError("Pipeline code changed; review and reauthorize continuation")
    path = manager.config.artifacts / task["id"] / execution.INPUT_FILE
    if digest(path) != task["input_hash"]:
        raise ValueError("Source input changed since admission")


def enqueue(
    manager: "Manager", task: models.TaskRecord, policy: models.ContinuationPolicy
) -> None:
    validate_snapshot(manager, task, policy)
    reviewed = preview(manager, task)
    if (
        reviewed["image_hash"] != task["image_hash"]
        or reviewed["pseudopotentials"] != task["pseudopotentials"]
    ):
        manager.previews.pop(reviewed["preview_id"], None)
        raise ValueError("Runtime image or pseudopotentials changed; preview again")
    successor_policy = models.ContinuationPolicy(
        remaining=policy["remaining"] - 1,
        pipeline_hash=policy["pipeline_hash"],
    )
    accept_successor(manager, task, reviewed, successor_policy)


def accept_successor(
    manager: "Manager",
    task: models.TaskRecord,
    reviewed: dict,
    policy: models.ContinuationPolicy,
) -> None:
    manager.queue(
        execution.TaskKind.QE,
        task["candidate"],
        system=task["system"],
        processes=task["processes"],
        target=execution.ExecutionMode.SLURM,
        runtime=task["runtime"],
        timeout=task["timeout"],
        memory_gib=task["resources"]["memory_bytes"] / execution.GIB,
        scheduler=task.get("scheduler_settings"),
        input_id=reviewed["input_id"],
        preview_id=reviewed["preview_id"],
        expected_hash=reviewed["input_hash"],
        _continuation_policy=policy,
        _continuation_parent=task["id"],
    )


def main(*, directory: Path, output: Path | None) -> None:
    assessment = inspect(directory)
    print(json.dumps(assessment, indent=2))
    if output is None:
        return
    if assessment["outcome"] != Outcome.INCOMPLETE:
        raise ValueError("Only complete, incomplete relaxations can be prepared here")
    text = optimization.continuation(directory)
    with output.open("x") as destination:
        destination.write(text)
