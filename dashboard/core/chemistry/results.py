"""Read QE output for the dashboard; execution belongs to the launcher or Slurm."""

import itertools
import json
import re
import shlex
from pathlib import Path

from dashboard.core import execution, models
from dashboard.core.chemistry.inputs import ram_reports
from dashboard.core.jobs.persistence import now
from dashboard.core.jobs.processes import evidence, tail


def write_entrypoint(
    task: models.TaskRecord, directory: Path, command: list[str]
) -> None:
    lines = [
        f"[{now()}] Task: {task['id']}",
        f"Kind: {task['kind']}",
        f"Runtime: {task['runtime']}",
        f"Working directory: {directory}",
        f"Entrypoint: {shlex.join(command)}",
        f"Processes: {task['processes']}",
        f"Timeout: {task['timeout']}",
        f"Image hash: {task.get('image_hash')}",
        f"Source hash: {task.get('source_hash')}",
        f"Input hash: {task.get('input_hash')}",
        f"Resources: {json.dumps(task['resources'])}",
    ]
    (directory / execution.ENTRYPOINT_LOG_FILE).write_text("\n".join(lines) + "\n")


def observed_mpi_processes(directory: Path) -> int | None:
    with (directory / execution.STDOUT_LOG_FILE).open(errors="replace") as stream:
        for line in stream:
            for pattern in execution.QE_MPI_PROCESS_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    return int(match[1])
    return None


def estimate_report(directory: Path) -> tuple[models.RAMReport, bool]:
    with (
        (directory / execution.STDOUT_LOG_FILE).open(errors="replace") as stdout,
        (directory / execution.STDERR_LOG_FILE).open(errors="replace") as stderr,
    ):
        return ram_reports(itertools.chain(stdout, stderr))


def estimate_problem(
    task: models.TaskRecord, directory: Path, report: models.RAMReport, qe_error: bool
) -> str:
    if qe_error:
        return "QE reported an error during RAM estimation"
    if report["per_process"] is None:
        return "QE did not report a per-process RAM estimate"
    if (
        task.get("target") == "slurm"
        and task["processes"] > 1
        and observed_mpi_processes(directory) != task["processes"]
    ):
        return "QE MPI process count did not match the requested RAM estimate"
    return ""


def validate_estimate(
    task: models.TaskRecord, directory: Path, result: models.TaskResult
) -> None:
    report, qe_error = estimate_report(directory)
    result["ram_estimate"] = report
    # Initialization-only QE may intentionally exit 255; errors still win.
    if (
        not result["error"]
        and result["status"] != execution.TaskStatus.CANCELED
        and result["exit_code"] in execution.QE_INITIALIZATION_EXIT_CODES
    ):
        problem = estimate_problem(task, directory, report, qe_error)
        result.update(
            status=execution.TaskStatus.FAILED
            if problem
            else execution.TaskStatus.SUCCEEDED,
            error=problem,
        )
    if result["status"] == execution.TaskStatus.FAILED and not result["error"]:
        result["error"] = f"Process exited with code {result['exit_code']}"


def scientific_output(directory: Path) -> str:
    markers = []
    with (directory / execution.STDOUT_LOG_FILE).open(errors="replace") as stream:
        for line in stream:
            if any(marker in line for marker in execution.QE_EVIDENCE_MARKERS):
                if execution.QE_ENERGY_MARKER in line:
                    markers = [
                        marker
                        for marker in markers
                        if execution.QE_ENERGY_MARKER not in marker
                    ]
                markers.append(line)
    return "".join(markers)


def validate_qe(
    task: models.TaskRecord, directory: Path, result: models.TaskResult
) -> None:
    calculation = re.search(
        r"calculation\s*=\s*['\"]([^'\"]+)",
        (directory / execution.INPUT_FILE).read_text(),
        re.IGNORECASE,
    )
    result["evidence"] = evidence(
        scientific_output(directory),
        calculation[1] if calculation else execution.QE_DEFAULT_CALCULATION,
    )
    # Scheduler completion alone must never certify scientific convergence.
    if (
        task.get("target") == "slurm"
        and result["status"] == execution.TaskStatus.SUCCEEDED
        and not result["evidence"]["confirmed"]
    ):
        result.update(
            status=execution.TaskStatus.FAILED,
            error="QE completion or convergence was not confirmed",
        )


def read_calculation_result(
    task: models.TaskRecord,
    directory: Path,
    code: int,
    status: str,
    error: str = "",
) -> models.TaskResult:
    """Interpret output without launching processes or changing queue state."""
    if (
        status == execution.TaskStatus.FAILED
        and not error
        and task["kind"] != execution.TaskKind.ESTIMATE_RAM
    ):
        detail = tail(
            directory / execution.STDERR_LOG_FILE, limit=execution.ERROR_TAIL_CHARACTERS
        ).strip()
        error = f"Process exited with code {code}" + (f": {detail}" if detail else "")
    result = models.TaskResult(status=status, exit_code=code, error=error, ended=now())
    if task["kind"] == execution.TaskKind.ESTIMATE_RAM:
        validate_estimate(task, directory, result)
    elif task["kind"] == execution.TaskKind.QE:
        validate_qe(task, directory, result)
    return result
