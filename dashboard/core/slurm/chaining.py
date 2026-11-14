"""Durable, opt-in advancement of the existing Slurm waiting queue."""

import json
import os
import shlex
import subprocess
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

from dashboard.core import config as configuration
from dashboard.core import execution, models
from dashboard.core.jobs.persistence import now
from dashboard.core.slurm import scheduler

if TYPE_CHECKING:
    from dashboard.core.jobs.tasks import Manager

DISPATCH_MODULE = "dashboard.app.dispatch"
DISPATCH_SCRIPT = "dispatch.sh"
DISPATCH_CONFIG = "config.json"
DISPATCH_TIME = "00:05:00"
DISPATCH_MEMORY = "256M"
PATH_FIELDS = ("artifacts", "image", "pseudos", "qe_inputs")


def config_arguments(config: configuration.Config) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def observation_error(
    record: models.SchedulerRecord, states: scheduler.JobStates
) -> str:
    if record["state"] in execution.TERMINAL_SLURM_STATES:
        return ""
    name = execution.SLURM_JOB_PREFIX + record["token"]
    if record["job_id"] is None:
        return (
            f"Submission {name} is unresolved. Check squeue/sacct by job name "
            "and recover after accounting catches up; automatic resubmission is blocked."
        )
    if (
        record["job_id"] not in states
        and record["state"] not in execution.TERMINAL_SLURM_STATES
    ):
        return (
            f"Slurm job {record['job_id']} ({name}) has no current observation. "
            "Wait for accounting and recover; queue advancement is blocked."
        )
    return ""


def load_config(path: Path) -> configuration.Config:
    values = json.loads(path.read_text())
    for key in PATH_FIELDS:
        if values.get(key) is not None:
            values[key] = Path(values[key])
    return configuration.Config(**values)


def main(*, config: Path, predecessor: str) -> None:
    from dashboard.core.jobs.tasks import Manager

    # sbatch exports the sanitized runtime environment; preserve this job's identity.
    manager = Manager(load_config(config), {}, recover=False)
    try:
        state = manager.chain.recover(predecessor=predecessor)
        if state["error"]:
            raise RuntimeError(state["error"])
    finally:
        manager.store.db.close()


def write_dispatch_script(directory: Path, predecessor: str) -> Path:
    arguments = [
        sys.executable,
        "-m",
        DISPATCH_MODULE,
        "--config",
        str(directory / DISPATCH_CONFIG),
        "--predecessor",
        predecessor,
    ]
    script = directory / DISPATCH_SCRIPT
    script.write_text(
        "#!/bin/sh\nset -eu\nexport PYTHONPATH="
        + shlex.quote(str(configuration.ROOT))
        + "\nexec "
        + shlex.join(arguments)
        + "\n"
    )
    return script


class Chain:
    def __init__(self, manager: "Manager") -> None:
        self.manager = manager

    def data(self) -> models.ChainState:
        return self.manager.store.chain()

    def configure(
        self, enabled: bool | None = None, paused: bool | None = None
    ) -> models.ChainState:
        with self.manager.store.mutation_lock:
            self.restore_dispatcher_intent()
            state = self.data()
            self.validate_mode_change(state, enabled)
            changes = models.ChainUpdate()
            if enabled is not None:
                changes["enabled"] = enabled
            if paused is not None:
                changes["paused"] = paused
            self.manager.store.set_chain(changes)
            return self.recover()

    def validate_mode_change(
        self, state: models.ChainState, enabled: bool | None
    ) -> None:
        if enabled is None or enabled == state["enabled"]:
            return
        if enabled:
            self.validate_waiting_groups()
        dispatcher = state["dispatcher"]
        dispatcher_pending = dispatcher and (
            dispatcher["state"] not in execution.TERMINAL_SLURM_STATES
        )
        observations, _ = self.dispatcher_observations()
        if self.outstanding_tasks() or dispatcher_pending or observations:
            raise ValueError(
                "Slurm compute or dispatcher submissions are outstanding; pause the "
                "chain and recover until Slurm confirms termination before changing mode"
            )

    def validate_waiting_groups(self) -> None:
        for task in self.manager.scheduler.waiting_tasks():
            batch_id = task.get("batch_id")
            batch = self.manager.store.batch(batch_id) if batch_id else None
            if batch and (
                batch.get("estimate_processes") or batch.get("estimate_systems")
            ):
                raise ValueError(
                    "Finish or remove grouped RAM batches before enabling chaining; "
                    "queue individual process counts for a chain"
                )

    def outstanding_tasks(self) -> list[models.TaskRecord]:
        return [
            task
            for task in self.manager.store.tasks()
            if task.get("target") == execution.ExecutionMode.SLURM
            and task.get("scheduler")
            and task["scheduler"]["state"] not in execution.TERMINAL_SLURM_STATES
        ]

    def recover(self, predecessor: str | None = None) -> models.ChainState:
        with self.manager.store.mutation_lock:
            state = self.data()
            if predecessor is not None and predecessor != state["active_task_id"]:
                self.audit(
                    "recover_skipped", "Predecessor no longer owns the active task"
                )
                return self.data()
            self.manager.store.set_chain({"last_attempt": now(), "error": ""})
            try:
                self.advance(predecessor)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                self.manager.store.set_chain({"error": str(error)})
            state = self.data()
            self.audit("recover_blocked" if state["error"] else "recover_completed")
            return self.data()

    def mark_dispatcher_rejected(self, token: str) -> models.ChainState:
        """Operator confirms rejection; absence alone never triggers this action."""
        with self.manager.store.mutation_lock:
            self.restore_dispatcher_intent()
            record = self.rejection_candidate(token)
            _, names = scheduler.query_observations(
                [cast(models.TaskRecord, {"scheduler": record})]
            )
            if names.get(execution.SLURM_JOB_PREFIX + token):
                raise ValueError(
                    "Slurm has a matching dispatcher; recover to reconnect"
                )
            record.update(
                state=execution.SlurmState.FAILED,
                reason=scheduler.CONFIRMED_REJECTION_REASON,
            )
            owner = self.data()["active_task_id"]
            if owner:
                self.save_dispatcher(owner, record)
            self.manager.store.set_chain(
                {
                    "dispatcher": record,
                    "error": "Dispatcher rejection confirmed by operator; use Recover to resume",
                }
            )
            self.audit("dispatcher_rejection_confirmed")
            return self.data()

    def rejection_candidate(self, token: str) -> models.SchedulerRecord:
        record = self.data()["dispatcher"]
        if not record or record["token"] != token:
            raise ValueError("Dispatcher changed; refresh before confirming rejection")
        if (
            record["job_id"] is not None
            or record["state"] != execution.SlurmState.SUBMISSION_UNKNOWN
        ):
            raise ValueError(
                "Only an unresolved dispatcher submission can be marked rejected"
            )
        return record.copy()

    def monitor(self) -> models.ChainState:
        """Refresh observations and diagnostics without submitting or advancing."""
        with self.manager.store.mutation_lock:
            previous = self.data()["error"]
            try:
                self.reconcile(None)
                message = self.monitor_error()
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                message = str(error)
            if message and message not in previous:
                combined = f"{previous}; {message}" if previous else message
                self.manager.store.set_chain({"error": combined})
                self.audit("monitor_blocked")
            return self.data()

    def monitor_error(self) -> str:
        state = self.data()
        record = state["dispatcher"]
        if not record or record["state"] not in execution.TERMINAL_SLURM_STATES:
            return ""
        failed = record["state"] != execution.SlurmState.COMPLETED
        waiting = (
            state["enabled"]
            and not state["paused"]
            and bool(self.manager.scheduler.waiting_tasks())
        )
        if not failed and not waiting:
            return ""
        return (
            f"Dispatcher {record['job_id']} ended with Slurm {record['state']}. "
            "Inspect its Slurm log and use Recover to resume queue advancement; "
            "no replacement dispatcher was submitted automatically."
        )

    def audit(self, action: str, detail: str = "") -> None:
        state = self.data()
        event: models.ChainEvent = {
            "timestamp": now(),
            "action": action,
            "error": state["error"],
            "detail": detail,
            "outcome": self.recovery_outcome(state),
            "enabled": state["enabled"],
            "paused": state["paused"],
            "active_task_id": state["active_task_id"],
            "dispatcher": state["dispatcher"],
        }
        self.manager.store.set_chain({"events": [*state.get("events", []), event]})

    def recovery_outcome(self, state: models.ChainState) -> str:
        if state["error"]:
            return "blocked"
        if not state["enabled"]:
            return "disabled"
        if state["paused"]:
            return "paused"
        if state["active_task_id"] is None:
            return "queue_empty"
        task = self.manager.store.get(state["active_task_id"])
        record = task.get("scheduler")
        return (
            "waiting_for_compute"
            if record and record["state"] not in execution.TERMINAL_SLURM_STATES
            else "waiting_for_dispatcher"
        )

    def save_dispatcher(self, task_id: str, record: models.SchedulerRecord) -> None:
        task = self.manager.store.get(task_id)
        history = task.get("chain_dispatchers", [])
        updated = list(history)
        # Replace in place so reconciliation never changes attempt chronology.
        if any(item["token"] == record["token"] for item in history):
            updated = [
                record if item["token"] == record["token"] else item for item in history
            ]
        else:
            updated.append(record)
        artifacts = dict(task["artifacts"])
        name = f"dispatcher-{record['token']}.log"
        artifacts[name] = self.manager.store.register(
            task_id + "-" + name,
            Path(record["directory"]) / execution.SLURM_LOG_FILE,
        )
        self.manager.store.update(
            task_id, chain_dispatchers=updated, artifacts=artifacts
        )

    def dispatcher_observations(self) -> tuple[list[models.TaskRecord], dict[str, str]]:
        observations: list[models.TaskRecord] = []
        owners: dict[str, str] = {}
        for task in self.manager.store.tasks():
            history = task.get("chain_dispatchers", [])
            for record in history:
                owners[record["token"]] = task["id"]
                if record["state"] not in execution.TERMINAL_SLURM_STATES:
                    observations.append(cast(models.TaskRecord, {"scheduler": record}))
        return observations, owners

    def restore_dispatcher_intent(self) -> None:
        state = self.data()
        active = state["active_task_id"]
        if active is None:
            return
        history = self.manager.store.get(active).get("chain_dispatchers", [])
        if not history:
            return
        # Task history is committed first and is authoritative across crashes.
        latest = history[-1]
        self.manager.store.set_chain({"dispatcher": latest})

    def reconciliation_tasks(self) -> tuple[list[models.TaskRecord], dict[str, str]]:
        self.restore_dispatcher_intent()
        state = self.data()
        tasks = self.outstanding_tasks()
        active = state["active_task_id"]
        if active and all(task["id"] != active for task in tasks):
            task = self.manager.store.get(active)
            if task.get("scheduler"):
                tasks.append(task)
        observations, owners = self.dispatcher_observations()
        tasks.extend(observations)
        dispatcher = state["dispatcher"]
        if dispatcher and not any(
            item["scheduler"]["token"] == dispatcher["token"] for item in observations
        ):
            tasks.append(cast(models.TaskRecord, {"scheduler": dispatcher}))
        if dispatcher and state["active_task_id"]:
            owners.setdefault(dispatcher["token"], state["active_task_id"])
        return tasks, owners

    def reconcile(self, predecessor: str | None) -> None:
        tasks, owners = self.reconciliation_tasks()
        if not tasks:
            return
        states, names = scheduler.query_observations(tasks)
        errors: list[str] = []
        for task in tasks:
            record = scheduler.reconcile_record(task["scheduler"], states, names)
            self.persist_observation(task, record, owners)
            own_dispatcher = "id" not in task and self.owns_dispatcher(
                record, predecessor
            )
            if not own_dispatcher:
                message = observation_error(record, states)
                if message:
                    errors.append(message)
        if errors:
            raise ValueError("; ".join(errors))

    def persist_observation(
        self,
        task: models.TaskRecord,
        record: models.SchedulerRecord,
        owners: dict[str, str],
    ) -> None:
        if "id" in task:
            self.manager.scheduler.import_attempt(task, record)
            return
        owner = owners.get(record["token"])
        if owner:
            self.save_dispatcher(owner, record)
        dispatcher = self.data()["dispatcher"]
        if dispatcher and record["token"] == dispatcher["token"]:
            self.manager.store.set_chain({"dispatcher": record})

    def owns_dispatcher(
        self, record: models.SchedulerRecord, predecessor: str | None
    ) -> bool:
        own_job = os.getenv(execution.SLURM_JOB_ID_ENV)
        return (
            predecessor is not None
            and own_job is not None
            and record["job_id"] == own_job
        )

    def dispatcher_blocks(self, predecessor: str | None) -> bool:
        record = self.data()["dispatcher"]
        if not record:
            return False
        if self.owns_dispatcher(record, predecessor):
            return False
        return record["state"] not in execution.TERMINAL_SLURM_STATES

    def advance(self, predecessor: str | None) -> None:
        self.reconcile(predecessor)
        state = self.data()
        if not state["enabled"] or state["paused"]:
            return
        active = state["active_task_id"]
        task = self.manager.store.get(active) if active else None
        if task and not task.get("scheduler"):
            self.manager.store.set_chain({"active_task_id": None})
            task = None
        if task and task["scheduler"]["state"] not in execution.TERMINAL_SLURM_STATES:
            self.ensure_dispatcher(task, predecessor)
            return
        if self.dispatcher_blocks(predecessor):
            return
        self.start_next()

    def start_next(self) -> None:
        for task in self.manager.scheduler.waiting_tasks():
            # Reserve identity first; definitive setup failures use existing handling.
            self.manager.store.set_chain(
                {"active_task_id": task["id"], "dispatcher": None}
            )
            if not self.manager.scheduler.submit_group([task], chain=True):
                continue
            task = self.manager.store.get(task["id"])
            if not task.get("scheduler"):
                self.manager.store.set_chain({"active_task_id": None})
                return
            self.ensure_dispatcher(task, None)
            return
        self.manager.store.set_chain({"active_task_id": None, "dispatcher": None})

    def ensure_dispatcher(
        self, task: models.TaskRecord, predecessor: str | None
    ) -> None:
        job_id = task["scheduler"]["job_id"]
        if job_id is None:
            raise ValueError(observation_error(task["scheduler"], {}))
        if self.dispatcher_blocks(predecessor):
            return
        record = self.dispatcher_record(task, job_id)
        self.save_dispatcher(task["id"], record)
        self.manager.store.set_chain({"dispatcher": record})
        self.manager.scheduler.run_submission(record)
        self.save_dispatcher(task["id"], record)
        self.manager.store.set_chain({"dispatcher": record})
        if record["job_id"] is None:
            raise ValueError(observation_error(record, {}))

    def dispatcher_record(
        self, task: models.TaskRecord, job_id: str
    ) -> models.SchedulerRecord:
        token = uuid.uuid4().hex
        directory = self.manager.config.artifacts / (
            execution.SLURM_DIRECTORY_PREFIX + token
        )
        directory.mkdir()
        config_path = directory / DISPATCH_CONFIG
        config_path.write_text(json.dumps(config_arguments(self.manager.config)))
        script = write_dispatch_script(directory, task["id"])
        return self.submission_record(task, token, directory, script, job_id)

    def submission_record(
        self,
        task: models.TaskRecord,
        token: str,
        directory: Path,
        script: Path,
        job_id: str,
    ) -> models.SchedulerRecord:
        """Keep the linear Slurm argument list together for auditing allocation limits.

        The length comes from declaring each required scheduler option explicitly;
        splitting that declaration would obscure the dependency and resource request.
        """
        arguments = [
            execution.SBATCH_COMMAND,
            "--parsable",
            "--export=ALL",
            "--no-requeue",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=1",
            f"--mem={DISPATCH_MEMORY}",
            f"--time={DISPATCH_TIME}",
            f"--dependency=afterany:{job_id}",
            f"--job-name={execution.SLURM_JOB_PREFIX}{token}",
            f"--output={directory / execution.SLURM_LOG_FILE}",
            f"--chdir={configuration.ROOT}",
        ]
        for attribute in scheduler.SCHEDULING_ATTRIBUTES:
            value = task["scheduler_settings"][attribute]
            if value:
                arguments.append(f"--{attribute}={value}")
        return models.SchedulerRecord(
            token=token,
            job_id=None,
            state=execution.SlurmState.SUBMITTING,
            reason="",
            directory=str(directory),
            submitted=now(),
            command=[*arguments, str(script)],
        )
