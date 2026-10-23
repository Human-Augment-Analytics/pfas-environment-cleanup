"""Independent single-node Slurm jobs, reconciled through shared artifacts."""

import json
import logging
import math
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from .config import ROOT
from .execution import (
    ACTIVE_TASK_STATUSES,
    APPLICATION_LOGGER_NAME,
    COMPUTE_WORKER_ENV,
    DEFAULT_ACCOUNT,
    DEFAULT_ESTIMATION_HOURS,
    DEFAULT_HEADROOM_PERCENT,
    DEFAULT_PARTITION,
    DEFAULT_QE_HOURS,
    DEFAULT_QOS,
    DISPATCH_POLL_SECONDS,
    ENTRYPOINT_LOG_FILE,
    EXECUTION_ENV,
    INPUT_FILE,
    MANIFEST_FILE,
    MAX_SLURM_HOURS,
    MAX_SUBMISSIONS_PER_CYCLE,
    MIB,
    PARENT_JOB_ENV_PREFIXES,
    SACCT_COMMAND,
    SBATCH_COMMAND,
    SCANCEL_COMMAND,
    SLURM_COMMAND_TIMEOUT,
    SLURM_CONFIG_ENV,
    SLURM_DIRECTORY_PREFIX,
    SLURM_JOB_PREFIX,
    SLURM_LOG_FILE,
    SLURM_POLL_SECONDS,
    SLURM_RESULT_FILE,
    SLURM_SCRIPT_FILE,
    SLURM_STARTED_FILE,
    SQUEUE_COMMAND,
    STDERR_LOG_FILE,
    STDOUT_LOG_FILE,
    TERMINAL_SLURM_STATES,
    ExecutionMode,
    Runtime,
    SlurmState,
    TaskKind,
    TaskStatus,
)
from .execution_models import (
    SchedulerRecord,
    SchedulerSettings,
    TaskRecord,
    TaskResult,
    WorkerManifest,
)
from .persistence import now

if TYPE_CHECKING:
    from .tasks import Manager

SCHEDULING_ATTRIBUTES = ("partition", "account", "qos")
SETTINGS_KEYS = (
    *SCHEDULING_ATTRIBUTES,
    "walltime_hours",
    "memory_mode",
    "headroom_percent",
)
MAX_HEADROOM_PERCENT = 1000
SQUEUE_FORMAT = "--format=%i|%.64j|%T|%R"
SACCT_FORMAT = "--format=JobIDRaw,JobName%64,State%40,ExitCode"
RESULT_FIELDS = (
    "status",
    "exit_code",
    "error",
    "ended",
    "started",
    "evidence",
    "ram_estimate",
    "usage",
    "command",
)
RESULT_ARTIFACTS = (
    ENTRYPOINT_LOG_FILE,
    STDOUT_LOG_FILE,
    STDERR_LOG_FILE,
    INPUT_FILE,
    SLURM_RESULT_FILE,
)
JobStates = dict[str, tuple[str, str]]
JobNames = dict[str, set[str]]


def scheduler_name(value: object) -> str:
    if not isinstance(value, str) or (
        value and not re.fullmatch(r"[A-Za-z0-9_.-]+", value)
    ):
        raise ValueError("Invalid partition, account, or QOS")
    return value


def settings(
    values: Mapping[str, object] | None = None, kind: str = TaskKind.QE
) -> SchedulerSettings:
    overrides = values or {}
    if set(overrides) - set(SETTINGS_KEYS):
        raise ValueError("Unknown scheduler setting")
    hours = overrides.get(
        "walltime_hours",
        DEFAULT_ESTIMATION_HOURS if kind == TaskKind.ESTIMATE_RAM else DEFAULT_QE_HOURS,
    )
    if (
        not isinstance(hours, int)
        or isinstance(hours, bool)
        or not 1 <= hours <= MAX_SLURM_HOURS
    ):
        raise ValueError(f"Walltime must be between 1 and {MAX_SLURM_HOURS} hours")
    memory_mode = overrides.get("memory_mode", "manual")
    if memory_mode not in ("manual", "estimate"):
        raise ValueError("Unknown memory mode")
    if kind == TaskKind.ESTIMATE_RAM and memory_mode != "manual":
        raise ValueError("RAM estimation requires an explicit memory reservation")
    headroom = overrides.get("headroom_percent", DEFAULT_HEADROOM_PERCENT)
    if (
        isinstance(headroom, bool)
        or not isinstance(headroom, (float, int))
        or not math.isfinite(headroom)
        or not 0 <= headroom <= MAX_HEADROOM_PERCENT
    ):
        raise ValueError(
            f"Headroom must be between 0 and {MAX_HEADROOM_PERCENT} percent"
        )
    return SchedulerSettings(
        partition=scheduler_name(overrides.get("partition", DEFAULT_PARTITION)),
        account=scheduler_name(overrides.get("account", DEFAULT_ACCOUNT)),
        qos=scheduler_name(overrides.get("qos", DEFAULT_QOS)),
        walltime_hours=hours,
        memory_mode=cast(Literal["manual", "estimate"], memory_mode),
        headroom_percent=float(headroom),
    )


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    # Preserve module/runtime paths, but never inherit a parent's allocation,
    # MPI session, or SBATCH overrides into a newly requested allocation.
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(PARENT_JOB_ENV_PREFIXES) or name in SLURM_CONFIG_ENV
    }
    return subprocess.run(
        arguments,
        capture_output=True,
        text=True,
        timeout=SLURM_COMMAND_TIMEOUT,
        check=False,
        env=environment,
    )


def build_submission(tasks: list[TaskRecord], token: str, directory: Path) -> list[str]:
    options = tasks[0]["scheduler_settings"]
    processes = max(task["processes"] for task in tasks)
    memory_bytes = max(task["resources"]["memory_bytes"] for task in tasks)
    arguments = [
        SBATCH_COMMAND,
        "--parsable",
        "--export=ALL",
        "--no-requeue",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={processes}",
        f"--mem={math.ceil(memory_bytes / MIB)}M",
        f"--time={options['walltime_hours']}:00:00",
        f"--job-name={SLURM_JOB_PREFIX}{token}",
        f"--comment={SLURM_JOB_PREFIX}{token}",
        f"--output={directory / SLURM_LOG_FILE}",
        f"--chdir={ROOT}",
    ]
    for attribute in SCHEDULING_ATTRIBUTES:
        value = options[attribute]
        if value:
            arguments.append(f"--{attribute}={value}")
    return [*arguments, str(directory / SLURM_SCRIPT_FILE)]


def parse_observations(outputs: tuple[str, str]) -> tuple[JobStates, JobNames]:
    states: JobStates = {}
    names: JobNames = {}
    # Live state takes precedence over accounting, which can lag.
    for output in outputs:
        for line in output.splitlines():
            fields = [field.strip() for field in line.split("|")]
            if len(fields) < 4 or not fields[0].isdigit():
                continue  # Ignore job steps; reconcile the allocation row.
            job_id, name, state, reason = fields[:4]
            if not state:
                continue
            states[job_id] = (state.split()[0].rstrip("+"), reason)
            names.setdefault(name, set()).add(job_id)
    return states, names


def query_observations(tasks: list[TaskRecord]) -> tuple[JobStates, JobNames]:
    owner = str(os.getuid())
    live = command([SQUEUE_COMMAND, "--noheader", "--user", owner, SQUEUE_FORMAT])
    job_ids = sorted(
        {
            task["scheduler"]["job_id"]
            for task in tasks
            if task["scheduler"]["job_id"] is not None
        }
    )
    arguments = [
        SACCT_COMMAND,
        "--noheader",
        "--parsable2",
        "--starttime",
        min(task["scheduler"]["submitted"][:10] for task in tasks),
        SACCT_FORMAT,
        "--user",
        owner,
    ]
    # Unknown submissions require a name search across this user's jobs.
    if job_ids and all(task["scheduler"]["job_id"] is not None for task in tasks):
        arguments += ["--jobs", ",".join(job_ids)]
    accounting = command(arguments)
    if live.returncode or accounting.returncode:
        raise ValueError(live.stderr or accounting.stderr)
    return parse_observations((accounting.stdout, live.stdout))


def reconcile_record(
    stored: SchedulerRecord, states: JobStates, names: JobNames
) -> SchedulerRecord:
    record = stored.copy()
    if record["job_id"] is None:
        matches = names.get(SLURM_JOB_PREFIX + record["token"], set())
        if len(matches) == 1:
            record["job_id"] = next(iter(matches))
        else:
            message = (
                "Submission outcome unresolved; no automatic retry. Check Slurm using job name "
                + SLURM_JOB_PREFIX
                + record["token"]
            )
            if record["reason"] and message not in record["reason"]:
                message = record["reason"] + "; " + message
            elif message in record["reason"]:
                message = record["reason"]
            record.update(
                state=SlurmState.SUBMISSION_UNKNOWN,
                reason=message,
            )
    job_id = record["job_id"]
    if job_id is not None and job_id in states:
        record["state"], record["reason"] = states[job_id]
    return record


def read_result(directory: Path, state: str) -> TaskResult:
    result_path = directory / SLURM_RESULT_FILE
    if result_path.is_file():
        report = json.loads(result_path.read_text())
        # Never import worker-supplied paths or change the accepted specification.
        return cast(
            TaskResult, {key: report[key] for key in RESULT_FIELDS if key in report}
        )
    if state in TERMINAL_SLURM_STATES:
        return TaskResult(
            status=TaskStatus.CANCELED
            if state == SlurmState.CANCELLED
            else TaskStatus.FAILED,
            ended=now(),
            error=f"Slurm {state}; no completed attempt result",
        )
    started_path = directory / SLURM_STARTED_FILE
    if started_path.is_file():
        return TaskResult(
            status=TaskStatus.RUNNING,
            started=json.loads(started_path.read_text())["started"],
        )
    return {}


class Scheduler:
    def __init__(self, manager: "Manager") -> None:
        self.manager = manager
        self.last_poll = -float("inf")

    def work(self) -> None:
        while not self.manager.stop.is_set():
            try:
                if not self.manager.settings["paused"]:
                    self.submit_waiting()
                self.poll()
            except (OSError, ValueError, subprocess.SubprocessError):
                logging.getLogger(APPLICATION_LOGGER_NAME).exception(
                    "Slurm dispatch failed"
                )
            self.manager.stop.wait(DISPATCH_POLL_SECONDS)

    def waiting_tasks(self) -> list[TaskRecord]:
        waiting: list[TaskRecord] = []
        for task in self.manager.store.queued():
            if task.get("target") != ExecutionMode.SLURM or task.get("scheduler"):
                continue
            batch = (
                self.manager.store.batch(task["batch_id"])
                if task.get("batch_id")
                else None
            )
            if not batch or not batch.get("accepting"):
                waiting.append(task)
        return waiting

    def submit_waiting(self) -> None:
        waiting = self.waiting_tasks()
        submissions = 0
        while (
            waiting
            and submissions < MAX_SUBMISSIONS_PER_CYCLE
            and not self.manager.stop.is_set()
        ):
            first, *waiting = waiting
            group = [first]
            if first["kind"] == TaskKind.ESTIMATE_RAM and first.get("batch_id"):
                group += [
                    task
                    for task in waiting
                    if task.get("batch_id") == first["batch_id"]
                ]
                waiting = [task for task in waiting if task not in group]
            try:
                self.submit(group)
                submissions += 1
            except (OSError, ValueError) as error:
                # Fail setup errors before submission, rather than endlessly
                # retrying a missing runtime or unwritable shared directory.
                for task in group:
                    if not self.manager.store.get(task["id"]).get("scheduler"):
                        self.manager.store.update(
                            task["id"],
                            status=TaskStatus.FAILED,
                            ended=now(),
                            error=str(error),
                        )

    def write_worker_files(self, tasks: list[TaskRecord], directory: Path) -> None:
        config = self.manager.config
        executable = (
            self.manager.executable(config.apptainer)
            if any(task["runtime"] == Runtime.APPTAINER for task in tasks)
            else config.apptainer
        )
        manifest = WorkerManifest(
            tasks=tasks,
            config={
                "artifacts": str(config.artifacts),
                "pseudos": str(config.pseudos),
                "apptainer": executable,
                "image": str(config.image) if config.image else None,
            },
        )
        atomic_json(directory / MANIFEST_FILE, manifest)
        script = (
            "#!/bin/bash\nset -euo pipefail\n"
            f"export {COMPUTE_WORKER_ENV}=1 {EXECUTION_ENV}={ExecutionMode.SLURM}\n"
            f"cd {shlex.quote(str(ROOT))}\n"
            f"exec {shlex.quote(sys.executable)} -m dashboard.slurm_worker {shlex.quote(str(directory / MANIFEST_FILE))}\n"
        )
        (directory / SLURM_SCRIPT_FILE).write_text(script)

    def claim_submission(
        self, tasks: list[TaskRecord], record: SchedulerRecord
    ) -> bool:
        manager = self.manager
        with manager.guard:
            if manager.settings["paused"] or manager.stop.is_set():
                return False
            for task in tasks:
                current = manager.store.get(task["id"])
                if current["status"] != TaskStatus.QUEUED or current.get("scheduler"):
                    return False
            # Commit intent before invoking sbatch; a crash must never cause
            # automatic resubmission of a job that Slurm may have accepted.
            for task in tasks:
                artifacts = dict(task["artifacts"])
                artifacts[SLURM_LOG_FILE] = manager.store.register(
                    task["id"] + "-" + SLURM_LOG_FILE,
                    Path(record["directory"]) / SLURM_LOG_FILE,
                )
                manager.store.update(task["id"], scheduler=record, artifacts=artifacts)
        return True

    def submit(self, tasks: list[TaskRecord]) -> None:
        token = uuid.uuid4().hex
        directory = self.manager.config.artifacts / (SLURM_DIRECTORY_PREFIX + token)
        directory.mkdir()
        self.write_worker_files(tasks, directory)
        arguments = build_submission(tasks, token, directory)
        record = SchedulerRecord(
            token=token,
            job_id=None,
            state=SlurmState.SUBMITTING,
            reason="",
            directory=str(directory),
            submitted=now(),
            command=arguments,
        )
        if not self.claim_submission(tasks, record):
            return
        try:
            result = command(arguments)
            match = re.fullmatch(r"(\d+)(?:;[A-Za-z0-9_.-]+)?", result.stdout.strip())
            if result.returncode or match is None:
                raise ValueError(result.stderr.strip() or "Missing Slurm job ID")
            record.update(job_id=match[1], state=SlurmState.PENDING)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            record.update(state=SlurmState.SUBMISSION_UNKNOWN, reason=str(error))
        for task in tasks:
            self.manager.store.update(task["id"], scheduler=record)

    def active_tasks(self) -> list[TaskRecord]:
        return [
            task
            for task in self.manager.store.tasks()
            if task.get("target") == ExecutionMode.SLURM
            and task["status"] in ACTIVE_TASK_STATUSES
            and task.get("scheduler")
        ]

    def import_attempt(self, task: TaskRecord, record: SchedulerRecord) -> None:
        manager = self.manager
        directory = manager.config.artifacts / task["id"]
        result = read_result(directory, record["state"])
        artifacts = dict(task["artifacts"])
        for name in RESULT_ARTIFACTS:
            if (directory / name).is_file():
                artifacts[name] = manager.store.register(
                    task["id"] + "-" + name, directory / name
                )
        manager.store.update(
            task["id"], **result, scheduler=record, artifacts=artifacts
        )

    def poll(self, force: bool = False) -> None:
        if not force and time.monotonic() - self.last_poll < SLURM_POLL_SECONDS:
            return
        self.last_poll = time.monotonic()
        tasks = self.active_tasks()
        if not tasks:
            return
        try:
            states, names = query_observations(tasks)
            canceled_jobs: set[str] = set()
            for task in tasks:
                record = reconcile_record(task["scheduler"], states, names)
                self.import_attempt(task, record)
                job_id = record["job_id"]
                if (
                    task.get("stop_requested")
                    and job_id is not None
                    and record["state"] not in TERMINAL_SLURM_STATES
                    and job_id not in canceled_jobs
                ):
                    command([SCANCEL_COMMAND, job_id])
                    canceled_jobs.add(job_id)
        except (OSError, ValueError, subprocess.SubprocessError):
            logging.getLogger(APPLICATION_LOGGER_NAME).exception(
                "Slurm monitoring unavailable; retaining last known state"
            )

    def cancel(self, task: TaskRecord) -> None:
        token = task["scheduler"]["token"]
        # Estimation attempts share an allocation. Stop all unfinished checks,
        # preserving the results of checks that have already completed.
        for item in self.manager.store.tasks():
            if (
                item.get("scheduler", {}).get("token") == token
                and item["status"] in ACTIVE_TASK_STATUSES
            ):
                self.manager.store.update(item["id"], stop_requested=True)
        job_id = task["scheduler"]["job_id"]
        if job_id is not None:
            result = command([SCANCEL_COMMAND, job_id])
            if result.returncode:
                raise ValueError("Slurm cancellation failed: " + result.stderr)

    def mark_rejected(self, task: TaskRecord) -> None:
        """Allow an explicit retry after the user confirms submission rejection."""
        record = task.get("scheduler")
        if (
            not record
            or record["job_id"] is not None
            or record["state"] != SlurmState.SUBMISSION_UNKNOWN
        ):
            raise ValueError("Only unresolved submissions can be marked rejected")
        _, names = query_observations([task])
        if names.get(SLURM_JOB_PREFIX + record["token"]):
            raise ValueError(
                "Slurm has a matching job; wait for monitoring to reconnect"
            )
        # Absence is not proof of rejection: the UI requires the user's
        # confirmation. Never invoke this recovery path automatically.
        for member in self.active_tasks():
            if member["scheduler"]["token"] == record["token"]:
                self.manager.store.update(
                    member["id"],
                    status=TaskStatus.FAILED,
                    ended=now(),
                    error="Submission marked rejected by user; preview and retry",
                )
