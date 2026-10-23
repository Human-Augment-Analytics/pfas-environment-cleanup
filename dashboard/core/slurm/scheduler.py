"""Independent single-node Slurm jobs, reconciled through shared artifacts."""

import json
import logging
import math
import os
import re
import subprocess
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from dashboard.core import config as configuration
from dashboard.core import execution, models
from dashboard.core.chemistry.results import read_calculation_result
from dashboard.core.jobs.persistence import now
from dashboard.core.slurm.export import write_direct_job

if TYPE_CHECKING:
    from dashboard.core.jobs.tasks import Manager

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
    execution.ENTRYPOINT_LOG_FILE,
    execution.STDOUT_LOG_FILE,
    execution.STDERR_LOG_FILE,
    execution.INPUT_FILE,
    execution.SLURM_RESULT_FILE,
    execution.SLURM_FINISHED_FILE,
    execution.SNAPSHOT_CHECK_LOG,
)
JobStates = dict[str, tuple[str, str]]
JobNames = dict[str, set[str]]


def scheduler_name(value: object) -> str:
    if not isinstance(value, str) or (
        value and not re.fullmatch(r"[A-Za-z0-9_.-]+", value)
    ):
        raise ValueError("Invalid partition, account, or QOS")
    return value


def walltime_hours(value: object) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= execution.MAX_SLURM_HOURS
    ):
        raise ValueError(
            f"Walltime must be between 1 and {execution.MAX_SLURM_HOURS} hours"
        )
    return value


def memory_mode(value: object, kind: str) -> Literal["manual", "estimate"]:
    if value not in ("manual", "estimate"):
        raise ValueError("Unknown memory mode")
    if kind == execution.TaskKind.ESTIMATE_RAM and value != "manual":
        raise ValueError("RAM estimation requires an explicit memory reservation")
    return cast(Literal["manual", "estimate"], value)


def memory_headroom(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or not 0 <= value <= MAX_HEADROOM_PERCENT
    ):
        raise ValueError(
            f"Headroom must be between 0 and {MAX_HEADROOM_PERCENT} percent"
        )
    return float(value)


def settings(
    values: Mapping[str, object] | None = None, kind: str = execution.TaskKind.QE
) -> models.SchedulerSettings:
    overrides = values or {}
    if set(overrides) - set(SETTINGS_KEYS):
        raise ValueError("Unknown scheduler setting")
    default_hours = (
        execution.DEFAULT_ESTIMATION_HOURS
        if kind == execution.TaskKind.ESTIMATE_RAM
        else execution.DEFAULT_QE_HOURS
    )
    # Normalize the complete scheduler record after validating each field.
    return models.SchedulerSettings(
        partition=scheduler_name(
            overrides.get("partition", execution.DEFAULT_PARTITION)
        ),
        account=scheduler_name(overrides.get("account", execution.DEFAULT_ACCOUNT)),
        qos=scheduler_name(overrides.get("qos", execution.DEFAULT_QOS)),
        walltime_hours=walltime_hours(overrides.get("walltime_hours", default_hours)),
        memory_mode=memory_mode(overrides.get("memory_mode", "manual"), kind),
        headroom_percent=memory_headroom(
            overrides.get("headroom_percent", execution.DEFAULT_HEADROOM_PERCENT)
        ),
    )


def command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    # Preserve module/runtime paths, but never inherit a parent's allocation,
    # MPI session, or SBATCH overrides into a newly requested allocation.
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(execution.PARENT_JOB_ENV_PREFIXES)
        or name in execution.SLURM_CONFIG_ENV
    }
    return subprocess.run(
        arguments,
        capture_output=True,
        text=True,
        timeout=execution.SLURM_COMMAND_TIMEOUT,
        check=False,
        env=environment,
    )


def build_submission(
    tasks: list[models.TaskRecord], token: str, directory: Path
) -> list[str]:
    options = tasks[0]["scheduler_settings"]
    processes = max(task["processes"] for task in tasks)
    memory_bytes = max(task["resources"]["memory_bytes"] for task in tasks)
    arguments = [
        execution.SBATCH_COMMAND,
        "--parsable",
        "--export=ALL",
        "--no-requeue",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={processes}",
        f"--mem={math.ceil(memory_bytes / execution.MIB)}M",
        f"--time={options['walltime_hours']}:00:00",
        f"--job-name={execution.SLURM_JOB_PREFIX}{token}",
        f"--comment={execution.SLURM_JOB_PREFIX}{token}",
        f"--output={directory / execution.SLURM_LOG_FILE}",
        f"--chdir={configuration.ROOT}",
    ]
    for attribute in SCHEDULING_ATTRIBUTES:
        value = options[attribute]
        if value:
            arguments.append(f"--{attribute}={value}")
    return [*arguments, str(directory / execution.SLURM_SCRIPT_FILE)]


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


def query_observations(tasks: list[models.TaskRecord]) -> tuple[JobStates, JobNames]:
    owner = str(os.getuid())
    live = command(
        [execution.SQUEUE_COMMAND, "--noheader", "--user", owner, SQUEUE_FORMAT]
    )
    job_ids = sorted(
        {
            task["scheduler"]["job_id"]
            for task in tasks
            if task["scheduler"]["job_id"] is not None
        }
    )
    arguments = [
        execution.SACCT_COMMAND,
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
    stored: models.SchedulerRecord, states: JobStates, names: JobNames
) -> models.SchedulerRecord:
    record = stored.copy()
    if record["job_id"] is None:
        matches = names.get(execution.SLURM_JOB_PREFIX + record["token"], set())
        if len(matches) == 1:
            record["job_id"] = next(iter(matches))
        else:
            message = (
                "Submission outcome unresolved; no automatic retry. Check Slurm using job name "
                + execution.SLURM_JOB_PREFIX
                + record["token"]
            )
            if record["reason"] and message not in record["reason"]:
                message = record["reason"] + "; " + message
            elif message in record["reason"]:
                message = record["reason"]
            record.update(
                state=execution.SlurmState.SUBMISSION_UNKNOWN,
                reason=message,
            )
    job_id = record["job_id"]
    if job_id is not None and job_id in states:
        record["state"], record["reason"] = states[job_id]
    return record


def finished_error(code: int, directory: Path) -> str:
    if code == execution.TIMEOUT_EXIT_CODE:
        return "Timed out"
    if code == execution.SNAPSHOT_CHECK_FAILED:
        details = (directory / execution.SNAPSHOT_CHECK_LOG).read_text()[
            -execution.ERROR_TAIL_CHARACTERS :
        ]
        return (
            "Captured input or pseudopotential failed checksum verification: " + details
        )
    return ""


def finished_result(task: models.TaskRecord, directory: Path) -> models.TaskResult:
    code_text, ended = (
        (directory / execution.SLURM_FINISHED_FILE).read_text().splitlines()
    )
    code = int(code_text)
    status = (
        execution.TaskStatus.SUCCEEDED if code == 0 else execution.TaskStatus.FAILED
    )
    result = read_calculation_result(
        task, directory, code, status, finished_error(code, directory)
    )
    result["ended"] = ended
    result["started"] = (
        (directory / execution.SLURM_STARTED_TEXT_FILE).read_text().strip()
    )
    return result


def incomplete_result(directory: Path, state: str) -> models.TaskResult:
    if state in execution.TERMINAL_SLURM_STATES:
        status = (
            execution.TaskStatus.CANCELED
            if state == execution.SlurmState.CANCELLED
            else execution.TaskStatus.FAILED
        )
        return models.TaskResult(
            status=status,
            ended=now(),
            error=f"Slurm {state}; no completed attempt result",
        )
    started = directory / execution.SLURM_STARTED_TEXT_FILE
    if started.is_file():
        return models.TaskResult(
            status=execution.TaskStatus.RUNNING, started=started.read_text().strip()
        )
    legacy = directory / execution.SLURM_STARTED_FILE
    if legacy.is_file():
        return models.TaskResult(
            status=execution.TaskStatus.RUNNING,
            started=json.loads(legacy.read_text())["started"],
        )
    return {}


def read_result(
    task: models.TaskRecord, directory: Path, state: str
) -> models.TaskResult:
    # Old completed worker results remain readable during the transition.
    legacy = directory / execution.SLURM_RESULT_FILE
    if legacy.is_file():
        report = json.loads(legacy.read_text())
        return cast(
            models.TaskResult,
            {key: report[key] for key in RESULT_FIELDS if key in report},
        )
    if (directory / execution.SLURM_FINISHED_FILE).is_file():
        return finished_result(task, directory)
    return incomplete_result(directory, state)


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
                logging.getLogger(execution.APPLICATION_LOGGER_NAME).exception(
                    "Slurm dispatch failed"
                )
            self.manager.stop.wait(execution.DISPATCH_POLL_SECONDS)

    def waiting_tasks(self) -> list[models.TaskRecord]:
        waiting: list[models.TaskRecord] = []
        for task in self.manager.store.queued():
            if task.get("target") != execution.ExecutionMode.SLURM or task.get(
                "scheduler"
            ):
                continue
            batch_id = task.get("batch_id")
            batch = self.manager.store.batch(batch_id) if batch_id else None
            if not batch or not batch.get("accepting"):
                waiting.append(task)
        return waiting

    def submit_waiting(self) -> None:
        waiting = self.waiting_tasks()
        submissions = 0
        while (
            waiting
            and submissions < execution.MAX_SUBMISSIONS_PER_CYCLE
            and not self.manager.stop.is_set()
        ):
            first, *waiting = waiting
            group = [first]
            if first["kind"] == execution.TaskKind.ESTIMATE_RAM and first.get(
                "batch_id"
            ):
                group += [
                    task
                    for task in waiting
                    if task.get("batch_id") == first["batch_id"]
                ]
                waiting = [task for task in waiting if task not in group]
            submissions += int(self.submit_group(group))

    def submit_group(self, tasks: list[models.TaskRecord]) -> bool:
        try:
            self.submit(tasks)
            return True
        except (OSError, ValueError) as error:
            # Setup failures have no Slurm job to reconcile or cancel.
            for task in tasks:
                if not self.manager.store.get(task["id"]).get("scheduler"):
                    self.manager.store.update(
                        task["id"],
                        status=execution.TaskStatus.FAILED,
                        ended=now(),
                        error=str(error),
                    )
            return False

    def claim_submission(
        self, tasks: list[models.TaskRecord], record: models.SchedulerRecord
    ) -> bool:
        manager = self.manager
        with manager.guard:
            if manager.settings["paused"] or manager.stop.is_set():
                return False
            for task in tasks:
                current = manager.store.get(task["id"])
                if current["status"] != execution.TaskStatus.QUEUED or current.get(
                    "scheduler"
                ):
                    return False
            # Commit intent before invoking sbatch; a crash must never cause
            # automatic resubmission of a job that Slurm may have accepted.
            for task in tasks:
                artifacts = dict(task["artifacts"])
                artifacts[execution.SLURM_LOG_FILE] = manager.store.register(
                    task["id"] + "-" + execution.SLURM_LOG_FILE,
                    Path(record["directory"]) / execution.SLURM_LOG_FILE,
                )
                manager.store.update(task["id"], scheduler=record, artifacts=artifacts)
        return True

    def submit(self, tasks: list[models.TaskRecord]) -> None:
        token = uuid.uuid4().hex
        directory = self.manager.config.artifacts / (
            execution.SLURM_DIRECTORY_PREFIX + token
        )
        directory.mkdir()
        write_direct_job(tasks, self.manager.config, self.manager.runtimes, directory)
        arguments = build_submission(tasks, token, directory)
        record = models.SchedulerRecord(
            token=token,
            job_id=None,
            state=execution.SlurmState.SUBMITTING,
            reason="",
            directory=str(directory),
            submitted=now(),
            command=arguments,
        )
        if not self.claim_submission(tasks, record):
            return
        self.run_submission(record)
        for task in tasks:
            self.manager.store.update(task["id"], scheduler=record)

    def run_submission(self, record: models.SchedulerRecord) -> None:
        # An ambiguous response stays unresolved until polling finds the job.
        try:
            result = command(record["command"])
            match = re.fullmatch(r"(\d+)(?:;[A-Za-z0-9_.-]+)?", result.stdout.strip())
            if result.returncode or match is None:
                raise ValueError(result.stderr.strip() or "Missing Slurm job ID")
            record.update(job_id=match[1], state=execution.SlurmState.PENDING)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            record.update(
                state=execution.SlurmState.SUBMISSION_UNKNOWN, reason=str(error)
            )

    def active_tasks(self) -> list[models.TaskRecord]:
        return [
            task
            for task in self.manager.store.tasks()
            if task.get("target") == execution.ExecutionMode.SLURM
            and task["status"] in execution.ACTIVE_TASK_STATUSES
            and task.get("scheduler")
        ]

    def import_attempt(
        self, task: models.TaskRecord, record: models.SchedulerRecord
    ) -> None:
        manager = self.manager
        directory = manager.config.artifacts / task["id"]
        result = read_result(task, directory, record["state"])
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
        if (
            not force
            and time.monotonic() - self.last_poll < execution.SLURM_POLL_SECONDS
        ):
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
                    and record["state"] not in execution.TERMINAL_SLURM_STATES
                    and job_id not in canceled_jobs
                ):
                    command([execution.SCANCEL_COMMAND, job_id])
                    canceled_jobs.add(job_id)
        except (OSError, ValueError, subprocess.SubprocessError):
            logging.getLogger(execution.APPLICATION_LOGGER_NAME).exception(
                "Slurm monitoring unavailable; retaining last known state"
            )

    def cancel(self, task: models.TaskRecord) -> None:
        token = task["scheduler"]["token"]
        # Estimation attempts share an allocation. Stop all unfinished checks,
        # preserving the results of checks that have already completed.
        for item in self.manager.store.tasks():
            if (
                item.get("scheduler", {}).get("token") == token
                and item["status"] in execution.ACTIVE_TASK_STATUSES
            ):
                self.manager.store.update(item["id"], stop_requested=True)
        job_id = task["scheduler"]["job_id"]
        if job_id is not None:
            result = command([execution.SCANCEL_COMMAND, job_id])
            if result.returncode:
                raise ValueError("Slurm cancellation failed: " + result.stderr)

    def mark_rejected(self, task: models.TaskRecord) -> None:
        """Allow an explicit retry after the user confirms submission rejection."""
        record = task.get("scheduler")
        if (
            not record
            or record["job_id"] is not None
            or record["state"] != execution.SlurmState.SUBMISSION_UNKNOWN
        ):
            raise ValueError("Only unresolved submissions can be marked rejected")
        _, names = query_observations([task])
        if names.get(execution.SLURM_JOB_PREFIX + record["token"]):
            raise ValueError(
                "Slurm has a matching job; wait for monitoring to reconnect"
            )
        # Absence is not proof of rejection: the UI requires the user's
        # confirmation. Never invoke this recovery path automatically.
        for member in self.active_tasks():
            if member["scheduler"]["token"] == record["token"]:
                self.manager.store.update(
                    member["id"],
                    status=execution.TaskStatus.FAILED,
                    ended=now(),
                    error="Submission marked rejected by user; preview and retry",
                )
