"""Execute and validate one QE calculation without accessing persistent state."""

import itertools
import json
import re
import shlex
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from .config import Config
from .execution import (
    ENTRYPOINT_LOG_FILE,
    ERROR_TAIL_CHARACTERS,
    INPUT_FILE,
    QE_DEFAULT_CALCULATION,
    QE_ENERGY_MARKER,
    QE_EVIDENCE_MARKERS,
    QE_INITIALIZATION_EXIT_CODES,
    QE_MPI_PROCESS_PATTERNS,
    STDERR_LOG_FILE,
    STDOUT_LOG_FILE,
    ExecutionMode,
    Runtime,
    TaskKind,
    TaskStatus,
)
from .execution_models import MemoryUsage, TaskRecord, TaskResult
from .inputs import ram_reports
from .persistence import now
from .processes import evidence, tail
from .runtimes import Runtimes

ProcessResult = tuple[int, str, str]
UsageCallback = Callable[[MemoryUsage], None]


class ProcessExecutor(Protocol):
    def __call__(
        self,
        command: list[str],
        directory: Path,
        cancel: threading.Event,
        timeout: float | None,
        *,
        on_usage: UsageCallback,
        container: bool,
    ) -> ProcessResult: ...


def write_entrypoint(task: TaskRecord, directory: Path, command: list[str]) -> None:
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
    (directory / ENTRYPOINT_LOG_FILE).write_text("\n".join(lines) + "\n")


def observed_mpi_processes(directory: Path) -> int | None:
    with (directory / STDOUT_LOG_FILE).open(errors="replace") as stream:
        for line in stream:
            for pattern in QE_MPI_PROCESS_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    return int(match[1])
    return None


def validate_estimate(task: TaskRecord, directory: Path, result: TaskResult) -> None:
    with (
        (directory / STDOUT_LOG_FILE).open(errors="replace") as stdout,
        (directory / STDERR_LOG_FILE).open(errors="replace") as stderr,
    ):
        report, qe_error = ram_reports(itertools.chain(stdout, stderr))
    result["ram_estimate"] = report
    # QE initialization-only mode deliberately returns 255 in some versions.
    if (
        not result["error"]
        and result["status"] != TaskStatus.CANCELED
        and result["exit_code"] in QE_INITIALIZATION_EXIT_CODES
    ):
        if qe_error:
            result.update(
                status=TaskStatus.FAILED,
                error="QE reported an error during RAM estimation",
            )
        elif report["per_process"] is None:
            result.update(
                status=TaskStatus.FAILED,
                error="QE did not report a per-process RAM estimate",
            )
        elif (
            task.get("target") == ExecutionMode.SLURM
            and task["processes"] > 1
            and observed_mpi_processes(directory) != task["processes"]
        ):
            result.update(
                status=TaskStatus.FAILED,
                error="QE MPI process count did not match the requested RAM estimate",
            )
        else:
            result.update(status=TaskStatus.SUCCEEDED, error="")
    if result["status"] == TaskStatus.FAILED and not result["error"]:
        result["error"] = f"Process exited with code {result['exit_code']}"


def validate_qe(task: TaskRecord, directory: Path, result: TaskResult) -> None:
    markers = []
    with (directory / STDOUT_LOG_FILE).open(errors="replace") as stream:
        for line in stream:
            if any(marker in line for marker in QE_EVIDENCE_MARKERS):
                if QE_ENERGY_MARKER in line:
                    markers = [
                        marker for marker in markers if QE_ENERGY_MARKER not in marker
                    ]
                markers.append(line)
    calculation = re.search(
        r"calculation\s*=\s*['\"]([^'\"]+)",
        (directory / INPUT_FILE).read_text(),
        re.IGNORECASE,
    )
    result["evidence"] = evidence(
        "".join(markers), calculation[1] if calculation else QE_DEFAULT_CALCULATION
    )
    if (
        task.get("target") == ExecutionMode.SLURM
        and result["status"] == TaskStatus.SUCCEEDED
        and not result["evidence"]["confirmed"]
    ):
        result.update(
            status=TaskStatus.FAILED,
            error="QE completion or convergence was not confirmed",
        )


def run_calculation(
    task: TaskRecord,
    config: Config,
    runtimes: Runtimes,
    cancel: threading.Event,
    executor: ProcessExecutor,
    on_usage: UsageCallback | None = None,
) -> TaskResult:
    directory = config.artifacts / task["id"]
    command = list(task["command"])
    result: TaskResult = {}

    def record_usage(usage: MemoryUsage) -> None:
        result["usage"] = usage
        if on_usage is not None:
            on_usage(usage)

    try:
        if task["runtime"] == Runtime.APPTAINER:
            command = runtimes.wrap(
                command,
                directory,
                task["resources"],
                task["image_hash"],
                scheduler=task.get("target") == ExecutionMode.SLURM,
            )
        write_entrypoint(task, directory, command)
        code, status, error = executor(
            command,
            directory,
            cancel,
            task["timeout"],
            on_usage=record_usage,
            container=task["runtime"] == Runtime.APPTAINER
            and not config.allocation
            and task.get("target") != ExecutionMode.SLURM,
        )
        if (
            status == TaskStatus.FAILED
            and not error
            and task["kind"] != TaskKind.ESTIMATE_RAM
        ):
            detail = tail(
                directory / STDERR_LOG_FILE, limit=ERROR_TAIL_CHARACTERS
            ).strip()
            error = f"Process exited with code {code}" + (
                f": {detail}" if detail else ""
            )
        result.update(
            status=status, exit_code=code, error=error, ended=now(), command=command
        )
        if task["kind"] == TaskKind.ESTIMATE_RAM:
            validate_estimate(task, directory, result)
        else:
            validate_qe(task, directory, result)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        result.update(
            status=TaskStatus.FAILED, error=str(error), ended=now(), command=command
        )
    return result
