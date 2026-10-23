"""Local process lifecycle, bounded log streaming, and memory observations.

Independent Slurm calculations bypass this module and execute directly.
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import IO

import psutil

from dashboard.core import config, execution, models
from dashboard.core.jobs.persistence import now
from dashboard.core.runtimes.cpu import compute_cpus

ProcessResult = tuple[int, str, str]
StopReason = tuple[str, str]
UsageCallback = Callable[[models.MemoryUsage], None]


@dataclass
class OutputStream:
    """The pipe, destination, and incomplete line for one output stream."""

    source: IO[bytes]
    destination: IO[bytes]
    pending: bytearray = field(default_factory=bytearray)


@dataclass
class MemoryState:
    """Observations retained between samples; never a resource reservation."""

    process_peak: int = 0
    cgroup_peak: int = 0
    last_sample: float = 0
    oom_error: str = ""


def process_children(pid: int) -> list[psutil.Process]:
    try:
        return psutil.Process(pid).children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []


def signal_children(children: list[psutil.Process], value: int) -> None:
    for child in children:
        try:
            child.send_signal(value)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass  # A child may exit between enumeration and signaling.


def signal_group(process: subprocess.Popen[bytes], value: int) -> None:
    try:
        os.killpg(process.pid, value)
    except ProcessLookupError:
        pass


def terminate(process: subprocess.Popen[bytes]) -> None:
    # Capture MPI descendants before the launcher exits and loses its children.
    children = process_children(process.pid)
    signal_children(children, signal.SIGTERM)
    signal_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=execution.TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    # Escalate for descendants that ignore graceful termination.
    signal_group(process, signal.SIGKILL)
    signal_children(children, signal.SIGKILL)
    process.wait()


def write_log_headers(directory: Path, stdout: IO[bytes], stderr: IO[bytes]) -> None:
    header = [f"Task: {directory.name}"]
    metadata = directory / execution.ENTRYPOINT_LOG_FILE
    if metadata.is_file():
        header.extend(
            line
            for line in metadata.read_text().splitlines()
            if line.startswith(execution.LOG_METADATA_PREFIXES)
        )
    # Preserve provenance in each downloadable stream, even after a failed run.
    for name, log in (("stdout", stdout), ("stderr", stderr)):
        for line in [*header, f"Stream: {name}"]:
            log.write(f"[{now()}] {line}\n".encode())
        log.flush()


def process_environment(env: dict[str, str] | None) -> dict[str, str]:
    environment = dict(env or os.environ)
    environment.update({name: "1" for name in execution.THREAD_ENV_VARS})
    # Child app entry points must resolve the package from an attempt's cwd.
    environment[execution.PYTHON_UNBUFFERED_ENV] = "1"
    environment[execution.PYTHON_PATH_ENV] = str(config.ROOT)
    return environment


def launch_process(
    command: list[str], directory: Path, env: dict[str, str] | None
) -> subprocess.Popen[bytes]:
    launch = [
        sys.executable,
        "-m",
        execution.CPU_APP,
        ",".join(map(str, compute_cpus())),
        *command,
    ]
    return subprocess.Popen(
        launch,
        cwd=directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env=process_environment(env),
    )


def output_streams(
    process: subprocess.Popen[bytes], stdout: IO[bytes], stderr: IO[bytes]
) -> list[OutputStream]:
    assert process.stdout is not None and process.stderr is not None
    streams = [
        OutputStream(process.stdout, stdout),
        OutputStream(process.stderr, stderr),
    ]
    for stream in streams:
        os.set_blocking(stream.source.fileno(), False)
    return streams


def write_complete_lines(stream: OutputStream) -> None:
    while (
        b"\n" in stream.pending or len(stream.pending) >= execution.STREAM_CHUNK_BYTES
    ):
        newline = stream.pending.find(b"\n")
        size = newline + 1 if newline >= 0 else execution.STREAM_CHUNK_BYTES
        stream.destination.write(f"[{now()}] ".encode() + bytes(stream.pending[:size]))
        del stream.pending[:size]


def read_available_output(stream: OutputStream) -> None:
    # A noisy process must not monopolize the cancellation/timeout loop.
    for _ in range(execution.STREAM_READS_PER_POLL):
        try:
            chunk = os.read(stream.source.fileno(), execution.STREAM_CHUNK_BYTES)
        except BlockingIOError:
            return
        if not chunk:
            return
        stream.pending.extend(chunk)
        write_complete_lines(stream)


def drain_output(streams: list[OutputStream], final: bool = False) -> None:
    for stream in streams:
        read_available_output(stream)
        # At shutdown retain a partial final line and close the pipe.
        if final:
            if stream.pending:
                stream.destination.write(
                    f"[{now()}] ".encode() + bytes(stream.pending) + b"\n"
                )
                stream.pending.clear()
            stream.source.close()
        stream.destination.flush()


def process_tree_rss(pid: int) -> int:
    total = 0
    try:
        processes = [psutil.Process(pid), *process_children(pid)]
        for process in processes:
            try:
                total += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return total


def read_container_report(directory: Path) -> tuple[dict, Path]:
    report = json.loads((directory / execution.RESOURCE_REPORT_FILE).read_text())
    path = Path(report["cgroup"]).resolve()
    if not path.is_relative_to(execution.CGROUP_ROOT) or not report["verified"]:
        raise ValueError("Invalid resource metadata")
    return report, path


def cgroup_oom_killed(report: dict, path: Path) -> bool:
    events = (
        (path / execution.CGROUP_EVENTS_FILE).read_text()
        if path.exists()
        else report.get("events_after", "")
    )
    before = dict(line.split() for line in report["events_before"].splitlines())
    after = dict(line.split() for line in events.splitlines())
    counter = execution.CGROUP_OOM_COUNTER
    return int(after.get(counter, 0)) > int(before.get(counter, 0))


def container_usage(
    directory: Path, memory: MemoryState, fallback: models.MemoryUsage
) -> models.MemoryUsage:
    try:
        report, path = read_container_report(directory)
        if cgroup_oom_killed(report, path):
            memory.oom_error = execution.CGROUP_OOM_ERROR
        # The cgroup may have disappeared after exit; retained metadata survives.
        current, peak = cgroup_memory(path)
        memory.cgroup_peak = max(
            memory.cgroup_peak, peak, report.get("peak_memory_bytes", 0)
        )
        return models.MemoryUsage(
            current_memory_bytes=current,
            peak_memory_bytes=memory.cgroup_peak,
            measurement=execution.CGROUP_MEASUREMENT,
        )
    except (OSError, ValueError, KeyError):
        return fallback  # The supervisor may not have published metadata yet.


def cgroup_memory(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    current = int((path / execution.CGROUP_CURRENT_FILE).read_text())
    peak = int((path / execution.CGROUP_PEAK_FILE).read_text())
    return current, peak


def sample_usage(
    process: subprocess.Popen[bytes],
    directory: Path,
    memory: MemoryState,
    container: bool,
    on_usage: UsageCallback,
) -> None:
    current = process_tree_rss(process.pid)
    memory.process_peak = max(memory.process_peak, current)
    usage = models.MemoryUsage(
        current_memory_bytes=current,
        peak_memory_bytes=memory.process_peak,
        measurement=execution.PROCESS_RSS_MEASUREMENT,
    )
    if container:
        usage = container_usage(directory, memory, usage)
    memory.last_sample = time.monotonic()
    on_usage(usage)


def stop_reason(
    memory: MemoryState, cancel: Event, deadline: float | None
) -> StopReason | None:
    if memory.oom_error:
        return execution.TaskStatus.FAILED, memory.oom_error
    if cancel.is_set():
        return execution.TaskStatus.CANCELED, "Stopped"
    if deadline is not None and time.monotonic() > deadline:
        return execution.TaskStatus.FAILED, "Timed out"
    return None


def monitor_process(
    process: subprocess.Popen[bytes],
    streams: list[OutputStream],
    directory: Path,
    cancel: Event,
    deadline: float | None,
    memory: MemoryState,
    container: bool,
    on_usage: UsageCallback,
) -> StopReason | None:
    # Drain first, sample periodically, and check Stop between bounded reads.
    while process.poll() is None:
        drain_output(streams)
        if time.monotonic() - memory.last_sample > execution.MEMORY_SAMPLE_SECONDS:
            sample_usage(process, directory, memory, container, on_usage)
        reason = stop_reason(memory, cancel, deadline)
        if reason is not None:
            terminate(process)
            return reason
        time.sleep(execution.PROCESS_POLL_SECONDS)
    return None


def process_result(
    process: subprocess.Popen[bytes], memory: MemoryState, reason: StopReason | None
) -> ProcessResult:
    code = process.wait()
    if memory.oom_error:
        return code, execution.TaskStatus.FAILED, memory.oom_error
    if reason is not None:
        return code, *reason
    status = (
        execution.TaskStatus.SUCCEEDED if code == 0 else execution.TaskStatus.FAILED
    )
    return code, status, ""


@contextmanager
def log_files(directory: Path) -> Iterator[tuple[IO[bytes], IO[bytes]]]:
    with (
        (directory / execution.STDOUT_LOG_FILE).open("wb") as stdout,
        (directory / execution.STDERR_LOG_FILE).open("wb") as stderr,
    ):
        yield stdout, stderr


def close_output(streams: list[OutputStream]) -> None:
    for stream in streams:
        if not stream.source.closed:
            drain_output([stream], final=True)


def observe_process(
    process: subprocess.Popen[bytes],
    streams: list[OutputStream],
    directory: Path,
    cancel: Event,
    deadline: float | None,
    container: bool,
    on_start: Callable[[subprocess.Popen[bytes]], object],
    on_usage: UsageCallback,
) -> ProcessResult:
    memory = MemoryState()
    try:
        on_start(process)
        reason = monitor_process(
            process, streams, directory, cancel, deadline, memory, container, on_usage
        )
        # Retain trailing output and final accounting after normal exit or Stop.
        drain_output(streams, final=True)
        sample_usage(process, directory, memory, container, on_usage)
        return process_result(process, memory, reason)
    finally:
        if process.poll() is None:
            terminate(process)  # Callback failures must not leak a live job.
        close_output(streams)


def execute_logged(
    command: list[str],
    directory: Path,
    cancel: Event,
    timeout: float | None,
    on_start: Callable[[subprocess.Popen[bytes]], object],
    env: dict[str, str] | None,
    on_usage: UsageCallback,
    container: bool,
    logs: tuple[IO[bytes], IO[bytes]],
) -> ProcessResult:
    deadline = time.monotonic() + timeout if timeout else None
    write_log_headers(directory, *logs)
    process = launch_process(command, directory, env)
    streams = output_streams(process, *logs)
    return observe_process(
        process, streams, directory, cancel, deadline, container, on_start, on_usage
    )


def execute(
    command: list[str],
    directory: Path,
    cancel: Event,
    timeout: float | None = None,
    on_start: Callable[[subprocess.Popen[bytes]], object] = lambda process: None,
    env: dict[str, str] | None = None,
    on_usage: UsageCallback = lambda usage: None,
    container: bool = False,
) -> ProcessResult:
    """Keep the existing callback API; only manage logs and delegate local execution.

    The signature is retained for existing callers; lifecycle steps live in
    focused helpers rather than being nested here. Slurm jobs bypass this path.
    """
    with log_files(directory) as logs:
        return execute_logged(
            command,
            directory,
            cancel,
            timeout,
            on_start,
            env,
            on_usage,
            container,
            logs,
        )


def last_energy(text: str) -> float | None:
    energies = re.findall(r"!\s+total energy\s*=\s*([-+\d.EeDd]+)\s+Ry", text)
    return float(energies[-1].replace("D", "E").replace("d", "e")) if energies else None


def evidence(text: str, calculation: str = "scf") -> models.Evidence:
    completed = "JOB DONE." in text
    scf = "convergence has been achieved" in text
    relax = (
        "End of BFGS Geometry Optimization" in text or "bfgs converged" in text.lower()
    )
    failed = "convergence NOT achieved" in text or "Error in routine" in text
    # Return raw markers alongside confirmation so the UI can explain failures.
    confirmed = (
        completed
        and scf
        and not failed
        and (calculation not in ("relax", "vc-relax") or relax)
    )
    return {
        "job_done": completed,
        "scf_converged": scf,
        "relaxation_completed": relax,
        "confirmed": confirmed,
        "energy_ry": last_energy(text),
        "provisional": not confirmed,
    }


def tail(path: Path, limit: int = 16000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read(limit).decode(errors="replace")
