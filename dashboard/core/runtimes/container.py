"""Small container supervisor: verify enforcement before executing any job."""

import json
import os
import subprocess
import sys
from pathlib import Path


def cgroup() -> Path:
    relative = Path("/proc/self/cgroup").read_text().split("0::", 1)[1].strip()
    return Path("/sys/fs/cgroup") / relative.lstrip("/")


def limits(limit: int) -> dict:
    path = cgroup()
    maximum = (path / "memory.max").read_text().strip()
    swap = (path / "memory.swap.max").read_text().strip()
    return {
        "verified": maximum != "max" and 0 < int(maximum) <= limit and swap == "0",
        "cgroup": str(path),
    }


def write_report(metadata: Path, report: dict) -> None:
    temporary = metadata.with_suffix(".tmp")
    temporary.write_text(json.dumps(report))
    temporary.replace(metadata)


def supervise(command: list[str], metadata: Path, report: dict) -> int:
    path = Path(report["cgroup"])
    report.update(
        events_before=(path / "memory.events").read_text(), host_pid=os.getpid()
    )
    write_report(metadata, report)
    # Retain before/after OOM counters even if the cgroup is gone at observation.
    command = command[1:] if command[0] == "--" else command
    code = subprocess.call(command)
    report.update(
        events_after=(path / "memory.events").read_text(),
        peak_memory_bytes=int((path / "memory.peak").read_text()),
    )
    write_report(metadata, report)
    return code if code >= 0 else 128 - code


def main(
    *,
    probe: int | None = None,
    limit: int | None = None,
    metadata: Path | None = None,
    command: list[str],
) -> int:
    effective_limit = probe or limit
    if effective_limit is None:
        raise ValueError("A probe or memory limit is required")
    report = limits(effective_limit)
    if probe:
        print(json.dumps(report))
        return 0 if report["verified"] else 1
    if not report["verified"]:
        print("Hard memory isolation unavailable; job was not started", file=sys.stderr)
        return 1
    if metadata is None:
        raise ValueError("Resource metadata path is required")
    return supervise(command, metadata, report)
