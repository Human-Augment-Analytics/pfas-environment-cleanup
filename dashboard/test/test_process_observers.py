"""Observer failures must never leave local jobs or output pipes alive."""

import subprocess
import sys
from pathlib import Path
from threading import Event

import pytest

from dashboard.core import models
from dashboard.core.jobs.processes import execute


@pytest.mark.parametrize("failure", ["start", "usage"])
def test_observer_failure_cleans_up_process_and_pipes(
    tmp_path: Path, failure: str
) -> None:
    captured: list[subprocess.Popen[bytes]] = []

    def started(process: subprocess.Popen[bytes]) -> None:
        captured.append(process)
        if failure == "start":
            raise RuntimeError("observer failed")

    def usage(value: models.MemoryUsage) -> None:
        if failure == "usage":
            raise RuntimeError("observer failed")

    # The process must outlive its observer unless execute performs cleanup.
    command = [sys.executable, "-c", "import time; time.sleep(10)"]
    with pytest.raises(RuntimeError, match="observer failed"):
        execute(command, tmp_path, Event(), on_start=started, on_usage=usage)
    process = captured[0]
    assert process.poll() is not None
    assert process.stdout is not None and process.stdout.closed
    assert process.stderr is not None and process.stderr.closed
