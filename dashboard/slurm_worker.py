"""Run frozen attempts sequentially; never open the dashboard database."""

import json
import logging
import sys
import threading
from pathlib import Path

from .calculation import run_calculation
from .config import Config
from .execution import (
    APPLICATION_LOGGER_NAME,
    INPUT_FILE,
    PSEUDOS_DIRECTORY,
    SLURM_RESULT_FILE,
    SLURM_STARTED_FILE,
    ExecutionMode,
    TaskStatus,
)
from .execution_models import TaskRecord, TaskResult, WorkerManifest
from .persistence import now
from .processes import execute
from .runtimes import Runtimes, file_hash
from .scheduler import atomic_json


def validate_snapshot(task: TaskRecord, directory: Path) -> None:
    if file_hash(directory / INPUT_FILE) != task["input_hash"]:
        raise ValueError("Captured input changed after submission")
    for name, expected_hash in task["pseudopotentials"].items():
        if file_hash(directory / PSEUDOS_DIRECTORY / name) != expected_hash:
            raise ValueError("Captured pseudopotential changed after submission")


def run(manifest: WorkerManifest) -> None:
    stored_config = manifest["config"]
    config = Config(
        artifacts=Path(stored_config["artifacts"]),
        pseudos=Path(stored_config["pseudos"]),
        apptainer=stored_config["apptainer"],
        image=Path(stored_config["image"]) if stored_config["image"] else None,
        execution=ExecutionMode.SLURM,
    )
    runtimes = Runtimes(config)
    for task in manifest["tasks"]:
        directory = config.artifacts / task["id"]
        started = now()
        atomic_json(directory / SLURM_STARTED_FILE, {"started": started})
        result: TaskResult
        try:
            validate_snapshot(task, directory)
            result = run_calculation(task, config, runtimes, threading.Event(), execute)
        except Exception as error:
            logging.getLogger(APPLICATION_LOGGER_NAME).exception(
                "Slurm attempt could not execute"
            )
            result = {
                "status": TaskStatus.FAILED,
                "error": str(error),
                "ended": now(),
            }
        result["started"] = started
        atomic_json(directory / SLURM_RESULT_FILE, result)


def main() -> None:
    manifest_path = Path(sys.argv[1])
    run(json.loads(manifest_path.read_text()))


if __name__ == "__main__":
    main()
