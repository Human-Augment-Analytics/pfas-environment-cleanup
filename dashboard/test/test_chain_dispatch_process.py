"""Run generated compute and dispatcher scripts with a filesystem Slurm double."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

import pytest

from dashboard.core import config as configuration
from dashboard.core import execution, models
from dashboard.core.jobs import persistence
from dashboard.core.jobs.tasks import Manager
from dashboard.core.slurm import chaining
from dashboard.test.test_scheduler import manager as manager_fixture  # noqa: F401
from dashboard.test.test_scheduler import queue

CONTROLLER_ENV = "PFAS_TEST_CHAIN_DISPATCH_PROCESS_CONTROLLER"
CONTROLLER_FILE = "slurm-controller.json"
EXECUTABLE_MODE = 0o755
SUBPROCESS_TIMEOUT_SECONDS = 10
PROCESS_TEST_TIMEOUT_SECONDS = 15
SLURM_PROGRAMS = ("sbatch", "squeue", "sacct", "scancel")

FAKE_SLURM_SCRIPT = """import fcntl
import json
import os
import sys
from pathlib import Path
from typing import TypedDict

CONTROLLER_ENV = "PFAS_TEST_CHAIN_DISPATCH_PROCESS_CONTROLLER"
FIRST_JOB_ID = 1000
JOB_NAME_OPTION = "--job-name="
DEPENDENCY_OPTION = "--dependency=afterany:"
LIVE_STATES = ("PENDING", "RUNNING")


class Job(TypedDict):
    name: str
    state: str
    arguments: list[str]
    dispatcher: bool


class Controller(TypedDict):
    jobs: dict[str, Job]
    calls: list[list[str]]


def submit(controller: Controller, arguments: list[str]) -> str:
    identifier = str(FIRST_JOB_ID + len(controller["jobs"]))
    name = next(arg.removeprefix(JOB_NAME_OPTION) for arg in arguments
                if arg.startswith(JOB_NAME_OPTION))
    controller["jobs"][identifier] = {
        "name": name, "state": "PENDING", "arguments": arguments,
        "dispatcher": any(arg.startswith(DEPENDENCY_OPTION) for arg in arguments),
    }
    return identifier + "\\n"


def observations(controller: Controller, program: str, arguments: list[str]) -> str:
    jobs = controller["jobs"]
    if "--jobs" in arguments:
        identifiers = arguments[arguments.index("--jobs") + 1].split(",")
        jobs = {identifier: job for identifier, job in jobs.items()
                if identifier in identifiers}
    if program == "squeue":
        jobs = {identifier: job for identifier, job in jobs.items()
                if job["state"] in LIVE_STATES}
    return "".join(f"{identifier}|{job['name']}|{job['state']}|0:0\\n"
                   for identifier, job in jobs.items())


def invoke(controller: Controller, program: str, arguments: list[str]) -> str:
    controller["calls"].append([program, *arguments])
    if program == "sbatch":
        return submit(controller, arguments)
    if program == "scancel":
        controller["jobs"][arguments[0]]["state"] = "CANCELLED"
        return ""
    if program in ("squeue", "sacct"):
        return observations(controller, program, arguments)
    raise ValueError(f"Unexpected scheduler command: {program}")


def main() -> None:
    path = Path(os.environ[CONTROLLER_ENV])
    # Serialize the controller independently of the dashboard's SQLite lock.
    with path.open("r+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        controller: Controller = json.load(stream)
        output = invoke(controller, Path(sys.argv[0]).name, sys.argv[1:])
        stream.seek(0)
        json.dump(controller, stream)
        stream.truncate()
    sys.stdout.write(output)


if __name__ == "__main__":
    main()
"""

PW_SCRIPT = """import os


def main() -> None:
    ranks = os.environ["TEST_MPI_RANKS"]
    print(f"Parallel version (MPI), running on {ranks} processors")
    print("Estimated max dynamical RAM per process > 2 MB")
    print("Estimated total dynamical RAM > 4 MB")
    raise SystemExit(255)


if __name__ == "__main__":
    main()
"""


class Job(TypedDict):
    name: str
    state: str
    arguments: list[str]
    dispatcher: bool


class Controller(TypedDict):
    jobs: dict[str, Job]
    calls: list[list[str]]


@pytest.fixture(name="manager")
def dispatch_manager(manager_fixture: Manager) -> Manager:  # noqa: F811
    return manager_fixture


def write_executable(path: Path, script: str) -> None:
    path.write_text(f"#!{sys.executable}\n{script}")
    path.chmod(EXECUTABLE_MODE)


@pytest.fixture
def controller(
    tmp_path: Path, manager: Manager, monkeypatch: pytest.MonkeyPatch
) -> Path:
    path = tmp_path / CONTROLLER_FILE
    path.write_text(json.dumps({"jobs": {}, "calls": []}))
    monkeypatch.setenv(CONTROLLER_ENV, str(path))
    # Reuse the fixture's PATH directory so its fake srun remains available.
    commands = tmp_path / "bin"
    for program in SLURM_PROGRAMS:
        write_executable(commands / program, FAKE_SLURM_SCRIPT)
    executable = tmp_path / "pw-estimate"
    write_executable(executable, PW_SCRIPT)
    manager.config.pw = str(executable)
    return path


def read_controller(path: Path) -> Controller:
    return json.loads(path.read_text())


def dispatcher_record(manager: Manager) -> models.SchedulerRecord:
    record = manager.chain.data()["dispatcher"]
    assert record is not None
    return record


def job_identifier(record: models.SchedulerRecord) -> str:
    identifier = record["job_id"]
    assert isinstance(identifier, str)
    return identifier


def run_script(path: Path, job_id: str) -> None:
    result = subprocess.run(
        ["bash", str(path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        env={**os.environ, execution.SLURM_JOB_ID_ENV: job_id},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def release_dispatcher(path: Path, compute_id: str, dispatcher_id: str) -> None:
    controller = read_controller(path)
    controller["jobs"][compute_id]["state"] = execution.SlurmState.COMPLETED
    controller["jobs"][dispatcher_id]["state"] = execution.SlurmState.RUNNING
    path.write_text(json.dumps(controller))


def assert_persisted_advancement(
    manager: Manager, controller: Path, first_id: str, second_id: str
) -> None:
    store = persistence.Store(
        manager.config.artifacts / configuration.TASK_DATABASE_FILE,
        recover=False,
    )
    try:
        first, second = store.get(first_id), store.get(second_id)
        assert first["status"] == execution.TaskStatus.SUCCEEDED
        assert first["ram_estimate"]["total"] is not None
        assert second["scheduler"]["job_id"] is not None
        state = store.chain()
        assert state["active_task_id"] == second_id and state["error"] == ""
        successor = state["dispatcher"]
        assert successor is not None
        job_identifier(successor)
        dependency = f"--dependency=afterany:{second['scheduler']['job_id']}"
        assert dependency in successor["command"]
    finally:
        store.db.close()
    jobs = read_controller(controller)["jobs"]
    assert len(jobs) == 4
    assert sum(job["dispatcher"] for job in jobs.values()) == 2


@pytest.mark.timeout(PROCESS_TEST_TIMEOUT_SECONDS)
def test_generated_dispatcher_advances_after_server_closes(
    manager: Manager,
    controller: Path,
) -> None:
    first, second = [
        queue(manager, kind="estimate_ram", processes=rank) for rank in (2, 4)
    ]
    state = manager.chain.configure(enabled=True, paused=False)
    assert not state["error"]
    compute = manager.store.get(first["id"])["scheduler"]
    dispatcher = dispatcher_record(manager)
    compute_id, dispatcher_id = job_identifier(compute), job_identifier(dispatcher)
    assert len(read_controller(controller)["jobs"]) == 2

    # Execute the real generated compute script, then release its dependency.
    run_script(Path(compute["directory"]) / execution.SLURM_SCRIPT_FILE, compute_id)
    release_dispatcher(controller, compute_id, dispatcher_id)
    manager.close()
    assert not manager.thread.is_alive() and not manager.scheduler_thread.is_alive()
    assert manager.store.get(first["id"])["status"] != execution.TaskStatus.SUCCEEDED

    # No server polling or in-process recovery can advance this stopped manager.
    script = Path(dispatcher["directory"]) / chaining.DISPATCH_SCRIPT
    run_script(script, dispatcher_id)
    assert_persisted_advancement(manager, controller, first["id"], second["id"])
