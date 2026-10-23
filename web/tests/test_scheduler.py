"""Slurm lifecycle tests without a scheduler or container installation."""

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict, cast

import pytest

from dashboard.candidates import TFA
from dashboard.config import Config
from dashboard.execution_models import TaskRecord, TaskResult
from dashboard.persistence import Store, now
from dashboard.runtimes import GIB, Runtimes, file_hash, memory_ceiling
from dashboard.scheduler import command
from dashboard.slurm_worker import run
from dashboard.tasks import Manager


class SlurmPreviewRequest(TypedDict):
    candidate: str
    system: str
    processes: int
    target: str
    memory_gib: float


class EstimateBatchRequest(TypedDict):
    kind: str
    candidates: list[str]
    target: str
    runtime: str
    memory_gib: float
    estimate_processes: list[int]


INPUT = """&CONTROL
calculation='scf', pseudo_dir='./Pseudopotentials', outdir='./Outputs'
/
&SYSTEM
nat=1, ntyp=1, ibrav=1, celldm(1)=12, ecutwfc=15
/
ATOMIC_SPECIES
H 1 H.UPF
ATOMIC_POSITIONS angstrom
H 0 0 0
K_POINTS gamma
"""


@pytest.fixture
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Manager]:
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "8192")
    commands = tmp_path / "bin"
    commands.mkdir()
    srun = commands / "srun"
    srun.write_text(
        f"#!{sys.executable}\nimport os,sys\nranks=next(arg.split('=',1)[1] for arg in sys.argv if arg.startswith('--ntasks='))\nos.environ['TEST_MPI_RANKS']=ranks\nprogram=sys.argv[sys.argv.index('--')+1:]\nos.execv(program[0],program)\n"
    )
    srun.chmod(0o755)
    monkeypatch.setenv("PATH", str(commands) + os.pathsep + os.environ["PATH"])
    config = Config(
        artifacts=tmp_path / "state",
        pseudos=tmp_path / "pseudos",
        qe_inputs=tmp_path / "inputs",
        pw=sys.executable,
        mpi=sys.executable,
    )
    monkeypatch.setattr(Config, "scheduler_enabled", property(lambda self: True))
    config.pseudos.mkdir()
    (config.pseudos / "H.UPF").write_text("pseudo")
    config.image = tmp_path / "chemistry.sif"
    config.image.write_bytes(b"test image")
    m = Manager(
        config,
        {"500": {"id": "500", "cid": "1", "smiles": "[H]", "fields": {}}, "tfa": TFA},
    )
    prepared = config.artifacts / "prepared"
    prepared.mkdir()
    (prepared / "candidate.in").write_text(INPUT)
    m.store.put(
        {
            "id": "prepared",
            "kind": "prepare",
            "candidate": "500",
            "system": "candidate",
            "status": "succeeded",
            "created": now(),
            "started": None,
            "ended": now(),
            "artifacts": {},
        }
    )
    yield m
    m.close()


def queue(manager: Manager, kind: str = "qe", processes: int = 16) -> TaskRecord:
    preview = manager.preview(
        "500", "candidate", processes, "slurm", kind=kind, memory_gib=64
    )
    task = manager.queue(
        kind,
        "500",
        processes=processes,
        target="slurm",
        memory_gib=64,
        expected_hash=preview["input_hash"],
        preview_id=preview["preview_id"],
    )
    return cast(TaskRecord, task)


def response(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], stdout=stdout, stderr=stderr, returncode=returncode
    )


def fake_slurm(
    monkeypatch: pytest.MonkeyPatch,
    manager: Manager,
    state: str = "PENDING",
    unknown: bool = False,
) -> list[list[str]]:
    calls: list[list[str]] = []

    def invoke(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[0] == "sbatch":
            if unknown:
                raise subprocess.TimeoutExpired(args, 30)
            return response("12345\n")
        tasks = [t for t in manager.store.tasks() if t.get("scheduler")]
        token = tasks[0]["scheduler"]["token"] if tasks else ""
        if args[0] == "squeue":
            return response(
                f"12345|pfas-{token}|{state}|Resources\n"
                if state == "PENDING" or state == "RUNNING"
                else ""
            )
        if args[0] == "sacct":
            return response(f"12345|pfas-{token}|{state}|0:0\n")
        return response()

    monkeypatch.setattr("dashboard.scheduler.command", invoke)
    return calls


def test_allocation_container_has_writable_attempt_and_no_cgroup_flags(
    manager: Manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/apptainer")
    runtime = Runtimes(manager.config)
    identity = runtime.verify()  # No D-Bus/cgroup probe required.
    directory = manager.config.artifacts / "attempt"
    args = runtime.wrap(
        ["pw.x"],
        directory,
        {"cpus": 1, "memory_bytes": GIB, "timeout": None, "hard_memory_limit": False},
        identity,
    )
    assert "--memory" not in args and "--cpus" not in args
    assert f"{directory}:{directory}:rw" in args
    assert args[-2:] == [str(manager.config.image), "pw.x"]
    assert not runtime.availability()["apptainer"]["hard_memory_limit"]
    monkeypatch.delenv("SLURM_JOB_ID")
    args = runtime.base(GIB, 1)
    assert "--memory" in args and "--cpuset-cpus" in args


def test_detection_and_allocation_resource_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "/bin/true")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert not Config().scheduler_enabled
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    assert Config().scheduler_enabled
    assert not Config(execution="local").scheduler_enabled
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "2048")
    assert memory_ceiling() <= 2 * GIB
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert not Config().scheduler_enabled
    with pytest.raises(ValueError, match="requires"):
        _ = Config(execution="slurm").scheduler_enabled


def test_remote_resources_preview_freeze_and_submission(
    manager: Manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert manager.settings["cpus"] <= 3
    with pytest.raises(ValueError, match="queue budget"):
        manager.resources("native", 16, 64, None)
    args: SlurmPreviewRequest = {
        "candidate": "500",
        "system": "candidate",
        "processes": 16,
        "target": "slurm",
        "memory_gib": 64,
    }
    preview = manager.preview(**args)
    with pytest.raises(ValueError, match="settings changed"):
        manager.queue(
            "qe",
            **args,
            scheduler={"walltime_hours": 1},
            expected_hash=preview["input_hash"],
            preview_id=preview["preview_id"],
        )
    task = manager.queue(
        "qe",
        **args,
        expected_hash=preview["input_hash"],
        preview_id=preview["preview_id"],
    )
    calls = fake_slurm(monkeypatch, manager)
    manager.scheduler.submit_waiting()
    manager.scheduler.submit_waiting()
    submits = [c for c in calls if c[0] == "sbatch"]
    assert len(submits) == 1
    assert "--cpus-per-task=16" in submits[0] and "--mem=65536M" in submits[0]
    assert "--nodes=1" in submits[0] and "--ntasks=1" in submits[0]
    assert manager.store.get(task["id"])["scheduler"]["job_id"] == "12345"
    manager.configure({"cpus": 1})  # Remote reservations do not constrain local budget.


def test_grouped_estimates_one_allocation_and_rank_specific_retry(
    manager: Manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    request: EstimateBatchRequest = {
        "kind": "estimate_ram",
        "candidates": ["500"],
        "target": "slurm",
        "runtime": "native",
        "memory_gib": 64,
        "estimate_processes": [16, 4, 1],
    }
    preview = manager.batches.review(**request)
    assert [e["processes"] for e in preview["entries"]] == [16, 4, 1]
    batch = manager.batches.submit(
        {**request, "expected": preview["entries"], "image_hash": None}
    )
    assert len(batch["task_ids"]) == 3
    calls = fake_slurm(monkeypatch, manager)
    manager.scheduler.submit_waiting()
    assert len([c for c in calls if c[0] == "sbatch"]) == 1
    tasks = [manager.store.get(id) for id in batch["task_ids"]]
    assert len({t["scheduler"]["job_id"] for t in tasks}) == 1
    assert "--time=4:00:00" in calls[0]
    for task in tasks:
        if task["processes"] == 4:
            manager.store.update(task["id"], status="failed")
        else:
            manager.store.update(
                task["id"],
                status="succeeded",
                ram_estimate={"per_process": {"bytes": GIB}, "total": None},
            )
    retry = manager.batches.retry(batch["id"])
    review = manager.batches.review(**retry)
    assert [(e["processes"], e["status"]) for e in review["entries"]] == [
        (16, "completed"),
        (4, "eligible"),
        (1, "completed"),
    ]
    reopened = Store(manager.config.artifacts / "tasks.sqlite")
    assert len(reopened.batch(batch["id"])["entries"]) == 3
    reopened.db.close()


def test_ambiguous_submission_reconciles_without_duplicate(
    manager: Manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = queue(manager)
    calls = fake_slurm(monkeypatch, manager, unknown=True)
    manager.scheduler.submit_waiting()
    assert manager.store.get(task["id"])["scheduler"]["state"] == "SUBMISSION_UNKNOWN"
    manager.scheduler.poll(force=True)
    assert manager.store.get(task["id"])["scheduler"]["job_id"] == "12345"
    manager.scheduler.submit_waiting()
    assert len([c for c in calls if c[0] == "sbatch"]) == 1


def test_pending_cancel_and_restart_survival(
    manager: Manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = queue(manager)
    calls = fake_slurm(monkeypatch, manager)
    manager.scheduler.submit_waiting()
    manager.scheduler.poll(force=True)
    manager.cancel_task(task["id"])
    assert ["scancel", "12345"] in calls
    assert manager.store.get(task["id"])["stop_requested"]
    assert manager.store.get(task["id"])["status"] == "queued"
    fake_slurm(monkeypatch, manager, state="CANCELLED")
    manager.scheduler.poll(force=True)
    assert manager.store.get(task["id"])["status"] == "canceled"

    # Separate jobs are never interrupted/canceled by stopping the dashboard.
    other = queue(manager, processes=4)
    fake_slurm(monkeypatch, manager)
    manager.scheduler.submit_waiting()
    manager.store.update(other["id"], status="running")
    manager.close()
    assert manager.store.get(other["id"])["status"] == "running"
    reopened = Store(manager.config.artifacts / "tasks.sqlite")
    assert reopened.get(other["id"])["status"] == "running"
    reopened.db.close()


def test_accounting_delay_outage_and_terminal_missing_result(
    manager: Manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = queue(manager)
    fake_slurm(monkeypatch, manager)
    manager.scheduler.submit_waiting()
    monkeypatch.setattr("dashboard.scheduler.command", lambda args: response())
    manager.scheduler.poll(force=True)
    assert manager.store.get(task["id"])["status"] == "queued"
    monkeypatch.setattr(
        "dashboard.scheduler.command",
        lambda args: response(stderr="offline", returncode=1),
    )
    manager.scheduler.poll(force=True)
    assert manager.store.get(task["id"])["status"] == "queued"
    fake_slurm(monkeypatch, manager, state="OUT_OF_MEMORY")
    manager.scheduler.poll(force=True)
    failed = manager.store.get(task["id"])
    assert failed["status"] == "failed" and "OUT_OF_MEMORY" in failed["error"]


def test_worker_incremental_results_and_image_change(
    manager: Manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = manager.config.artifacts / "pw-stub"
    executable.write_text(
        f"#!{sys.executable}\nprint('Estimated max dynamical RAM per process > 12.50 MB')\nprint('Estimated total dynamical RAM > 1.25 GB')\nraise SystemExit(255)\n"
    )
    executable.chmod(0o755)
    manager.config.pw = str(executable)
    task = queue(manager, kind="estimate_ram", processes=1)
    fake_slurm(monkeypatch, manager)
    manager.scheduler.submit_waiting()
    record = manager.store.get(task["id"])["scheduler"]
    manifest = json.loads((Path(record["directory"]) / "manifest.json").read_text())
    # Prove no worker constructs Store or writes SQLite.
    monkeypatch.setattr(
        Store, "__init__", lambda *args: pytest.fail("worker opened SQLite")
    )
    run(manifest)
    fake_slurm(monkeypatch, manager, state="RUNNING")
    manager.scheduler.poll(force=True)
    done = manager.store.get(task["id"])
    assert (
        done["status"] == "succeeded"
        and done["ram_estimate"]["total"]["bytes"] == 1.25 * GIB
    )
    assert done["artifacts"]["slurm-result.json"]
    assert manager.config.image is not None
    manager.config.image.write_text("changed")
    with pytest.raises(ValueError, match="image changed"):
        manager.runtimes.wrap(
            ["pw.x"],
            manager.config.artifacts,
            {
                "cpus": 1,
                "memory_bytes": GIB,
                "timeout": None,
                "hard_memory_limit": False,
            },
            file_hash(Path(record["directory"]) / "manifest.json"),
            scheduler=True,
        )


def test_submission_environment_excludes_parent_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    monkeypatch.setenv("SBATCH_MEM_PER_NODE", "2048")
    monkeypatch.setenv("SLURM_CONF", "/etc/slurm/slurm.conf")

    def invoke(args: list[str], **kwargs: dict) -> subprocess.CompletedProcess[str]:
        assert "SLURM_JOB_ID" not in kwargs["env"]
        assert "SBATCH_MEM_PER_NODE" not in kwargs["env"]
        assert kwargs["env"]["SLURM_CONF"] == "/etc/slurm/slurm.conf"
        assert kwargs["env"]["PATH"] == os.environ["PATH"]
        return response()

    monkeypatch.setattr(subprocess, "run", invoke)
    command(["sbatch", "script"])


def test_group_worker_executes_all_ranks_and_imports_partial_results(
    manager: Manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    pw = manager.config.artifacts / "pw-stub"
    pw.write_text(
        f"#!{sys.executable}\nimport os\nranks=int(os.getenv('TEST_MPI_RANKS','1'))\nprint(f'Parallel version (MPI), running on {{ranks}} processors')\nprint('Estimated max dynamical RAM per process > 2 MB')\nprint(f'Estimated total dynamical RAM > {{ranks * 2}} MB')\nraise SystemExit(2 if ranks==4 else 255)\n"
    )
    pw.chmod(0o755)
    manager.config.pw = str(pw)
    request: EstimateBatchRequest = {
        "kind": "estimate_ram",
        "candidates": ["500"],
        "target": "slurm",
        "runtime": "native",
        "memory_gib": 64,
        "estimate_processes": [16, 4, 1],
    }
    preview = manager.batches.review(**request)
    batch = manager.batches.submit(
        {**request, "expected": preview["entries"], "image_hash": None}
    )
    fake_slurm(monkeypatch, manager)
    manager.scheduler.submit_waiting()
    record = manager.store.get(batch["task_ids"][0])["scheduler"]
    manifest = json.loads((Path(record["directory"]) / "manifest.json").read_text())
    monkeypatch.setenv("PFAS_COMPUTE_WORKER", "1")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    run(manifest)
    manager.scheduler.poll(force=True)
    attempts = [manager.store.get(id) for id in batch["task_ids"]]
    assert [(t["processes"], t["status"]) for t in attempts] == [
        (16, "succeeded"),
        (4, "failed"),
        (1, "succeeded"),
    ]
    assert attempts[0]["ram_estimate"]["total"]["bytes"] == 32 * 1024**2
    assert attempts[2]["ram_estimate"]["total"]["bytes"] == 2 * 1024**2
    assert not (Path(record["directory"]) / "tasks.sqlite").exists()


def test_changed_snapshot_is_rejected_and_other_attempt_continues(
    manager: Manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = queue(manager, kind="estimate_ram", processes=1)
    fake_slurm(monkeypatch, manager)
    manager.scheduler.submit_waiting()
    record = manager.store.get(first["id"])["scheduler"]
    manifest = json.loads((Path(record["directory"]) / "manifest.json").read_text())
    (manager.config.artifacts / first["id"] / "input.in").write_text("changed")
    run(manifest)
    manager.scheduler.poll(force=True)
    assert "Captured input changed" in manager.store.get(first["id"])["error"]


def test_rank_mismatch_is_not_a_successful_estimate(manager: Manager) -> None:
    from dashboard.calculation import validate_estimate

    task = queue(manager, kind="estimate_ram")
    directory = manager.config.artifacts / task["id"]
    (directory / "stdout.log").write_text(
        "Parallel version (MPI), running on 4 processors\nEstimated max dynamical RAM per process > 2 MB\n"
    )
    result: TaskResult = {"status": "failed", "exit_code": 255, "error": ""}
    validate_estimate(task, directory, result)
    assert result["status"] == "failed" and "process count" in result["error"]


def test_rejected_submission_recovery_requires_absent_job(
    manager: Manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = queue(manager)
    fake_slurm(monkeypatch, manager, unknown=True)
    manager.scheduler.submit_waiting()
    task = manager.store.get(task["id"])
    with pytest.raises(ValueError, match="matching job"):
        manager.scheduler.mark_rejected(task)
    monkeypatch.setattr("dashboard.scheduler.command", lambda args: response())
    manager.scheduler.mark_rejected(task)
    assert manager.store.get(task["id"])["status"] == "failed"


def test_matching_rank_memory_sizing_and_stale_inputs(manager: Manager) -> None:
    request: EstimateBatchRequest = {
        "kind": "estimate_ram",
        "candidates": ["500"],
        "target": "slurm",
        "runtime": "native",
        "memory_gib": 64,
        "estimate_processes": [16, 1],
    }
    review = manager.batches.review(**request)
    batch = manager.batches.submit(
        {**request, "expected": review["entries"], "image_hash": None}
    )
    for task_id in batch["task_ids"]:
        task = manager.store.get(task_id)
        total = 12 * GIB if task["processes"] == 16 else 24 * GIB
        manager.store.update(
            task_id,
            status="succeeded",
            ram_estimate={"per_process": {"bytes": GIB}, "total": {"bytes": total}},
        )
    preview = manager.preview(
        "500", "candidate", 16, "slurm", scheduler={"memory_mode": "estimate"}
    )
    assert preview["resources"]["memory_bytes"] == 18 * GIB
    assert preview["memory_estimate"]["processes"] == 16
    (manager.config.pseudos / "H.UPF").write_text("new pseudo")
    with pytest.raises(ValueError, match="No matching RAM estimate"):
        manager.preview(
            "500", "candidate", 16, "slurm", scheduler={"memory_mode": "estimate"}
        )


def test_slurm_api_freezes_resources_and_exposes_scheduler_state(
    manager: Manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from dashboard.server import create_app

    app = create_app(manager.config)
    client = TestClient(app)
    request = {
        "candidate": "500",
        "kind": "qe",
        "target": "slurm",
        "runtime": "native",
        "processes": 16,
        "memory_gib": 64,
        "scheduler": {"walltime_hours": 2},
    }
    reviewed = client.post("/api/preview", json=request)
    assert reviewed.status_code == 200
    preview = reviewed.json()
    assert preview["resources"]["cpus"] == 16
    assert preview["command"][0].endswith("/srun")
    assert "--ntasks=16" in preview["command"]
    accepted = client.post(
        "/api/tasks",
        json={
            **request,
            "preview_id": preview["preview_id"],
            "expected_hash": preview["input_hash"],
        },
    )
    assert accepted.status_code == 200
    task = accepted.json()
    fake_slurm(monkeypatch, app.state.manager)
    app.state.manager.scheduler.submit_waiting()
    app.state.manager.scheduler.poll(force=True)
    current = next(
        t for t in client.get("/api/queue").json()["tasks"] if t["id"] == task["id"]
    )
    assert current["target"] == "slurm"
    assert current["scheduler"]["job_id"] == "12345"
    assert current["scheduler_settings"]["walltime_hours"] == 2
    assert client.post(f"/api/tasks/{task['id']}/stop").status_code == 200
    app.state.manager.close()
