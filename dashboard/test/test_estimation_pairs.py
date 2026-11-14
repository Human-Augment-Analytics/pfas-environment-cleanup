"""Paired estimates preserve per-system results within one allocation."""

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI

from dashboard.app import server
from dashboard.core import execution, models
from dashboard.core.jobs.persistence import Store
from dashboard.core.jobs.tasks import Manager
from dashboard.test.test_chaining import post
from dashboard.test.test_scheduler import INPUT, fake_slurm
from dashboard.test.test_scheduler import manager as manager_fixture  # noqa: F401

PAIRED_SYSTEMS = ["candidate", "complex"]
PROCESS_COUNTS = [4, 1]


@pytest.fixture(name="manager")
def paired_manager(manager_fixture: Manager) -> Manager:  # noqa: F811
    prepared = manager_fixture.config.artifacts / "prepared"
    (prepared / "complex.in").write_text(INPUT.replace("ecutwfc=15", "ecutwfc=20"))
    return manager_fixture


def paired_request(*, grouped_ranks: bool = True) -> models.BatchReviewRequest:
    request: models.BatchReviewRequest = {
        "kind": "estimate_ram",
        "candidates": ["500"],
        "target": "slurm",
        "runtime": "native",
        "system": "candidate",
        "processes": 1,
        "memory_gib": 64,
        "estimate_systems": PAIRED_SYSTEMS,
    }
    if grouped_ranks:
        request["estimate_processes"] = PROCESS_COUNTS
    return request


def accept_pair(manager: Manager, *, grouped_ranks: bool = True) -> dict:
    request = paired_request(grouped_ranks=grouped_ranks)
    preview = manager.batches.review(**request)
    return manager.batches.submit(
        {**request, "expected": preview["entries"], "image_hash": None}
    )


def test_paired_review_and_submission_keep_distinct_inputs(
    manager: Manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview = manager.batches.review(**paired_request())
    entries = preview["entries"]
    assert {(entry["system"], entry["processes"]) for entry in entries} == {
        (system, ranks) for system in PAIRED_SYSTEMS for ranks in PROCESS_COUNTS
    }
    assert len({entry["preview_id"] for entry in entries}) == 4
    assert len({entry["input_hash"] for entry in entries}) == 2
    batch = accept_pair(manager)
    calls = fake_slurm(monkeypatch, manager)
    manager.scheduler.submit_waiting()
    tasks = [manager.store.get(task_id) for task_id in batch["task_ids"]]
    assert len(tasks) == 4
    assert len({task["scheduler"]["job_id"] for task in tasks}) == 1
    assert len([call for call in calls if call[0] == execution.SBATCH_COMMAND]) == 1
    assert "--cpus-per-task=4" in calls[0]
    assert "--mem=65536M" in calls[0]


def write_estimation_program(path: Path, *, fail_candidate: bool) -> None:
    path.write_text(
        f"#!{sys.executable}\n"
        "import os\nfrom pathlib import Path\n"
        "ranks = os.getenv('TEST_MPI_RANKS', '1')\n"
        "print(f'Parallel version (MPI), running on {ranks} processors')\n"
        "print('Estimated max dynamical RAM per process > 2 MB')\n"
        "print('Estimated total dynamical RAM > 4 MB')\n"
        f"failed = {fail_candidate!r} and 'ecutwfc=15' in Path('input.in').read_text()\n"
        "raise SystemExit(2 if failed else 255)\n"
    )
    path.chmod(0o755)


@pytest.mark.parametrize("fail_candidate", [False, True])
def test_generated_script_runs_both_and_preserves_individual_outcomes(
    manager: Manager, monkeypatch: pytest.MonkeyPatch, fail_candidate: bool
) -> None:
    program = manager.config.artifacts / "paired-pw"
    write_estimation_program(program, fail_candidate=fail_candidate)
    manager.config.pw = str(program)
    batch = accept_pair(manager, grouped_ranks=False)
    fake_slurm(monkeypatch, manager)
    manager.scheduler.submit_waiting()
    first = manager.store.get(batch["task_ids"][0])
    script = Path(first["scheduler"]["directory"]) / execution.SLURM_SCRIPT_FILE
    subprocess.run(["bash", str(script)], check=True, timeout=1)
    fake_slurm(monkeypatch, manager, state="COMPLETED")
    manager.scheduler.poll(force=True)
    results = {
        task["system"]: task
        for task in (manager.store.get(id) for id in batch["task_ids"])
    }
    assert results["complex"]["status"] == execution.TaskStatus.SUCCEEDED
    expected = (
        execution.TaskStatus.FAILED
        if fail_candidate
        else execution.TaskStatus.SUCCEEDED
    )
    assert results["candidate"]["status"] == expected
    assert results["complex"]["ram_estimate"]["total"] is not None


def test_restart_preserves_both_systems_and_all_ranks(manager: Manager) -> None:
    batch = accept_pair(manager)
    store = Store(manager.config.artifacts / "tasks.sqlite")
    try:
        recovered = store.batch(batch["id"])
        assert recovered is not None
        assert len(recovered["task_ids"]) == 4
        assert {
            (entry["system"], entry["processes"]) for entry in recovered["entries"]
        } == {(system, ranks) for system in PAIRED_SYSTEMS for ranks in PROCESS_COUNTS}
    finally:
        store.db.close()


def test_pair_without_rank_group_blocks_chain_enable(manager: Manager) -> None:
    accept_pair(manager, grouped_ranks=False)
    with pytest.raises(ValueError, match="grouped RAM"):
        manager.chain.configure(enabled=True, paused=False)


def test_chain_mode_rejects_paired_system_request(manager: Manager) -> None:
    manager.chain.configure(enabled=True, paused=True)
    with pytest.raises(ValueError, match="chain mode"):
        manager.batches.review(**paired_request(grouped_ranks=False))


@pytest.mark.parametrize(
    "systems", [[], ["candidate", "candidate"], ["tfa"], ["unknown"]]
)
def test_invalid_system_groups_do_not_queue_work(
    manager: Manager, systems: list[str]
) -> None:
    request = paired_request(grouped_ranks=False)
    request["estimate_systems"] = systems
    with pytest.raises(ValueError, match="systems"):
        manager.batches.review(**request)
    assert not manager.store.ordered_ids()


def test_retry_keeps_pair_and_skips_successful_system(manager: Manager) -> None:
    batch = accept_pair(manager, grouped_ranks=False)
    tasks = [manager.store.get(task_id) for task_id in batch["task_ids"]]
    for task in tasks:
        if task["system"] == "candidate":
            manager.store.update(
                task["id"],
                status=execution.TaskStatus.SUCCEEDED,
                ram_estimate={"per_process": {"bytes": execution.GIB}, "total": None},
            )
        else:
            manager.store.update(task["id"], status=execution.TaskStatus.FAILED)
    retry = manager.batches.retry(batch["id"])
    assert retry["estimate_systems"] == PAIRED_SYSTEMS
    preview = manager.batches.review(**retry)
    assert {entry["system"]: entry["status"] for entry in preview["entries"]} == {
        "candidate": "completed",
        "complex": "eligible",
    }


def test_api_accepts_paired_estimates(manager: Manager) -> None:
    app = FastAPI()
    app.state.manager = manager
    app.include_router(server.get_router())
    request = dict(paired_request(grouped_ranks=False))
    preview = post(app, "/api/batches/preview", request)
    assert preview.status_code == 200
    response = post(
        app,
        "/api/batches",
        {
            **request,
            "expected": preview.json()["entries"],
            "image_hash": None,
        },
    )
    assert response.status_code == 200
    task_ids = response.json()["task_ids"]
    assert len(task_ids) == 2
    assert {manager.store.get(task_id)["system"] for task_id in task_ids} == set(
        PAIRED_SYSTEMS
    )
