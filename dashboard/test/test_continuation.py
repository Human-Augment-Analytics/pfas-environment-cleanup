"""Exercise safe continuation decisions and durable queue admission."""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dashboard.app import server
from dashboard.core import execution
from dashboard.core.chemistry import optimization
from dashboard.core.jobs import continuation
from dashboard.core.jobs.tasks import Manager
from dashboard.core.slurm import chaining
from dashboard.test.test_chaining import Slurm, dispatch
from dashboard.test.test_chaining import slurm as slurm_fixture  # noqa: F401
from dashboard.test.test_optimization import INPUT, OUTPUT
from dashboard.test.test_scheduler import manager as manager_fixture  # noqa: F401
from dashboard.test.test_scheduler import queue

INCOMPLETE = OUTPUT.split("End of BFGS")[0]


@pytest.fixture(name="manager")
def relaxation_manager(manager_fixture: Manager) -> Manager:  # noqa: F811
    path = manager_fixture.config.artifacts / "prepared" / "candidate.in"
    path.write_text(INPUT)
    return manager_fixture


def finished(manager: Manager, output: str = INCOMPLETE) -> str:
    task = queue(manager, processes=1)
    directory = manager.config.artifacts / task["id"]
    (directory / execution.STDOUT_LOG_FILE).write_text(output)
    manager.store.update(
        task["id"],
        status=execution.TaskStatus.FAILED,
        scheduler={"state": execution.SlurmState.TIMEOUT},
    )
    return task["id"]


def advance(manager: Manager, task_id: str) -> None:
    with manager.guard:
        continuation.advance(manager, manager.store.get(task_id))


@pytest.mark.parametrize(
    "output,expected",
    [
        (OUTPUT, continuation.Outcome.CONVERGED),
        (INCOMPLETE, continuation.Outcome.INCOMPLETE),
        (INCOMPLETE + "convergence NOT achieved", continuation.Outcome.FAILED),
        (INCOMPLETE + "Error in routine", continuation.Outcome.FAILED),
        (
            INCOMPLETE + "ATOMIC_POSITIONS angstrom\nH 0 0 0\n",
            continuation.Outcome.UNCERTAIN,
        ),
        ("", continuation.Outcome.UNCERTAIN),
    ],
)
def test_output_decisions(manager: Manager, output: str, expected: str) -> None:
    task_id = finished(manager, output)
    assert (
        continuation.inspect(manager.config.artifacts / task_id)["outcome"] == expected
    )


def test_fresh_electronic_settings_override_file_reads() -> None:
    text = INPUT.replace(
        "&SYSTEM", "&ELECTRONS\n startingpot='file', startingwfc='file',\n/\n&SYSTEM"
    )
    continued = optimization.fresh_start(text)
    assert "startingpot='atomic'" in continued
    assert "startingwfc='atomic+random'" in continued
    assert "='file'" not in continued


def test_missing_electrons_is_inserted_in_qe_namelist_order() -> None:
    continued = optimization.fresh_start(INPUT)
    assert continued.index("&SYSTEM") < continued.index("&ELECTRONS")
    assert continued.index("&ELECTRONS") < continued.index("ATOMIC_SPECIES")


def test_successor_is_fresh_bounded_and_duplicate_safe(manager: Manager) -> None:
    parent = finished(manager)
    continuation.configure(manager, parent, 2)
    advance(manager, parent)
    child = continuation.existing_successor(manager, parent)
    assert child is not None
    assert child["continuation_policy"]["remaining"] == 1
    assert child["continuation_parent"] == parent
    assert child["resources"] == manager.store.get(parent)["resources"]
    directory = manager.config.artifacts / child["id"]
    assert list((directory / "Outputs").iterdir()) == []
    assert (
        "restart_mode='from_scratch'" in (directory / execution.INPUT_FILE).read_text()
    )
    restarted = Manager(manager.config, manager.candidates, recover=False)
    try:
        advance(restarted, parent)
        recovered = continuation.existing_successor(restarted, parent)
        assert recovered is not None
        assert recovered["id"] == child["id"]
        assert len(restarted.store.tasks()) == len(manager.store.tasks())
    finally:
        restarted.store.db.close()


@pytest.mark.parametrize(
    "change,problem",
    [
        ("pipeline", "Pipeline code changed"),
        ("pseudo", "pseudopotentials changed"),
        ("input", "Source input changed"),
        ("partial", "Partial coordinates"),
        ("cancel", "requires a known timeout"),
    ],
)
def test_changes_require_review(
    manager: Manager, monkeypatch: pytest.MonkeyPatch, change: str, problem: str
) -> None:
    parent = finished(manager)
    continuation.configure(manager, parent, 1)
    directory = manager.config.artifacts / parent
    if change == "pipeline":
        monkeypatch.setattr(continuation, "pipeline_hash", lambda: "changed")
    elif change == "pseudo":
        (manager.config.pseudos / "H.UPF").write_text("changed")
    elif change == "input":
        (directory / execution.INPUT_FILE).write_text(INPUT + "\n")
    elif change == "partial":
        (directory / execution.STDOUT_LOG_FILE).write_text(
            INCOMPLETE + "ATOMIC_POSITIONS angstrom\nH 0 0 0\n"
        )
    else:
        manager.store.update(parent, stop_requested=True)
    with pytest.raises(ValueError, match=problem):
        advance(manager, parent)
    assert continuation.existing_successor(manager, parent) is None
    assert manager.store.get(parent)["continuation_error"]


@pytest.mark.parametrize("limit", [0, 1])
def test_disabled_and_converged_do_not_continue(manager: Manager, limit: int) -> None:
    parent = finished(manager, OUTPUT)
    continuation.configure(manager, parent, limit)
    advance(manager, parent)
    assert continuation.existing_successor(manager, parent) is None


def test_chain_dispatches_one_successor_and_respects_limit(
    manager: Manager,
    slurm_fixture: Slurm,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = queue(manager, processes=1)
    continuation.configure(manager, parent["id"], 1)
    manager.chain.configure(enabled=True, paused=False)
    (manager.config.artifacts / parent["id"] / execution.STDOUT_LOG_FILE).write_text(
        INCOMPLETE
    )
    dispatch(manager, slurm_fixture, monkeypatch, parent["id"], "TIMEOUT")
    assert not manager.chain.data()["error"]
    child = continuation.existing_successor(manager, parent["id"])
    assert child is not None
    assert manager.chain.data()["active_task_id"] == child["id"]
    (manager.config.artifacts / child["id"] / execution.STDOUT_LOG_FILE).write_text(
        INCOMPLETE
    )
    dispatch(manager, slurm_fixture, monkeypatch, child["id"], "TIMEOUT")
    assert continuation.existing_successor(manager, child["id"]) is None
    assert manager.chain.data()["active_task_id"] is None


def test_downloaded_run_preparation_refuses_overwrite(
    manager: Manager, tmp_path: Path
) -> None:
    parent = finished(manager)
    output = tmp_path / "continued.in"
    continuation.main(directory=manager.config.artifacts / parent, output=output)
    assert "startingpot='atomic'" in output.read_text()
    with pytest.raises(FileExistsError):
        continuation.main(directory=manager.config.artifacts / parent, output=output)


def test_offline_recovery_restores_candidate_and_continues(
    manager: Manager,
    slurm_fixture: Slurm,  # noqa: F811
) -> None:
    parent = queue(manager, processes=1)
    continuation.configure(manager, parent["id"], 1)
    manager.chain.configure(enabled=True, paused=False)
    record = manager.chain.data()["dispatcher"]
    assert record is not None
    job_id = manager.store.get(parent["id"])["scheduler"]["job_id"]
    assert job_id is not None and record["job_id"] is not None
    slurm_fixture.jobs[job_id].state = "TIMEOUT"
    slurm_fixture.jobs[record["job_id"]].state = "FAILED"
    (manager.config.artifacts / parent["id"] / execution.STDOUT_LOG_FILE).write_text(
        INCOMPLETE
    )
    chaining.main(config=Path(record["directory"]) / chaining.DISPATCH_CONFIG)
    child = continuation.existing_successor(manager, parent["id"])
    assert child is not None
    assert manager.chain.data()["active_task_id"] == child["id"]


def test_authorization_api_validates_limit_and_can_disable(manager: Manager) -> None:
    parent = queue(manager, processes=1)
    app = FastAPI()
    app.state.manager = manager
    app.include_router(server.get_router())
    client = TestClient(app)
    path = f"/api/tasks/{parent['id']}/continuation"
    for invalid in (True, -1, continuation.MAX_CONTINUATIONS + 1):
        assert client.post(path, json={"limit": invalid}).status_code == 422
    assert client.post(path, json={"limit": 2}).status_code == 200
    assert manager.store.get(parent["id"])["continuation_policy"]["remaining"] == 2
    assert client.post(path, json={"limit": 0}).status_code == 200
    assert manager.store.get(parent["id"])["continuation_policy"]["remaining"] == 0


def test_crash_after_successor_commit_does_not_duplicate(manager: Manager) -> None:
    parent = finished(manager)
    continuation.configure(manager, parent, 1)
    # Admission uses Store.put; interruption occurs after the child is durable.
    with (
        patch.object(manager.store, "update", side_effect=OSError("interrupted")),
        pytest.raises(OSError, match="interrupted"),
    ):
        advance(manager, parent)
    child = continuation.existing_successor(manager, parent)
    assert child is not None
    advance(manager, parent)
    recovered = continuation.existing_successor(manager, parent)
    assert recovered is not None and recovered["id"] == child["id"]
    assert (
        len(
            [
                task
                for task in manager.store.tasks()
                if task.get("continuation_parent") == parent
            ]
        )
        == 1
    )


def test_changed_container_image_requires_review(manager: Manager) -> None:
    reviewed = manager.preview(
        "500", "candidate", 1, "slurm", runtime="apptainer", memory_gib=64
    )
    task = manager.queue(
        "qe",
        "500",
        target="slurm",
        runtime="apptainer",
        memory_gib=64,
        expected_hash=reviewed["input_hash"],
        preview_id=reviewed["preview_id"],
    )
    parent = task["id"]
    (manager.config.artifacts / parent / execution.STDOUT_LOG_FILE).write_text(
        INCOMPLETE
    )
    manager.store.update(parent, status="failed", scheduler={"state": "TIMEOUT"})
    continuation.configure(manager, parent, 1)
    assert manager.config.image is not None
    manager.config.image.write_bytes(b"changed image")
    with pytest.raises(ValueError, match="Runtime image"):
        advance(manager, parent)
    assert continuation.existing_successor(manager, parent) is None
