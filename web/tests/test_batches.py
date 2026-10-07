import json
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from dashboard.candidates import TFA
from dashboard.config import Config
from dashboard.persistence import now
from dashboard.runtimes import GIB
from dashboard.server import create_app
from dashboard.tasks import Manager

pytestmark = pytest.mark.timeout(8)


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr("dashboard.tasks.memory_ceiling", lambda: 16 * GIB)
    config = Config(artifacts=tmp_path / "artifacts", pseudos=tmp_path / "pseudos")
    config.pseudos.mkdir()
    (config.pseudos / "H.UPF").write_text("hydrogen")
    candidates = {
        id: {"id": id, "cid": id, "smiles": "O", "fields": {}}
        for id in ("500", "501", "502")
    }
    candidates["tfa"] = TFA
    m = Manager(config, candidates)
    yield m
    m.close()


def request(ids, kind="diagram", **kwargs):
    return {
        "candidates": ids,
        "kind": kind,
        "system": "candidate",
        "runtime": "native",
        "processes": 1,
        "memory_gib": 4,
        **kwargs,
    }


def submit(manager, body):
    review = manager.batches.review(**body)
    return manager.batches.submit(
        {**body, "expected": review["entries"], "image_hash": review["image_hash"]}
    )


def wait_tasks(manager, limit=5):
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        if all(t["status"] not in ("queued", "running") for t in manager.store.tasks()):
            return
        time.sleep(0.02)
    raise AssertionError(manager.store.tasks())


def prepare_qe(manager, candidate):
    id = "prepared-" + candidate
    directory = manager.config.artifacts / id
    directory.mkdir()
    (directory / "candidate.in").write_text(
        "&CONTROL\ncalculation='scf', pseudo_dir='./Pseudopotentials', outdir='./Outputs'\n/\nATOMIC_SPECIES\nH 1 H.UPF\n"
    )
    manager.store.put(
        {
            "id": id,
            "kind": "prepare",
            "candidate": candidate,
            "system": "candidate",
            "status": "succeeded",
            "artifacts": {},
            "created": now(),
            "started": None,
            "ended": None,
        }
    )
    return directory / "candidate.in"


def test_order_dedup_failure_continuation_and_retry(manager, tmp_path):
    executable = tmp_path / "diagram-python"
    executable.write_text(
        f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\nif sys.argv[2]=='bad':\n print('invalid molecule',file=sys.stderr);sys.exit(2)\nPath(sys.argv[3]).write_bytes(b'png')\n"
    )
    executable.chmod(0o755)
    manager.config.python = str(executable)
    manager.candidates["501"]["smiles"] = "bad"
    body = request(["501", "500", "502"], id="ordered")
    batch = submit(manager, body)
    assert [t["candidate"] for t in manager.store.queued()] == body["candidates"]
    assert submit(manager, body) == batch
    assert len(manager.store.tasks()) == 3
    assert (
        manager.batches.review(**request(["500"]))["entries"][0]["status"] != "eligible"
    )
    manager.start()
    wait_tasks(manager)
    statuses = {t["candidate"]: t["status"] for t in manager.store.tasks()}
    assert statuses == {"501": "failed", "500": "succeeded", "502": "succeeded"}
    failed = manager.store.get(batch["task_ids"][0])
    assert "invalid molecule" in failed["error"]
    assert failed["artifacts"]["stderr.log"]
    retry = manager.batches.retry(batch["id"])
    assert retry["candidates"] == ["501"]
    assert (
        manager.batches.review(**request(["500"]))["entries"][0]["status"]
        == "completed"
    )
    manager.configure({"paused": True})
    retried = submit(manager, retry)
    assert retried["task_ids"][0] != failed["id"]


def test_bulk_preparation_common_preflight_once(manager, monkeypatch):
    calls = []
    monkeypatch.setattr(
        manager.runtimes,
        "chemistry_probe",
        lambda *args, **kwargs: (
            calls.append(args) or SimpleNamespace(returncode=0, stderr="")
        ),
    )
    monkeypatch.setattr(
        manager, "preflight", lambda _: pytest.fail("Per-entry synchronous preflight")
    )
    batch = submit(manager, request(["500", "501"], "prepare"))
    assert len(batch["task_ids"]) == 2
    assert (
        len(calls) == 2
    )  # Preview and acceptance each validate the common environment.
    assert all(t["resources"]["timeout"] == 900 for t in manager.store.tasks())


def test_qe_review_skip_hash_changes_and_immutable_inputs(manager):
    manager.config.pw = sys.executable
    source = prepare_qe(manager, "500")
    body = request(["500", "501"], "qe")
    review = manager.batches.review(**body)
    assert [e["status"] for e in review["entries"]] == ["eligible", "unavailable"]
    batch = manager.batches.submit({**body, "expected": review["entries"]})
    task = manager.store.get(batch["task_ids"][0])
    manager.store.update(task["id"], status="succeeded", evidence={"confirmed": True})
    assert manager.batches.review(**body)["entries"][0]["status"] == "completed"
    source.write_text(source.read_text() + "! edited\n")
    assert (
        manager.config.artifacts / task["id"] / "input.in"
    ).read_text() != source.read_text()
    assert manager.batches.review(**body)["entries"][0]["status"] == "eligible"
    changed = manager.batches.submit({**body, "expected": review["entries"]})
    assert changed["task_ids"] == []
    assert "changed" in changed["entries"][0]["reason"]
    current = manager.batches.review(**body)
    (manager.config.pseudos / "H.UPF").write_text("changed pseudo")
    assert (
        manager.batches.submit({**body, "expected": current["entries"]})["task_ids"]
        == []
    )


@pytest.mark.parametrize("runtime,maximum", [("native", 1), ("apptainer", 2)])
def test_scheduler_reservations_and_native_serial(
    manager, monkeypatch, runtime, maximum
):
    monkeypatch.setattr(manager.runtimes, "verify", lambda: "image")
    manager.configure(
        {"paused": True, "concurrency": 3, "memory_bytes": 8 * GIB, "cpus": 2}
    )
    running, observed, order = set(), [], []
    lock = threading.Lock()
    release = threading.Event()

    def run(task, cancel):
        with lock:
            running.add(task["id"])
            observed.append(len(running))
            order.append(task["candidate"])
        release.wait(1)
        with lock:
            running.remove(task["id"])
        manager.store.update(task["id"], status="succeeded", ended=now())

    monkeypatch.setattr(manager, "run", run)
    for id in ("501", "500", "502"):
        manager.queue("diagram", id, runtime=runtime)
    manager.start()
    time.sleep(0.1)
    assert not order
    manager.configure({"paused": False})
    deadline = time.monotonic() + 1
    while len(order) < maximum and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(order) == maximum
    release.set()
    wait_tasks(manager)
    assert max(observed) == maximum
    assert order == ["501", "500", "502"]
    assert "stdout_tail" not in json.dumps(manager.queue_data())


def test_cancel_batch_restart_and_persisted_pause(manager):
    manager.configure({"paused": True})
    batch = submit(manager, request(["500", "501"]))
    for id in batch["task_ids"]:
        manager.cancel_task(id)
    assert all(t["status"] == "canceled" for t in manager.store.tasks())
    task = manager.queue("diagram", "502")
    restarted = Manager(manager.config, manager.candidates)
    assert restarted.settings["paused"]
    assert restarted.store.get(task["id"])["status"] == "interrupted"
    assert restarted.store.batch(batch["id"]) == batch


def test_container_missing_and_image_changed_do_not_fallback(
    manager, monkeypatch, tmp_path
):
    with pytest.raises(ValueError, match="SIF"):
        manager.queue("diagram", "500", runtime="apptainer")
    assert not manager.store.tasks()
    image = tmp_path / "image.sif"
    image.write_bytes(b"original")
    manager.config.image = image
    identity = manager.runtimes.image_identity()
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="image changed"):
        manager.runtimes.wrap(
            ["pw.x"], tmp_path, {"cpus": 1, "memory_bytes": GIB}, identity
        )
    assert not manager.store.tasks()
    with pytest.raises(ValueError, match="budget"):
        manager.resources("apptainer", 1, 20, None)


def test_container_argv_and_no_host_mpi(manager, monkeypatch, tmp_path):
    manager.config.image = tmp_path / "image.sif"
    manager.config.image.write_bytes(b"image")
    monkeypatch.setattr(
        "dashboard.runtimes.shutil.which", lambda _: "/usr/bin/apptainer"
    )
    command = manager.runtimes.wrap(
        ["mpirun", "-np", "2", "pw.x", "-in", "input.in"],
        tmp_path,
        {"cpus": 2, "memory_bytes": 4 * GIB},
        manager.runtimes.image_identity(),
    )
    assert command[:2] == ["/usr/bin/apptainer", "exec"]
    assert (
        command[command.index("--memory") + 1]
        == command[command.index("--memory-swap") + 1]
        == str(4 * GIB)
    )
    assert command[-6:] == ["mpirun", "-np", "2", "pw.x", "-in", "input.in"]
    assert "--pid" in command and "--cleanenv" in command


def test_api_batch_queue_and_origins(manager, monkeypatch):
    app = create_app(manager.config)
    monkeypatch.setattr(app.state.manager, "start", lambda: None)
    with TestClient(app) as client:
        body = request(["501", "500"])
        review = client.post("/api/batches/preview", json=body).json()
        assert client.get("/api/data").json()["tasks"] == []
        submitted = client.post(
            "/api/batches", json={**body, "expected": review["entries"], "id": "api"}
        ).json()
        assert len(submitted["task_ids"]) == 2
        data = client.get("/api/queue").json()
        assert data["batches"][0]["id"] == "api"
        assert all("stdout_tail" not in t for t in data["tasks"])
        assert client.post("/api/queue/settings", json={"paused": True}).json()[
            "paused"
        ]
        assert client.post("/api/batches/api/cancel").status_code == 200
        assert client.get("/api/batches/api/retry").json()["candidates"] == [
            "501",
            "500",
        ]
        assert (
            client.post(
                "/api/batches", json=body, headers={"origin": "https://evil.test"}
            ).status_code
            == 403
        )
        assert client.get("/api/tasks/missing/logs").status_code == 400


def test_resource_probe_requires_actual_memory_and_swap_limits(tmp_path, monkeypatch):
    from dashboard.container_runner import limits

    (tmp_path / "memory.max").write_text(str(GIB))
    (tmp_path / "memory.swap.max").write_text("0")
    monkeypatch.setattr("dashboard.container_runner.cgroup", lambda: tmp_path)
    assert limits(GIB)["verified"]
    (tmp_path / "memory.max").write_text("max")
    assert not limits(GIB)["verified"]
    (tmp_path / "memory.max").write_text(str(GIB))
    (tmp_path / "memory.swap.max").write_text(str(GIB))
    assert not limits(GIB)["verified"]


def test_recover_partial_batch_acceptance(manager):
    batch = submit(manager, request(["500", "501"]))
    # Simulate a crash after the first task was saved but before its entry was saved.
    manager.store.put_batch(
        {**batch, "entries": [], "task_ids": [], "candidates": ["500", "501", "502"]}
    )
    recovered = Manager(manager.config, manager.candidates)
    batch = recovered.store.batch(batch["id"])
    assert len(batch["task_ids"]) == 2
    assert batch["entries"][-1]["status"] == "unavailable"
    assert recovered.batches.retry(batch["id"])["candidates"] == ["500", "501", "502"]


def test_smiles_and_pseudo_preflight_does_not_block_other_entries(manager, monkeypatch):
    monkeypatch.setattr(
        manager.runtimes,
        "chemistry_probe",
        lambda *a, **k: SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps({"500": None, "501": ["Unknown"], "502": ["H"]}),
        ),
    )
    review = manager.batches.review(**request(["500", "501", "502"], "prepare"))
    assert [e["status"] for e in review["entries"]] == [
        "unavailable",
        "unavailable",
        "eligible",
    ]
    assert "SMILES" in review["entries"][0]["reason"]
    assert "pseudopotentials" in review["entries"][1]["reason"]


def test_cgroup_oom_evidence_and_unknown_kill_are_distinct(tmp_path):
    from dashboard.processes import execute

    report = {
        "verified": True,
        "cgroup": "/sys/fs/cgroup/pfas-test-nonexistent",
        "events_before": "oom_kill 0\n",
        "events_after": "oom_kill 1\n",
        "peak_memory_bytes": 123456,
    }
    (tmp_path / "resource.json").write_text(json.dumps(report))
    samples = []
    code, state, error = execute(
        [sys.executable, "-c", "pass"],
        tmp_path,
        threading.Event(),
        container=True,
        on_usage=samples.append,
        on_start=lambda process: process.wait(timeout=2),
    )
    assert code == 0 and state == "failed" and "Memory limit" in error
    assert samples[-1]["peak_memory_bytes"] >= 123456
    (tmp_path / "resource.json").unlink()
    _, state, error = execute(
        [sys.executable, "-c", "import os,signal;os.kill(os.getpid(),signal.SIGKILL)"],
        tmp_path,
        threading.Event(),
        container=True,
    )
    assert state == "failed" and "Memory limit" not in error


def test_worker_finalization_error_releases_slot(manager, monkeypatch):
    order = []

    def run(task, cancel):
        order.append(task["candidate"])
        if task["candidate"] == "500":
            raise OSError("artifact registration failed")
        manager.store.update(task["id"], status="succeeded", ended=now())

    monkeypatch.setattr(manager, "run", run)
    manager.queue("diagram", "500")
    manager.queue("diagram", "501")
    manager.start()
    wait_tasks(manager)
    assert order == ["500", "501"]
    assert manager.store.tasks()[0]["status"] == "succeeded"
    assert manager.store.tasks()[1]["status"] == "failed"


def test_remote_batch_rejected_before_work(manager):
    with pytest.raises(NotImplementedError):
        manager.batches.review(**request(["500"], target="slurm"))
    with pytest.raises(NotImplementedError):
        manager.queue("prepare", "500", target="slurm")
    assert manager.store.tasks() == []
