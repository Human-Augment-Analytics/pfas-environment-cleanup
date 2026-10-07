import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard.candidates import TFA, load
from dashboard.config import Config
from dashboard.export import export
from dashboard.persistence import now
from dashboard.processes import evidence, execute, tail
from dashboard.server import create_app
from dashboard.tasks import Manager


@pytest.fixture
def manager(tmp_path):
    config = Config(
        artifacts=tmp_path / "artifacts",
        pseudos=tmp_path / "pseudos",
        pw=sys.executable,
        mpi=sys.executable,
    )
    config.pseudos.mkdir()
    (config.pseudos / "H.UPF").write_text("pseudo")
    return Manager(config, {c["id"]: c for c in [*load(), TFA]})


def prepared(manager):
    directory = manager.config.artifacts / "prepared"
    directory.mkdir()
    text = "&CONTROL\n calculation='scf', pseudo_dir='./Pseudopotentials', outdir='./Outputs'\n/\nATOMIC_SPECIES\nH 1.0 H.UPF\n"
    (directory / "candidate.in").write_text(text)
    manager.store.put(
        {
            "id": "prepared",
            "kind": "prepare",
            "candidate": "500",
            "system": "candidate",
            "status": "succeeded",
            "artifacts": {},
            "created": now(),
            "started": None,
            "ended": None,
        }
    )
    return directory / "candidate.in"


def wait(manager, id):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        task = manager.store.get(id)
        if task["status"] not in ("queued", "running"):
            return task
        time.sleep(0.01)
    raise AssertionError("Task did not finish")


def test_selection():
    candidates = load()
    assert [c["id"] for c in candidates] == list(map(str, range(1000)))
    assert len(candidates[0]["fields"]) == 12
    assert candidates[0]["smiles"] == candidates[0]["fields"]["medoid_SMILES"]


def test_commands_snapshots_dedup_and_remote(manager):
    source = prepared(manager)
    one = manager.preview("500", "candidate", 1, "local")
    two = manager.preview("500", "candidate", 2, "local")
    assert one["command"] == [sys.executable, "-in", "input.in"]
    assert two["command"] == [
        sys.executable,
        "-np",
        "2",
        sys.executable,
        "-in",
        "input.in",
    ]
    task = manager.queue("qe", "500", expected_hash=one["input_hash"])
    assert (
        manager.queue("qe", "500", expected_hash=one["input_hash"])["id"] == task["id"]
    )
    directory = manager.config.artifacts / task["id"]
    source.write_text("edited")
    (manager.config.pseudos / "H.UPF").write_text("edited pseudo")
    assert (directory / "input.in").read_text() == one["input"]
    assert (directory / "Pseudopotentials/H.UPF").read_text() == "pseudo"
    assert (directory / "Outputs").is_dir()
    manager.cancel_task(task["id"])
    with pytest.raises(NotImplementedError):
        manager.queue("qe", "500", target="slurm")
    assert len(manager.store.tasks()) == 2
    with pytest.raises(ValueError):
        manager.preview("500", "candidate", 0, "local")


def test_queue_failures_retries_restart(manager):
    manager.config.python = "/nonexistent/chem-python"
    first = manager.queue("diagram", "500")
    second = manager.queue("diagram", "501")
    manager.start()
    assert wait(manager, first["id"])["status"] == "failed"
    assert wait(manager, second["id"])["status"] == "failed"
    retry = manager.queue("diagram", "500")
    assert retry["id"] != first["id"]
    manager.close()
    manager.store.update(retry["id"], status="queued")
    restarted = Manager(manager.config, manager.candidates)
    assert restarted.store.get(retry["id"])["status"] == "interrupted"


def test_process_timeout_cancel_and_tail(tmp_path):
    code, state, error = execute(
        [sys.executable, "-c", "import time;time.sleep(20)"],
        tmp_path,
        threading.Event(),
        0.1,
    )
    assert state == "failed" and error == "Timed out" and code != 0
    event = threading.Event()
    event.set()
    assert (
        execute([sys.executable, "-c", "import time;time.sleep(20)"], tmp_path, event)[
            1
        ]
        == "canceled"
    )
    (tmp_path / "stdout.log").write_text("x" * 30000)
    assert len(tail(tmp_path / "stdout.log")) == 16000


def test_group_cancellation(tmp_path):
    # Child ignores SIGTERM; terminating the leader still kills its children.
    script = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(20)']); print(p.pid,flush=True);time.sleep(20)"
    execute([sys.executable, "-c", script], tmp_path, threading.Event(), 0.15)
    pid = int((tmp_path / "stdout.log").read_text())
    stat = Path(f"/proc/{pid}/stat")
    deadline = time.monotonic() + 0.5
    while (
        stat.exists()
        and stat.read_text().split()[2] != "Z"
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert not stat.exists() or stat.read_text().split()[2] == "Z"


@pytest.mark.parametrize(
    "text,calc,confirmed",
    [
        ("! total energy = -1 Ry", "scf", False),
        (
            "convergence has been achieved\n! total energy = -2 Ry\nJOB DONE.",
            "scf",
            True,
        ),
        ("convergence has been achieved\nJOB DONE.", "relax", False),
        (
            "convergence has been achieved\nEnd of BFGS Geometry Optimization\nJOB DONE.",
            "relax",
            True,
        ),
        ("convergence NOT achieved\nJOB DONE.", "scf", False),
    ],
)
def test_evidence(text, calc, confirmed):
    assert evidence(text, calc)["confirmed"] is confirmed


@pytest.mark.timeout(5)
def test_api_readonly_bulk_and_origins(manager, monkeypatch):
    app = create_app(manager.config)
    with TestClient(app) as client:
        assert len(client.get("/api/data").json()["candidates"]) == 1000
        assert client.get("/api/data").json()["tasks"] == []
        assert (
            client.post(
                "/api/tasks",
                json={"kind": "diagram", "candidate": "500"},
                headers={"origin": "https://evil.test"},
            ).status_code
            == 403
        )
        assert client.get("/api/artifacts/not-registered").status_code == 404
        assert (
            client.post(
                "/api/preview",
                json={"kind": "qe", "candidate": "500", "target": "slurm-array"},
            ).status_code
            == 501
        )
        manager2 = app.state.manager
        manager2.config.python = "/missing"
        existing = manager2.config.artifacts / "existing.png"
        existing.write_bytes(b"png")
        url = manager2.store.register("existing-diagram.png", existing)
        manager2.store.put(
            {
                "id": "existing",
                "kind": "diagram",
                "candidate": "500",
                "system": "candidate",
                "status": "succeeded",
                "artifacts": {"diagram.png": url},
                "created": now(),
                "started": None,
                "ended": None,
            }
        )
        # Exercise selection and skip logic without starting 999 subprocess attempts.
        monkeypatch.setattr(
            manager2,
            "queue",
            lambda kind, candidate: {"kind": kind, "candidate": candidate},
        )
        queued = client.post("/api/diagrams").json()
        assert len(queued) == 999
        assert "500" not in {task["candidate"] for task in queued}


def test_export_is_readonly_and_redacted(manager, tmp_path):
    task = manager.queue("diagram", "500")
    manager.store.update(
        task["id"], error="/private/machine/secret", command=["/private/python"]
    )
    before = (manager.config.artifacts / "tasks.sqlite").read_bytes()
    output = tmp_path / "public"
    data = export(manager.config, output)
    assert data["mode"] == "snapshot"
    assert len(data["candidates"]) == 1000
    text = (output / "data.json").read_text()
    assert (
        "/private" not in text and "stdout_tail" not in text and "command" not in text
    )
    assert (manager.config.artifacts / "tasks.sqlite").read_bytes() == before
    assert manager.store.get(task["id"])["status"] == "queued"


def test_missing_executables_and_changed_preview(manager):
    source = prepared(manager)
    preview = manager.preview("500", "candidate", 1, "local")
    source.write_text(source.read_text() + "\n! external edit\n")
    with pytest.raises(ValueError, match="preview again"):
        manager.queue("qe", "500", expected_hash=preview["input_hash"])
    manager.config.pw = "/missing/pw.x"
    with pytest.raises(ValueError, match="Executable unavailable"):
        manager.preview("500", "candidate", 1, "local")
    manager.config.pw = sys.executable
    manager.config.mpi = "/missing/mpirun"
    manager.preview("500", "candidate", 1, "local")
    with pytest.raises(ValueError, match="Executable unavailable"):
        manager.preview("500", "candidate", 2, "local")
    manager.config.prepare_python = "/missing/python"
    with pytest.raises(OSError):
        manager.queue("prepare", "500")
    assert len(manager.store.tasks()) == 1


def test_shutdown_active_and_queued(manager, tmp_path):
    prepared(manager)
    executable = tmp_path / "fake-pw"
    executable.write_text(
        f'#!{sys.executable}\nimport os,time\nprint(os.environ["OMP_NUM_THREADS"],flush=True)\ntime.sleep(20)\n'
    )
    executable.chmod(0o755)
    manager.config.pw = str(executable)
    preview = manager.preview("500", "candidate", 1, "local")
    active = manager.queue("qe", "500", expected_hash=preview["input_hash"])
    queued = manager.queue("diagram", "501")
    manager.start()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        if tail(manager.config.artifacts / active["id"] / "stdout.log"):
            break
        time.sleep(0.01)
    manager.close()
    assert manager.store.get(active["id"])["status"] == "interrupted"
    assert manager.store.get(queued["id"])["status"] == "interrupted"
    assert tail(manager.config.artifacts / active["id"] / "stdout.log").strip() == "1"
    assert manager.store.get(active["id"])["artifacts"]["stderr.log"]


def test_bulk_failure_does_not_block_success(manager, monkeypatch):
    def fake(command, directory, cancel, timeout, env, **kwargs):
        assert timeout == 60
        if command[2] == manager.candidates["500"]["smiles"]:
            return -11, "failed", "crashed"
        (directory / "diagram.png").write_bytes(b"png")
        return 0, "succeeded", ""

    monkeypatch.setattr("dashboard.tasks.execute", fake)
    first = manager.queue("diagram", "500")
    second = manager.queue("diagram", "501")
    manager.start()
    try:
        assert wait(manager, first["id"])["status"] == "failed"
        assert wait(manager, second["id"])["status"] == "succeeded"
    finally:
        manager.close()
    assert manager.store.get(second["id"])["artifacts"]["diagram.png"]


def test_preparation_interpreter_selection(monkeypatch):

    monkeypatch.delenv("PFAS_CHEM_PYTHON", raising=False)
    assert Config().prepare_python == sys.executable
    assert Config().python == sys.executable
    monkeypatch.setenv("PFAS_CHEM_PYTHON", "/custom/chem/python")
    assert Config().prepare_python == "/custom/chem/python"


def test_existing_preparation_geometry_and_snapshot(manager, tmp_path):
    source = prepared(manager)
    source.write_text("nat=1\nATOMIC_POSITIONS angstrom\nH 1 2 3\n")
    manager.store.register("prepared-candidate.in", source)
    live = manager.data()["tasks"][0]["geometries"]["candidate"]
    assert live["atoms"] == [{"element": "H", "position": [1, 2, 3]}]
    before = (manager.config.artifacts / "tasks.sqlite").read_bytes()
    snapshot = export(manager.config, tmp_path / "snapshot")
    assert snapshot["tasks"][0]["geometries"]["candidate"] == live
    assert (manager.config.artifacts / "tasks.sqlite").read_bytes() == before
    source.write_text("nat=1\nATOMIC_POSITIONS angstrom\nH 4 5 6\n")
    edited = manager.data()["tasks"][0]["geometries"]["candidate"]
    assert edited["atoms"][0]["position"] == [4, 5, 6]
    assert edited["input_hash"] != live["input_hash"]


def test_export_builds_snapshot_without_replacing_live_frontend(tmp_path, monkeypatch):
    from dashboard import __main__ as launcher

    frontend = tmp_path / "web/frontend"
    frontend.mkdir(parents=True)
    (frontend / "dist").mkdir()
    live = frontend / "dist/index.html"
    live.write_text("live build")
    output = tmp_path / "public"
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["dashboard", "export", "--output", str(output)])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "build" in command:
            assert kwargs["env"]["PFAS_DATA_MODE"] == "snapshot"
            Path(command[-1], "index.html").write_text("snapshot build")

    def snapshot(config, destination):
        destination.mkdir(parents=True)
        (destination / "data.json").write_text('{"mode":"snapshot"}')

    monkeypatch.setattr(launcher.subprocess, "run", run)
    monkeypatch.setattr(launcher, "export", snapshot)
    launcher.main()
    assert calls[0][1] == "ci"
    assert live.read_text() == "live build"
    assert (output / "index.html").read_text() == "snapshot build"
    assert (output / "snapshot/data.json").is_file()
