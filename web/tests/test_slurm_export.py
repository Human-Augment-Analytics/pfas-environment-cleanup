import hashlib
import io
import json
import os
import subprocess
import zipfile

import pytest
from fastapi.testclient import TestClient

from dashboard.config import Config
from dashboard.server import create_app

INPUT = """&CONTROL
 calculation='relax', pseudo_dir='./Pseudopotentials', outdir='./Outputs'
/
&SYSTEM
 nat=1, ntyp=1, ibrav=1, celldm(1)=12, ecutwfc=15
/
ATOMIC_SPECIES
H 1.0 H.UPF
ATOMIC_POSITIONS angstrom
H 0 0 0
K_POINTS gamma
"""


@pytest.fixture
def client(tmp_path):
    pseudos = tmp_path / "pseudos"
    pseudos.mkdir()
    (pseudos / "H.UPF").write_bytes(b"original pseudo")
    # Export must work without a local QE executable or a configured SIF.
    app = create_app(
        Config(
            artifacts=tmp_path / "artifacts",
            pseudos=pseudos,
            pw="missing-pw.x",
            image=None,
        )
    )
    manager = app.state.manager
    for candidate in [*map(str, range(500, 508)), "tfa"]:
        directory = manager.config.artifacts / ("prepared-" + candidate)
        directory.mkdir()
        for system in ("candidate", "complex", "tfa"):
            (directory / (system + ".in")).write_text(INPUT)
        manager.store.put(
            {
                "id": directory.name,
                "kind": "prepare",
                "candidate": candidate,
                "status": "succeeded",
                "artifacts": {},
            }
        )
    yield TestClient(app)
    manager.close()


def export(client, **extra):
    request = {
        "candidates": list(map(str, range(500, 508))),
        "memory_mode": "manual",
        **extra,
    }
    response = client.post("/api/slurm/preview", json=request)
    assert response.status_code == 200, response.text
    reviewed = response.json()
    response = client.post("/api/slurm/export", json={**request, "expected": reviewed})
    return response, reviewed


def test_eight_molecules_portable_bundle_and_no_jobs(client, tmp_path):
    response, reviewed = export(client)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert len(reviewed["entries"]) == 16
    manager = client.app.state.manager
    assert all(t["kind"] == "prepare" for t in manager.store.tasks())
    assert manager.store.batches() == []
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    assert not any(n.endswith(".sif") for n in archive.namelist())
    for entry in reviewed["entries"]:
        prefix = "pfas-slurm-batch/" + entry["directory"]
        content = archive.read(prefix + "/input.in")
        assert content == INPUT.encode()
        assert hashlib.sha256(content).hexdigest() == entry["input_hash"]
        assert archive.read(prefix + "/Pseudopotentials/H.UPF") == b"original pseudo"
        assert prefix + "/Outputs/" in archive.namelist()
    root = tmp_path / "folder with spaces"
    archive.extractall(root)
    batch = root / "pfas-slurm-batch"
    for script in ("run.sbatch", "submit.sh"):
        subprocess.run(["bash", "-n", str(batch / script)], check=True)
    manifest = json.loads((batch / "manifest.json").read_text())
    assert manifest == reviewed

    # Exercise the generated submission script with a fake scheduler, preserving
    # paths containing spaces and recording all 16 accepted IDs.
    binary = tmp_path / "bin"
    binary.mkdir()
    scheduler = binary / "sbatch"
    scheduler.write_text("""#!/bin/bash
set -eu
[[ "$*" == *"--ntasks=1"* && "$*" == *"--cpus-per-task=8"* && "$*" == *"--mem=16384"* ]]
for arg in "$@"; do
  if [[ "$arg" == --chdir=* ]]; then
    test -f "${arg#--chdir=}/input.in"
    test -f "${arg#--chdir=}/Pseudopotentials/H.UPF"
  fi
done
echo 12345
""")
    scheduler.chmod(0o755)
    (batch / "chemistry.sif").write_text("fake image")
    env = {**os.environ, "PATH": str(binary) + ":" + os.environ["PATH"]}
    result = subprocess.run(
        ["bash", str(batch / "submit.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert len((batch / "jobs.tsv").read_text().splitlines()) == 16
    repeated = subprocess.run(
        ["bash", str(batch / "submit.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert repeated.returncode != 0
    assert "submitted jobs" in repeated.stderr

    # Slurm copies the job script into its spool directory. The script must use
    # the exported batch argument, rather than locating data beside that copy.
    launcher = binary / "apptainer"
    launcher.write_text("""#!/bin/bash
set -eu
[[ "$*" == *"-np 8 pw.x -in input.in"* ]]
[[ "$*" == *"--mca plm isolated"* ]]
test -f input.in
test -d Outputs
echo 'MPI job launched'
""")
    launcher.chmod(0o755)
    spool = tmp_path / "slurm-spool"
    spool.mkdir()
    script = spool / "slurm_script"
    script.write_bytes((batch / "run.sbatch").read_bytes())
    run_dir = batch / "runs/500-candidate"
    subprocess.run(
        ["bash", str(script), str(batch / "chemistry.sif"), str(run_dir), str(batch)],
        env=env,
        check=True,
    )
    assert (run_dir / "pw.out").read_text() == "MPI job launched\n"


def test_one_process_runs_pw_directly(client):
    response, reviewed = export(client, processes=1)
    assert response.status_code == 200
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    script = archive.read("pfas-slurm-batch/run.sbatch").decode()
    assert "mpirun" not in script
    assert "mpirun" not in reviewed["entries"][0]["command"]


def test_tfa_once_and_configurable_reservations(client):
    response, reviewed = export(
        client,
        include_tfa=True,
        memory_gib=24.5,
        walltime_hours=4,
        partition="",
        account="my-account",
        qos="",
        processes=16,
    )
    assert response.status_code == 200
    assert len(reviewed["entries"]) == 17
    assert sum(e["system"] == "tfa" for e in reviewed["entries"]) == 1
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    script = archive.read("pfas-slurm-batch/submit.sh").decode()
    assert "--mem=25088" in script and "--time=4:00:00" in script
    assert "--account=my-account" in script
    assert "--partition" not in script and "--qos" not in script
    assert "--cpus-per-task=16" in script
    run = archive.read("pfas-slurm-batch/run.sbatch").decode()
    assert "-np 16 pw.x" in run
    assert "--mca plm isolated" in run
    assert "mpirun" in reviewed["entries"][0]["command"]


@pytest.mark.parametrize("change", ["input", "pseudo", "settings"])
def test_changed_preview_requires_review(client, change):
    request = {"candidates": ["500"], "memory_mode": "manual"}
    reviewed = client.post("/api/slurm/preview", json=request).json()
    manager = client.app.state.manager
    if change == "input":
        manager.input("500", "complex").write_text(INPUT + "! edited\n")
    elif change == "pseudo":
        (manager.config.pseudos / "H.UPF").write_text("changed")
    else:
        request["memory_gib"] = 32
    result = client.post("/api/slurm/export", json={**request, "expected": reviewed})
    assert result.status_code == 400
    assert "preview again" in result.json()["detail"]


def test_incomplete_pair_blocks_export(client):
    client.app.state.manager.input("500", "complex").unlink()
    response, reviewed = export(client)
    assert reviewed["entries"][1]["status"] == "unavailable"
    assert response.status_code == 400
    assert "Prepare both inputs" in response.json()["detail"]


@pytest.mark.parametrize(
    "extra",
    [
        {"candidates": []},
        {"candidates": ["500", "500"]},
        {"candidates": ["../500"]},
        {"candidates": ["tfa"]},
        {"memory_gib": 0},
        {"walltime_hours": 0},
        {"walltime_hours": 1.5},
        {"processes": 0},
        {"processes": 1.5},
        {"processes": 257},
        {"account": "coc\n--wrap=evil"},
        {"partition": "ice;echo evil"},
    ],
)
def test_invalid_export_settings(client, extra):
    response = client.post("/api/slurm/preview", json={"candidates": ["500"], **extra})
    assert response.status_code in (400, 422)


def test_unreviewed_and_cross_origin_export_rejected(client):
    assert (
        client.post("/api/slurm/export", json={"candidates": ["500"]}).status_code
        == 400
    )
    assert (
        client.post(
            "/api/slurm/export",
            json={"candidates": ["500"]},
            headers={"Origin": "https://example.com"},
        ).status_code
        == 403
    )


def add_estimate(
    client,
    candidate="500",
    system="candidate",
    amount=512 * 1024**2,
    processes=1,
    total=None,
    **overrides,
):
    manager = client.app.state.manager
    task = {
        "id": f"estimate-{len(manager.store.tasks())}",
        "kind": "estimate_ram",
        "status": "succeeded",
        "candidate": candidate,
        "system": system,
        "processes": processes,
        "source_hash": hashlib.sha256(
            manager.input(candidate, system).read_bytes()
        ).hexdigest(),
        "pseudopotentials": {
            "H.UPF": hashlib.sha256(
                (manager.config.pseudos / "H.UPF").read_bytes()
            ).hexdigest()
        },
        "ram_estimate": {
            "per_process": {"bytes": amount},
            "total": {"bytes": total} if total is not None else None,
        },
        **overrides,
    }
    manager.store.put(task)
    return task


@pytest.mark.parametrize("processes", [8, 16])
def test_automatic_memory_serial_baseline_independent_of_mpi_count(client, processes):
    add_estimate(client)
    add_estimate(client, system="complex", amount=1024**3)
    add_estimate(client, candidate="tfa", system="tfa", amount=2 * 1024**3)
    response, reviewed = export(
        client,
        candidates=["500"],
        include_tfa=True,
        memory_mode="estimate",
        processes=processes,
    )
    assert response.status_code == 200
    assert reviewed["settings"]["headroom_percent"] == 50
    assert [e["memory_mib"] for e in reviewed["entries"]] == [1024, 2048, 3072]
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    script = archive.read("pfas-slurm-batch/submit.sh").decode()
    assert all(f"--mem={m}" in script for m in (1024, 2048, 3072))
    assert all(e["memory_estimate"]["processes"] == 1 for e in reviewed["entries"])


def test_serial_total_preferred_and_custom_headroom(client):
    for system in ("candidate", "complex"):
        add_estimate(client, system=system, amount=1024**3, total=3 * 1024**3)
    response, reviewed = export(
        client, candidates=["500"], memory_mode="estimate", headroom_percent=100
    )
    assert response.status_code == 200
    assert all(e["memory_mib"] == 6144 for e in reviewed["entries"])


def test_optional_tfa_without_estimate_blocks_only_when_included(client):
    for system in ("candidate", "complex"):
        add_estimate(client, system=system)
    response, reviewed = export(
        client, candidates=["500"], memory_mode="estimate", include_tfa=True
    )
    assert response.status_code == 400
    assert [e["status"] for e in reviewed["entries"]] == [
        "eligible",
        "eligible",
        "unavailable",
    ]
    assert "No matching RAM estimate" in reviewed["entries"][-1]["reason"]
    response, _ = export(
        client, candidates=["500"], memory_mode="estimate", include_tfa=False
    )
    assert response.status_code == 200


def test_matching_rank_total_preferred_over_newer_serial_and_rounding(client):
    for system in ("candidate", "complex"):
        add_estimate(
            client, system=system, processes=8, amount=256 * 1024**2, total=2 * 1024**3
        )
        add_estimate(client, system=system, amount=10 * 1024**3)
    response, reviewed = export(client, candidates=["500"], memory_mode="estimate")
    assert response.status_code == 200
    assert [e["memory_mib"] for e in reviewed["entries"]] == [3072, 3072]
    assert all(e["memory_estimate"]["processes"] == 8 for e in reviewed["entries"])


@pytest.mark.parametrize(
    "overrides",
    [
        {"source_hash": "stale"},
        {"pseudopotentials": {}},
        {"status": "failed"},
        {"processes": 4},
        {"ram_estimate": {"per_process": {"bytes": -1}}},
    ],
)
def test_automatic_memory_rejects_unusable_estimates(client, overrides):
    add_estimate(client, **overrides)
    add_estimate(client, system="complex")
    response, reviewed = export(client, candidates=["500"], memory_mode="estimate")
    assert response.status_code == 400
    assert reviewed["entries"][0]["status"] == "unavailable"
    assert "No matching RAM estimate" in reviewed["entries"][0]["reason"]


def test_new_estimate_invalidates_preview(client):
    for system in ("candidate", "complex"):
        add_estimate(client, system=system)
    request = {"candidates": ["500"]}
    reviewed = client.post("/api/slurm/preview", json=request).json()
    add_estimate(client, amount=1024**3)
    result = client.post("/api/slurm/export", json={**request, "expected": reviewed})
    assert result.status_code == 400
    assert "preview again" in result.json()["detail"]


def test_missing_estimates_can_use_manual_and_minimum_memory(client):
    response, reviewed = export(client, candidates=["500"], memory_mode="estimate")
    assert response.status_code == 400
    assert all(e["status"] == "unavailable" for e in reviewed["entries"])
    response, _ = export(client, candidates=["500"], memory_mode="manual")
    assert response.status_code == 200
    for system in ("candidate", "complex"):
        add_estimate(client, system=system, amount=1024**2)
    response, reviewed = export(
        client, candidates=["500"], memory_mode="estimate", headroom_percent=0
    )
    assert response.status_code == 200
    assert all(e["memory_mib"] == 1024 for e in reviewed["entries"])
