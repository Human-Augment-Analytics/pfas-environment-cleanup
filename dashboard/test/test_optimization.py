"""Saved QE logs must support inspection and safe coordinate-only continuation."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dashboard.app import server
from dashboard.core import candidates, config, execution
from dashboard.core.chemistry import optimization, results
from dashboard.core.jobs import processes
from dashboard.core.jobs.tasks import Manager

INPUT = """&CONTROL
 calculation='relax', restart_mode='restart',
 pseudo_dir='./Pseudopotentials', outdir='./Outputs',
 forc_conv_thr=1.d-3, etot_conv_thr=1.d-4,
/
&SYSTEM
 ibrav=0, nat=2, ntyp=1,
/
ATOMIC_SPECIES
H 1.0 H.UPF
CELL_PARAMETERS angstrom
10 0 0
0 10 0
0 0 10
ATOMIC_POSITIONS angstrom
H 0 0 0 0 0 0
H 1 0 0 1 1 1
K_POINTS gamma
"""
OUTPUT = """convergence has been achieved in 12 iterations
!    total energy = -10.5D+00 Ry
Total force = 0.020 Total SCF correction = 0.001
ATOMIC_POSITIONS (angstrom)
H 0 0 0
H 0.8 0 0
convergence has been achieved in 4 iterations
!    total energy = -10.6 Ry
Total force = 0.0002 Total SCF correction = 0.00001
End of BFGS Geometry Optimization
Begin final coordinates
ATOMIC_POSITIONS (angstrom)
H 0 0 0
H 0.7 0 0
End final coordinates
The total energy is the sum of the following terms:
JOB DONE.
"""


def saved_run(directory: Path) -> None:
    (directory / execution.INPUT_FILE).write_text(INPUT)
    (directory / execution.STDOUT_LOG_FILE).write_text(OUTPUT)


def test_statistics_and_latest_geometry(tmp_path: Path) -> None:
    saved_run(tmp_path)
    report = optimization.report(tmp_path)
    assert report["steps"] == [
        {"energy_ry": -10.5, "total_force_ry_bohr": 0.02, "scf_iterations": 12},
        {"energy_ry": -10.6, "total_force_ry_bohr": 0.0002, "scf_iterations": 4},
    ]
    assert report["latest_geometry"] is not None
    assert report["latest_geometry"]["atoms"][1]["position"] == [0.7, 0, 0]
    assert report["force_threshold_ry_bohr"] == 0.001
    assert report["energy_threshold_ry"] == 0.0001
    assert report["continuation_error"] == ""


def test_continuation_preserves_input_constraints_and_source(tmp_path: Path) -> None:
    saved_run(tmp_path)
    continued = optimization.continuation(tmp_path)
    assert "H 0.7 0 0 1 1 1" in continued
    assert "H 0 0 0 0 0 0" in continued
    assert "restart_mode='from_scratch'" in continued
    assert "K_POINTS gamma" in continued
    assert (tmp_path / execution.INPUT_FILE).read_text() == INPUT
    assert "restart_mode='from_scratch'" in optimization.fresh_start(
        INPUT.replace(", restart_mode='restart'", "")
    )


def test_partial_output_keeps_last_complete_positions() -> None:
    partial = OUTPUT + "ATOMIC_POSITIONS (angstrom)\nH 0 0 0\n"
    output = optimization.read_output(iter(partial.splitlines()), 2)
    assert output["positions"][-1] == "H 0.7 0 0"
    assert len(output["steps"]) == 2


@pytest.mark.parametrize(
    "text, positions, problem",
    [
        (INPUT, [], "No complete"),
        (
            INPUT.replace("'relax'", "'vc-relax'"),
            ["ATOMIC_POSITIONS angstrom"],
            "fixed-cell",
        ),
        (INPUT, ["ATOMIC_POSITIONS angstrom", "F 0 0 0", "H 1 0 0"], "atom order"),
        (INPUT, ["ATOMIC_POSITIONS angstrom", "H 0 0 0"], "atom count"),
    ],
)
def test_unsafe_continuation_is_rejected(
    text: str, positions: list[str], problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        optimization.replace_positions(text, positions)


def test_missing_force_and_failed_scf_are_explicit() -> None:
    output = optimization.read_output(
        ["! total energy = -1 Ry", "convergence NOT achieved after 100 iterations"], 0
    )
    assert output["steps"][0]["total_force_ry_bohr"] is None
    assert output["steps"][0]["scf_iterations"] is None
    assert output["failed_scf_cycles"] == 1


def test_energy_summary_does_not_erase_final_energy(tmp_path: Path) -> None:
    saved_run(tmp_path)
    evidence = processes.evidence(results.scientific_output(tmp_path), "relax")
    assert evidence["energy_ry"] == -10.6
    assert evidence["confirmed"]


@pytest.fixture
def manager(tmp_path: Path) -> Manager:
    configuration = config.Config(
        execution="local",
        artifacts=tmp_path / "artifacts",
        pseudos=tmp_path / "pseudos",
        qe_inputs=tmp_path / "inputs",
    )
    configuration.pseudos.mkdir()
    (configuration.pseudos / "H.UPF").write_text("pseudo")
    manager = Manager(
        configuration, {record["id"]: record for record in candidates.load()}
    )
    directory = configuration.artifacts / "original"
    directory.mkdir()
    saved_run(directory)
    manager.store.put(
        {
            "id": "original",
            "kind": "qe",
            "candidate": "500",
            "system": "candidate",
            "status": "succeeded",
            "artifacts": {},
        }
    )
    return manager


def test_continuation_is_previewed_and_snapshotted(manager: Manager) -> None:
    version = manager.continuation_versions()[0]["input_id"]
    preview = manager.preview("500", "candidate", 1, "local", input_id=version)
    task = manager.queue(
        "qe",
        "500",
        input_id=version,
        expected_hash=preview["input_hash"],
        preview_id=preview["preview_id"],
    )
    copied = manager.config.artifacts / task["id"] / execution.INPUT_FILE
    assert copied.read_text() == preview["input"]
    assert "H 0.7 0 0 1 1 1" in copied.read_text()
    assert task["input_id"] == version
    assert manager.store.get("original")["status"] == "succeeded"


def test_continuation_rejects_changed_output_after_preview(manager: Manager) -> None:
    version = manager.continuation_versions()[0]["input_id"]
    preview = manager.preview("500", "candidate", 1, "local", input_id=version)
    log = manager.config.artifacts / "original" / execution.STDOUT_LOG_FILE
    log.write_text(OUTPUT.replace("H 0.7 0 0", "H 0.6 0 0"))
    with pytest.raises(ValueError, match="preview again"):
        manager.queue(
            "qe",
            "500",
            input_id=version,
            expected_hash=preview["input_hash"],
            preview_id=preview["preview_id"],
        )


def test_continuation_rejects_other_system_or_active_task(manager: Manager) -> None:
    version = manager.continuation_versions()[0]["input_id"]
    with pytest.raises(ValueError, match="candidate/system"):
        manager.preview("501", "candidate", 1, "local", input_id=version)
    manager.store.update("original", status="running")
    assert manager.continuation_versions() == []
    with pytest.raises(ValueError, match="finished QE"):
        manager.preview("500", "candidate", 1, "local", input_id=version)


def test_statistics_api_reads_existing_history(manager: Manager) -> None:
    app = FastAPI()
    app.state.manager = manager
    app.include_router(server.get_router())
    client = TestClient(app)
    response = client.get("/api/tasks/original/optimization")
    assert response.status_code == 200
    assert response.json()["steps"][-1]["energy_ry"] == -10.6
    assert client.get("/api/tasks/missing/optimization").status_code == 400
    manager.store.update("original", kind="prepare")
    assert client.get("/api/tasks/original/optimization").status_code == 400
