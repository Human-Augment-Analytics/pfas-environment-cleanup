"""Tests for pw.x exit-code handling and leg-resume in run_adsorption_case.py.

The NGC quantum-espresso GPU container can exit with a nonzero status
(benign IEEE exception flags, e.g. exit code 3) after a fully successful
run that printed "JOB DONE." — the pipeline must trust the marker, and
must not re-run legs whose output already reports completion.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PIPE = REPO / "qespresso_pipeline"


def _install_stubs() -> None:
    """Provide minimal stand-ins so the module imports without chemistry deps."""
    for name in ("numpy", "pymatgen", "pymatgen.core", "pymatgen.io", "pymatgen.io.cif", "pymatgen.io.ase"):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            if name == "pymatgen.core":
                stub.Molecule = object
                stub.Lattice = object
                stub.Structure = object
            elif name == "pymatgen.io.cif":
                stub.CifWriter = object
            elif name == "pymatgen.io.ase":
                stub.AseAtomsAdaptor = object
            sys.modules[name] = stub

    if "smiles_to_qe" not in sys.modules:
        stub = types.ModuleType("smiles_to_qe")
        for fn in ("run_obabel", "mol_to_cif_pymatgen", "run_cif2cell", "modify_qe_input"):
            setattr(stub, fn, lambda *a, **k: None)
        sys.modules["smiles_to_qe"] = stub


def _load_rac(name: str = "run_adsorption_case_under_test"):
    try:
        import numpy  # noqa: F401
        import pymatgen.core  # noqa: F401
    except ImportError:
        _install_stubs()
    else:
        sys.path.insert(0, str(PIPE))
        try:
            import smiles_to_qe  # noqa: F401
        except ImportError:
            _install_stubs()
        finally:
            sys.path.remove(str(PIPE))

    spec = importlib.util.spec_from_file_location(name, PIPE / "run_adsorption_case.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture()
def rac():
    return _load_rac()


class _Result:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


def test_pwx_completed_detects_marker(tmp_path, rac):
    out = tmp_path / "leg.out"
    out.write_text("some output\n     JOB DONE.\n")
    assert rac._pwx_completed(out) is True
    out.write_text("crashed mid-SCF\n")
    assert rac._pwx_completed(out) is False
    assert rac._pwx_completed(tmp_path / "missing.out") is False


def test_run_pwscf_accepts_benign_gpu_exit_code(tmp_path, monkeypatch, rac):
    """Exit code 3 + 'JOB DONE.' (NGC container IEEE-flag artifact) => success."""
    inp = tmp_path / "leg.in"
    inp.write_text("&CONTROL\n")

    def fake_run(cmd, cwd, stdout, stderr, **kwargs):
        stdout.write("     total energy = -232.79 Ry\n\n     JOB DONE.\n")
        stderr.write("Warning: ieee_inexact is signaling\n")
        return _Result(3)

    monkeypatch.setattr(rac.subprocess, "run", fake_run)
    out = rac.run_pwscf(inp, pw_command="pw.x")
    assert out.exists() and "JOB DONE." in out.read_text()


def test_run_pwscf_raises_on_real_failure(tmp_path, monkeypatch, rac):
    inp = tmp_path / "leg.in"
    inp.write_text("&CONTROL\n")

    def fake_run(cmd, cwd, stdout, stderr, **kwargs):
        stdout.write("Error in routine addusdens_gpu (1):\n")
        return _Result(1)

    monkeypatch.setattr(rac.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="Return code: 1"):
        rac.run_pwscf(inp, pw_command="pw.x")


def test_run_pwscf_zero_exit_still_ok(tmp_path, monkeypatch, rac):
    inp = tmp_path / "leg.in"
    inp.write_text("&CONTROL\n")

    def fake_run(cmd, cwd, stdout, stderr, **kwargs):
        stdout.write("     JOB DONE.\n")
        return _Result(0)

    monkeypatch.setattr(rac.subprocess, "run", fake_run)
    out = rac.run_pwscf(inp, pw_command="pw.x")
    assert "JOB DONE." in out.read_text()


def test_run_pwscf_skips_completed_leg(tmp_path, monkeypatch, rac):
    """Resume: a leg whose .out already says 'JOB DONE.' must not re-run."""
    inp = tmp_path / "leg.in"
    inp.write_text("&CONTROL\n")
    out = tmp_path / "leg.out"
    out.write_text("     JOB DONE.\n")

    calls: list[str] = []

    def fake_run(*a, **k):
        calls.append("ran")
        raise AssertionError("pw.x must not be invoked for a completed leg")

    monkeypatch.setattr(rac.subprocess, "run", fake_run)
    result = rac.run_pwscf(inp, pw_command="pw.x")
    assert calls == []
    assert result == out
