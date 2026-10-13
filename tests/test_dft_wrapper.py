"""Fast smoke test for the dependency-free DFT submission CLI."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]

_SPEC = importlib.util.spec_from_file_location(
    "dft_wrapper_under_test", ROOT / "scripts" / "dft_wrapper.py"
)
assert _SPEC is not None and _SPEC.loader is not None
WRAPPER = importlib.util.module_from_spec(_SPEC)
# Register before exec: the module's dataclasses resolve annotations via
# sys.modules[cls.__module__] under `from __future__ import annotations`.
sys.modules[_SPEC.name] = WRAPPER
_SPEC.loader.exec_module(WRAPPER)

TFA = "OC(=O)C(F)(F)F"  # 7 heavy atoms
CTAB = "CCCCCCCCCCCCCCCC[N+](C)(C)C.[Br-]"  # 21 heavy atoms
PFOA = "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"  # 25 heavy atoms


def test_dft_wrapper_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/dft_wrapper.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "DFT wrapper: submit-if-missing and fetch results." in result.stdout


def test_ssh_opts_control_windows_space_safe() -> None:
    """Windows: no Control* options (spaces in %USERPROFILE% would break
    Win32-OpenSSH's -o parsing); POSIX: full multiplexing options present."""
    c = WRAPPER.Cluster(
        ssh_target="user@host",
        control_path="/home/spaced user/.ssh/cm_sockets/cm-user@host",
    )
    opts = WRAPPER.ssh_opts_control(c)
    joined = " ".join(opts)
    if sys.platform == "win32" or os.name == "nt":
        assert "ControlMaster" not in joined
        assert "ControlPersist" not in joined
        assert "ControlPath" not in joined
    else:
        assert "ControlMaster=auto" in joined
        assert "ControlPersist=10m" in joined
        assert "ControlPath=/home/spaced user/.ssh/cm_sockets/cm-user@host" in joined
    # Both platforms keep the keepalive + quiet options.
    assert "ServerAliveInterval=30" in joined
    assert "LogLevel=ERROR" in joined
    assert "-q" in opts


def test_heavy_atom_count() -> None:
    assert WRAPPER.heavy_atom_count("CCO") == 3  # ethanol
    assert WRAPPER.heavy_atom_count("c1ccccc1") == 6  # benzene, aromatic subset
    assert WRAPPER.heavy_atom_count(CTAB) == 21  # 16-chain + N+ + 3 methyl C + Br-
    assert WRAPPER.heavy_atom_count("[13C](C)(C)C") == 4  # isotope prefix ignored
    assert WRAPPER.heavy_atom_count("") == 0
    assert WRAPPER.heavy_atom_count(None) == 0
    assert WRAPPER.heavy_atom_count("C1CC1") == 3  # ring-closure digits ignored


def test_memory_warnings_small_pair_fits_32() -> None:
    # A TFA-sized pair (~14 heavy atoms) needs no warning at 32 GB.
    assert (
        WRAPPER.memory_warnings(adsorbent_smiles=TFA, pfas_smiles=TFA, mem_gb=32) == []
    )


def test_memory_warnings_suggests_64() -> None:
    # CTAB-sized single molecule at 32 GB: suggest the README's 64 GB row.
    warnings = WRAPPER.memory_warnings(
        adsorbent_smiles=CTAB, pfas_smiles=TFA, mem_gb=32
    )
    assert len(warnings) == 1
    assert "--mem-gb 32 looks small" in warnings[0]
    assert "--mem-gb 64" in warnings[0]


def test_memory_warnings_large_complex_137_note() -> None:
    # Very large complex at the 64 GB default: the rc-137 / 96 GB note.
    warnings = WRAPPER.memory_warnings(
        adsorbent_smiles=CTAB, pfas_smiles=PFOA, mem_gb=64
    )
    assert len(warnings) == 1
    assert "return code 137" in warnings[0]
    assert "--mem-gb 96" in warnings[0]


def test_memory_warnings_respects_skip_and_energy_flags() -> None:
    # Skipping the adsorbent side leaves only the small PFAS SMILES.
    assert (
        WRAPPER.memory_warnings(
            adsorbent_smiles=CTAB,
            pfas_smiles=TFA,
            skip_ads=True,
            skip_complex=True,
            mem_gb=32,
        )
        == []
    )
    # --pfas-energy-ry replaces the PFAS SMILES calculation.
    assert (
        WRAPPER.memory_warnings(
            adsorbent_smiles=TFA,
            pfas_smiles=PFOA,
            pfas_energy_ry=-1.0,
            mem_gb=32,
        )
        == []
    )
    # A CIF-sourced adsorbent has no SMILES: only the PFAS side is visible.
    assert (
        WRAPPER.memory_warnings(adsorbent_smiles=None, pfas_smiles=TFA, mem_gb=32) == []
    )


def _render(**backend_kwargs):
    """Render a job.sbatch with production-shaped positionals plus the
    keyword-only GPU knobs (backend/gres/ntasks/cpus_per_task)."""
    c = WRAPPER.Cluster(ssh_target="user@host", root="/remote/root")
    return WRAPPER.render_job_sbatch(
        c,
        "case_TFA",
        "/remote/root/scripts/run_dft_workflow.sh",
        "ice-cpu",
        "coc",
        "coc-ice",
        "18:00:00",
        8,
        96,
        False,
        False,
        False,
        "ads",
        "TFA",
        "C",
        TFA,
        None,
        "smiles",
        None,
        "molecule",
        "production",
        **backend_kwargs,
    )


def test_render_job_sbatch_gpu_shape() -> None:
    # GPU track: 1 task x N threads on 1 GPU (ntasks=cpus with srun would
    # launch N container copies on one GPU), a type-pinned gres line, and
    # the PFAS_DFT_BACKEND export consumed by run_dft_workflow.sh.
    text = _render(
        backend="gpu",
        gres="gpu:h100:1",
        ntasks=1,
        cpus_per_task=8,
    )
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --cpus-per-task=8" in text
    assert "#SBATCH --gres=gpu:h100:1" in text
    assert "export PFAS_DFT_BACKEND=gpu" in text


def test_render_job_sbatch_cpu_default_unchanged() -> None:
    # CPU default (no backend kwargs): byte-for-byte legacy shape - no
    # gres line, no backend export, ntasks=cpus / cpus-per-task=1.
    text = _render()
    assert "#SBATCH --ntasks=8" in text
    assert "#SBATCH --cpus-per-task=1" in text
    assert "--gres" not in text
    assert "PFAS_DFT_BACKEND" not in text
