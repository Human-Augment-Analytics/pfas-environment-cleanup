"""Fast smoke test for the dependency-free DFT submission CLI."""

import importlib.util
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
