"""Fast offline tests for the single-case DFT runner (no SSH, no network)."""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dft_single.py"


def _load_single(name: str = "dft_single_under_test"):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


TFA = "OC(=O)C(F)(F)F"  # 7 heavy atoms
A17 = "CCCCCCCCCCCCCCCCN"  # 17 heavy atoms
A21 = "CCCCCCCCCCCCCCCCCCCCC"  # 21 heavy atoms
P25 = "CCCCCCCCCCCCCCCCCCCCCCCCC"  # 25 heavy atoms


def test_dft_single_help() -> None:
    # build_parser() constructs both subparsers, so the top-level and
    # run help cover the CLI surface; two interpreter starts keep this
    # comfortably inside the 2 s per-test timeout.
    for argv in (["--help"], ["run", "--help"]):
        p = subprocess.run(
            [sys.executable, str(SCRIPT), *argv],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert p.returncode == 0, p.stderr


def test_wrapper_is_reused_not_duplicated() -> None:
    single = _load_single()
    assert hasattr(single.wrapper, "render_job_sbatch")
    assert hasattr(single.wrapper, "Cluster")


def test_sanitize_name_matches_batch_script_rule() -> None:
    single = _load_single()
    assert single.sanitize_name("Crown Ether / 18-Crown-6") == "Crown_Ether_18-Crown-6"
    assert single.sanitize_name(" foo! ") == "foo"
    assert single.sanitize_name("CTAB") == "CTAB"


def test_default_case_name() -> None:
    single = _load_single()
    assert single.default_case_name("CTAB", "TFA") == "CTAB_TFA"
    assert single.default_case_name("Crown Ether", "TFA") == "Crown_Ether_TFA"


def test_suggest_mem_gb_bands() -> None:
    single = _load_single()
    assert single.suggest_mem_gb(TFA, TFA) == 32  # pair total 14
    assert single.suggest_mem_gb(A21, TFA) == 96  # 28 -> medium band
    assert single.suggest_mem_gb(A17, P25) == 128  # 42 -> large band
    assert single.suggest_mem_gb(A17, TFA) == 96  # 24 -> medium band


def test_classify_e_ads_rubric() -> None:
    single = _load_single()
    assert single.classify_e_ads(None)[0] == "INCONCLUSIVE"
    assert single.classify_e_ads(math.nan)[0] == "INCONCLUSIVE"
    assert single.classify_e_ads(0.5)[0] == "NOT BINDING"
    assert single.classify_e_ads(0.0)[0] == "NOT BINDING"
    assert single.classify_e_ads(-0.4)[0] == "PLAUSIBLE (weak)"
    assert single.classify_e_ads(-2.7)[0] == "PLAUSIBLE"
    assert single.classify_e_ads(-6.35)[0] == "SUSPECT"


def test_parse_sbatch_submission() -> None:
    single = _load_single()
    assert single.parse_sbatch_submission("Submitted batch job 6056851\n") == 6056851
    assert single.parse_sbatch_submission("sbatch: error\n") is None


def test_parse_sacct_output() -> None:
    single = _load_single()
    text = (
        "6056851|RUNNING|1:0||\n"
        "6056851.batch|RUNNING|1:0|77G|01:14:25\n"
        "6056851.extern|PENDING|0:0||\n"
    )
    rows = single.parse_sacct_output(text)
    assert single.sacct_main_row(rows, 6056851)["state"] == "RUNNING"
    batch = next(r for r in rows if r["jobid"] == "6056851.batch")
    assert batch["maxrss"] == "77G"
    assert single.sacct_main_row([], 1) is None


def test_parse_squeue_line_takes_latest() -> None:
    single = _load_single()
    assert single.parse_squeue_line("6056851 RUNNING\n6057000 PENDING\n") == (
        6057000,
        "PENDING",
    )
    assert single.parse_squeue_line("") is None


def test_is_oom_row() -> None:
    single = _load_single()
    assert single.is_oom_row({"state": "OUT_OF_MEMORY", "exit_code": "0:0"})
    assert single.is_oom_row({"state": "FAILED", "exit_code": "137:0"})
    assert not single.is_oom_row({"state": "COMPLETED", "exit_code": "0:0"})


def test_format_results_report_verdict_and_warnings() -> None:
    single = _load_single()
    data = {
        "case_name": "CTAB_TFA",
        "adsorbent_name": "CTAB",
        "pfas_name": "TFA",
        "pfas_energy_source": "parsed",
        "mode": "production",
        "system_type": "molecule",
        "energies_ry": {
            "adsorbent": -1.0,
            "pfas": -2.0,
            "complex": -3.5,
            "adsorption": -0.5,
        },
        "energies_ev": {"adsorption": -6.8},
    }
    report = single.format_results_report(data)
    assert "SUSPECT" in report
    assert "-6.800 eV" in report
    assert "-656 kJ/mol" in report  # -6.8 * 96.485

    data["energies_ry"]["adsorbent"] = 0.0
    report = single.format_results_report(data)
    assert "adsorbent energy is 0.0" in report


def test_format_results_report_accepts_batch_array() -> None:
    # The workflow writes results.json as a JSON array (batch format);
    # a single-case run dir contains one entry, which must render the same.
    single = _load_single()
    data = {
        "case_name": "TFA_dimer_selfcheck_TFA",
        "adsorbent_name": "TFA_dimer_selfcheck",
        "pfas_name": "TFA",
        "pfas_energy_source": "parsed",
        "mode": "production",
        "system_type": "molecule",
        "energies_ry": {
            "adsorbent": -299.48344051,
            "pfas": -299.48344053,
            "complex": -598.96793458,
            "adsorption": -0.00105354,
        },
        "energies_ev": {"adsorption": -0.014334},
    }
    from_dict = single.format_results_report(dict(data))
    from_array = single.format_results_report([dict(data)])
    assert from_dict == from_array
    assert "TFA_dimer_selfcheck_TFA" in from_array
    assert "-0.014 eV" in from_array
    assert "PLAUSIBLE" in from_array  # weak H-bond dimer without dispersion

    try:
        single.format_results_report([])
    except ValueError as exc:
        assert "empty" in str(exc)
    else:  # pragma: no cover - should not happen
        raise AssertionError("empty array must raise ValueError")


def test_run_parser_backend_flags() -> None:
    single = _load_single()
    parser = single.build_parser()
    # Defaults: cpu backend, no gres (legacy CPU submissions unchanged).
    args = parser.parse_args(
        ["run", "--user", "u", "--name", "X", "--smiles", "CCO"]
    )
    assert args.backend == "cpu"
    assert args.gres is None
    # Explicit GPU track survives parsing.
    args = parser.parse_args(
        [
            "run", "--user", "u", "--name", "X", "--smiles", "CCO",
            "--backend", "gpu", "--gres", "gpu:h100:1",
        ]
    )
    assert args.backend == "gpu"
    assert args.gres == "gpu:h100:1"


def test_apply_backend_defaults_gpu_flips_cpu_defaults() -> None:
    # --backend gpu without explicit scheduling flags must never render an
    # ice-cpu 18h job (it would either never schedule with a gres request
    # or hit the ice-gpu 16h MaxTime rejection).
    single = _load_single()
    args = single.build_parser().parse_args(
        ["run", "--user", "u", "--name", "X", "--smiles", "CCO", "--backend", "gpu"]
    )
    single.apply_backend_defaults(args)
    assert args.partition == single.DEFAULT_GPU_PARTITION
    assert args.time == single.DEFAULT_GPU_TIME
    assert args.gres == single.DEFAULT_GPU_GRES


def test_apply_backend_defaults_explicit_flags_win() -> None:
    # Explicit user scheduling flags are never overridden by the GPU defaults.
    single = _load_single()
    args = single.build_parser().parse_args(
        [
            "run", "--user", "u", "--name", "X", "--smiles", "CCO",
            "--backend", "gpu",
            "--partition", "ice-gpu-2",
            "--time", "4:00:00",
            "--gres", "gpu:v100:1",
        ]
    )
    single.apply_backend_defaults(args)
    assert args.partition == "ice-gpu-2"
    assert args.time == "4:00:00"
    assert args.gres == "gpu:v100:1"


def test_apply_backend_defaults_cpu_is_noop() -> None:
    single = _load_single()
    args = single.build_parser().parse_args(
        ["run", "--user", "u", "--name", "X", "--smiles", "CCO"]
    )
    before = (args.partition, args.time, args.gres)
    single.apply_backend_defaults(args)
    assert (args.partition, args.time, args.gres) == before
