"""Fast offline tests for the seed-campaign driver (no SSH, no network)."""

import contextlib
import csv
import importlib.util
import io
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load_seed(name: str = "dft_seed_under_test"):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / "dft_seed.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_dft_seed_help() -> None:
    p = subprocess.run(
        [sys.executable, str(SCRIPTS / "dft_seed.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert p.returncode == 0
    for sub in ("submit", "status", "harvest"):
        assert sub in p.stdout


def test_dft_seed_reuses_dft_single() -> None:
    seed = _load_seed()
    single = sys.modules["dft_single_for_seed"]
    assert seed.single is single
    # same physics constants flow through (accuracy by construction)
    assert seed.single.DEFAULT_PFAS_NAME == "TFA"
    assert math.isclose(seed.single.RY_TO_EV, 13.605693009)


def _csv_file(tmp: Path, rows: list[dict]) -> Path:
    path = tmp / "seed.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ID", "Name", "SMILES", "Category"])
        w.writeheader()
        w.writerows(rows)
    return path


def test_read_seed_csv_and_plan() -> None:
    seed = _load_seed()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        path = _csv_file(
            tmp,
            [
                {
                    "ID": "s0001",
                    "Name": "seed_c0_CID5079070",
                    "SMILES": "CC1=CC(=C(C(=C1)C)NC(=S)N(CCC(C)C)C(C)C)",
                    "Category": "seed_neutral",
                },
                {
                    "ID": "s0002",
                    "Name": "seed_c1_CID893290",
                    "SMILES": "CC1=NC2=C(O1)C=CC(=C2)NC(=O)NCC",
                    "Category": "seed_neutral",
                },
            ],
        )
        cases = seed.read_seed_csv(path, "TFA")
        assert [c.ident for c in cases] == ["s0001", "s0002"]
        assert cases[0].case_name == "seed_c0_CID5079070_TFA"

        planned = seed.plan_cases(cases, seed.single.DEFAULT_PFAS_SMILES, order="size")
        # size order: s0002 (fewer heavy atoms) first
        assert [c.ident for c in planned] == ["s0002", "s0001"]
        # csv order (the function default) preserves file order
        csv_order = seed.plan_cases(cases, seed.single.DEFAULT_PFAS_SMILES)
        assert [c.ident for c in csv_order] == ["s0001", "s0002"]
        assert planned[0].mem_gb > 0 and planned[0].heavy_atoms > 0

        only = seed.plan_cases(cases, seed.single.DEFAULT_PFAS_SMILES, only={"s0001"})
        assert [c.ident for c in only] == ["s0001"]

        after = seed.plan_cases(
            cases, seed.single.DEFAULT_PFAS_SMILES, start_after="s0001"
        )
        assert [c.ident for c in after] == ["s0002"]


def test_read_seed_csv_rejects_duplicates_and_missing() -> None:
    seed = _load_seed()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        dup = _csv_file(
            tmp,
            [
                {"ID": "a", "Name": "x", "SMILES": "CCO", "Category": "c"},
                {"ID": "b", "Name": "x", "SMILES": "CCC", "Category": "c"},
            ],
        )
        try:
            seed.read_seed_csv(dup, "TFA")
        except ValueError as exc:
            assert "duplicate" in str(exc)
        else:
            raise AssertionError("duplicate case names must raise")

        bad = tmp / "bad.csv"
        bad.write_text("ID,SMILES\n1,CCO\n", encoding="utf-8")
        try:
            seed.read_seed_csv(bad, "TFA")
        except ValueError as exc:
            assert "missing column" in str(exc)
        else:
            raise AssertionError("missing Name column must raise")


def test_campaign_state_roundtrip_and_attempts_cap() -> None:
    seed = _load_seed()
    with tempfile.TemporaryDirectory() as td:
        st = seed.CampaignState.load(Path(td) / "nested" / "state.json")
        case = seed.SeedCase(
            ident="s0001",
            adsorbent_name="x",
            smiles="CCO",
            category="c",
            case_name="x_TFA",
        )
        st.update(case, status="submitted", job_id=42, mem_gb=32, attempts=1)
        loaded = seed.CampaignState.load(Path(td) / "nested" / "state.json")
        entry = loaded.cases["x_TFA"]
        assert entry["id"] == "s0001"
        assert entry["attempts"] == 1
        assert entry["job_id"] == 42

        # a spent escalation budget parks instead of retrying
        entry["attempts"] = seed.single.MAX_ESCALATIONS + 1
        entry["mem_gb"] = 256
        assert seed.next_mem_gb(entry, case) is None
        # a fresh case doubles once, then parks at the cap
        fresh = {**entry, "attempts": 1, "mem_gb": 96}
        assert seed.next_mem_gb(fresh, case) == 192


def test_next_mem_gb_caps_at_escalation_cap() -> None:
    seed = _load_seed()
    case = seed.SeedCase("i", "n", "CCO", "c", "n_TFA")
    entry = {"attempts": 1, "mem_gb": 192}
    assert seed.next_mem_gb(entry, case) == 256
    entry["mem_gb"] = 256
    assert seed.next_mem_gb(entry, case) is None


def test_live_jobs_map_parses_squeue() -> None:
    seed = _load_seed()

    class FakeProc:
        stdout = "123|RUNNING|dft_caseA_TFA\n456|PENDING|otherjob\n789|PENDING|dft_caseB_TFA\n"

    class FakeSingle:
        def _ssh(self, c, cmd, check=False, capture=False):
            return FakeProc()

    fake = FakeSingle()
    orig = seed.single._ssh
    seed.single._ssh = fake._ssh
    try:
        m = seed.live_jobs_map(None)
    finally:
        seed.single._ssh = orig
    assert m == {
        "caseA_TFA": ("123", "RUNNING"),
        "caseB_TFA": ("789", "PENDING"),
    }


def test_verdict_row_flattens_results() -> None:
    seed = _load_seed()
    case = seed.SeedCase("s0001", "med", "CCO", "seed_neutral", "med_TFA")
    data = {
        "pfas_energy_source": "provided",
        "mode": "production",
        "energies_ry": {
            "adsorbent": -299.48344051,
            "pfas": -299.48344053,
            "complex": -598.98,
            "adsorption": -0.013119,
        },
        "energies_ev": {"adsorption": -0.178537},
    }
    row = seed.verdict_row(case, data)
    assert row["ID"] == "s0001"
    assert row["case_name"] == "med_TFA"
    assert math.isclose(row["E_ads_eV"], -0.178537)
    assert row["verdict"] == "PLAUSIBLE (weak)"
    assert row["pfas_energy_source"] == "provided"


def test_render_summary_flags_suspect() -> None:
    seed = _load_seed()
    rows = [
        {"ID": "a", "verdict": "PLAUSIBLE"},
        {"ID": "b", "verdict": "SUSPECT"},
        {"ID": "c", "verdict": "PLAUSIBLE"},
    ]
    text = seed.render_summary(rows)
    assert "cases with results: 3" in text
    assert "SUSPECT: 1" in text
    assert "b" in text  # the suspect id is listed for manual inspection


def test_probe_pfas_energy_parses_pw_line() -> None:
    seed = _load_seed()

    class FakeProc:
        stdout = "!    total energy              =     -299.48344053 Ry\n"

    class FakeSingle:
        def _ssh(self, c, cmd, check=False, capture=False):
            return FakeProc()

    fake = FakeSingle()
    orig = seed.single._ssh
    seed.single._ssh = fake._ssh
    try:
        val = seed.probe_pfas_energy(None, "/root", "TFA")
    finally:
        seed.single._ssh = orig
    assert math.isclose(val, -299.48344053)


def test_probe_pfas_energy_disagreement_disables_reuse() -> None:
    # Observed on PACE-ICE: case legs (-299.48344053) vs the foreign
    # root-level compounds/ cache (-299.47595132) differ by ~0.1 eV.
    # If legs themselves ever disagree, reuse must be disabled.
    seed = _load_seed()

    class FakeProc:
        stdout = (
            "!    total energy              =     -299.48344053 Ry\n"
            "!    total energy              =     -299.47595132 Ry\n"
        )

    class FakeSingle:
        def _ssh(self, c, cmd, check=False, capture=False):
            return FakeProc()

    fake = FakeSingle()
    orig = seed.single._ssh
    seed.single._ssh = fake._ssh
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            val = seed.probe_pfas_energy(None, "/root", "TFA")
    finally:
        seed.single._ssh = orig
    assert val is None
    assert "WARNING" in buf.getvalue()


def test_probe_pfas_energy_no_legs_returns_none() -> None:
    # First case of a campaign bootstraps: no legs exist yet, so the
    # probe returns None and the case computes all three legs itself.
    seed = _load_seed()

    class FakeProc:
        stdout = ""

    class FakeSingle:
        def _ssh(self, c, cmd, check=False, capture=False):
            return FakeProc()

    fake = FakeSingle()
    orig = seed.single._ssh
    seed.single._ssh = fake._ssh
    try:
        val = seed.probe_pfas_energy(None, "/root", "TFA")
    finally:
        seed.single._ssh = orig
    assert val is None


def test_harvest_cli_wiring() -> None:
    # harvest itself needs the cluster; the local CSV write path is covered
    # by the verdict_row/render_summary tests, so here we only check the
    # CLI accepts --out / --fetch.
    p = subprocess.run(
        [sys.executable, str(SCRIPTS / "dft_seed.py"), "harvest", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert p.returncode == 0
    assert "--out" in p.stdout
    assert "--fetch" in p.stdout


def test_state_json_is_valid_json_with_sorted_cases() -> None:
    seed = _load_seed()
    with tempfile.TemporaryDirectory() as td:
        st = seed.CampaignState(path=Path(td) / "s.json")
        for i in range(3):
            case = seed.SeedCase(f"id{i}", f"n{i}", "CCO", "c", f"case{i}")
            st.update(case, status="submitted")
        raw = json.loads((Path(td) / "s.json").read_text(encoding="utf-8"))
        assert list(raw["cases"]) == ["case0", "case1", "case2"]


def test_max_attempts_constant() -> None:
    seed = _load_seed()
    assert seed.MAX_ATTEMPTS >= 1
    assert seed.MAX_ATTEMPTS <= 5  # sanity: not an unbounded retry loop
