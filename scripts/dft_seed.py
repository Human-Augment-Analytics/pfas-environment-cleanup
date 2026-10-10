#!/usr/bin/env python3
"""Seed-campaign driver on top of dft_single.py.

Runs the seed-set adsorption campaign (one adsorbent from a CSV x the TFA
reference PFAS) with the same production settings dft_single.py validated
end-to-end on PACE-ICE (job 6124763): production mode, molecule system,
ice-cpu, 18 h, 8 cpus, memory auto-sized from the SMILES.

Accuracy is preserved by construction — no shortcut settings are used:

- every case runs the full production workflow (same ecutwfc/conv_thr the
  anchors validated);
- the PFAS reference leg is skipped only by passing the *actual computed*
  TFA energy parsed from compounds/pfas/TFA/pfas.out on the cluster
  (identical number the leg would recompute; run_adsorption_case.py then
  records pfas_energy_source="provided");
- every harvested case gets the same verdict rubric as dft_single.py
  (NOT BINDING / PLAUSIBLE weak / PLAUSIBLE / SUSPECT / INCONCLUSIVE).

Sub-commands:

  submit   submit pending cases (idempotent: skips cases with results on
           the cluster or a live job; OOM cases retry with doubled memory,
           TIMEOUT cases stay parked unless --retry-timeout)
  status   one-line table of every case (state, verdict if finished)
  harvest  read results.json for finished cases, write a campaign CSV +
           summary report; --fetch also pulls the full evidence bundles

CSV schema: ID,Name,SMILES,Category (plans/seed_campaign_v1.csv).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def _load_single(name: str = "dft_single_for_seed"):
    spec = importlib.util.spec_from_file_location(name, HERE / "dft_single.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


single = _load_single()

# ------------------------------------------------------------ campaign data


@dataclass
class SeedCase:
    ident: str
    adsorbent_name: str
    smiles: str
    category: str
    case_name: str
    heavy_atoms: int = 0
    mem_gb: int = 0

    def meta(self) -> dict[str, Any]:
        return {
            "seed_id": self.ident,
            "category": self.category,
            "adsorbent_name": self.adsorbent_name,
        }


CSV_COLUMNS = ("ID", "Name", "SMILES", "Category")

# Hard cap on (re)submissions per case across the whole campaign, so a
# permanently failing case parks instead of resubmitting every invocation.
# OOM retries additionally respect dft_single's escalation budget.
MAX_ATTEMPTS = 3


def read_seed_csv(path: Path, pfas_name: str) -> list[SeedCase]:
    """Read the seed CSV into campaign cases (validated, deduped)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in CSV_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")
        rows = list(reader)

    seen: dict[str, str] = {}
    cases: list[SeedCase] = []
    for r in rows:
        name = single.sanitize_name(r["Name"].strip())
        smiles = r["SMILES"].strip()
        if not name or not smiles:
            raise ValueError(f"{path}: row {r.get('ID', '?')} has empty Name/SMILES")
        case_name = single.default_case_name(name, pfas_name)
        if case_name in seen:
            raise ValueError(
                f"{path}: duplicate case name {case_name} "
                f"(rows {seen[case_name]} and {r['ID']})"
            )
        seen[case_name] = r["ID"]
        cases.append(
            SeedCase(
                ident=r["ID"].strip(),
                adsorbent_name=name,
                smiles=smiles,
                category=r.get("Category", "").strip(),
                case_name=case_name,
            )
        )
    return cases


def plan_cases(
    cases: list[SeedCase],
    pfas_smiles: str,
    *,
    order: str = "csv",
    only: set[str] | None = None,
    start_after: str | None = None,
) -> list[SeedCase]:
    """Select/annotate cases: size-sort, --only ids, --start-after id."""
    for c in cases:
        c.heavy_atoms = single.wrapper.heavy_atom_count(c.smiles)
        c.mem_gb = single.suggest_mem_gb(c.smiles, pfas_smiles)

    out = cases
    if start_after is not None:
        idx = [i for i, c in enumerate(cases) if c.ident == start_after]
        if not idx:
            raise ValueError(f"--start-after: no case with ID {start_after!r}")
        out = out[idx[0] + 1 :]
    if only is not None:
        by_id = {c.ident: c for c in out}
        missing = sorted(only - set(by_id))
        if missing:
            raise ValueError(f"--only: unknown ID(s) {', '.join(missing)}")
        out = [by_id[i] for i in sorted(only)]
    if order == "size":
        out = sorted(out, key=lambda c: (c.heavy_atoms, c.ident))
    return out


# ----------------------------------------------------------------- state io


@dataclass
class CampaignState:
    path: Path
    cases: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> CampaignState:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            return cls(path=path, cases=raw.get("cases", {}))
        return cls(path=path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"cases": self.cases}, f, indent=2, sort_keys=True)
            f.write("\n")

    def update(self, case: SeedCase, **fields: Any) -> None:
        entry = self.cases.setdefault(
            case.case_name,
            {"id": case.ident, "name": case.adsorbent_name, "attempts": 0},
        )
        entry.update(fields)
        entry["updated_at"] = single.wrapper.now_utc_iso()
        self.save()


def next_mem_gb(entry: dict[str, Any], case: SeedCase) -> int | None:
    """Doubled-memory retry for OOM cases, None when the budget is spent."""
    attempts = int(entry.get("attempts", 0))
    if attempts > single.MAX_ESCALATIONS:
        return None
    current = int(entry.get("mem_gb") or case.mem_gb)
    new_mem = min(current * 2, single.ESCALATION_CAP_GB)
    return new_mem if new_mem > current else None


# ------------------------------------------------------------- ssh helpers


def live_jobs_map(c) -> dict[str, tuple[str, str]]:
    """All of the user's queued/running jobs: case_name -> (job_id, state)."""
    p = single._ssh(
        c, "squeue -h -u $(whoami) -o '%i|%T|%j'", check=False, capture=True
    )
    out: dict[str, tuple[str, str]] = {}
    for line in p.stdout.splitlines():
        parts = [x.strip() for x in line.split("|")]
        if len(parts) != 3 or not parts[2].startswith("dft_"):
            continue
        out[parts[2][len("dft_") :]] = (parts[0], parts[1])
    return out


def results_present_map(c, root: str, case_names: list[str]) -> dict[str, bool]:
    """One ssh round-trip per chunk: which cases already have results.json."""
    out: dict[str, bool] = {}
    for i in range(0, len(case_names), 50):
        checks = "; ".join(
            f"if [ -f {shlex.quote(single.case_results_path(root, n))} ]; "
            f"then echo {shlex.quote(n)}; fi"
            for n in case_names[i : i + 50]
        )
        p = single._ssh(c, checks, check=False, capture=True)
        for name in p.stdout.split():
            out[name] = True
    return out


def probe_pfas_energy(c, root: str, pfas_name: str) -> float | None:
    """Parse the cached PFAS reference energy from compounds/ (Ry), if any.

    This is the number the PFAS leg would recompute (same input settings);
    passing it via PFAS_ENERGY_RY skips a redundant identical calculation.
    """
    p = single._ssh(
        c,
        f"grep -m1 '^!' {shlex.quote(f'{root}/compounds/pfas/{pfas_name}/pfas.out')}",
        check=False,
        capture=True,
    )
    line = p.stdout.strip()
    if "=" not in line:
        return None
    try:
        return float(line.split("=")[1].split()[0])
    except (IndexError, ValueError):
        return None


def final_state_for(c, job_id: str) -> str:
    """sacct end-state for a job that left the queue (PENDING/RUNNING gone)."""
    p = single._ssh(
        c,
        f"sacct -n -P -j {shlex.quote(job_id)} -o State,ExitCode "
        "| grep -v ' \\.' | head -1",
        check=False,
        capture=True,
    )
    return p.stdout.strip().split()[0] if p.stdout.strip() else "UNKNOWN"


# --------------------------------------------------------------- reporting


def verdict_row(case: SeedCase, data: dict[str, Any]) -> dict[str, Any]:
    """Flatten one results.json entry into a CSV row with its verdict."""
    ry = data.get("energies_ry", {})
    ev = (data.get("energies_ev") or {}).get("adsorption")
    verdict, _ = single.classify_e_ads(ev)
    return {
        "ID": case.ident,
        "Name": case.adsorbent_name,
        "Category": case.category,
        "case_name": case.case_name,
        "E_ads_Ry": ry.get("adsorption"),
        "E_ads_eV": ev,
        "verdict": verdict,
        "pfas_energy_source": data.get("pfas_energy_source", ""),
        "mode": data.get("mode", ""),
    }


def render_summary(rows: list[dict[str, Any]]) -> str:
    """Campaign-level rollup with the accuracy flags spelled out."""
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    lines = [f"cases with results: {len(rows)}"]
    for verdict in sorted(counts):
        lines.append(f"  {verdict}: {counts[verdict]}")
    suspect = [r["ID"] for r in rows if r["verdict"] == "SUSPECT"]
    if suspect:
        lines.append(
            f"SUSPECT cases to inspect before use (ecutwfc/conv_thr/pseudos): "
            f"{', '.join(suspect)}"
        )
    lines.append(
        "note: PLAUSIBLE (weak) is expected for small neutral molecules "
        "without dispersion correction; SUSPECT values need manual checks."
    )
    return "\n".join(lines)


# -------------------------------------------------------------- subcommands


def cmd_submit(args: argparse.Namespace) -> int:
    c = single.make_cluster(
        args.user, args.cluster, args.cluster_root, args.runs_subdir
    )
    root = c.root.rstrip("/")
    cases = plan_cases(
        read_seed_csv(Path(args.csv), args.pfas_name),
        args.pfas_smiles,
        order=args.order,
        only=set(args.only.split(",")) if args.only else None,
        start_after=args.start_after,
    )
    state = CampaignState.load(Path(args.state))
    single._open_master(c)

    pfas_energy = args.pfas_energy_ry
    if pfas_energy is None and not args.no_pfas_reuse:
        pfas_energy = probe_pfas_energy(c, root, args.pfas_name)
        if pfas_energy is not None:
            print(
                f"[seed] reusing computed {args.pfas_name} reference energy "
                f"{pfas_energy:.8f} Ry (parsed from compounds/ on the cluster)"
            )
        else:
            print(
                "[seed] no cached PFAS energy found - each case will compute "
                "its own PFAS reference leg"
            )

    live = live_jobs_map(c)
    done = results_present_map(c, root, [x.case_name for x in cases])
    workflow_script = args.workflow_script or f"{root}/scripts/run_dft_workflow.sh"

    submitted = skipped_done = skipped_live = parked = 0
    for case in cases:
        if done.get(case.case_name):
            state.update(case, status="done")
            skipped_done += 1
            continue
        if case.case_name in live:
            job_id, slurm_state = live[case.case_name]
            state.update(case, status="running", job_id=job_id, state=slurm_state)
            skipped_live += 1
            continue

        entry = state.cases.get(case.case_name, {})
        status = entry.get("status", "")
        mem_gb = case.mem_gb
        attempts = int(entry.get("attempts", 0))
        if status == "oom":
            retry = next_mem_gb(entry, case)
            if retry is None:
                parked += 1
                continue
            mem_gb = retry
        elif status == "timeout":
            if not args.retry_timeout:
                parked += 1
                continue  # A3 lesson: walltime failures do not retry-loop
        elif status in ("running", "submitted") and entry.get("job_id"):
            # Left the queue without results -> resolve why before resubmit.
            end = final_state_for(c, str(entry["job_id"]))
            if end == "OUT_OF_MEMORY":
                state.update(case, status="oom", last_state=end)
                retry = next_mem_gb(entry, case)
                if retry is None:
                    parked += 1
                    continue
                mem_gb, status = retry, "oom"
            elif end in ("TIMEOUT", "DEADLINE"):
                state.update(case, status="timeout", last_state=end)
                if not args.retry_timeout:
                    parked += 1
                    continue
            elif end not in ("COMPLETED", "CANCELLED", "UNKNOWN", "FAILED"):
                state.update(case, status="running", state=end)
                skipped_live += 1
                continue
            elif end == "COMPLETED":
                state.update(case, status="results_missing", last_state=end)
                parked += 1
                continue
            elif end in ("CANCELLED", "FAILED", "UNKNOWN"):
                if attempts >= MAX_ATTEMPTS:
                    state.update(case, status="failed", last_state=end)
                    parked += 1
                    continue
                # otherwise: treat like a fresh attempt and resubmit below

        if submitted >= args.limit:
            parked += len(cases) - (skipped_done + skipped_live + parked + submitted)
            break
        if sum(
            1 for s in live.values() if s[1] in single.ACTIVE_STATES
        ) + submitted >= (args.max_inflight):
            parked += 1
            continue

        print(
            f"[seed] submitting {case.case_name} ({case.ident}, "
            f"{case.heavy_atoms} heavy atoms, {mem_gb}G)"
        )
        job_id = single.submit_job(
            c,
            case_name=case.case_name,
            adsorbent_name=case.adsorbent_name,
            pfas_name=args.pfas_name,
            adsorbent_smiles=case.smiles,
            pfas_smiles=args.pfas_smiles,
            pfas_energy_ry=pfas_energy,
            adsorbent_source="smiles",
            adsorbent_cif=None,
            system_type=args.system_type,
            mode=args.mode,
            partition=args.partition,
            account=args.account,
            qos=args.qos,
            time_limit=args.time,
            cpus=args.cpus,
            mem_gb=mem_gb,
            workflow_script=workflow_script,
        )
        single.write_meta(
            c,
            case.case_name,
            {
                "case_name": case.case_name,
                "adsorbent_name": case.adsorbent_name,
                "pfas_name": args.pfas_name,
                "adsorbent_smiles": case.smiles,
                "pfas_smiles": args.pfas_smiles,
                "pfas_energy_ry": pfas_energy,
                "adsorbent_source": "smiles",
                "adsorbent_cif": None,
                "system_type": args.system_type,
                "mode": args.mode,
                "mem_gb": mem_gb,
                "cpus": args.cpus,
                "time_limit": args.time,
                "partition": args.partition,
                "account": args.account,
                "qos": args.qos,
                "workflow_script": workflow_script,
                "job_id": job_id,
                "submitted_via": "scripts/dft_seed.py",
                "campaign": Path(args.csv).name,
                **case.meta(),
            },
        )
        state.update(
            case,
            status="submitted",
            job_id=job_id,
            mem_gb=mem_gb,
            attempts=attempts + 1,
        )
        live[case.case_name] = (str(job_id), "SUBMITTED")
        submitted += 1
        time.sleep(args.submit_gap)

    print(
        f"[seed] submitted {submitted}, already-done {skipped_done}, "
        f"already-live {skipped_live}, parked {parked} "
        f"(state: {args.state})"
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    c = single.make_cluster(
        args.user, args.cluster, args.cluster_root, args.runs_subdir
    )
    root = c.root.rstrip("/")
    cases = plan_cases(
        read_seed_csv(Path(args.csv), args.pfas_name),
        args.pfas_smiles,
        only=set(args.only.split(",")) if args.only else None,
        start_after=args.start_after,
    )
    single._open_master(c)

    live = live_jobs_map(c)
    done = results_present_map(c, root, [x.case_name for x in cases])

    rows = [
        (
            x.ident,
            x.case_name,
            "results" if done.get(x.case_name) else "none",
            (live.get(x.case_name) or ("", "-"))[1],
            x.heavy_atoms,
            x.mem_gb,
        )
        for x in cases
    ]
    print(f"{'ID':<7} {'case':<38} {'results':<8} {'job':<10} {'atm':>4} {'GB':>4}")
    for r in rows:
        print(f"{r[0]:<7} {r[1]:<38.38} {r[2]:<8} {r[3]:<10} {r[4]:>4} {r[5]:>4}")
    n_done = sum(1 for r in rows if r[2] == "results")
    n_live = sum(1 for r in rows if r[3] != "-")
    print(f"\n[seed] {len(rows)} case(s): {n_done} with results, {n_live} live")
    return 0


def cmd_harvest(args: argparse.Namespace) -> int:
    c = single.make_cluster(
        args.user, args.cluster, args.cluster_root, args.runs_subdir
    )
    root = c.root.rstrip("/")
    cases = plan_cases(
        read_seed_csv(Path(args.csv), args.pfas_name),
        args.pfas_smiles,
        only=set(args.only.split(",")) if args.only else None,
    )
    single._open_master(c)

    done = results_present_map(c, root, [x.case_name for x in cases])
    rows: list[dict[str, Any]] = []
    missing = 0
    for case in cases:
        if not done.get(case.case_name):
            missing += 1
            continue
        data = single.read_remote_results(c, root, case.case_name)
        if data is None:
            missing += 1
            continue
        rows.append(verdict_row(case, data))
        if args.fetch:
            single.fetch_case(c, case.case_name, args.local_cache)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"[seed] wrote {len(rows)} result row(s) to {out} ({missing} missing)")
    print(render_summary(rows))
    return 0 if rows else 1


# ---------------------------------------------------------------------- CLI


def add_shared_args(sp: argparse.ArgumentParser) -> None:
    single.add_cluster_args(sp)
    sp.add_argument("--csv", required=True, help="Seed CSV (ID,Name,SMILES,Category).")
    sp.add_argument(
        "--state",
        default=None,
        help="Campaign state file (default: <local-cache>/seed_state.json).",
    )
    sp.add_argument("--local-cache", default=single.wrapper.DEFAULT_LOCAL_CACHE)
    sp.add_argument("--pfas-name", default=single.DEFAULT_PFAS_NAME)
    sp.add_argument("--pfas-smiles", default=single.DEFAULT_PFAS_SMILES)
    sp.add_argument("--only", help="Comma-separated seed IDs to restrict to.")
    sp.add_argument("--start-after", help="Skip until after this seed ID.")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="dft_seed.py",
        description="Seed-set DFT campaign: submit, monitor, harvest, evaluate.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    submit = sub.add_parser("submit", help="Submit pending cases (idempotent).")
    add_shared_args(submit)
    submit.add_argument("--limit", type=int, default=10, help="Max new submissions.")
    submit.add_argument(
        "--max-inflight",
        type=int,
        default=10,
        help="Stop submitting once this many of our jobs are queued/running.",
    )
    submit.add_argument(
        "--order",
        choices=["csv", "size"],
        default="size",
        help="Submission order (size = smallest systems first for early wins).",
    )
    submit.add_argument(
        "--no-pfas-reuse",
        action="store_true",
        help="Recompute the PFAS reference leg in every case.",
    )
    submit.add_argument(
        "--pfas-energy-ry",
        type=float,
        default=None,
        help="Known PFAS energy (Ry); default: probe compounds/ on the cluster.",
    )
    submit.add_argument("--retry-timeout", action="store_true")
    submit.add_argument(
        "--mode", choices=["lowmem", "cluster", "production"], default="production"
    )
    submit.add_argument(
        "--system-type", choices=["molecule", "periodic"], default="molecule"
    )
    submit.add_argument("--time", default=single.DEFAULT_TIME)
    submit.add_argument("--cpus", type=int, default=single.DEFAULT_CPUS)
    submit.add_argument("--partition", default=single.DEFAULT_PARTITION)
    submit.add_argument("--account", default=single.DEFAULT_ACCOUNT)
    submit.add_argument("--qos", default=single.DEFAULT_QOS)
    submit.add_argument("--workflow-script", default=None)
    submit.add_argument(
        "--submit-gap",
        type=float,
        default=2.0,
        help="Seconds between sbatch calls (scheduler courtesy).",
    )
    submit.set_defaults(func=cmd_submit)

    status = sub.add_parser("status", help="Per-case table: results + live job.")
    add_shared_args(status)
    status.set_defaults(func=cmd_status)

    harvest = sub.add_parser(
        "harvest", help="Collect results into a CSV + summary verdict report."
    )
    add_shared_args(harvest)
    harvest.add_argument(
        "--out", default="seed_campaign_results.csv", help="Output CSV path."
    )
    harvest.add_argument(
        "--fetch",
        action="store_true",
        help="Also download full evidence bundles into --local-cache.",
    )
    harvest.set_defaults(func=cmd_harvest)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "state", None) is None:
        base = Path(getattr(args, "local_cache", single.wrapper.DEFAULT_LOCAL_CACHE))
        args.state = str(base / "seed_state.json")
    try:
        return args.func(args)
    except RuntimeError as exc:
        print(f"[seed] error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "\n[seed] interrupted (jobs keep running on the cluster)", file=sys.stderr
        )
        return 130


if __name__ == "__main__":
    sys.exit(main())
