#!/usr/bin/env python3
"""Single-case DFT runner for team members.

Submits one adsorption case to PACE-ICE, watches it until it finishes,
fetches the results, and prints an adsorption-energy verdict. No Slurm
knowledge required:

    python scripts/dft_single.py run --user <gt-user> --name "CTAB" --smiles "<SMILES>"
    python scripts/dft_single.py status --user <gt-user> --case CTAB_TFA

`run` is idempotent: if the case already has results it skips straight
to fetch + verdict; if a job is already queued/running it attaches and
monitors; only otherwise does it submit. Memory is sized automatically
from the SMILES heavy-atom counts (32/96/128 GB bands), and an
out-of-memory job is resubmitted with doubled memory (finished legs
are reused automatically by the workflow's compounds cache).

Defaults target the TFA reference PFAS in production mode on PACE-ICE
(ice-cpu / coc / coc-ice). Every knob is still a flag; for full
control use scripts/dft_wrapper.py, and for CSV batch screening use
scripts/run_batch_screening.sh - both remain unchanged.

Exit codes: 0 finished (any verdict), 1 completed but no results,
2 failed / timed out / escalation exhausted, 3 nothing to attach to,
130 interrupted (re-run the same command to resume monitoring).

Works from Windows cmd.exe as well as POSIX shells: every ssh/scp/tar
call below is a list-form subprocess argv, never a shell string.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import re
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent


def _load_wrapper():
    """Reuse the wrapper's Cluster plumbing, sbatch template, and size
    heuristics instead of duplicating them. Registered in sys.modules
    before exec so the dataclass resolves (same pattern as the tests).
    """
    path = SCRIPTS_DIR / "dft_wrapper.py"
    spec = importlib.util.spec_from_file_location("dft_wrapper_single", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot import dft_wrapper from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dft_wrapper_single"] = mod
    spec.loader.exec_module(mod)
    return mod


wrapper = _load_wrapper()

# ---------------------------------------------------------------- config

DEFAULT_PFAS_NAME = "TFA"
DEFAULT_PFAS_SMILES = "FC(F)(F)C(=O)O"  # matches run_batch_screening.sh
DEFAULT_PARTITION = "ice-cpu"
DEFAULT_ACCOUNT = "coc"
DEFAULT_QOS = "coc-ice"
DEFAULT_TIME = "18:00:00"  # ice-cpu walltime cap
DEFAULT_CPUS = 8
DEFAULT_POLL_SECONDS = 120

RY_TO_EV = 13.605693009
EV_TO_KJ_MOL = 96.485

# Memory bands (README table, one notch above the floor: the CTAB pilot
# measured 77 GB where the table suggested 64 GB and OOM'd there).
SMALL_BAND_GB = 32  # pair total <= 14 heavy atoms
MEDIUM_BAND_GB = 96  # <= 28 heavy atoms
LARGE_BAND_GB = 128  # anything larger
ESCALATION_CAP_GB = 256
MAX_ESCALATIONS = 2

ACTIVE_STATES = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "SUSPENDED",
    "REQUEUED",
    "REQUEUE_HOLD",
}


# ---------------------------------------------------------- pure helpers


def sanitize_name(name: str) -> str:
    """Match run_batch_screening.sh: keep [A-Za-z0-9._-], else '_'."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def default_case_name(adsorbent_name: str, pfas_name: str) -> str:
    return f"{sanitize_name(adsorbent_name)}_{sanitize_name(pfas_name)}"


def suggest_mem_gb(adsorbent_smiles: str | None, pfas_smiles: str | None) -> int:
    """Pick the memory band from the largest system the job will build.

    The complex contains both molecules, so its heavy-atom count is the
    adsorbent + PFAS sum; the individual legs set the floor.
    """
    ads = wrapper.heavy_atom_count(adsorbent_smiles)
    pfas = wrapper.heavy_atom_count(pfas_smiles)
    largest = max(ads, pfas, ads + pfas)
    if largest <= wrapper._SMALL_SYSTEM_HEAVY_ATOMS:
        return SMALL_BAND_GB
    if largest <= wrapper._LARGE_SYSTEM_HEAVY_ATOMS:
        return MEDIUM_BAND_GB
    return LARGE_BAND_GB


def classify_e_ads(ev: float | None) -> tuple[str, str]:
    """Plain-language plausibility verdict for an adsorption energy.

    Typical molecular adsorption energies land between 0 and -4 eV;
    much stronger values are usually a convergence/pseudopotential
    artifact (e.g. the 6.35 eV ecutwfc-30 CTAB_TFA case).
    """
    if ev is None or not math.isfinite(ev):
        return "INCONCLUSIVE", "no finite adsorption energy available"
    if ev >= 0:
        return (
            "NOT BINDING",
            ("E_ads >= 0: the complex is higher-energy than the separated molecules"),
        )
    if ev > -1:
        return "PLAUSIBLE (weak)", "-1 < E_ads < 0 eV: weak physisorption"
    if ev >= -4:
        return (
            "PLAUSIBLE",
            "-1 to -4 eV: typical range for molecular adsorption",
        )
    return (
        "SUSPECT",
        (
            "E_ads < -4 eV is unusually strong; check convergence "
            "(ecutwfc, conv_thr) and pseudopotentials before trusting it"
        ),
    )


def format_results_report(data: dict[str, Any] | list[Any]) -> str:
    """Human-readable summary + verdict from a results.json dict.

    The workflow writes results.json as a JSON array (one dict per case,
    batch format); for a single-case run dir the last entry is the case.
    """
    if isinstance(data, list):
        if not data:
            raise ValueError("results.json array is empty")
        data = data[-1]
    ry = data.get("energies_ry", {})
    ev = data.get("energies_ev", {}).get("adsorption")
    verdict, why = classify_e_ads(ev)
    kj = ev * EV_TO_KJ_MOL if ev is not None and math.isfinite(ev) else None

    lines = [
        f"=== DFT results: {data.get('case_name', '?')} ===",
        (
            f"adsorbent: {data.get('adsorbent_name', '?')}    "
            f"pfas: {data.get('pfas_name', '?')} "
            f"(reference: {data.get('pfas_energy_source', '?')})"
        ),
        (f"mode: {data.get('mode', '?')} ({data.get('system_type', '?')})"),
        f"E(adsorbent) = {ry.get('adsorbent', float('nan')):.6f} Ry",
        f"E(pfas)      = {ry.get('pfas', float('nan')):.6f} Ry",
        f"E(complex)   = {ry.get('complex', float('nan')):.6f} Ry",
    ]
    if kj is None:
        lines.append("E_ads        = n/a")
    else:
        lines.append(f"E_ads        = {ev:.3f} eV  ({kj:.0f} kJ/mol)")
    lines.append(f"verdict: {verdict} - {why}")

    for label in ("adsorbent", "pfas", "complex"):
        if ry.get(label) == 0.0:
            lines.append(
                f"warning: {label} energy is 0.0 (leg skipped or "
                "failed) - E_ads is not meaningful"
            )
    return "\n".join(lines)


def parse_sbatch_submission(text: str) -> int | None:
    """Extract the job id from sbatch output ('Submitted batch job N')."""
    m = re.search(r"Submitted batch job\s+(\d+)", text)
    return int(m.group(1)) if m else None


def parse_sacct_output(text: str) -> list[dict[str, str]]:
    """Parse `sacct -n -P -o JobID,State,ExitCode,MaxRSS,Elapsed` rows."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = (line.split("|") + [""] * 5)[:5]
        rows.append(
            {
                "jobid": parts[0],
                "state": parts[1],
                "exit_code": parts[2],
                "maxrss": parts[3],
                "elapsed": parts[4],
            }
        )
    return rows


def sacct_main_row(rows: list[dict[str, str]], job_id: int) -> dict[str, str] | None:
    """The allocation row (no '.batch'/'.extern' suffix) for a job id."""
    for r in rows:
        if r["jobid"] == str(job_id):
            return r
    return rows[0] if rows else None


def parse_squeue_line(text: str) -> tuple[int, str] | None:
    """Last '<id> <STATE>' line from `squeue -h -n <name> -o '%A %T'`."""
    last = None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            last = (int(parts[0]), parts[1])
    return last


def is_oom_row(row: dict[str, str]) -> bool:
    return "OUT_OF_MEMORY" in row["state"] or row["exit_code"].split(":")[0] == "137"


# ------------------------------------------------- remote plumbing (cmd.exe-safe)


def make_cluster(user: str, host: str, root: str, runs_subdir: str):
    return wrapper.Cluster(
        ssh_target=f"{user}@{host}",
        root=root,
        runs_subdir=runs_subdir,
        control_path=wrapper.make_control_path(user, host),
    )


def _ssh(
    c, remote_cmd: str, *, tty: bool = False, check: bool = True, capture: bool = False
) -> subprocess.CompletedProcess:
    """Run one remote command via list-form argv (no local shell)."""
    argv = ["ssh"] + wrapper.ssh_opts_control(c)
    argv += ["-tt" if tty else "-T", c.ssh_target, remote_cmd]
    p = subprocess.run(
        argv,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=False,
    )
    if check and p.returncode != 0:
        detail = ""
        if capture:
            detail = f"\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"
        raise RuntimeError(f"ssh command failed ({p.returncode}): {remote_cmd}{detail}")
    return p


def _open_master(c) -> None:
    print("[single] warming SSH (enter your password/passkey once if asked)...")
    _ssh(c, "echo AUTH_OK", tty=True, check=True)
    print("[single] SSH OK (connection reused for the rest of this run)")


def _remote_file_exists(c, remote_path: str) -> bool:
    return _ssh(c, f"test -f {shlex.quote(remote_path)}", check=False).returncode == 0


def _read_remote_json(c, remote_path: str) -> dict[str, Any] | None:
    p = _ssh(c, f"cat {shlex.quote(remote_path)}", check=False, capture=True)
    if p.returncode != 0 or not p.stdout.strip():
        return None
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return None


def live_job(c, case_name: str) -> tuple[int, str] | None:
    p = _ssh(
        c,
        f"squeue -h -n dft_{case_name} -o '%A %T'",
        check=False,
        capture=True,
    )
    return parse_squeue_line(p.stdout)


def case_results_path(root: str, case_name: str) -> str:
    return f"{root.rstrip('/')}/dft_cases/{case_name}/results.json"


def read_remote_results(c, root: str, case_name: str) -> dict[str, Any] | None:
    return _read_remote_json(c, case_results_path(root, case_name))


def _extract_tgz(tgz: Path, dst: Path) -> None:
    with (
        tarfile.open(tgz, "r:gz") as tf,
        contextlib.suppress((tarfile.TarError, OSError)),
    ):
        tf.extractall(dst, filter="data")


def fetch_case(c, case_name: str, local_cache: str) -> Path:
    """Pull the run bundle + the compounds/<leg> references into the cache.

    Layout on the cluster (run_dft_workflow.sh):
      dft_cases/<case>/        results.json meta.json DONE job.sbatch
                              outputs/ (slurm log) complex/ (QE in/out/cif)
      compounds/adsorbents/<a>/  adsorbent.{mol,cif,in,err,out}
      compounds/pfas/<p>/        pfas.{cif,in,err,out}
    The QE scratch dirs (Outputs/, wavefunctions, hundreds of MB) and the
    shared Pseudopotentials symlink are excluded; only what reproduces the
    numbers is fetched. Two remote tars -> scp (list argv) -> local stdlib
    tarfile extract, so no shell pipes run on the submitting machine.
    """
    dst = Path(local_cache).resolve() / case_name
    dst.mkdir(parents=True, exist_ok=True)
    run_dir = wrapper.rdir(c, case_name)
    root = c.root.rstrip("/")
    ts = int(time.time())
    t_run = f"/tmp/dft_single_{case_name}_{ts}_run.tgz"
    t_legs = f"/tmp/dft_single_{case_name}_{ts}_legs.tgz"

    _ssh(
        c,
        f"cd {shlex.quote(run_dir)} && tar -czf {shlex.quote(t_run)} "
        "--exclude Outputs --exclude Pseudopotentials . 2>/dev/null",
        check=False,
    )
    # Reference-leg names come from the run's own meta.json.
    meta = _read_remote_json(c, f"{run_dir}/meta.json") or {}
    ads = meta.get("adsorbent_name")
    pfas = meta.get("pfas_name")
    if ads and pfas:
        _ssh(
            c,
            f"cd {shlex.quote(root)} && tar -czf {shlex.quote(t_legs)} "
            f"--exclude Outputs --exclude Pseudopotentials "
            f"compounds/adsorbents/{shlex.quote(str(ads))} "
            f"compounds/pfas/{shlex.quote(str(pfas))} 2>/dev/null",
            check=False,
        )

    for remote_tgz, required in ((t_run, True), (t_legs, False)):
        local_tgz = dst / remote_tgz.rsplit("/", 1)[-1]
        p = subprocess.run(
            ["scp"]
            + wrapper.ssh_opts_control(c)
            + [f"{c.ssh_target}:{remote_tgz}", str(local_tgz)],
            check=False,
        )
        if p.returncode != 0 and required:
            raise RuntimeError(
                f"fetch failed: could not retrieve {remote_tgz} "
                f"(does the run directory {run_dir} exist on the cluster?)"
            )
        if local_tgz.exists():
            if local_tgz.stat().st_size == 0:
                local_tgz.unlink()
            else:
                _extract_tgz(local_tgz, dst)
                local_tgz.unlink()
    _ssh(c, f"rm -f {shlex.quote(t_run)} {shlex.quote(t_legs)}", check=False)
    return dst


# ------------------------------------------------------------- submit path


def submit_job(
    c,
    *,
    case_name: str,
    adsorbent_name: str,
    pfas_name: str,
    adsorbent_smiles: str | None,
    pfas_smiles: str | None,
    pfas_energy_ry: float | None,
    adsorbent_source: str,
    adsorbent_cif: str | None,
    system_type: str,
    mode: str,
    partition: str | None,
    account: str | None,
    qos: str | None,
    time_limit: str,
    cpus: int,
    mem_gb: int,
    workflow_script: str,
) -> int:
    """Create run dirs, job.sbatch (wrapper template), submit, return id."""
    run_dir = wrapper.rdir(c, case_name)
    # Three explicit dirs (no brace expansion: the previous
    # '"/{{inputs,outputs,results}}"' literal leaked unexpanded braces
    # through Python concatenation and bash created '{inputs}'-style dirs).
    _ssh(
        c,
        f"mkdir -p {shlex.quote(run_dir)}/inputs"
        f" {shlex.quote(run_dir)}/outputs"
        f" {shlex.quote(run_dir)}/results",
    )
    jobfile = f"{run_dir}/job.sbatch"
    body = wrapper.render_job_sbatch(
        c,
        case_name,
        workflow_script,
        partition,
        account,
        qos,
        time_limit,
        cpus,
        mem_gb,
        False,
        False,
        False,  # skip_ads, skip_pfas, skip_complex
        adsorbent_name,
        pfas_name,
        adsorbent_smiles,
        pfas_smiles,
        pfas_energy_ry,
        adsorbent_source,
        adsorbent_cif,
        system_type,
        mode,
    )
    _write_remote_file(c, jobfile, body)

    # sbatch resolves relative #SBATCH --output/--error paths against the
    # submit cwd (ssh lands in $HOME), so cd into the run dir first.
    p = _ssh(
        c,
        f"cd {shlex.quote(run_dir)} && sbatch {shlex.quote(jobfile)}",
        check=True,
        capture=True,
    )
    job_id = parse_sbatch_submission(f"{p.stdout}\n{p.stderr}")
    if job_id is None:
        # sbatch succeeded but the id scrolled past; fall back to the
        # named queue.
        job_id = (live_job(c, case_name) or (None,))[0]
    if job_id is None:
        raise RuntimeError(
            "sbatch reported success but the job id could not be "
            f"determined; check `squeue -n dft_{case_name}` on the cluster"
        )
    return job_id


def _write_remote_file(c, remote_path: str, content: str) -> None:
    heredoc = f"cat > {shlex.quote(remote_path)} << 'EOF'\n{content}\nEOF"
    _ssh(c, heredoc, check=True)


def write_meta(c, case_name: str, meta: dict[str, Any]) -> None:
    path = f"{wrapper.rdir(c, case_name)}/meta.json"
    _write_remote_file(c, path, json.dumps(meta, indent=2) + "\n")


# ------------------------------------------------------------ monitoring


def sacct_rows_for(c, job_id: int) -> list[dict[str, str]]:
    p = _ssh(
        c,
        f"sacct -n -P -j {job_id} -o JobID,State,ExitCode,MaxRSS,Elapsed",
        check=False,
        capture=True,
    )
    return parse_sacct_output(p.stdout)


def _heartbeat(rows: list[dict[str, str]], job_id: int) -> dict[str, str] | None:
    row = sacct_main_row(rows, job_id)
    if row is None:
        return None
    rss = next(
        (r["maxrss"] for r in rows if r["jobid"] == f"{job_id}.batch" and r["maxrss"]),
        "",
    )
    note = f" rss {rss}" if rss else ""
    print(
        f"[single] job {job_id}: {row['state']} elapsed {row['elapsed'] or '?'}{note}",
        flush=True,
    )
    return row


def wait_for_job(c, job_id: int, poll_seconds: int) -> dict[str, str]:
    """Poll sacct until the job leaves the active states; return its row."""
    while True:
        rows = sacct_rows_for(c, job_id)
        row = _heartbeat(rows, job_id)
        if row is not None and row["state"] not in ACTIVE_STATES:
            return row
        time.sleep(poll_seconds)


# ------------------------------------------------------------ commands


def resume_hint(args, case_name: str) -> str:
    return (
        f"python scripts/dft_single.py run --user {args.user} "
        f"--cluster {args.cluster} --case {case_name}"
    )


def cmd_run(args: argparse.Namespace) -> int:
    c = make_cluster(args.user, args.cluster, args.cluster_root, args.runs_subdir)
    root = c.root.rstrip("/")

    if not args.case and not args.name:
        print(
            "[single] provide --name (new case) or --case (existing)", file=sys.stderr
        )
        return 2
    case_name = args.case or default_case_name(args.name, args.pfas_name)
    run_dir = wrapper.rdir(c, case_name)
    attach_only = args.case is not None and args.name is None

    if not attach_only and not args.smiles and not args.adsorbent_cif:
        print(
            "[single] a new case needs --smiles (or --adsorbent-cif); "
            "use --case <name> to attach to an existing case instead",
            file=sys.stderr,
        )
        return 2

    _open_master(c)

    # Idempotency 1: results already on the cluster -> straight to verdict.
    data = read_remote_results(c, root, case_name)
    if data is not None:
        print(f"[single] {case_name}: results already exist - fetching")
        dst = fetch_case(c, case_name, args.local_cache)
        print(f"[single] fetched results to {dst}")
        print(format_results_report(data))
        return 0

    # Idempotency 2: a job is already queued/running -> attach to it.
    live = live_job(c, case_name)
    if live is not None:
        job_id, state = live
        print(f"[single] {case_name}: job {job_id} already {state} - attaching")
        # Carry the original submit provenance so a later OOM escalation
        # can re-create the job exactly even though no flags were given
        # (attach mode), and so escalation doubles the *actual* job
        # memory instead of a band re-derived from defaults.
        meta = _read_remote_json(c, f"{run_dir}/meta.json")
        if meta:
            if args.mem_gb is None and meta.get("mem_gb"):
                args.mem_gb = int(meta["mem_gb"])
            if args.name is None:
                args.name = meta.get("adsorbent_name")
            if args.smiles is None:
                args.smiles = meta.get("adsorbent_smiles")
            if args.adsorbent_cif is None:
                args.adsorbent_cif = meta.get("adsorbent_cif")
            if meta.get("pfas_smiles"):
                args.pfas_smiles = meta["pfas_smiles"]
            if meta.get("adsorbent_source"):
                args.adsorbent_source = meta["adsorbent_source"]
    else:
        if attach_only:
            print(
                f"[single] no live job and no results for case {case_name} "
                "on the cluster; submit with --name/--smiles first",
                file=sys.stderr,
            )
            return 3

        mem_gb = args.mem_gb or suggest_mem_gb(args.smiles, args.pfas_smiles)
        mem_note = "" if args.mem_gb is not None else " (auto-sized from SMILES)"
        workflow_script = args.workflow_script or f"{root}/scripts/run_dft_workflow.sh"
        print(
            f"[single] submitting {case_name}: adsorbent "
            f"{args.name!r} + {args.pfas_name} "
            f"({args.mode}/{args.system_type}), {mem_gb}G{mem_note}"
        )
        job_id = submit_job(
            c,
            case_name=case_name,
            adsorbent_name=args.name or case_name,
            pfas_name=args.pfas_name,
            adsorbent_smiles=args.smiles,
            pfas_smiles=args.pfas_smiles,
            pfas_energy_ry=args.pfas_energy_ry,
            adsorbent_source=args.adsorbent_source,
            adsorbent_cif=args.adsorbent_cif,
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
        write_meta(
            c,
            case_name,
            {
                "case_name": case_name,
                "adsorbent_name": args.name,
                "pfas_name": args.pfas_name,
                "adsorbent_smiles": args.smiles,
                "adsorbent_cif": args.adsorbent_cif,
                "pfas_smiles": args.pfas_smiles,
                "pfas_energy_ry": args.pfas_energy_ry,
                "adsorbent_source": args.adsorbent_source,
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
                "submitted_via": "scripts/dft_single.py",
                "submitted_at": wrapper.now_utc_iso(),
            },
        )
        print(f"[single] submitted job {job_id} ({mem_gb}G, {args.time})")
        if args.no_wait:
            print(f"[single] to resume later:  {resume_hint(args, case_name)}")
            print(
                f"[single] to check on it:   python scripts/dft_single.py "
                f"status --user {args.user} --case {case_name}"
            )
            return 0

    poll = max(5, args.poll_interval)
    escalations = 0
    while True:
        try:
            row = wait_for_job(c, job_id, poll)
        except KeyboardInterrupt:
            print(
                f"\n[single] interrupted - the job keeps running on the "
                f"cluster. Resume with:\n  {resume_hint(args, case_name)}",
                file=sys.stderr,
            )
            return 130

        state = row["state"]
        if is_oom_row(row):
            current = args.mem_gb or suggest_mem_gb(args.smiles, args.pfas_smiles)
            rss = next(
                (
                    r["maxrss"]
                    for r in sacct_rows_for(c, job_id)
                    if r["jobid"] == f"{job_id}.batch" and r["maxrss"]
                ),
                "unknown",
            )
            new_mem = min(current * 2, ESCALATION_CAP_GB)
            if escalations < MAX_ESCALATIONS and new_mem > current:
                escalations += 1
                print(
                    f"[single] job {job_id} ran out of memory (state "
                    f"{state}, peak rss {rss}); resubmitting with "
                    f"{new_mem}G (auto-escalation {escalations}/"
                    f"{MAX_ESCALATIONS}; finished legs are reused "
                    "automatically)"
                )
                args.mem_gb = new_mem
                job_id = resubmit_with_mem(c, args, case_name, new_mem)
                continue
            print(
                f"[single] job {job_id} ran out of memory and the "
                "escalation budget is exhausted (cap "
                f"{ESCALATION_CAP_GB}G); inspect the fetched logs",
                file=sys.stderr,
            )
            fetch_case(c, case_name, args.local_cache)
            return 2

        if state == "COMPLETED":
            data = read_remote_results(c, root, case_name)
            if data is not None:
                dst = fetch_case(c, case_name, args.local_cache)
                print(f"[single] fetched results to {dst}")
                print(format_results_report(data))
                return 0
            fetch_case(c, case_name, args.local_cache)
            if _remote_file_exists(c, f"{run_dir}/DONE"):
                print(
                    "[single] the workflow finished but results.json is "
                    "missing - a leg was skipped or produced no energy; "
                    "check outputs/slurm-*.out in the fetched cache",
                    file=sys.stderr,
                )
                return 1
            print(
                "[single] the job 'completed' but the workflow exited "
                "early (no DONE marker); check outputs/slurm-*.err in "
                "the fetched cache",
                file=sys.stderr,
            )
            return 2

        # TIMEOUT / FAILED / CANCELLED / NODE_FAIL / ...
        hint = {
            "TIMEOUT": "walltime exceeded - rerun with a larger --time; "
            "finished legs are reused automatically",
        }.get(state, "check the fetched slurm logs (out and err)")
        fetch_case(c, case_name, args.local_cache)
        print(
            f"[single] job {job_id} ended in state {state}; {hint}",
            file=sys.stderr,
        )
        return 2


def resubmit_with_mem(c, args: argparse.Namespace, case_name: str, new_mem: int) -> int:
    workflow_script = args.workflow_script or (
        f"{c.root.rstrip('/')}/scripts/run_dft_workflow.sh"
    )
    job_id = submit_job(
        c,
        case_name=case_name,
        adsorbent_name=args.name or case_name,
        pfas_name=args.pfas_name,
        adsorbent_smiles=args.smiles,
        pfas_smiles=args.pfas_smiles,
        pfas_energy_ry=args.pfas_energy_ry,
        adsorbent_source=args.adsorbent_source,
        adsorbent_cif=args.adsorbent_cif,
        system_type=args.system_type,
        mode=args.mode,
        partition=args.partition,
        account=args.account,
        qos=args.qos,
        time_limit=args.time,
        cpus=args.cpus,
        mem_gb=new_mem,
        workflow_script=workflow_script,
    )
    print(f"[single] resubmitted as job {job_id}")
    return job_id


def cmd_status(args: argparse.Namespace) -> int:
    c = make_cluster(args.user, args.cluster, args.cluster_root, args.runs_subdir)
    root = c.root.rstrip("/")
    run_dir = wrapper.rdir(c, args.case)
    _open_master(c)

    print(f"case:   {args.case}")
    print(
        "run dir exists: "
        f"{_ssh(c, f'test -d {shlex.quote(run_dir)}', check=False).returncode == 0}"
    )
    print(f"DONE marker:    {_remote_file_exists(c, f'{run_dir}/DONE')}")

    data = read_remote_results(c, root, args.case)
    if data is not None:
        print(format_results_report(data))
    else:
        print("results.json:   not on the cluster yet")

    live = live_job(c, args.case)
    if live is not None:
        print(f"live job: {live[0]} {live[1]}")
    else:
        print("live job: none")

    p = _ssh(
        c,
        f"sacct -n -P -S $(date -d '7 days ago' +%F) "
        f"--name=dft_{args.case} -o JobID,State,ExitCode,MaxRSS,Elapsed",
        check=False,
        capture=True,
    )
    if p.stdout.strip():
        print("recent sacct rows:")
        print("\n".join("  " + ln for ln in p.stdout.strip().splitlines()))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    c = make_cluster(args.user, args.cluster, args.cluster_root, args.runs_subdir)
    _open_master(c)
    dst = fetch_case(c, args.case, args.local_cache)
    print(f"[single] fetched to {dst}")
    data = read_remote_results(c, c.root.rstrip("/"), args.case)
    if data is not None:
        print(format_results_report(data))
    return 0


# ------------------------------------------------------------------ CLI


def add_cluster_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--user", required=True, help="Cluster username.")
    sp.add_argument("--cluster", default=wrapper.DEFAULT_CLUSTER_HOST)
    sp.add_argument("--cluster-root", default=wrapper.DEFAULT_CLUSTER_ROOT)
    sp.add_argument("--runs-subdir", default=wrapper.DEFAULT_RUNS_SUBDIR)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="dft_single.py",
        description="One-command single-case DFT: submit, monitor, fetch, verdict.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Submit (if needed), monitor, fetch, verdict.")
    add_cluster_args(run)
    run.add_argument("--name", help="Adsorbent name (new case).")
    run.add_argument("--case", help="Explicit case name (attach to existing).")
    run.add_argument("--smiles", help="Adsorbent SMILES.")
    run.add_argument(
        "--adsorbent-cif",
        help="Path to an adsorbent CIF on the cluster (with --adsorbent-source cif).",
    )
    run.add_argument("--adsorbent-source", choices=["smiles", "cif"], default="smiles")
    run.add_argument("--pfas-name", default=DEFAULT_PFAS_NAME)
    run.add_argument("--pfas-smiles", default=DEFAULT_PFAS_SMILES)
    run.add_argument(
        "--pfas-energy-ry",
        type=float,
        default=None,
        help="Skip the PFAS reference calc with a known energy (Ry).",
    )
    run.add_argument(
        "--mode",
        choices=["lowmem", "cluster", "production"],
        default="production",
    )
    run.add_argument(
        "--system-type", choices=["molecule", "periodic"], default="molecule"
    )
    run.add_argument(
        "--mem-gb",
        type=int,
        default=None,
        help="Override the automatic 32/96/128 GB sizing.",
    )
    run.add_argument("--time", default=DEFAULT_TIME)
    run.add_argument("--cpus", type=int, default=DEFAULT_CPUS)
    run.add_argument("--partition", default=DEFAULT_PARTITION)
    run.add_argument("--account", default=DEFAULT_ACCOUNT)
    run.add_argument("--qos", default=DEFAULT_QOS)
    run.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_SECONDS)
    run.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit and exit; resume later with the printed command.",
    )
    run.add_argument("--local-cache", default=wrapper.DEFAULT_LOCAL_CACHE)
    run.add_argument(
        "--workflow-script",
        default=None,
        help="Path to run_dft_workflow.sh on the cluster "
        "(default: <cluster-root>/scripts/run_dft_workflow.sh).",
    )
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="Show run state, results, live job.")
    add_cluster_args(status)
    status.add_argument("--case", required=True)
    status.set_defaults(func=cmd_status)

    fetch = sub.add_parser(
        "fetch", help="Download results + slurm logs + dft_cases/<case> tree."
    )
    add_cluster_args(fetch)
    fetch.add_argument("--case", required=True)
    fetch.add_argument("--local-cache", default=wrapper.DEFAULT_LOCAL_CACHE)
    fetch.set_defaults(func=cmd_fetch)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as exc:
        print(f"[single] error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[single] interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
