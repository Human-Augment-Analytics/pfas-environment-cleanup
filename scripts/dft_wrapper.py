#!/usr/bin/env python3

# a wrapper that can be used to run DFT simulations
# skips if DFT is ran for the simulation already
"""
Docstring for scripts.dft_wrapper

Check status:
python3 scripts/dft_wrapper.py --host user@ice-login --compound-id IKHGUXGNUITLKF-UHFFFAOYSA-N --status

Submit if results are missing:
python3 scripts/dft_wrapper.py --host user@ice-login --compound-id IKHGUXGNUITLKF-UHFFFAOYSA-N --submit-if-missing

Fetch if results exist:
python3 scripts/dft_wrapper.py --host user@ice-login --compound-id IKHGUXGNUITLKF-UHFFFAOYSA-N --fetch
"""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_CLUSTER_ROOT = "/storage/ice-shared/cs8903onl/mussmann-pfas"
DEFAULT_RUNS_SUBDIR = "dft_runs"
DEFAULT_LOCAL_CACHE = "./data/dft_cache"
DEFAULT_CLUSTER_HOST = "login-ice.pace.gatech.edu"

def run(cmd: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """
    Runs a shell command.
    capture=False keeps it interactive; capture=True captures stdout/stderr.
    """
    if capture:
        p = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        p = subprocess.run(cmd, shell=True)

    if check and p.returncode != 0:
        if capture:
            raise RuntimeError(
                f"Command failed ({p.returncode}): {cmd}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
            )
        raise RuntimeError(f"Command failed ({p.returncode}): {cmd}")
    return p


@dataclass(frozen=True)
class Cluster:
    ssh_target: str
    root: str = DEFAULT_CLUSTER_ROOT
    runs_subdir: str = DEFAULT_RUNS_SUBDIR
    control_path: str = ""

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def rdir(c: Cluster, compound_id: str) -> str:
    return f"{c.root.rstrip('/')}/{c.runs_subdir.strip('/')}/{compound_id}"

def ensure_control_dir() -> Path:
    d = Path.home() / ".ssh" / "cm_sockets"
    d.mkdir(parents=True, exist_ok=True)
    return d

def make_control_path(user: str, host: str) -> str:
    d = ensure_control_dir()
    return str(d / f"cm-{user}@{host}")

def ssh_opts_control(c: Cluster) -> list[str]:
    return [
        "-o", "ControlMaster=auto",
        "-o", "ControlPersist=10m",
        "-o", f"ControlPath={c.control_path}",
        "-o", "ServerAliveInterval=30",
        "-o", "LogLevel=ERROR",
        "-q",
    ]

def ssh_cmd_interactive(c: Cluster, remote_cmd: str) -> str:
    opts = ssh_opts_control(c) + ["-tt"]
    return "ssh " + " ".join(shlex.quote(x) for x in opts) + " " + shlex.quote(c.ssh_target) + " " + shlex.quote(remote_cmd)

def ssh_cmd_quiet(c: Cluster, remote_cmd: str) -> str:
    opts = ssh_opts_control(c) + ["-T"]
    return "ssh " + " ".join(shlex.quote(x) for x in opts) + " " + shlex.quote(c.ssh_target) + " " + shlex.quote(remote_cmd)

def open_master_connection(c: Cluster) -> None:
    print("[AUTH] Warming SSH (you should be prompted once if needed)...")
    run(ssh_cmd_interactive(c, "echo AUTH_OK"), check=True, capture=False)
    print("[AUTH] SSH OK (ControlMaster enabled).")

def remote_file_exists(c: Cluster, remote_path: str) -> bool:
    cmd = ssh_cmd_quiet(c, f"test -f {shlex.quote(remote_path)}")
    p = run(cmd, check=False, capture=False)
    return p.returncode == 0

def remote_dir_exists(c: Cluster, remote_path: str) -> bool:
    cmd = ssh_cmd_quiet(c, f"test -d {shlex.quote(remote_path)}")
    p = run(cmd, check=False, capture=False)
    return p.returncode == 0

def ensure_remote_dirs(c: Cluster, compound_id: str) -> None:
    cmd = ssh_cmd_quiet(c, f"mkdir -p {shlex.quote(rdir(c, compound_id))}/{{inputs,outputs,results}}")
    run(cmd, check=True, capture=False)

def write_remote_file(c: Cluster, remote_path: str, content: str) -> None:
    cmd = ssh_cmd_quiet(c, f"cat > {shlex.quote(remote_path)} << 'EOF'\n{content}\nEOF")
    run(cmd, check=True, capture=False)

def submit_slurm_job(
    c: Cluster,
    compound_id: str,
    workflow_script: str,
    partition: Optional[str],
    time_limit: str,
    cpus: int,
    mem_gb: int,
) -> None:
    run_dir = rdir(c, compound_id)
    jobfile = f"{run_dir}/job.sbatch"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=dft_{compound_id}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem_gb}G",
        f"#SBATCH --time={time_limit}",
        "#SBATCH --output=outputs/slurm-%j.out",
        "#SBATCH --error=outputs/slurm-%j.err",
    ]
    if partition:
        lines.append(f"#SBATCH --partition={partition}")

    lines += [
        "set -euo pipefail",
        f"cd {shlex.quote(run_dir)}",
        "echo \"[DFT] Starting workflow at $(date)\"",
        f"{shlex.quote(workflow_script)} {shlex.quote(compound_id)}",
        "echo \"[DFT] Finished at $(date)\"",
        "echo \"DONE\" > DONE",
    ]

    write_remote_file(c, jobfile, "\n".join(lines) + "\n")
    run(ssh_cmd_quiet(c, f"sbatch {shlex.quote(jobfile)}"), check=True, capture=False)

def fetch_with_tar(c: Cluster, run_dir: str, dst: Path) -> None:
    if not remote_dir_exists(c, run_dir):
        raise RuntimeError(
            f"Remote run directory does not exist:\n  {run_dir}\n"
            "You likely need to run --submit-if-missing first, or your compound_id is wrong."
        )

    remote_tar_cmd = (
        f"cd {shlex.quote(run_dir)} && "
        f"tar -czf - "
        f"results meta.json DONE outputs/slurm-*.out outputs/slurm-*.err "
        f"2>/dev/null"
    )

    local_extract_cmd = f"tar -xzf - -C {shlex.quote(str(dst))}"
    cmd = f"{ssh_cmd_quiet(c, remote_tar_cmd)} | {local_extract_cmd}"
    run(cmd, check=True, capture=False)


def fetch(c: Cluster, compound_id: str, local_cache: str) -> Path:
    dst = Path(local_cache).resolve() / compound_id
    dst.mkdir(parents=True, exist_ok=True)

    run_dir = rdir(c, compound_id)

    fetch_with_tar(c, run_dir, dst)
    return dst

def main() -> int:
    ap = argparse.ArgumentParser(description="DFT wrapper: submit-if-missing and fetch results.")
    ap.add_argument("--user", required=True, help="Cluster username.")
    ap.add_argument("--cluster", default=DEFAULT_CLUSTER_HOST, help="Cluster SSH hostname.")
    ap.add_argument("--compound-id", required=True, help="Stable ID (InChIKey/CID/etc.).")
    ap.add_argument("--cluster-root", default=DEFAULT_CLUSTER_ROOT)
    ap.add_argument("--runs-subdir", default=DEFAULT_RUNS_SUBDIR)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--submit-if-missing", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--local-cache", default=DEFAULT_LOCAL_CACHE)
    ap.add_argument("--workflow-script", default=f"{DEFAULT_CLUSTER_ROOT}/run_dft_workflow.sh")
    ap.add_argument("--partition", default=None)
    ap.add_argument("--time", default="12:00:00")
    ap.add_argument("--cpus", type=int, default=8)
    ap.add_argument("--mem-gb", type=int, default=32)
    ap.add_argument("--no-auth-check", action="store_true")

    args = ap.parse_args()

    ssh_target = f"{args.user}@{args.cluster}"
    control_path = make_control_path(args.user, args.cluster)
    c = Cluster(ssh_target=ssh_target, root=args.cluster_root, runs_subdir=args.runs_subdir, control_path=control_path)

    if not args.no_auth_check:
        open_master_connection(c)

    run_dir = rdir(c, args.compound_id)
    done = remote_file_exists(c, f"{run_dir}/DONE")
    summary = remote_file_exists(c, f"{run_dir}/results/summary.json")

    if args.status:
        print(json.dumps(
            {"compound_id": args.compound_id, "run_dir": run_dir, "done": done, "summary": summary},
            indent=2
        ))
        return 0

    if args.submit_if_missing:
        if done and summary:
            print(f"[SKIP] {args.compound_id}: results already exist (DONE + summary.json).")
        else:
            print(f"[SUBMIT] {args.compound_id}: preparing directory and submitting SLURM job.")
            ensure_remote_dirs(c, args.compound_id)

            meta = {
                "compound_id": args.compound_id,
                "created_utc": now_utc_iso(),
                "cluster_root": args.cluster_root,
                "runs_subdir": args.runs_subdir,
                "run_dir": run_dir,
                "workflow_script": args.workflow_script,
                "slurm": {
                    "partition": args.partition,
                    "time": args.time,
                    "cpus": args.cpus,
                    "mem_gb": args.mem_gb,
                },
            }
            write_remote_file(c, f"{run_dir}/meta.json", json.dumps(meta, indent=2) + "\n")

            submit_slurm_job(
                c=c,
                compound_id=args.compound_id,
                workflow_script=args.workflow_script,
                partition=args.partition,
                time_limit=args.time,
                cpus=args.cpus,
                mem_gb=args.mem_gb,
            )
            print("[SUBMIT] Job submitted.")

    if args.fetch:
        dst = fetch(c, args.compound_id, args.local_cache)
        print(f"[FETCHED] {args.compound_id} -> {dst}")

    if not (args.status or args.submit_if_missing or args.fetch):
        ap.print_help()
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())