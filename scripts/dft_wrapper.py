#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
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


def rdir(c: Cluster, case_name: str) -> str:
    return f"{c.root.rstrip('/')}/{c.runs_subdir.strip('/')}/{case_name}"


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


def ensure_remote_dirs(c: Cluster, case_name: str) -> None:
    cmd = ssh_cmd_quiet(c, f"mkdir -p {shlex.quote(rdir(c, case_name))}/{{inputs,outputs,results}}")
    run(cmd, check=True, capture=False)


def write_remote_file(c: Cluster, remote_path: str, content: str) -> None:
    cmd = ssh_cmd_quiet(c, f"cat > {shlex.quote(remote_path)} << 'EOF'\n{content}\nEOF")
    run(cmd, check=True, capture=False)


def submit_slurm_job(
    c: Cluster,
    case_name: str,
    workflow_script: str,
    partition: Optional[str],
    time_limit: str,
    cpus: int,
    mem_gb: int,
    skip_ads: bool,
    skip_pfas: bool,
    skip_complex: bool,
    adsorbent_name: str,
    pfas_name: str,
    adsorbent_smiles: Optional[str],
    pfas_smiles: Optional[str],
    pfas_energy_ry: Optional[float],
    adsorbent_source: str,
    adsorbent_cif: Optional[str],
    system_type: str,
    mode: str,
) -> None:
    run_dir = rdir(c, case_name)
    jobfile = f"{run_dir}/job.sbatch"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=dft_{case_name}",
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
        "export OMP_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        f"export CASE_NAME={shlex.quote(case_name)}",
        f"export ADSORBENT_NAME={shlex.quote(adsorbent_name)}",
        f"export PFAS_NAME={shlex.quote(pfas_name)}",
        f"export COMPOUND_ROOT={shlex.quote(c.root.rstrip('/') + '/compounds')}",
        f"export WORKDIR={shlex.quote(c.root.rstrip('/') + '/dft_cases')}",
        f"export SKIP_ADS={'1' if skip_ads else '0'}",
        f"export SKIP_PFAS={'1' if skip_pfas else '0'}",
        f"export SKIP_COMPLEX={'1' if skip_complex else '0'}",
    ]

    if adsorbent_smiles:
        lines.append(f"export ADSORBENT_SMILES={shlex.quote(adsorbent_smiles)}")
    if pfas_smiles:
        lines.append(f"export PFAS_SMILES={shlex.quote(pfas_smiles)}")
    if pfas_energy_ry is not None:
        lines.append(f"export PFAS_ENERGY_RY={shlex.quote(str(pfas_energy_ry))}")

    lines += [
        f"export ADSORBENT_SOURCE={shlex.quote(adsorbent_source)}",
        f"export SYSTEM_TYPE={shlex.quote(system_type)}",
        f"export MODE={shlex.quote(mode)}",
    ]
    if adsorbent_cif:
        lines.append(f"export ADSORBENT_CIF={shlex.quote(adsorbent_cif)}")
        
    lines += [
        "echo \"[DFT] Starting workflow at $(date)\"",
        "echo \"[DFT] CASE_NAME=$CASE_NAME ADSORBENT_NAME=$ADSORBENT_NAME PFAS_NAME=$PFAS_NAME\"",
        "echo \"[DFT] SKIP_ADS=$SKIP_ADS SKIP_PFAS=$SKIP_PFAS SKIP_COMPLEX=$SKIP_COMPLEX\"",
        shlex.quote(workflow_script),
        "echo \"[DFT] Finished at $(date)\"",
        "echo \"DONE\" > DONE",
    ]

    write_remote_file(c, jobfile, "\n".join(lines) + "\n")
    run(ssh_cmd_quiet(c, f"sbatch {shlex.quote(jobfile)}"), check=True, capture=False)


def fetch_with_tar(c: Cluster, run_dir: str, dst: Path) -> None:
    if not remote_dir_exists(c, run_dir):
        raise RuntimeError(
            f"Remote run directory does not exist:\n  {run_dir}\n"
            "You likely need to run --submit-if-missing first, or your case_name is wrong."
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


def fetch(c: Cluster, case_name: str, local_cache: str) -> Path:
    dst = Path(local_cache).resolve() / case_name
    dst.mkdir(parents=True, exist_ok=True)

    run_dir = rdir(c, case_name)
    fetch_with_tar(c, run_dir, dst)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description="DFT wrapper: submit-if-missing and fetch results.")
    ap.add_argument("--user", required=True, help="Cluster username.")
    ap.add_argument("--cluster", default=DEFAULT_CLUSTER_HOST, help="Cluster SSH hostname.")
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

    ap.add_argument("--skip-ads", action="store_true")
    ap.add_argument("--skip-pfas", action="store_true")
    ap.add_argument("--skip-complex", action="store_true")

    ap.add_argument("--case-name", required=True)
    ap.add_argument("--adsorbent-name", required=True)
    ap.add_argument("--pfas-name", required=True)
    ap.add_argument("--adsorbent-smiles", required=False)
    ap.add_argument("--pfas-smiles", required=False)
    ap.add_argument("--pfas-energy-ry", type=float, default=None)
    ap.add_argument("--adsorbent-source", choices=["smiles", "cif"], required=True)
    ap.add_argument("--adsorbent-cif", default=None)
    ap.add_argument("--system-type", choices=["molecule", "periodic"], default="molecule")
    ap.add_argument("--mode", choices=["lowmem", "cluster", "production"], default="cluster")
    args = ap.parse_args()

    ssh_target = f"{args.user}@{args.cluster}"
    control_path = make_control_path(args.user, args.cluster)
    c = Cluster(
        ssh_target=ssh_target,
        root=args.cluster_root,
        runs_subdir=args.runs_subdir,
        control_path=control_path,
    )

    if not args.no_auth_check:
        open_master_connection(c)

    run_dir = rdir(c, args.case_name)
    done = remote_file_exists(c, f"{run_dir}/DONE")
    summary = remote_file_exists(c, f"{run_dir}/results/summary.json")

    if args.status:
        print(json.dumps(
            {
                "case_name": args.case_name,
                "run_dir": run_dir,
                "done": done,
                "summary": summary,
            },
            indent=2
        ))
        return 0

    if args.submit_if_missing:
        if done and summary:
            print(f"[SKIP] {args.case_name}: results already exist (DONE + summary.json).")
        else:
            print(f"[SUBMIT] {args.case_name}: preparing directory and submitting SLURM job.")
            ensure_remote_dirs(c, args.case_name)

            ads_done = remote_file_exists(c, f"{c.root}/compounds/adsorbents/{args.adsorbent_name}/adsorbent.out")
            pfas_done = remote_file_exists(c, f"{c.root}/compounds/pfas/{args.pfas_name}/pfas.out")
            complex_done = remote_file_exists(c, f"{c.root}/dft_cases/{args.case_name}/complex/complex.out")

            skip_ads = args.skip_ads or ads_done
            skip_pfas = args.skip_pfas or pfas_done or (args.pfas_energy_ry is not None)
            skip_complex = args.skip_complex or complex_done

            meta = {
                "case_name": args.case_name,
                "adsorbent_name": args.adsorbent_name,
                "pfas_name": args.pfas_name,
                "created_utc": now_utc_iso(),
                "cluster_root": args.cluster_root,
                "runs_subdir": args.runs_subdir,
                "run_dir": run_dir,
                "workflow_script": args.workflow_script,
                "pfas_energy_ry": args.pfas_energy_ry,
                "skip_flags": {
                    "skip_ads": skip_ads,
                    "skip_pfas": skip_pfas,
                    "skip_complex": skip_complex,
                },
                "existing_outputs": {
                    "ads_done": ads_done,
                    "pfas_done": pfas_done,
                    "complex_done": complex_done,
                },
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
                case_name=args.case_name,
                workflow_script=args.workflow_script,
                partition=args.partition,
                time_limit=args.time,
                cpus=args.cpus,
                mem_gb=args.mem_gb,
                skip_ads=skip_ads,
                skip_pfas=skip_pfas,
                skip_complex=skip_complex,
                adsorbent_name=args.adsorbent_name,
                pfas_name=args.pfas_name,
                adsorbent_smiles=args.adsorbent_smiles,
                pfas_smiles=args.pfas_smiles,
                pfas_energy_ry=args.pfas_energy_ry,
                adsorbent_source=args.adsorbent_source,
                adsorbent_cif=args.adsorbent_cif,
                system_type=args.system_type,
                mode=args.mode,
            )
            print("[SUBMIT] Job submitted.")

    if args.fetch:
        dst = fetch(c, args.case_name, args.local_cache)
        print(f"[FETCHED] {args.case_name} -> {dst}")

    if not (args.status or args.submit_if_missing or args.fetch):
        ap.print_help()
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())