"""Portable, reviewed SLURM bundles; exporting never queues or executes jobs."""

import hashlib
import io
import json
import math
import re
import shlex
import zipfile

from .inputs import validate_references
from .preparation import pseudo_names


def review(
    manager,
    candidates,
    include_tfa=False,
    processes=8,
    memory_gib=16,
    memory_mode="estimate",
    headroom_percent=50,
    walltime_hours=18,
    partition="ice-cpu",
    account="coc",
    qos="coc-ice",
):
    if (
        not candidates
        or len(candidates) > 256
        or len(set(candidates)) != len(candidates)
    ):
        raise ValueError("Select between 1 and 256 distinct candidates")
    if any(
        c not in manager.candidates
        or not re.fullmatch(r"[A-Za-z0-9_-]+", c)
        or c == "tfa"
        for c in candidates
    ):
        raise ValueError("Unknown candidate")
    if not math.isfinite(memory_gib) or not 0 < memory_gib <= 65536:
        raise ValueError("Memory must be positive and at most 65536 GiB")
    if memory_mode not in ("estimate", "manual"):
        raise ValueError("Unknown memory mode")
    if not math.isfinite(headroom_percent) or not 0 <= headroom_percent <= 1000:
        raise ValueError("Memory headroom must be between 0 and 1000 percent")
    if type(walltime_hours) is not int or not 1 <= walltime_hours <= 168:
        raise ValueError("Walltime must be between 1 and 168 hours")
    if type(processes) is not int or not 1 <= processes <= 256:
        raise ValueError("Process count must be between 1 and 256")
    for value in (partition, account, qos):
        if value and not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise ValueError("Invalid partition, account, or QOS")
    settings = {
        "memory_mib": math.ceil(memory_gib * 1024),
        "memory_mode": memory_mode,
        "headroom_percent": headroom_percent,
        "walltime_hours": walltime_hours,
        "partition": partition,
        "account": account,
        "qos": qos,
        "processes": processes,
    }
    pairs = [(c, s) for c in candidates for s in ("candidate", "complex")]
    if include_tfa:
        pairs.append(("tfa", "tfa"))
    entries = []
    estimates = manager.store.tasks() if memory_mode == "estimate" else []
    for candidate, system in pairs:
        entry = {
            "candidate": candidate,
            "system": system,
            "directory": f"runs/{candidate}-{system}",
        }
        try:
            source = manager.input(candidate, system)
            content = source.read_bytes()
            text = content.decode()
            validate_references(text)
            hashes = {
                name: hashlib.sha256(
                    (manager.config.pseudos / name).read_bytes()
                ).hexdigest()
                for name in pseudo_names(text)
            }
            entry.update(
                status="eligible",
                input=text,
                input_hash=hashlib.sha256(content).hexdigest(),
                pseudopotentials=hashes,
                command=[
                    "apptainer",
                    "exec",
                    "chemistry.sif",
                    *mpi_command(processes),
                    "pw.x",
                    "-in",
                    "input.in",
                ],
            )
            if memory_mode == "estimate":
                entry.update(
                    estimate_memory(estimates, entry, processes, headroom_percent)
                )
            else:
                entry["memory_mib"] = settings["memory_mib"]
        except (ValueError, OSError, UnicodeError) as error:
            entry.update(status="unavailable", reason=str(error))
        entries.append(entry)
    return {"settings": settings, "entries": entries}


def estimate_memory(tasks, entry, processes, headroom_percent):
    # History is newest first. Prefer matching-rank estimates to a serial baseline.
    for ranks in dict.fromkeys((processes, 1)):
        for task in tasks:
            if not (
                task.get("kind") == "estimate_ram"
                and task.get("status") == "succeeded"
                and task.get("candidate") == entry["candidate"]
                and task.get("system") == entry["system"]
                and task.get("processes") == ranks
                and task.get("source_hash") == entry["input_hash"]
                and task.get("pseudopotentials") == entry["pseudopotentials"]
            ):
                continue
            report = task.get("ram_estimate") or {}
            per_process = (report.get("per_process") or {}).get("bytes")
            total = (report.get("total") or {}).get("bytes")

            def valid(value):
                return (
                    type(value) in (int, float) and math.isfinite(value) and value > 0
                )

            # QE's total is preferred for matching ranks; accept a per-rank
            # maximum when the total is absent. A serial report already covers
            # the complete calculation, so use it as the job baseline directly.
            if ranks == processes:
                if valid(total):
                    base = total
                    basis = "Matching MPI total estimate"
                elif valid(per_process):
                    base = per_process * processes
                    basis = "Matching per-process estimate × MPI processes"
                else:
                    continue
            else:
                if valid(total):
                    base = total
                elif valid(per_process):
                    base = per_process
                else:
                    continue
                basis = "One-process estimate used as total job baseline"
            memory_gib = max(
                1, math.ceil(base * (1 + headroom_percent / 100) / 1024**3)
            )
            if memory_gib > 65536:
                raise ValueError(
                    "Estimated reservation exceeds 65536 GiB; use manual memory"
                )
            return {
                "memory_mib": memory_gib * 1024,
                "memory_estimate": {
                    "task_id": task["id"],
                    "processes": ranks,
                    "base_bytes": base,
                    "basis": basis,
                },
            }
    raise ValueError(
        "No matching RAM estimate; estimate this input first or select manual memory"
    )


def mpi_command(processes):
    # The image contains Open MPI 4. One container runs on one allocated node;
    # isolated launching prevents MPI from trying to invoke host Slurm binaries.
    return (
        []
        if processes == 1
        else [
            "mpirun",
            "--mca",
            "plm",
            "isolated",
            "--mca",
            "ras",
            "^slurm",
            "--bind-to",
            "none",
            "-np",
            str(processes),
        ]
    )


RUN_SCRIPT = """#!/bin/bash
set -euo pipefail
image=$1
run_dir=$2
batch_root=$3
cd -- "$run_dir"
mkdir -p Outputs
exec apptainer exec --cleanenv --bind "$batch_root:$batch_root" --pwd "$run_dir" \\
    --env OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1 \\
    "$image" {command} pw.x -in input.in > pw.out 2> pw.err
"""


def bundle(manager, expected, **request):
    current = review(manager, **request)
    if not expected or current != expected:
        raise ValueError("Inputs, pseudopotentials, or settings changed; preview again")
    if any(e["status"] != "eligible" for e in current["entries"]):
        raise ValueError(
            "Prepare both inputs and resolve unavailable RAM estimates before exporting"
        )
    settings = current["settings"]
    options = [
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={settings['processes']}",
        f"--time={settings['walltime_hours']}:00:00",
    ]
    for key in ("partition", "account", "qos"):
        if settings[key]:
            options.append(f"--{key}={settings[key]}")
    submit = """#!/bin/bash
set -euo pipefail
batch_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
image=$(realpath -- "${1:-$batch_root/chemistry.sif}")
test -f "$image" || { echo "Missing image: $image" >&2; exit 1; }
command -v sbatch >/dev/null
if [[ -s "$batch_root/jobs.tsv" ]]; then
    echo "This batch has submitted jobs. Check jobs.tsv before resubmitting; use a fresh export for a new batch." >&2
    exit 1
fi
"""
    for entry in current["entries"]:
        name = f"{entry['candidate']}-{entry['system']}"
        submit += f'run_dir="$batch_root/{entry["directory"]}"\n'
        submit += f"job_id=$(sbatch --parsable {shlex.join(options)} --mem={entry['memory_mib']} --job-name=pfas-{name} "
        submit += '--chdir="$run_dir" --output="$run_dir/slurm-%j.out" --error="$run_dir/slurm-%j.err" '
        submit += '"$batch_root/run.sbatch" "$image" "$run_dir" "$batch_root")\n'
        submit += f'printf \'%s\\t%s\\n\' {shlex.quote(name)} "$job_id" | tee -a "$batch_root/jobs.tsv"\n'
    readme = """PFAS SLURM batch

Each selected molecule has an isolated-candidate and TFA-complex calculation.
The optional isolated TFA reference appears once. Each job runs on one node.
The selected MPI process count is also the number of CPUs reserved per job.
The container's Open MPI launches all ranks locally, with one thread per rank.
The isolated launcher avoids depending on the host MPI or Slurm libraries.
This launch mode is specific to Open MPI 4 in containers/chemistry.def.
More ranks do not guarantee better QE performance; compare 8 and 16 in practice.
Memory and walltime are reservations chosen in the dashboard, not measured usage.
Automatic memory uses matching input/pseudopotential RAM estimates plus headroom,
rounded up to whole GiB (minimum 1 GiB). Matching MPI total estimates are preferred;
otherwise a one-process estimate is used directly as the total job baseline,
without multiplying by the selected MPI count. Default headroom is 50 percent.
Manifest entries record each job's memory, estimate source, and calculation basis.
Existing local results do not remove calculations from this export.

1. Extract this ZIP locally or on PACE.
2. Put your built chemistry.sif in this folder (the image is NOT in this ZIP).
3. Upload the whole folder from your local terminal, for example:
   scp -r pfas-slurm-batch GTUSER@login-ice.pace.gatech.edu:~/
4. Log in to PACE, load Apptainer if your cluster requires a module, then:
   cd ~/pfas-slurm-batch
   bash submit.sh
   # Or: bash submit.sh /absolute/path/to/chemistry.sif

Confirm partition/account/QOS and resources in manifest.json and submit.sh.
sbatch submits each calculation separately; it does not wait for completion.
jobs.tsv records accepted job IDs. Use squeue -u "$USER" to monitor them.
Each run folder receives pw.out, pw.err, slurm logs, and QE scratch in Outputs/.
Look for JOB DONE and convergence in pw.out; exit zero alone is not sufficient.
If submission fails partway, jobs.tsv lists the accepted jobs. Submit only the
remaining jobs manually using the commands in submit.sh to avoid duplicates.
The dashboard does not monitor or import remote results in this version.
Keep inputs, pseudopotentials, and the same image together for reproducibility.
"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:

        def write(name, content):
            archive.writestr("pfas-slurm-batch/" + name, content)

        write(
            "run.sbatch",
            RUN_SCRIPT.replace(
                "{command}", shlex.join(mpi_command(settings["processes"]))
            ),
        )
        write("submit.sh", submit)
        write("README.txt", readme)
        write("manifest.json", json.dumps(current, indent=2))
        for entry in current["entries"]:
            directory = entry["directory"]
            write(directory + "/input.in", entry["input"])
            write(directory + "/Outputs/", b"")
            for name, expected_hash in entry["pseudopotentials"].items():
                content = (manager.config.pseudos / name).read_bytes()
                if hashlib.sha256(content).hexdigest() != expected_hash:
                    raise ValueError(
                        "Pseudopotentials changed during export; preview again"
                    )
                write(directory + "/Pseudopotentials/" + name, content)
    return buffer.getvalue()
