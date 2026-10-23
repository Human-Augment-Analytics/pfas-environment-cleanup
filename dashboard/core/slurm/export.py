"""Portable, reviewed SLURM bundles; exporting never queues or executes jobs."""

import hashlib
import io
import json
import math
import re
import shlex
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, TypeGuard, Unpack

from dashboard.core import execution, models
from dashboard.core.chemistry.results import write_entrypoint

if TYPE_CHECKING:
    from dashboard.core.config import Config
    from dashboard.core.jobs.tasks import Manager
    from dashboard.core.runtimes.local import Runtimes

from dashboard.core.chemistry.inputs import validate_references
from dashboard.core.chemistry.preparation import pseudo_names


def review(
    manager: "Manager",
    candidates: list[str],
    include_tfa: bool = False,
    processes: int = 8,
    memory_gib: float = 16,
    memory_mode: str = "estimate",
    headroom_percent: float = 50,
    walltime_hours: int = 18,
    partition: str = "ice-cpu",
    account: str = "coc",
    qos: str = "coc-ice",
) -> dict:
    """Preserve export options; validate settings and review each input separately."""
    validate_selection(manager, candidates)
    settings = export_settings(
        processes,
        memory_gib,
        memory_mode,
        headroom_percent,
        walltime_hours,
        partition,
        account,
        qos,
    )
    pairs = [
        (candidate, system)
        for candidate in candidates
        for system in ("candidate", "complex")
    ]
    if include_tfa:
        pairs.append(("tfa", "tfa"))
    estimates = manager.store.tasks() if memory_mode == "estimate" else []
    entries = [
        review_entry(manager, candidate, system, settings, estimates)
        for candidate, system in pairs
    ]
    return {"settings": settings, "entries": entries}


def validate_selection(manager: "Manager", candidates: list[str]) -> None:
    if (
        not candidates
        or len(candidates) > execution.MAX_SLURM_PROCESSES
        or len(set(candidates)) != len(candidates)
    ):
        raise ValueError("Select between 1 and 256 distinct candidates")
    if any(
        candidate not in manager.candidates
        or not re.fullmatch(r"[A-Za-z0-9_-]+", candidate)
        or candidate == "tfa"
        for candidate in candidates
    ):
        raise ValueError("Unknown candidate")


def validate_export_resources(
    processes: int, memory_gib: float, memory_mode: str, headroom: float, hours: int
) -> None:
    if (
        not math.isfinite(memory_gib)
        or not 0 < memory_gib <= execution.MAX_SLURM_MEMORY_GIB
    ):
        raise ValueError("Memory must be positive and at most 65536 GiB")
    if memory_mode not in ("estimate", "manual"):
        raise ValueError("Unknown memory mode")
    if not math.isfinite(headroom) or not 0 <= headroom <= 1000:
        raise ValueError("Memory headroom must be between 0 and 1000 percent")
    if type(hours) is not int or not 1 <= hours <= execution.MAX_SLURM_HOURS:
        raise ValueError("Walltime must be between 1 and 168 hours")
    if (
        type(processes) is not int
        or not 1 <= processes <= execution.MAX_SLURM_PROCESSES
    ):
        raise ValueError("Process count must be between 1 and 256")


def export_settings(
    processes: int,
    memory_gib: float,
    memory_mode: str,
    headroom: float,
    hours: int,
    partition: str,
    account: str,
    qos: str,
) -> dict:
    validate_export_resources(processes, memory_gib, memory_mode, headroom, hours)
    for value in (partition, account, qos):
        if value and not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise ValueError("Invalid partition, account, or QOS")
    return {
        "memory_mib": math.ceil(memory_gib * 1024),
        "memory_mode": memory_mode,
        "headroom_percent": headroom,
        "walltime_hours": hours,
        "partition": partition,
        "account": account,
        "qos": qos,
        "processes": processes,
    }


def export_input(
    manager: "Manager", candidate: str, system: str, processes: int
) -> dict:
    content = manager.input(candidate, system).read_bytes()
    text = content.decode()
    validate_references(text)
    hashes = {
        name: hashlib.sha256((manager.config.pseudos / name).read_bytes()).hexdigest()
        for name in pseudo_names(text)
    }
    return {
        "status": "eligible",
        "input": text,
        "input_hash": hashlib.sha256(content).hexdigest(),
        "pseudopotentials": hashes,
        "command": [
            "apptainer",
            "exec",
            "chemistry.sif",
            *mpi_command(processes),
            "pw.x",
            "-in",
            "input.in",
        ],
    }


def review_entry(
    manager: "Manager",
    candidate: str,
    system: str,
    settings: dict,
    estimates: list[models.TaskRecord],
) -> dict:
    entry = {
        "candidate": candidate,
        "system": system,
        "directory": f"runs/{candidate}-{system}",
    }
    try:
        entry.update(export_input(manager, candidate, system, settings["processes"]))
        if settings["memory_mode"] == "estimate":
            entry.update(
                estimate_memory(
                    estimates,
                    entry,
                    settings["processes"],
                    settings["headroom_percent"],
                )
            )
        else:
            entry["memory_mib"] = settings["memory_mib"]
    except (ValueError, OSError, UnicodeError) as error:
        entry.update(status="unavailable", reason=str(error))
    return entry


def valid_memory(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def matches_estimate(task: models.TaskRecord, entry: dict, ranks: int) -> bool:
    return (
        task.get("kind") == "estimate_ram"
        and task.get("status") == "succeeded"
        and task.get("candidate") == entry["candidate"]
        and task.get("system") == entry["system"]
        and task.get("processes") == ranks
        and task.get("source_hash") == entry["input_hash"]
        and task.get("pseudopotentials") == entry["pseudopotentials"]
    )


def memory_baseline(
    report: models.RAMReport | dict, ranks: int, processes: int
) -> tuple[float, str] | None:
    total = (report.get("total") or {}).get("bytes")
    per_process = (report.get("per_process") or {}).get("bytes")
    if valid_memory(total):
        base, basis = total, "Matching MPI total estimate"
    elif valid_memory(per_process):
        base = per_process * processes if ranks == processes else per_process
        basis = "Matching per-process estimate × MPI processes"
    else:
        return None
    # A serial estimate is already a whole-job baseline; never multiply it.
    if ranks != processes:
        basis = "One-process estimate used as total job baseline"
    return base, basis


def sized_memory(
    task_id: str, ranks: int, base: float, basis: str, headroom: float
) -> dict:
    memory_gib = max(1, math.ceil(base * (1 + headroom / 100) / execution.GIB))
    if memory_gib > execution.MAX_SLURM_MEMORY_GIB:
        raise ValueError("Estimated reservation exceeds 65536 GiB; use manual memory")
    return {
        "memory_mib": memory_gib * 1024,
        "memory_estimate": {
            "task_id": task_id,
            "processes": ranks,
            "base_bytes": base,
            "basis": basis,
        },
    }


def estimate_memory(
    tasks: list[models.TaskRecord], entry: dict, processes: int, headroom_percent: float
) -> dict:
    # Newest matching-rank estimate first; serial is an explicitly labeled fallback.
    for ranks in dict.fromkeys((processes, 1)):
        for task in tasks:
            if matches_estimate(task, entry, ranks):
                baseline = memory_baseline(
                    task.get("ram_estimate") or {}, ranks, processes
                )
                if baseline is not None:
                    return sized_memory(task["id"], ranks, *baseline, headroom_percent)
    raise ValueError(
        "No matching RAM estimate; estimate this input first or select manual memory"
    )


def mpi_command(processes: int) -> list[str]:
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


EXPORT_ROOT = "pfas-slurm-batch/"
EXPORT_README = """PFAS SLURM batch

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
SUBMIT_HEADER = """#!/bin/bash
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


def submission_options(settings: dict) -> list[str]:
    options = [
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={settings['processes']}",
        f"--time={settings['walltime_hours']}:00:00",
    ]
    options.extend(
        f"--{key}={settings[key]}"
        for key in ("partition", "account", "qos")
        if settings[key]
    )
    return options


def submission_line(entry: dict, options: list[str]) -> str:
    name = f"{entry['candidate']}-{entry['system']}"
    script = f'run_dir="$batch_root/{entry["directory"]}"\n'
    script += f"job_id=$(sbatch --parsable {shlex.join(options)} --mem={entry['memory_mib']} --job-name=pfas-{name} "
    script += '--chdir="$run_dir" --output="$run_dir/slurm-%j.out" --error="$run_dir/slurm-%j.err" '
    script += '"$batch_root/run.sbatch" "$image" "$run_dir" "$batch_root")\n'
    script += f'printf \'%s\\t%s\\n\' {shlex.quote(name)} "$job_id" | tee -a "$batch_root/jobs.tsv"\n'
    return script


def write_file(archive: zipfile.ZipFile, name: str, content: str | bytes) -> None:
    archive.writestr(EXPORT_ROOT + name, content)


def write_export_run(manager: "Manager", archive: zipfile.ZipFile, entry: dict) -> None:
    directory = entry["directory"]
    write_file(archive, directory + "/input.in", entry["input"])
    write_file(archive, directory + "/Outputs/", b"")
    # Recheck content while packaging, after the preview's earlier hash check.
    for name, expected in entry["pseudopotentials"].items():
        content = (manager.config.pseudos / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("Pseudopotentials changed during export; preview again")
        write_file(archive, directory + "/Pseudopotentials/" + name, content)


def write_export_files(
    manager: "Manager", archive: zipfile.ZipFile, current: dict
) -> None:
    settings = current["settings"]
    script = SUBMIT_HEADER + "".join(
        submission_line(entry, submission_options(settings))
        for entry in current["entries"]
    )
    write_file(
        archive,
        "run.sbatch",
        RUN_SCRIPT.replace("{command}", shlex.join(mpi_command(settings["processes"]))),
    )
    write_file(archive, "submit.sh", script)
    write_file(archive, "README.txt", EXPORT_README)
    write_file(archive, "manifest.json", json.dumps(current, indent=2))
    for entry in current["entries"]:
        write_export_run(manager, archive, entry)


def bundle(
    manager: "Manager",
    expected: dict | None,
    **request: Unpack[models.SlurmExportRequest],
) -> bytes:
    current = review(manager, **request)
    if not expected or current != expected:
        raise ValueError("Inputs, pseudopotentials, or settings changed; preview again")
    if any(entry["status"] != "eligible" for entry in current["entries"]):
        raise ValueError(
            "Prepare both inputs and resolve unavailable RAM estimates before exporting"
        )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        write_export_files(manager, archive, current)
    return buffer.getvalue()


DIRECT_ATTEMPT_SCRIPT = r"""run_attempt() (
    cd -- "$1" || exit 1
    duration=$2
    shift 2
    date -u +%Y-%m-%dT%H:%M:%SZ > {started}
    if [ "$duration" != 0 ]; then
        set -- timeout --signal=TERM --kill-after={kill_seconds} "$duration" "$@"
    fi
    if sha256sum --check {checksums} > {check_log} 2>&1; then
        if "$@" > {stdout} 2> {stderr}; then code=0; else code=$?; fi
    else
        code={checksum_failure}
    fi
    printf '%s\n%s\n' "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > {finished}.tmp
    mv -- {finished}.tmp {finished}
)
"""


def image_preflight(
    tasks: list[models.TaskRecord],
    config: "Config",
    runtimes: "Runtimes",
    directory: Path,
) -> list[str]:
    container_tasks = [
        task for task in tasks if task["runtime"] == execution.Runtime.APPTAINER
    ]
    if not container_tasks:
        return []
    identity = runtimes.image_identity()
    if any(task["image_hash"] != identity for task in container_tasks):
        raise ValueError("Container image changed; preview and retry explicitly")
    checksums = directory / execution.IMAGE_CHECKSUMS_FILE
    checksums.write_text(f"{identity}  {config.image}\n")
    return [f"sha256sum --check {shlex.quote(str(checksums))}"]


def direct_attempt_template() -> str:
    """Keep the shell template's filename substitution together for auditability."""
    return DIRECT_ATTEMPT_SCRIPT.format(
        started=execution.SLURM_STARTED_TEXT_FILE,
        finished=execution.SLURM_FINISHED_FILE,
        checksums=execution.SNAPSHOT_CHECKSUMS_FILE,
        check_log=execution.SNAPSHOT_CHECK_LOG,
        stdout=execution.STDOUT_LOG_FILE,
        stderr=execution.STDERR_LOG_FILE,
        kill_seconds=execution.TIMEOUT_KILL_SECONDS,
        checksum_failure=execution.SNAPSHOT_CHECK_FAILED,
    )


def write_input_checksums(task: models.TaskRecord, directory: Path) -> None:
    checksums = [f"{task['input_hash']}  {execution.INPUT_FILE}"]
    checksums.extend(
        f"{digest}  {execution.PSEUDOS_DIRECTORY}/{name}"
        for name, digest in task["pseudopotentials"].items()
    )
    (directory / execution.SNAPSHOT_CHECKSUMS_FILE).write_text(
        "\n".join(checksums) + "\n"
    )


def direct_attempt(
    task: models.TaskRecord, config: "Config", runtimes: "Runtimes"
) -> str:
    directory = config.artifacts / task["id"]
    write_input_checksums(task, directory)
    command = task["command"]
    if task["runtime"] == execution.Runtime.APPTAINER:
        command = runtimes.allocation_command(command, directory)
    write_entrypoint(task, directory, command)
    return (
        shlex.join(["run_attempt", str(directory), str(task["timeout"] or 0), *command])
        + "\n"
    )


def write_direct_job(
    tasks: list[models.TaskRecord],
    config: "Config",
    runtimes: "Runtimes",
    directory: Path,
) -> Path:
    """Write direct commands; the compute node needs no dashboard Python worker."""
    header = ["#!/bin/bash", "set -euo pipefail"]
    header.extend(f"export {name}=1" for name in execution.THREAD_ENV_VARS)
    header.extend(image_preflight(tasks, config, runtimes, directory))
    script = "\n".join(header) + "\n" + direct_attempt_template()
    # Each check publishes its own result before the next selected check starts.
    script += "".join(direct_attempt(task, config, runtimes) for task in tasks)
    output = directory / execution.SLURM_SCRIPT_FILE
    output.write_text(script)
    return output
