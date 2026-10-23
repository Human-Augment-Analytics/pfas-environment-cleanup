import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

from .batches import Batches, validate_positive
from .calculation import run_calculation
from .config import Config
from .cpu import compute_cpus
from .execution import (
    APPLICATION_LOGGER_NAME,
    CALCULATION_KINDS,
    MAX_PREVIEWS,
    MAX_SLURM_MEMORY_GIB,
    MAX_SLURM_PROCESSES,
    SRUN_COMMAND,
    ExecutionMode,
    TaskStatus,
)
from .execution_models import CandidateRecord, Resources, SchedulerOverrides, TaskRecord
from .inputs import discover, dry_input, validate_references
from .persistence import Store, now
from .preparation import digest, pseudo_names
from .processes import execute, tail
from .runtimes import GIB, Runtimes, memory_ceiling
from .scheduler import Scheduler
from .scheduler import settings as scheduler_settings


class Manager:
    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(APPLICATION_LOGGER_NAME)

    def __init__(self, config: Config, candidates: dict[str, CandidateRecord]) -> None:
        self.config = config
        self.scheduler_enabled = config.scheduler_enabled
        config.artifacts.mkdir(parents=True, exist_ok=True)
        self.store = Store(config.artifacts / "tasks.sqlite")
        self.candidates = candidates
        self.chem_env = {
            **os.environ,
            "PATH": str(Path(config.prepare_python).absolute().parent)
            + os.pathsep
            + os.environ["PATH"],
        }
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.guard = threading.RLock()
        self.running = {}
        self.previews = {}
        self.runtimes = Runtimes(config)
        self.batches = Batches(self)
        self.scheduler = Scheduler(self)
        self.settings = self.store.settings(
            {
                "paused": False,
                "concurrency": 1,
                "memory_bytes": min(8 * GIB, memory_ceiling() * 9 // 10),
                "cpus": min(4, len(compute_cpus())),
            }
        )
        self.settings["memory_bytes"] = min(
            self.settings["memory_bytes"], memory_ceiling() * 9 // 10
        )
        available_cpus = len(compute_cpus())
        self.settings["cpus"] = min(self.settings["cpus"], available_cpus)
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.scheduler_thread = threading.Thread(
            target=self.scheduler.work, daemon=True
        )

    def start(self) -> None:
        self.thread.start()
        if self.scheduler_enabled:
            self.scheduler_thread.start()

    def close(self) -> None:
        with self.guard:
            self.stop.set()
            for _, cancel, _ in self.running.values():
                cancel.set()
            self.wake.set()
        if self.thread.is_alive():
            self.thread.join()
        if self.scheduler_thread.is_alive():
            self.scheduler_thread.join()
        for task in self.store.tasks():
            if task["status"] == "running" and task.get("target") != "slurm":
                self.store.update(
                    task["id"],
                    status="interrupted",
                    ended=now(),
                    error="Server stopped; retry explicitly",
                )

    def executable(self, name: str) -> str:
        path = shutil.which(name)
        if not path:
            raise ValueError(f"Executable unavailable: {name}")
        return path

    def preflight(self, candidate: CandidateRecord) -> None:
        probe = subprocess.run(
            [
                self.config.prepare_python,
                "-c",
                "import shutil; import rdkit, pymatgen.core, ase, openbabel; assert shutil.which('obabel'); assert shutil.which('cif2cell')",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            env=self.chem_env,
        )
        if probe.returncode:
            raise ValueError(
                f"Preparation interpreter {self.config.prepare_python} requires RDKit, pymatgen, ASE, Open Babel and cif2cell. "
                "Start with uv run --no-default-groups --group web --group preparation python -m dashboard, or set PFAS_CHEM_PYTHON to a chemistry interpreter. Details: "
                + probe.stderr[-2000:]
            )
        # RDKit runs in the configured interpreter, keeping the web environment minimal.
        result = subprocess.run(
            [
                self.config.prepare_python,
                "-c",
                "from rdkit import Chem; import sys; m=Chem.MolFromSmiles(sys.argv[1]); assert m is not None; print(' '.join(sorted({a.GetSymbol() for a in m.GetAtoms()} | {'H','C','O','F'})))",
                candidate["smiles"],
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            env=self.chem_env,
        )
        if result.returncode:
            raise ValueError("Invalid representative SMILES")
        missing = [
            s + ".UPF"
            for s in result.stdout.split()
            if not (self.config.pseudos / (s + ".UPF")).is_file()
        ]
        if missing:
            raise ValueError("Missing pseudopotentials: " + ", ".join(missing))

    def input(
        self, candidate: str, system: str, tasks: list[TaskRecord] | None = None
    ) -> Path:
        owner = "tfa" if system == "tfa" else candidate
        for task in self.store.prepared(candidate, system) if tasks is None else tasks:
            if (
                task["kind"] == "prepare"
                and task["status"] == "succeeded"
                and (task["candidate"] == owner or system == "tfa")
            ):
                path = self.config.artifacts / task["id"] / (system + ".in")
                if path.exists():
                    return path
        raise ValueError("Prepare inputs first")

    def versions(self) -> list[dict[str, str]]:
        return [
            {
                "input_id": identifier,
                "candidate": owner,
                "system": system,
                "label": path.name,
            }
            for identifier, (owner, system, path) in discover(
                self.config.qe_inputs, self.candidates
            ).items()
        ]

    def preview(
        self,
        candidate: str,
        system: str,
        processes: int,
        target: str,
        runtime: str = "native",
        input_id: str | None = None,
        kind: str = "qe",
        issue_token: bool = True,
        _tasks: list[TaskRecord] | None = None,
        _pseudo_hashes: dict | None = None,
        scheduler: SchedulerOverrides | None = None,
        memory_gib: float = 4,
        timeout: float | None = None,
    ) -> dict:
        if runtime not in ("native", "apptainer"):
            raise ValueError("Unknown runtime")
        self.validate_target(target, kind)
        if type(processes) is not int or processes < 1:
            raise ValueError("Process count must be a positive integer")
        if system not in ("tfa", "candidate", "complex"):
            raise ValueError("Unknown system")
        if candidate not in self.candidates or (candidate == "tfa" and system != "tfa"):
            raise ValueError("Unknown candidate/system")
        if kind not in ("qe", "estimate_ram"):
            raise ValueError("Unknown preview task")
        if input_id is None:
            path = self.input(candidate, system, tasks=_tasks)
            label = "Prepared default"
        else:
            entry = discover(self.config.qe_inputs, self.candidates).get(input_id)
            if not entry or entry[:2] != (
                "tfa" if system == "tfa" else candidate,
                system,
            ):
                raise ValueError("Invalid input identifier for candidate/system")
            path = entry[2]
            label = path.name
        input_bytes = path.read_bytes()
        text = input_bytes.decode()
        validate_references(text)
        source_hash = hashlib.sha256(input_bytes).hexdigest()
        source_text = text
        if kind == "estimate_ram":
            text = dry_input(text)
        names = pseudo_names(text)
        missing = [n for n in names if not (self.config.pseudos / n).is_file()]
        if missing:
            raise ValueError("Missing pseudopotentials: " + ", ".join(missing))
        hashes = {}
        for name in names:
            path = self.config.pseudos / name
            if _pseudo_hashes is None:
                hashes[name] = digest(path)
            else:
                stat = path.stat()
                key = (path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                if key not in _pseudo_hashes:
                    _pseudo_hashes[key] = digest(path)
                hashes[name] = _pseudo_hashes[key]
        pw = "pw.x" if runtime == "apptainer" else self.executable(self.config.pw)
        if runtime == "apptainer" and (
            target == ExecutionMode.SLURM or self.config.allocation
        ):
            from .slurm import mpi_command

            command = [*mpi_command(processes), pw, "-in", "input.in"]
        elif target == ExecutionMode.SLURM and processes > 1:
            # Host MPI ranks belong to a Slurm step. The container branch above
            # uses its own MPI and must never receive this second launcher.
            command = [
                self.executable(SRUN_COMMAND),
                f"--ntasks={processes}",
                "--cpus-per-task=1",
                "--exact",
                "--cpu-bind=cores",
                "--",
                pw,
                "-in",
                "input.in",
            ]
        elif processes == 1:
            command = [pw, "-in", "input.in"]
        else:
            mpi = (
                "mpirun" if runtime == "apptainer" else self.executable(self.config.mpi)
            )
            command = [mpi, "-np", str(processes), pw, "-in", "input.in"]
        from .geometry import geometry

        try:
            initial_geometry = geometry(source_text)
        except (ValueError, IndexError) as error:
            initial_geometry = {"error": str(error)}
        result = {
            "input_id": input_id,
            "version": label,
            "source_hash": source_hash,
            "executed_input_hash": hashlib.sha256(text.encode()).hexdigest(),
            "geometry": initial_geometry,
            "image_hash": self.runtimes.image_identity()
            if runtime == "apptainer"
            else None,
            "command": command,
            "input": text,
            "input_hash": hashlib.sha256(text.encode()).hexdigest(),
            "pseudopotentials": hashes,
        }

        if target == "slurm":
            timeout = (
                timeout
                if timeout is not None
                else 120
                if kind == "estimate_ram"
                else None
            )
            opts = scheduler_settings(scheduler, kind)
            resource = self.resources(runtime, processes, memory_gib, timeout, target)
            if opts["memory_mode"] == "estimate":
                from .slurm import estimate_memory

                matching = [
                    t
                    for t in self.store.tasks()
                    if t.get("runtime", "native") == runtime
                    and t.get("image_hash") == result["image_hash"]
                ]
                sizing = estimate_memory(
                    matching,
                    {
                        "candidate": candidate,
                        "system": system,
                        "input_hash": source_hash,
                        "pseudopotentials": hashes,
                    },
                    processes,
                    opts["headroom_percent"],
                )
                resource["memory_bytes"] = sizing["memory_mib"] * 1024**2
                result["memory_estimate"] = sizing["memory_estimate"]
            result.update(scheduler_settings=opts, resources=resource, target=target)
        if issue_token:
            token = uuid.uuid4().hex
            with self.guard:
                if len(self.previews) >= MAX_PREVIEWS:
                    self.previews.pop(next(iter(self.previews)))
                self.previews[token] = (
                    (candidate, system, processes, target, runtime, input_id, kind),
                    result.copy(),
                )
            result["preview_id"] = token
        return result

    def queue(
        self,
        kind: str,
        candidate: str,
        system: str = "candidate",
        processes: int = 1,
        target: str = ExecutionMode.LOCAL,
        timeout: float | None = None,
        expected_hash: str | None = None,
        runtime: str = "native",
        memory_gib: float = 4,
        batch_id: str | None = None,
        position: int | None = None,
        checked_preflight: bool = False,
        expected_pseudos: dict[str, str] | None = None,
        input_id: str | None = None,
        preview_id: str | None = None,
        scheduler: SchedulerOverrides | None = None,
    ) -> dict:
        self.validate_target(target, kind)
        if candidate not in self.candidates:
            raise ValueError("Unknown candidate")
        if kind not in ("diagram", "prepare", "qe", "estimate_ram"):
            raise ValueError("Unknown task")
        if timeout is not None and (
            not isinstance(timeout, (float, int)) or timeout <= 0
        ):
            raise ValueError("Timeout must be positive")
        if kind not in ("qe", "estimate_ram"):
            processes = 1
            system = "tfa" if candidate == "tfa" else "candidate"
        timeout = (
            timeout
            if timeout is not None
            else self.config.prepare_timeout
            if kind == "prepare"
            else 60
            if kind == "diagram"
            else 120
            if kind == "estimate_ram"
            else None
        )
        resources = self.resources(
            runtime,
            processes if kind in ("qe", "estimate_ram") else 1,
            memory_gib,
            timeout,
            target,
        )
        image_hash = (
            (
                self.runtimes.image_identity()
                if target == "slurm"
                else self.runtimes.verify()
            )
            if runtime == "apptainer"
            else None
        )
        with self.guard:
            existing = self.store.active(kind, candidate, system)
            if (
                existing
                and input_id is None
                and kind != "estimate_ram"
                and preview_id is None
            ):
                return existing
            preview = None
            if kind == "prepare" and not checked_preflight:
                try:
                    if runtime == "native":
                        self.preflight(self.candidates[candidate])
                    else:
                        report = self.runtimes.chemistry_probe(
                            runtime,
                            "import shutil; import rdkit,pymatgen.core,ase,openbabel; assert shutil.which('obabel'); assert shutil.which('cif2cell')",
                        )
                        if report.returncode:
                            raise ValueError(
                                "Preparation tools unavailable: "
                                + report.stderr[-1000:]
                            )
                except (ValueError, OSError, subprocess.SubprocessError) as error:
                    self.logger.warning(
                        "Preparation rejected for cluster %s: %s", candidate, error
                    )
                    raise
            if kind in ("qe", "estimate_ram"):
                preview = self.preview(
                    candidate,
                    system,
                    processes,
                    target,
                    runtime,
                    input_id,
                    kind,
                    issue_token=False,
                    scheduler=scheduler,
                    memory_gib=memory_gib,
                    timeout=timeout if target == "slurm" else None,
                )
                if (
                    preview_id is not None
                    or target == "slurm"
                    or input_id is not None
                    or kind == "estimate_ram"
                ):
                    reviewed = self.previews.get(preview_id)
                    if not reviewed or reviewed != (
                        (candidate, system, processes, target, runtime, input_id, kind),
                        preview,
                    ):
                        raise ValueError(
                            "Input, pseudopotentials, or execution settings changed; preview again"
                        )
                if (
                    preview_id is None
                    and input_id is None
                    and kind == "qe"
                    and expected_pseudos is None
                ):
                    specification = (
                        candidate,
                        system,
                        processes,
                        target,
                        runtime,
                        input_id,
                        kind,
                    )
                    if not any(
                        record == (specification, preview)
                        for record in self.previews.values()
                    ):
                        raise ValueError(
                            "Input or pseudopotentials changed; preview again"
                        )
                if (
                    expected_pseudos is not None
                    and expected_pseudos != preview["pseudopotentials"]
                ):
                    raise ValueError("Pseudopotentials changed; preview again")
                if expected_hash != preview["input_hash"]:
                    raise ValueError(
                        "Input changed or was not previewed; preview again"
                    )
            id = uuid.uuid4().hex
            directory = self.config.artifacts / id
            directory.mkdir()
            task = {
                "id": id,
                "kind": kind,
                "candidate": candidate,
                "system": system,
                "processes": processes,
                "status": "queued",
                "created": now(),
                "started": None,
                "ended": None,
                "timeout": timeout,
                "runtime": runtime,
                "target": target,
                "resources": resources,
                "image_hash": image_hash,
                "batch_id": batch_id,
                "position": position,
                "source": {
                    "cid": self.candidates[candidate]["cid"],
                    "smiles": self.candidates[candidate]["smiles"],
                },
                "artifacts": {},
                "error": "",
            }
            if preview:
                (directory / "input.in").write_text(preview.pop("input"))
                (directory / "Outputs").mkdir()
                (directory / "Pseudopotentials").mkdir()
                # Link immutable copies, so later external changes cannot alter a queued run.
                (directory / "pseudo-snapshots").mkdir()
                for name, hash in preview["pseudopotentials"].items():
                    source = self.config.pseudos / name
                    dest = directory / "pseudo-snapshots" / name
                    shutil.copyfile(source, dest)
                    if digest(dest) != hash:
                        raise ValueError("Pseudopotential changed while queuing")
                    (directory / "Pseudopotentials" / name).symlink_to(dest)
                preview.pop("geometry", None)
                task.update(preview)
                if preview_id:
                    self.previews.pop(preview_id, None)
            for name in ("stdout.log", "stderr.log"):
                path = directory / name
                path.touch()
                task["artifacts"][name] = self.store.register(id + "-" + name, path)
            if preview:
                task["artifacts"]["input.in"] = self.store.register(
                    id + "-input.in", directory / "input.in"
                )
            self.store.put(task)
            self.logger.info(
                "Task %s queued: %s for cluster %s (%s)", id, kind, candidate, system
            )
            self.wake.set()
            return task

    def validate_target(self, target: str, kind: str) -> None:
        if target not in ("local", "slurm"):
            raise NotImplementedError(
                "Only local and single-node Slurm execution are supported"
            )
        if target == "slurm":
            if not self.scheduler_enabled:
                raise ValueError(
                    "Slurm unavailable; use PFAS_EXECUTION=slurm with scheduler commands installed"
                )
            if kind not in ("qe", "estimate_ram"):
                raise ValueError(
                    "Diagrams and preparation use the dashboard allocation"
                )

    def cancel_task(self, id: str) -> None:
        with self.guard:
            task = self.store.get(id)
            if task.get("stop_requested"):
                return
            if task.get("scheduler") and task["status"] in ("queued", "running"):
                self.scheduler.cancel(task)
            elif task["status"] == "queued":
                self.store.update(id, status="canceled", ended=now())
                self.logger.info("Task %s canceled before starting", id)
            elif task["status"] == "running" and id in self.running:
                self.logger.info("Task %s stop requested", id)
                self.running[id][1].set()
                self.store.update(id, stop_requested=True)

    def clear_queue(self) -> dict[str, int]:
        # Hold admission while canceling the snapshot so waiting jobs cannot start.
        with self.guard:
            tasks = self.store.queued()
            for task in tasks:
                self.cancel_task(task["id"])
            return {"canceled_count": len(tasks)}

    def resources(
        self,
        runtime: str,
        cpus: int,
        memory_gib: float,
        timeout: float | None,
        target: str = ExecutionMode.LOCAL,
    ) -> Resources:
        if runtime not in ("native", "apptainer"):
            raise ValueError("Unknown runtime")
        validate_positive(cpus, "CPU count", integer=True)
        validate_positive(memory_gib, "Memory limit")
        if timeout is not None:
            validate_positive(timeout, "Timeout")
        memory_bytes = int(memory_gib * GIB)
        if target == "slurm" and (
            cpus > MAX_SLURM_PROCESSES or memory_gib > MAX_SLURM_MEMORY_GIB
        ):
            raise ValueError("Slurm jobs support at most 256 processes and 65536 GiB")
        if target != "slurm" and cpus > self.settings["cpus"]:
            raise ValueError("Job CPU count exceeds queue budget")
        if (
            target != "slurm"
            and runtime == "apptainer"
            and memory_bytes > self.settings["memory_bytes"]
        ):
            raise ValueError("Job memory limit exceeds queue budget")
        return {
            "cpus": int(cpus),
            "memory_bytes": memory_bytes,
            "timeout": timeout,
            "hard_memory_limit": runtime == "apptainer"
            and target != "slurm"
            and not self.config.allocation,
        }

    def configure(self, values: dict) -> dict:
        with self.guard:
            settings = {**self.settings, **values}
            validate_positive(settings["concurrency"], "Concurrency", integer=True)
            validate_positive(settings["cpus"], "CPU budget", integer=True)
            validate_positive(settings["memory_bytes"], "Memory budget", integer=True)
            available_cpus = len(compute_cpus())
            if settings["cpus"] > available_cpus:
                raise ValueError(
                    f"CPU budget cannot exceed {available_cpus}; "
                    "one CPU is reserved for web requests on multicore hosts"
                )
            memory_limit = memory_ceiling() * 9 // 10
            if settings["memory_bytes"] > memory_limit:
                raise ValueError(
                    f"Memory budget cannot exceed {memory_limit / GIB:.2f} GiB "
                    f"({memory_limit} bytes; 90% of detected host memory)"
                )
            running_tasks = [value[2] for value in self.running.values()]
            if (
                sum(t["resources"]["cpus"] for t in running_tasks) > settings["cpus"]
                or sum(
                    t["resources"]["memory_bytes"]
                    for t in running_tasks
                    if t.get("runtime") == "apptainer"
                )
                > settings["memory_bytes"]
            ):
                raise ValueError(
                    "New budget is below current running reservations; stop jobs or wait for them to finish"
                )
            for task in self.store.queued():
                if task.get("target") == "slurm":
                    continue
                resource = task.get(
                    "resources", {"cpus": task["processes"], "memory_bytes": 4 * GIB}
                )
                if resource["cpus"] > settings["cpus"] or (
                    task.get("runtime") == "apptainer"
                    and resource["memory_bytes"] > settings["memory_bytes"]
                ):
                    raise ValueError(
                        "New budget cannot accommodate an existing queued job"
                    )
            self.settings = self.store.set_settings(settings)
            self.wake.set()
            return self.settings

    def queue_data(self) -> dict:
        tasks = self.store.tasks()
        # No log tails, geometries, or raw input text in queue polling.
        keys = (
            "id",
            "kind",
            "candidate",
            "system",
            "processes",
            "status",
            "stop_requested",
            "created",
            "started",
            "ended",
            "runtime",
            "resources",
            "target",
            "scheduler",
            "scheduler_settings",
            "batch_id",
            "position",
            "error",
            "exit_code",
            "usage",
            "evidence",
            "version",
            "source_hash",
            "executed_input_hash",
            "ram_estimate",
            "artifacts",
        )
        return {
            "settings": self.settings,
            "cpu_capacity": len(compute_cpus()),
            "runtimes": self.runtimes.availability(),
            "batches": self.store.batches(),
            "tasks": [{k: t[k] for k in keys if k in t} for t in tasks],
            "running_count": len(self.running)
            + sum(
                t.get("target") == ExecutionMode.SLURM
                and t["status"] == TaskStatus.RUNNING
                for t in tasks
            ),
        }

    def worker(self) -> None:
        while not self.stop.is_set():
            with self.guard:
                for id, (thread, _, _) in list(self.running.items()):
                    if not thread.is_alive():
                        thread.join()
                        del self.running[id]
                if self.stop.is_set():
                    break
                if not self.settings["paused"]:
                    for task in self.store.queued():
                        if task.get("target") == "slurm":
                            continue
                        # Strict FIFO admission; starts follow the reviewed view order.
                        runtime = task.get("runtime", "native")
                        resources = task.get(
                            "resources",
                            {"cpus": task["processes"], "memory_bytes": 4 * GIB},
                        )
                        if resources["cpus"] > len(compute_cpus()):
                            self.store.update(
                                task["id"],
                                status="failed",
                                ended=now(),
                                error="Job CPU count exceeds compute capacity after "
                                "reserving a CPU for web requests; retry with fewer CPUs",
                            )
                            continue
                        running_tasks = [value[2] for value in self.running.values()]
                        if runtime == "native" and running_tasks:
                            break
                        if any(
                            t.get("runtime", "native") == "native"
                            for t in running_tasks
                        ):
                            break
                        if len(running_tasks) >= (
                            1
                            if self.config.allocation
                            else self.settings["concurrency"]
                        ):
                            break
                        if (
                            sum(t["resources"]["cpus"] for t in running_tasks)
                            + resources["cpus"]
                            > self.settings["cpus"]
                        ):
                            break
                        if (
                            runtime == "apptainer"
                            and sum(
                                t["resources"]["memory_bytes"] for t in running_tasks
                            )
                            + resources["memory_bytes"]
                            > self.settings["memory_bytes"]
                        ):
                            break
                        cancel = threading.Event()
                        thread = threading.Thread(
                            target=self.execute_attempt,
                            args=(task, cancel),
                            daemon=True,
                        )
                        self.store.update(task["id"], status="running", started=now())
                        self.running[task["id"]] = (thread, cancel, task)
                        thread.start()
                        if runtime == "native":
                            break
            self.wake.wait(0.1)
            self.wake.clear()
        for thread, cancel, _ in list(self.running.values()):
            cancel.set()
            thread.join()
        self.running.clear()

    def execute_attempt(self, task: TaskRecord, cancel: threading.Event) -> None:
        try:
            self.run(task, cancel)
        except Exception as error:
            # Catch artifact registration/finalization errors too; release the slot.
            self.logger.exception("Task %s failed during finalization", task["id"])
            self.store.update(
                task["id"], status="failed", error=str(error), ended=now()
            )
        finally:
            self.wake.set()

    def run(self, task: TaskRecord, cancel: threading.Event | None = None) -> None:
        cancel = cancel or threading.Event()
        id = task["id"]
        directory = self.config.artifacts / id
        self.logger.info(
            "Task %s started: %s for cluster %s (%s)",
            id,
            task["kind"],
            task["candidate"],
            task["system"],
        )
        self.logger.info(
            "Task %s logs: stdout=%s stderr=%s",
            id,
            directory / "stdout.log",
            directory / "stderr.log",
        )
        try:
            if task["kind"] in CALCULATION_KINDS:
                result = run_calculation(
                    task,
                    self.config,
                    self.runtimes,
                    cancel,
                    execute,
                    on_usage=lambda usage: self.store.update(id, usage=usage),
                )
                if self.stop.is_set():
                    result["status"] = TaskStatus.INTERRUPTED
                self.store.update(id, **result)
                return
            candidate = self.candidates[task["candidate"]]
            timeout = task["timeout"]
            if task["kind"] == "diagram":
                command = [
                    self.config.python,
                    str(Path(__file__).with_name("diagram.py")),
                    candidate["smiles"],
                    str(directory / "diagram.png"),
                ]
                timeout = task["timeout"] or 60
            elif task["kind"] == "prepare":
                request = {
                    "candidate": task["candidate"],
                    "smiles": candidate["smiles"],
                    "cid": candidate["cid"],
                    "pseudos": str(self.config.pseudos),
                }
                for previous in self.store.tasks():
                    if (
                        previous["kind"] == "prepare"
                        and previous["status"] == "succeeded"
                    ):
                        shared = self.config.artifacts / previous["id"]
                        if (shared / "tfa.mol").exists() and (
                            shared / "tfa.in"
                        ).exists():
                            try:
                                manifest = json.loads(
                                    (shared / "manifest.json").read_text()
                                )
                                if digest(shared / "tfa.in") == manifest["inputs"][
                                    "tfa.in"
                                ] and all(
                                    digest(self.config.pseudos / n) == h
                                    for n, h in manifest["pseudopotentials"].items()
                                ):
                                    request["shared_tfa"] = str(shared)
                                    break
                            except (OSError, KeyError, ValueError):
                                self.logger.warning(
                                    "Ignoring unavailable shared TFA from attempt %s",
                                    previous["id"],
                                )
                (directory / "request.json").write_text(json.dumps(request))
                command = [
                    self.config.prepare_python,
                    str(Path(__file__).with_name("preparation.py")),
                    str(directory / "request.json"),
                    str(directory),
                ]
                timeout = task["timeout"] or self.config.prepare_timeout
            else:
                command = task["command"]
            runtime = task.get("runtime", "native")
            if runtime == "apptainer":
                if task["kind"] not in ("qe", "estimate_ram"):
                    command[0] = "/opt/venv/bin/python"
                command = self.runtimes.wrap(
                    command,
                    directory,
                    task["resources"],
                    task["image_hash"],
                    scheduler=task.get("target") == "slurm",
                )
            entrypoint = directory / "entrypoint.log"
            entrypoint.write_text(
                f"[{now()}] Task: {id}\nKind: {task['kind']}\nRuntime: {runtime}\n"
                f"Working directory: {directory}\n"
                f"Entrypoint: {shlex.join(command)}\n"
                f"Processes: {task['processes']}\nTimeout: {timeout}\n"
                f"Image hash: {task.get('image_hash')}\n"
                f"Source hash: {task.get('source_hash')}\n"
                f"Input hash: {task.get('input_hash')}\n"
                f"Resources: {json.dumps(task.get('resources', {}))}\n"
            )
            artifacts = {
                **task["artifacts"],
                "entrypoint.log": self.store.register(
                    id + "-entrypoint.log", entrypoint
                ),
            }
            self.store.update(id, command=command, artifacts=artifacts)

            def usage(value: dict) -> None:
                self.store.update(id, usage=value)

            self.logger.info(
                "Task %s entrypoint: %s (cwd=%s)", id, shlex.join(command), directory
            )
            code, status, error = execute(
                command,
                directory,
                cancel,
                timeout,
                env=self.chem_env
                if task["kind"] not in ("qe", "estimate_ram")
                else None,
                on_usage=usage,
                container=runtime == "apptainer"
                and not self.config.allocation
                and task.get("target") != "slurm",
            )
            if status == "failed" and not error and task["kind"] != "estimate_ram":
                detail = tail(directory / "stderr.log", limit=2000).strip()
                error = f"Process exited with code {code}" + (
                    f": {detail}" if detail else ""
                )
            values = {
                "status": "interrupted" if self.stop.is_set() else status,
                "exit_code": code,
                "error": error,
                "ended": now(),
            }
            self.store.update(id, **values)
        except Exception as error:
            self.logger.exception("Task %s could not execute", id)
            self.store.update(id, status="failed", error=str(error), ended=now())
        finally:
            artifacts = {}
            for name in (
                "entrypoint.log",
                "stdout.log",
                "stderr.log",
                "diagram.png",
                "candidate.in",
                "complex.in",
                "tfa.in",
                "input.in",
                "manifest.json",
            ):
                path = directory / name
                if path.is_file():
                    artifacts[name] = self.store.register(id + "-" + name, path)
            result = self.store.update(id, artifacts=artifacts)
            self.logger.info(
                "Task %s finished: %s (exit=%s)",
                id,
                result["status"],
                result.get("exit_code"),
            )
            if result.get("error"):
                self.logger.error("Task %s: %s", id, result["error"])

    def data(self) -> dict:
        from .geometry import prepared_geometries

        tasks = self.store.tasks()
        for task in tasks:

            def registered_input(name: str, task_id: str = task["id"]) -> Path | None:
                path = self.store.artifact(task_id + "-" + name)
                return Path(path) if path else None

            task["geometries"] = prepared_geometries(task, registered_input)
            if task["started"]:
                from datetime import datetime

                task["elapsed_seconds"] = (
                    datetime.fromisoformat(task["ended"] or now())
                    - datetime.fromisoformat(task["started"])
                ).total_seconds()
            directory = self.config.artifacts / task["id"]
            task["stdout_tail"] = tail(directory / "stdout.log")
            task["stderr_tail"] = tail(directory / "stderr.log")
        return {
            "mode": "live",
            "queue": {
                "settings": self.settings,
                "runtimes": self.runtimes.availability(),
            },
            "candidates": [c for c in self.candidates.values() if c["id"] != "tfa"],
            "reference": self.candidates["tfa"],
            "tasks": tasks,
            "input_versions": self.versions(),
        }
