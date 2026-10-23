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

from dashboard.core import config as configuration
from dashboard.core import execution, models
from dashboard.core.chemistry.inputs import discover, dry_input, validate_references
from dashboard.core.chemistry.preparation import digest, pseudo_names
from dashboard.core.chemistry.results import read_calculation_result, write_entrypoint
from dashboard.core.config import Config
from dashboard.core.jobs.batches import Batches, validate_positive
from dashboard.core.jobs.persistence import Store, now
from dashboard.core.jobs.processes import execute, tail
from dashboard.core.runtimes.cpu import compute_cpus
from dashboard.core.runtimes.local import Runtimes, memory_ceiling
from dashboard.core.slurm.scheduler import Scheduler
from dashboard.core.slurm.scheduler import settings as scheduler_settings

QUEUE_FIELDS = (
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


class Manager:
    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(execution.APPLICATION_LOGGER_NAME)

    def __init__(
        self, config: Config, candidates: dict[str, models.CandidateRecord]
    ) -> None:
        self.config, self.candidates = config, candidates
        self.scheduler_enabled = config.scheduler_enabled
        config.artifacts.mkdir(parents=True, exist_ok=True)
        self.store = Store(config.artifacts / "tasks.sqlite")
        self.chem_env = self.chemistry_environment()
        # Initialize shared synchronization before adapters and their threads.
        self.initialize_state()
        self.runtimes = Runtimes(config)
        self.batches, self.scheduler = Batches(self), Scheduler(self)
        self.settings = self.restore_settings()
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.scheduler_thread = threading.Thread(
            target=self.scheduler.work, daemon=True
        )

    def chemistry_environment(self) -> dict[str, str]:
        return {
            **os.environ,
            "PYTHONPATH": str(configuration.ROOT),
            "PATH": str(Path(self.config.prepare_python).absolute().parent)
            + os.pathsep
            + os.environ["PATH"],
        }

    def initialize_state(self) -> None:
        self.wake, self.stop = threading.Event(), threading.Event()
        self.guard = threading.RLock()
        self.running = {}
        self.previews = {}

    def restore_settings(self) -> dict:
        ceiling = memory_ceiling() * 9 // 10
        cpus = len(compute_cpus())
        defaults = {
            "paused": False,
            "concurrency": 1,
            "memory_bytes": min(8 * execution.GIB, ceiling),
            "cpus": min(4, cpus),
        }
        settings = self.store.settings(defaults)
        # Saved budgets cannot exceed this process's current allocation.
        settings["memory_bytes"] = min(settings["memory_bytes"], ceiling)
        settings["cpus"] = min(settings["cpus"], cpus)
        return settings

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

    def chemistry_probe(
        self, code: str, input_text: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.config.prepare_python, "-c", code],
            input=input_text,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            env=self.chem_env,
        )

    def preflight_tools(self) -> None:
        probe = self.chemistry_probe(execution.PREPARATION_TOOL_PROBE)
        if probe.returncode:
            raise ValueError(
                f"Preparation interpreter {self.config.prepare_python} requires RDKit, pymatgen, ASE, Open Babel and cif2cell. "
                "Start with uv run --no-default-groups --group web --group preparation python -m dashboard, or set PFAS_CHEM_PYTHON to a chemistry interpreter. Details: "
                + probe.stderr[-2000:]
            )

    def required_elements(self, candidate: models.CandidateRecord) -> list[str]:
        code = "from rdkit import Chem; import sys,json; m=Chem.MolFromSmiles(json.load(sys.stdin)); assert m is not None; print(' '.join(sorted({a.GetSymbol() for a in m.GetAtoms()} | {'H','C','O','F'})))"
        result = self.chemistry_probe(code, json.dumps(candidate["smiles"]))
        if result.returncode:
            raise ValueError("Invalid representative SMILES")
        return result.stdout.split()

    def preflight(self, candidate: models.CandidateRecord) -> None:
        self.preflight_tools()
        # Resolve elements in the chemistry interpreter, not the web process.
        missing = [
            element + ".UPF"
            for element in self.required_elements(candidate)
            if not (self.config.pseudos / (element + ".UPF")).is_file()
        ]
        if missing:
            raise ValueError("Missing pseudopotentials: " + ", ".join(missing))

    def input(
        self, candidate: str, system: str, tasks: list[models.TaskRecord] | None = None
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
        _tasks: list[models.TaskRecord] | None = None,
        _pseudo_hashes: dict | None = None,
        scheduler: models.SchedulerSettings | None = None,
        memory_gib: float = 4,
        timeout: float | None = None,
    ) -> dict:
        """Retain preview options while delegating parsing, commands, and scheduler sizing."""
        self.validate_preview(candidate, system, processes, target, runtime, kind)
        result = self.input_preview(
            candidate, system, input_id, kind, _tasks, _pseudo_hashes
        )
        result["command"] = self.qe_command(runtime, target, processes)
        result["image_hash"] = (
            self.runtimes.image_identity() if runtime == "apptainer" else None
        )
        # Scheduler resources are reviewed separately from the dashboard budget.
        if target == "slurm":
            self.slurm_preview(
                result,
                candidate,
                system,
                processes,
                runtime,
                kind,
                memory_gib,
                timeout,
                scheduler,
            )
        if issue_token:
            self.issue_preview(
                (candidate, system, processes, target, runtime, input_id, kind), result
            )
        return result

    def validate_preview(
        self,
        candidate: str,
        system: str,
        processes: int,
        target: str,
        runtime: str,
        kind: str,
    ) -> None:
        if runtime not in execution.Runtime:
            raise ValueError("Unknown runtime")
        self.validate_target(target, kind)
        if type(processes) is not int or processes < 1:
            raise ValueError("Process count must be a positive integer")
        if system not in ("tfa", "candidate", "complex"):
            raise ValueError("Unknown system")
        if candidate not in self.candidates or (candidate == "tfa" and system != "tfa"):
            raise ValueError("Unknown candidate/system")
        if kind not in execution.CALCULATION_KINDS:
            raise ValueError("Unknown preview task")

    def preview_source(
        self,
        candidate: str,
        system: str,
        input_id: str | None,
        tasks: list[models.TaskRecord] | None,
    ) -> tuple[Path, str]:
        if input_id is None:
            return self.input(candidate, system, tasks=tasks), "Prepared default"
        entry = discover(self.config.qe_inputs, self.candidates).get(input_id)
        owner = "tfa" if system == "tfa" else candidate
        if not entry or entry[:2] != (owner, system):
            raise ValueError("Invalid input identifier for candidate/system")
        return entry[2], entry[2].name

    def pseudo_hash(self, path: Path, cache: dict | None) -> str:
        if cache is None:
            return digest(path)
        stat = path.stat()
        key = (path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key not in cache:
            cache[key] = digest(path)
        return cache[key]

    def input_pseudopotentials(self, text: str, cache: dict | None) -> dict[str, str]:
        names = pseudo_names(text)
        missing = [name for name in names if not (self.config.pseudos / name).is_file()]
        if missing:
            raise ValueError("Missing pseudopotentials: " + ", ".join(missing))
        return {
            name: self.pseudo_hash(self.config.pseudos / name, cache) for name in names
        }

    def input_preview(
        self,
        candidate: str,
        system: str,
        input_id: str | None,
        kind: str,
        tasks: list[models.TaskRecord] | None,
        cache: dict | None,
    ) -> dict:
        from dashboard.core.chemistry.geometry import preview_geometry

        path, label = self.preview_source(candidate, system, input_id, tasks)
        source = path.read_bytes()
        text = source.decode()
        validate_references(text)
        executed = dry_input(text) if kind == "estimate_ram" else text
        executed_hash = hashlib.sha256(executed.encode()).hexdigest()
        # Keep the response's provenance fields together; all validation is above.
        return {
            "input_id": input_id,
            "version": label,
            "source_hash": hashlib.sha256(source).hexdigest(),
            "executed_input_hash": executed_hash,
            "geometry": preview_geometry(text),
            "input": executed,
            "input_hash": executed_hash,
            "pseudopotentials": self.input_pseudopotentials(executed, cache),
        }

    def native_slurm_command(self, pw: str, processes: int) -> list[str]:
        return [
            self.executable(execution.SRUN_COMMAND),
            f"--ntasks={processes}",
            "--cpus-per-task=1",
            "--exact",
            "--cpu-bind=cores",
            "--",
            pw,
            "-in",
            "input.in",
        ]

    def qe_command(self, runtime: str, target: str, processes: int) -> list[str]:
        pw = "pw.x" if runtime == "apptainer" else self.executable(self.config.pw)
        # Container MPI stays inside the image; native MPI uses a Slurm step.
        if runtime == "apptainer" and (target == "slurm" or self.config.allocation):
            from dashboard.core.slurm.export import mpi_command

            return [*mpi_command(processes), pw, "-in", "input.in"]
        if target == "slurm" and processes > 1:
            return self.native_slurm_command(pw, processes)
        if processes == 1:
            return [pw, "-in", "input.in"]
        mpi = "mpirun" if runtime == "apptainer" else self.executable(self.config.mpi)
        return [mpi, "-np", str(processes), pw, "-in", "input.in"]

    def estimate_preview_memory(
        self,
        result: dict,
        candidate: str,
        system: str,
        processes: int,
        runtime: str,
        headroom: float,
    ) -> dict:
        from dashboard.core.slurm.export import estimate_memory

        matching = [
            task
            for task in self.store.tasks()
            if task.get("runtime", "native") == runtime
            and task.get("image_hash") == result["image_hash"]
        ]
        entry = {
            "candidate": candidate,
            "system": system,
            "input_hash": result["source_hash"],
            "pseudopotentials": result["pseudopotentials"],
        }
        return estimate_memory(matching, entry, processes, headroom)

    def slurm_preview(
        self,
        result: dict,
        candidate: str,
        system: str,
        processes: int,
        runtime: str,
        kind: str,
        memory_gib: float,
        timeout: float | None,
        scheduler: models.SchedulerSettings | None,
    ) -> None:
        timeout = self.default_timeout(kind, timeout)
        options = scheduler_settings(scheduler, kind)
        resources = self.resources(runtime, processes, memory_gib, timeout, "slurm")
        if options["memory_mode"] == "estimate":
            sizing = self.estimate_preview_memory(
                result,
                candidate,
                system,
                processes,
                runtime,
                options["headroom_percent"],
            )
            resources["memory_bytes"] = sizing["memory_mib"] * execution.MIB
            result["memory_estimate"] = sizing["memory_estimate"]
        result.update(scheduler_settings=options, resources=resources, target="slurm")

    def issue_preview(self, specification: tuple, result: dict) -> None:
        token = uuid.uuid4().hex
        with self.guard:
            if len(self.previews) >= execution.MAX_PREVIEWS:
                self.previews.pop(next(iter(self.previews)))
            self.previews[token] = (specification, result.copy())
        result["preview_id"] = token

    def queue(
        self,
        kind: str,
        candidate: str,
        system: str = "candidate",
        processes: int = 1,
        target: str = execution.ExecutionMode.LOCAL,
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
        scheduler: models.SchedulerSettings | None = None,
    ) -> dict:
        """Keep the existing API options; delegate validation, snapshots, and persistence."""
        request = models.TaskRequest(
            kind=kind,
            candidate=candidate,
            system=system,
            processes=processes,
            target=target,
            timeout=timeout,
            expected_hash=expected_hash,
            runtime=runtime,
            memory_gib=memory_gib,
            batch_id=batch_id,
            position=position,
            checked_preflight=checked_preflight,
            expected_pseudos=expected_pseudos,
            input_id=input_id,
            preview_id=preview_id,
            scheduler=scheduler,
        )
        return self.queue_request(request)

    def default_timeout(self, kind: str, timeout: float | None) -> float | None:
        if timeout is not None:
            return timeout
        defaults = {
            "prepare": self.config.prepare_timeout,
            "diagram": execution.DEFAULT_DIAGRAM_TIMEOUT,
            "estimate_ram": execution.DEFAULT_ESTIMATION_TIMEOUT,
        }
        return defaults.get(kind)

    def normalize_request(self, request: models.TaskRequest) -> models.TaskRequest:
        self.validate_target(request["target"], request["kind"])
        if request["candidate"] not in self.candidates:
            raise ValueError("Unknown candidate")
        if request["kind"] not in execution.TaskKind:
            raise ValueError("Unknown task")
        timeout = request["timeout"]
        if timeout is not None and (
            not isinstance(timeout, (float, int)) or timeout <= 0
        ):
            raise ValueError("Timeout must be positive")
        request = request.copy()
        if request["kind"] not in execution.CALCULATION_KINDS:
            request["processes"] = 1
            request["system"] = "tfa" if request["candidate"] == "tfa" else "candidate"
        request["timeout"] = self.default_timeout(request["kind"], timeout)
        return request

    def runtime_identity(self, runtime: str, target: str) -> str | None:
        if runtime != execution.Runtime.APPTAINER:
            return None
        return (
            self.runtimes.image_identity()
            if target == execution.ExecutionMode.SLURM
            else self.runtimes.verify()
        )

    def active_request(self, request: models.TaskRequest) -> models.TaskRecord | None:
        if (
            request["input_id"] is not None
            or request["preview_id"] is not None
            or request["kind"] == "estimate_ram"
        ):
            return None
        return self.store.active(
            request["kind"], request["candidate"], request["system"]
        )

    def preparation_preflight(self, request: models.TaskRequest) -> None:
        if request["kind"] != "prepare" or request["checked_preflight"]:
            return
        try:
            if request["runtime"] == "native":
                self.preflight(self.candidates[request["candidate"]])
            else:
                report = self.runtimes.chemistry_probe(
                    request["runtime"], execution.PREPARATION_TOOL_PROBE
                )
                if report.returncode:
                    raise ValueError(
                        "Preparation tools unavailable: " + report.stderr[-1000:]
                    )
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            self.logger.warning(
                "Preparation rejected for cluster %s: %s", request["candidate"], error
            )
            raise

    def queue_request(self, request: models.TaskRequest) -> dict:
        request = self.normalize_request(request)
        resources = self.resources(
            request["runtime"],
            request["processes"],
            request["memory_gib"],
            request["timeout"],
            request["target"],
        )
        identity = self.runtime_identity(request["runtime"], request["target"])
        # Serialize admission so duplicate detection and preview acceptance agree.
        with self.guard:
            existing = self.active_request(request)
            if existing:
                return dict(existing)
            self.preparation_preflight(request)
            preview = self.queued_preview(request)
            task = self.new_task(request, resources, identity)
            self.accept_task(task, preview, request["preview_id"])
            return task

    def queued_preview(self, request: models.TaskRequest) -> dict | None:
        if request["kind"] not in execution.CALCULATION_KINDS:
            return None
        preview = self.preview(
            request["candidate"],
            request["system"],
            request["processes"],
            request["target"],
            request["runtime"],
            request["input_id"],
            request["kind"],
            issue_token=False,
            scheduler=request["scheduler"],
            memory_gib=request["memory_gib"],
            timeout=request["timeout"] if request["target"] == "slurm" else None,
        )
        self.validate_queued_preview(request, preview)
        return preview

    def request_specification(self, request: models.TaskRequest) -> tuple:
        return (
            request["candidate"],
            request["system"],
            request["processes"],
            request["target"],
            request["runtime"],
            request["input_id"],
            request["kind"],
        )

    def require_preview_token(self, request: models.TaskRequest, preview: dict) -> None:
        reviewed = self.previews.get(request["preview_id"])
        if not reviewed or reviewed != (self.request_specification(request), preview):
            raise ValueError(
                "Input, pseudopotentials, or execution settings changed; preview again"
            )

    def require_previous_review(
        self, request: models.TaskRequest, preview: dict
    ) -> None:
        expected = (self.request_specification(request), preview)
        if not any(record == expected for record in self.previews.values()):
            raise ValueError("Input or pseudopotentials changed; preview again")

    def validate_queued_preview(
        self, request: models.TaskRequest, preview: dict
    ) -> None:
        # Versioned inputs, estimates, and scheduler jobs require the exact token.
        if (
            request["preview_id"] is not None
            or request["input_id"] is not None
            or request["kind"] == "estimate_ram"
            or request["target"] == "slurm"
        ):
            self.require_preview_token(request, preview)
        elif request["kind"] == "qe" and request["expected_pseudos"] is None:
            self.require_previous_review(request, preview)
        # Legacy clients may send reviewed hashes instead of a token.
        if (
            request["expected_pseudos"] is not None
            and request["expected_pseudos"] != preview["pseudopotentials"]
        ):
            raise ValueError("Pseudopotentials changed; preview again")
        if request["expected_hash"] != preview["input_hash"]:
            raise ValueError("Input changed or was not previewed; preview again")

    def new_task(
        self,
        request: models.TaskRequest,
        resources: models.Resources,
        identity: str | None,
    ) -> dict:
        task: dict = {key: request[key] for key in execution.PERSISTED_REQUEST_FIELDS}
        candidate = self.candidates[request["candidate"]]
        # This is the persisted record's schema; no I/O occurs while assembling it.
        task.update(
            id=uuid.uuid4().hex,
            status="queued",
            created=now(),
            started=None,
            ended=None,
            resources=resources,
            image_hash=identity,
            source={"cid": candidate["cid"], "smiles": candidate["smiles"]},
            artifacts={},
            error="",
        )
        return task

    def copy_pseudopotentials(self, hashes: dict[str, str], directory: Path) -> None:
        (directory / "Pseudopotentials").mkdir()
        (directory / "pseudo-snapshots").mkdir()
        # Verify copies before linking them into the immutable attempt snapshot.
        for name, expected in hashes.items():
            destination = directory / "pseudo-snapshots" / name
            shutil.copyfile(self.config.pseudos / name, destination)
            if digest(destination) != expected:
                raise ValueError("Pseudopotential changed while queuing")
            (directory / "Pseudopotentials" / name).symlink_to(destination)

    def capture_preview(
        self, task: dict, preview: dict, directory: Path, preview_id: str | None
    ) -> None:
        (directory / "input.in").write_text(preview.pop("input"))
        (directory / "Outputs").mkdir()
        self.copy_pseudopotentials(preview["pseudopotentials"], directory)
        preview.pop("geometry", None)
        task.update(preview)
        if preview_id:
            self.previews.pop(preview_id, None)

    def accept_task(
        self, task: dict, preview: dict | None, preview_id: str | None
    ) -> None:
        directory = self.config.artifacts / task["id"]
        directory.mkdir()
        if preview:
            self.capture_preview(task, preview, directory, preview_id)
        # Register logs immediately so accepted jobs have inspectable artifacts.
        for name in (execution.STDOUT_LOG_FILE, execution.STDERR_LOG_FILE):
            path = directory / name
            path.touch()
            task["artifacts"][name] = self.store.register(task["id"] + "-" + name, path)
        if preview:
            task["artifacts"]["input.in"] = self.store.register(
                task["id"] + "-input.in", directory / "input.in"
            )
        self.store.put(task)
        self.logger.info(
            "Task %s queued: %s for cluster %s (%s)",
            task["id"],
            task["kind"],
            task["candidate"],
            task["system"],
        )
        self.wake.set()

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

    def validate_resource_budget(
        self, runtime: str, cpus: int, memory_gib: float, target: str
    ) -> None:
        if target == "slurm":
            if (
                cpus > execution.MAX_SLURM_PROCESSES
                or memory_gib > execution.MAX_SLURM_MEMORY_GIB
            ):
                raise ValueError(
                    "Slurm jobs support at most 256 processes and 65536 GiB"
                )
            return
        if cpus > self.settings["cpus"]:
            raise ValueError("Job CPU count exceeds queue budget")
        if (
            runtime == "apptainer"
            and int(memory_gib * execution.GIB) > self.settings["memory_bytes"]
        ):
            raise ValueError("Job memory limit exceeds queue budget")

    def resources(
        self,
        runtime: str,
        cpus: int,
        memory_gib: float,
        timeout: float | None,
        target: str = execution.ExecutionMode.LOCAL,
    ) -> models.Resources:
        if runtime not in execution.Runtime:
            raise ValueError("Unknown runtime")
        validate_positive(cpus, "CPU count", integer=True)
        validate_positive(memory_gib, "Memory limit")
        if timeout is not None:
            validate_positive(timeout, "Timeout")
        self.validate_resource_budget(runtime, cpus, memory_gib, target)
        return models.Resources(
            cpus=int(cpus),
            memory_bytes=int(memory_gib * execution.GIB),
            timeout=timeout,
            hard_memory_limit=runtime == "apptainer"
            and target != "slurm"
            and not self.config.allocation,
        )

    def validate_queue_limits(self, settings: dict) -> None:
        for key, label in (
            ("concurrency", "Concurrency"),
            ("cpus", "CPU budget"),
            ("memory_bytes", "Memory budget"),
        ):
            validate_positive(settings[key], label, integer=True)
        available = len(compute_cpus())
        if settings["cpus"] > available:
            raise ValueError(
                f"CPU budget cannot exceed {available}; one CPU is reserved for web requests on multicore hosts"
            )
        memory_limit = memory_ceiling() * 9 // 10
        if settings["memory_bytes"] > memory_limit:
            raise ValueError(
                f"Memory budget cannot exceed {memory_limit / execution.GIB:.2f} GiB "
                f"({memory_limit} bytes; 90% of detected host memory)"
            )

    def validate_running_budget(self, settings: dict) -> None:
        running = [value[2] for value in self.running.values()]
        cpus = sum(task["resources"]["cpus"] for task in running)
        memory = sum(
            task["resources"]["memory_bytes"]
            for task in running
            if task.get("runtime") == "apptainer"
        )
        if cpus > settings["cpus"] or memory > settings["memory_bytes"]:
            raise ValueError(
                "New budget is below current running reservations; stop jobs or wait for them to finish"
            )

    def validate_waiting_budget(self, settings: dict) -> None:
        for task in self.store.queued():
            if task.get("target") == "slurm":
                continue
            resource = self.task_resources(task)
            if resource["cpus"] > settings["cpus"] or (
                task.get("runtime") == "apptainer"
                and resource["memory_bytes"] > settings["memory_bytes"]
            ):
                raise ValueError("New budget cannot accommodate an existing queued job")

    def configure(self, values: dict) -> dict:
        with self.guard:
            settings = {**self.settings, **values}
            # Validate the host ceiling, running reservations, then waiting work.
            self.validate_queue_limits(settings)
            self.validate_running_budget(settings)
            self.validate_waiting_budget(settings)
            self.settings = self.store.set_settings(settings)
            self.wake.set()
            return self.settings

    def queue_data(self) -> dict:
        tasks = self.store.tasks()
        # No log tails, geometries, or raw input text in queue polling.
        return {
            "settings": self.settings,
            "cpu_capacity": len(compute_cpus()),
            "runtimes": self.runtimes.availability(),
            "batches": self.store.batches(),
            "tasks": [{k: t[k] for k in QUEUE_FIELDS if k in t} for t in tasks],
            "running_count": len(self.running)
            + sum(
                t.get("target") == execution.ExecutionMode.SLURM
                and t["status"] == execution.TaskStatus.RUNNING
                for t in tasks
            ),
        }

    def reap_finished(self) -> None:
        for task_id, (thread, _, _) in list(self.running.items()):
            if not thread.is_alive():
                thread.join()
                del self.running[task_id]

    def task_resources(self, task: models.TaskRecord) -> models.Resources:
        return task.get(
            "resources",
            models.Resources(
                cpus=task["processes"],
                memory_bytes=4 * execution.GIB,
                timeout=None,
                hard_memory_limit=False,
            ),
        )

    def valid_compute_capacity(self, task: models.TaskRecord) -> bool:
        if self.task_resources(task)["cpus"] <= len(compute_cpus()):
            return True
        self.store.update(
            task["id"],
            status="failed",
            ended=now(),
            error="Job CPU count exceeds compute capacity after reserving a CPU for web requests; retry with fewer CPUs",
        )
        return False

    def can_start_local(
        self, task: models.TaskRecord, running: list[models.TaskRecord]
    ) -> bool:
        runtime = task.get("runtime", "native")
        if (runtime == "native" and running) or any(
            item.get("runtime", "native") == "native" for item in running
        ):
            return False
        limit = 1 if self.config.allocation else self.settings["concurrency"]
        resources = self.task_resources(task)
        if (
            len(running) >= limit
            or sum(item["resources"]["cpus"] for item in running) + resources["cpus"]
            > self.settings["cpus"]
        ):
            return False
        memory = sum(item["resources"]["memory_bytes"] for item in running)
        return (
            runtime != "apptainer"
            or memory + resources["memory_bytes"] <= self.settings["memory_bytes"]
        )

    def start_local(self, task: models.TaskRecord) -> None:
        cancel = threading.Event()
        thread = threading.Thread(
            target=self.execute_attempt, args=(task, cancel), daemon=True
        )
        self.store.update(task["id"], status="running", started=now())
        self.running[task["id"]] = (thread, cancel, task)
        thread.start()

    def admit_local_tasks(self) -> None:
        # Preserve strict FIFO admission instead of skipping a blocked large job.
        for task in self.store.queued():
            if task.get("target") == "slurm":
                continue
            if not self.valid_compute_capacity(task):
                continue
            running = [value[2] for value in self.running.values()]
            if not self.can_start_local(task, running):
                break
            self.start_local(task)
            if task.get("runtime", "native") == "native":
                break

    def stop_local_tasks(self) -> None:
        for thread, cancel, _ in list(self.running.values()):
            cancel.set()
            thread.join()
        self.running.clear()

    def worker(self) -> None:
        while not self.stop.is_set():
            with self.guard:
                self.reap_finished()
                if self.stop.is_set():
                    break
                if not self.settings["paused"]:
                    self.admit_local_tasks()
            self.wake.wait(0.1)
            self.wake.clear()
        self.stop_local_tasks()

    def execute_attempt(self, task: models.TaskRecord, cancel: threading.Event) -> None:
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

    def valid_tfa_reference(self, directory: Path) -> bool:
        if not (directory / "tfa.mol").exists() or not (directory / "tfa.in").exists():
            return False
        manifest = json.loads((directory / "manifest.json").read_text())
        return digest(directory / "tfa.in") == manifest["inputs"]["tfa.in"] and all(
            digest(self.config.pseudos / name) == expected
            for name, expected in manifest["pseudopotentials"].items()
        )

    def shared_tfa_reference(self) -> Path | None:
        for task in self.store.tasks():
            if task["kind"] != "prepare" or task["status"] != "succeeded":
                continue
            directory = self.config.artifacts / task["id"]
            try:
                if self.valid_tfa_reference(directory):
                    return directory
            except (OSError, KeyError, ValueError):
                self.logger.warning(
                    "Ignoring unavailable shared TFA from attempt %s", task["id"]
                )
        return None

    def preparation_command(
        self, task: models.TaskRecord, directory: Path
    ) -> list[str]:
        candidate = self.candidates[task["candidate"]]
        request = {
            "candidate": task["candidate"],
            "smiles": candidate["smiles"],
            "cid": candidate["cid"],
            "pseudos": str(self.config.pseudos),
        }
        shared = self.shared_tfa_reference()
        if shared is not None:
            request["shared_tfa"] = str(shared)
        (directory / "request.json").write_text(json.dumps(request))
        return [
            self.config.prepare_python,
            str(configuration.ROOT / "dashboard/app/preparation.py"),
            str(directory / "request.json"),
            str(directory),
        ]

    def task_command(self, task: models.TaskRecord, directory: Path) -> list[str]:
        if task["kind"] == "diagram":
            return [
                self.config.python,
                str(configuration.ROOT / "dashboard/app/diagram.py"),
                self.candidates[task["candidate"]]["smiles"],
                str(directory / "diagram.png"),
            ]
        if task["kind"] == "prepare":
            return self.preparation_command(task, directory)
        return list(task["command"])

    def runtime_command(self, task: models.TaskRecord, directory: Path) -> list[str]:
        command = self.task_command(task, directory)
        if task.get("runtime", "native") == "apptainer":
            if task["kind"] not in execution.CALCULATION_KINDS:
                command[0] = "/opt/venv/bin/python"
            command = self.runtimes.wrap(
                command,
                directory,
                task["resources"],
                task["image_hash"],
                scheduler=task.get("target") == "slurm",
            )
        return command

    def register_entrypoint(
        self, task: models.TaskRecord, directory: Path, command: list[str]
    ) -> None:
        write_entrypoint(task, directory, command)
        name = execution.ENTRYPOINT_LOG_FILE
        artifacts = {
            **task["artifacts"],
            name: self.store.register(task["id"] + "-" + name, directory / name),
        }
        self.store.update(task["id"], command=command, artifacts=artifacts)
        self.logger.info(
            "Task %s entrypoint: %s (cwd=%s)",
            task["id"],
            shlex.join(command),
            directory,
        )

    def execute_local(
        self, task: models.TaskRecord, directory: Path, cancel: threading.Event
    ) -> None:
        command = self.runtime_command(task, directory)
        self.register_entrypoint(task, directory, command)

        def usage(value: models.MemoryUsage) -> None:
            self.store.update(task["id"], usage=value)

        # Only workstation containers report a separate cgroup memory limit.
        container = (
            task.get("runtime", "native") == "apptainer"
            and not self.config.allocation
            and task.get("target") != "slurm"
        )
        env = self.chem_env if task["kind"] not in execution.CALCULATION_KINDS else None
        code, status, error = execute(
            command,
            directory,
            cancel,
            self.default_timeout(task["kind"], task["timeout"]),
            env=env,
            on_usage=usage,
            container=container,
        )
        values = read_calculation_result(task, directory, code, status, error)
        values["command"] = command
        if self.stop.is_set():
            values["status"] = execution.TaskStatus.INTERRUPTED
        self.store.update(task["id"], **values)

    def finish_attempt(self, task_id: str, directory: Path) -> None:
        artifacts = {}
        for name in execution.LOCAL_ARTIFACT_FILES:
            path = directory / name
            if path.is_file():
                artifacts[name] = self.store.register(task_id + "-" + name, path)
        result = self.store.update(task_id, artifacts=artifacts)
        self.logger.info(
            "Task %s finished: %s (exit=%s)",
            task_id,
            result["status"],
            result.get("exit_code"),
        )
        if result.get("error"):
            self.logger.error("Task %s: %s", task_id, result["error"])

    def run(
        self, task: models.TaskRecord, cancel: threading.Event | None = None
    ) -> None:
        cancel = cancel or threading.Event()
        directory = self.config.artifacts / task["id"]
        self.logger.info(
            "Task %s started: %s for cluster %s (%s)",
            task["id"],
            task["kind"],
            task["candidate"],
            task["system"],
        )
        self.logger.info(
            "Task %s logs: stdout=%s stderr=%s",
            task["id"],
            directory / "stdout.log",
            directory / "stderr.log",
        )
        try:
            self.execute_local(task, directory, cancel)
        except Exception as error:
            self.logger.exception("Task %s could not execute", task["id"])
            self.store.update(
                task["id"], status="failed", error=str(error), ended=now()
            )
        finally:
            self.finish_attempt(task["id"], directory)

    def registered_input(self, task_id: str, name: str) -> Path | None:
        path = self.store.artifact(task_id + "-" + name)
        return Path(path) if path else None

    def elapsed_seconds(self, started: str, ended: str | None) -> float:
        from datetime import datetime

        return (
            datetime.fromisoformat(ended or now()) - datetime.fromisoformat(started)
        ).total_seconds()

    def task_data(self, task: models.TaskRecord) -> models.TaskRecord:
        from dashboard.core.chemistry.geometry import prepared_geometries

        task["geometries"] = prepared_geometries(
            task, lambda name: self.registered_input(task["id"], name)
        )
        if task["started"]:
            task["elapsed_seconds"] = self.elapsed_seconds(
                task["started"], task["ended"]
            )
        directory = self.config.artifacts / task["id"]
        task["stdout_tail"] = tail(directory / execution.STDOUT_LOG_FILE)
        task["stderr_tail"] = tail(directory / execution.STDERR_LOG_FILE)
        return task

    def data(self) -> dict:
        return {
            "mode": "live",
            "queue": {
                "settings": self.settings,
                "runtimes": self.runtimes.availability(),
            },
            "candidates": [
                candidate
                for candidate in self.candidates.values()
                if candidate["id"] != "tfa"
            ],
            "reference": self.candidates["tfa"],
            "tasks": [self.task_data(task) for task in self.store.tasks()],
            "input_versions": self.versions(),
        }
