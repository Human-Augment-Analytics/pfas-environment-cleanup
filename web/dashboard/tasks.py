import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

from .persistence import Store, now
from .preparation import digest, pseudo_names
from .processes import evidence, execute, tail


class Manager:
    def __init__(self, config, candidates):
        self.config = config
        config.artifacts.mkdir(parents=True, exist_ok=True)
        self.store = Store(config.artifacts / "tasks.sqlite")
        self.candidates = candidates
        self.chem_env = {
            **os.environ,
            "PATH": str(Path(config.python).absolute().parent)
            + os.pathsep
            + os.environ["PATH"],
        }
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.cancel = threading.Event()
        self.guard = threading.RLock()
        self.active = None
        self.thread = threading.Thread(target=self.worker, daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        with self.guard:
            self.stop.set()
            self.cancel.set()
            self.wake.set()
        if self.thread.is_alive():
            self.thread.join()
        for task in self.store.tasks():
            if task["status"] in ("queued", "running"):
                self.store.update(task["id"], status="interrupted", ended=now())

    def executable(self, name):
        path = shutil.which(name)
        if not path:
            raise ValueError(f"Executable unavailable: {name}")
        return path

    def preflight(self, candidate):
        probe = subprocess.run(
            [
                self.config.python,
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
                "Chemistry interpreter requires RDKit, pymatgen, ASE, Open Babel and cif2cell: "
                + probe.stderr[-2000:]
            )
        # RDKit runs in the configured interpreter, keeping the web environment minimal.
        result = subprocess.run(
            [
                self.config.python,
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

    def input(self, candidate, system):
        owner = "tfa" if system == "tfa" else candidate
        for task in self.store.tasks():
            if (
                task["kind"] == "prepare"
                and task["status"] == "succeeded"
                and (task["candidate"] == owner or system == "tfa")
            ):
                path = self.config.artifacts / task["id"] / (system + ".in")
                if path.exists():
                    return path
        raise ValueError("Prepare inputs first")

    def preview(self, candidate, system, processes, target):
        if target != "local":
            raise NotImplementedError(
                "Slurm and Slurm array execution are not implemented"
            )
        if type(processes) is not int or processes < 1:
            raise ValueError("Process count must be a positive integer")
        if system not in ("tfa", "candidate", "complex"):
            raise ValueError("Unknown system")
        path = self.input(candidate, system)
        input_bytes = path.read_bytes()
        text = input_bytes.decode()
        # Keep all file references inside the new run directory, including edited inputs.
        for key, expected in (
            ("pseudo_dir", "./Pseudopotentials"),
            ("outdir", "./Outputs"),
        ):
            match = re.search(
                rf"\b{key}\s*=\s*['\"]([^'\"]+)['\"]", text, re.IGNORECASE
            )
            if not match or match[1] != expected:
                raise ValueError(f"Input must use {key}='{expected}'")
        names = pseudo_names(text)
        hashes = {n: digest(self.config.pseudos / n) for n in names}
        pw = self.executable(self.config.pw)
        command = (
            [pw, "-in", "input.in"]
            if processes == 1
            else [
                self.executable(self.config.mpi),
                "-np",
                str(processes),
                pw,
                "-in",
                "input.in",
            ]
        )
        return {
            "command": command,
            "input": text,
            "input_hash": hashlib.sha256(input_bytes).hexdigest(),
            "pseudopotentials": hashes,
        }

    def queue(
        self,
        kind,
        candidate,
        system="candidate",
        processes=1,
        target="local",
        timeout=None,
        expected_hash=None,
    ):
        if kind == "qe" and target != "local":
            raise NotImplementedError(
                "Slurm and Slurm array execution are not implemented"
            )
        if candidate not in self.candidates:
            raise ValueError("Unknown candidate")
        if kind not in ("diagram", "prepare", "qe"):
            raise ValueError("Unknown task")
        if timeout is not None and (
            not isinstance(timeout, (float, int)) or timeout <= 0
        ):
            raise ValueError("Timeout must be positive")
        if kind != "qe":
            system = "tfa" if candidate == "tfa" else "candidate"
        with self.guard:
            for task in self.store.tasks():
                if (task["kind"], task["candidate"], task["system"]) == (
                    kind,
                    candidate,
                    system,
                ) and task["status"] in ("queued", "running"):
                    return task
            preview = None
            if kind == "prepare":
                self.preflight(self.candidates[candidate])
            if kind == "qe":
                preview = self.preview(candidate, system, processes, target)
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
                task.update(preview)
            self.store.put(task)
            self.wake.set()
            return task

    def cancel_task(self, id):
        with self.guard:
            task = self.store.get(id)
            if task["status"] == "queued":
                self.store.update(id, status="canceled", ended=now())
            elif task["status"] == "running" and self.active == id:
                self.cancel.set()

    def worker(self):
        while not self.stop.is_set():
            with self.guard:
                if self.stop.is_set():
                    break
                queued = [
                    t for t in reversed(self.store.tasks()) if t["status"] == "queued"
                ]
                task = queued[0] if queued else None
                if task:
                    self.active = task["id"]
                    self.cancel.clear()
                    self.store.update(task["id"], status="running", started=now())
            if task is None:
                self.wake.wait(0.2)
                self.wake.clear()
                continue
            self.run(task)
            with self.guard:
                self.active = None

    def run(self, task):
        id = task["id"]
        directory = self.config.artifacts / id
        try:
            candidate = self.candidates[task["candidate"]]
            timeout = task["timeout"]
            if task["kind"] == "diagram":
                command = [
                    self.config.python,
                    str(Path(__file__).with_name("diagram.py")),
                    candidate["smiles"],
                    str(directory / "diagram.png"),
                ]
                timeout = 60
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
                (directory / "request.json").write_text(json.dumps(request))
                command = [
                    self.config.python,
                    str(Path(__file__).with_name("preparation.py")),
                    str(directory / "request.json"),
                    str(directory),
                ]
                timeout = self.config.prepare_timeout
            else:
                command = task["command"]
            self.store.update(id, command=command)
            code, status, error = execute(
                command,
                directory,
                self.cancel,
                timeout,
                env=self.chem_env if task["kind"] != "qe" else None,
            )
            values = {
                "status": "interrupted" if self.stop.is_set() else status,
                "exit_code": code,
                "error": error,
                "ended": now(),
            }
            if task["kind"] == "qe":
                # Retain only scientific markers, bounding memory for long QE output.
                markers = []
                with (directory / "stdout.log").open(errors="replace") as stream:
                    for line in stream:
                        if any(
                            m in line
                            for m in (
                                "total energy",
                                "JOB DONE.",
                                "convergence",
                                "BFGS Geometry",
                                "bfgs converged",
                                "Error in routine",
                            )
                        ):
                            if "total energy" in line:
                                markers = [
                                    m for m in markers if "total energy" not in m
                                ]
                            markers.append(line)
                text = "".join(markers)
                calculation = re.search(
                    r"calculation\s*=\s*['\"]([^'\"]+)",
                    (directory / "input.in").read_text(),
                    re.IGNORECASE,
                )
                values["evidence"] = evidence(
                    text, calculation[1] if calculation else "scf"
                )
            self.store.update(id, **values)
        except Exception as error:  # noqa: BLE001 -- isolate failed attempts; keep queue alive
            self.store.update(id, status="failed", error=str(error), ended=now())
        finally:
            artifacts = {}
            for name in (
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
            self.store.update(id, artifacts=artifacts)

    def data(self):
        tasks = self.store.tasks()
        for task in tasks:
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
            "candidates": [c for c in self.candidates.values() if c["id"] != "tfa"],
            "reference": self.candidates["tfa"],
            "tasks": tasks,
        }
