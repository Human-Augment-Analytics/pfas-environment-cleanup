"""Local execution adapters; the job specification is independent of the scheduler."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from .config import ROOT, Config
from .cpu import compute_cpus
from .execution import (
    EXECUTION_ENV,
    GIB,
    MIB,
    SLURM_CPUS_PER_TASK_ENV,
    SLURM_JOB_ID_ENV,
    SLURM_MEM_PER_CPU_ENV,
    SLURM_MEM_PER_NODE_ENV,
    THREAD_ENV_VARS,
    ExecutionMode,
)
from .execution_models import Resources, RuntimeAvailability

__all__ = ["GIB", "Runtimes", "file_hash", "memory_ceiling"]


def file_hash(path: Path | str) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def memory_ceiling() -> int:
    import psutil

    ceiling = psutil.virtual_memory().total
    if os.getenv(SLURM_JOB_ID_ENV) and os.getenv(EXECUTION_ENV) != ExecutionMode.LOCAL:
        if os.getenv(SLURM_MEM_PER_NODE_ENV):
            ceiling = min(ceiling, int(os.environ[SLURM_MEM_PER_NODE_ENV]) * MIB)
        elif os.getenv(SLURM_MEM_PER_CPU_ENV):
            ceiling = min(
                ceiling,
                int(os.environ[SLURM_MEM_PER_CPU_ENV])
                * MIB
                * int(os.getenv(SLURM_CPUS_PER_TASK_ENV, "1")),
            )
    try:
        relative = Path("/proc/self/cgroup").read_text().split("0::", 1)[1].strip()
        path = Path("/sys/fs/cgroup") / relative.lstrip("/")
        for parent in [path, *path.parents]:
            if not parent.is_relative_to("/sys/fs/cgroup"):
                break
            value = (parent / "memory.max").read_text().strip()
            if value != "max":
                ceiling = min(ceiling, int(value))
    except (OSError, ValueError, IndexError):
        pass
    return ceiling


class Runtimes:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.verified: tuple[tuple[str, int, int], str] | None = None

    def availability(self) -> RuntimeAvailability:
        executable = shutil.which(self.config.apptainer)
        image = self.config.image
        verified = bool(
            self.verified
            and image
            and image.is_file()
            and self.verified[0]
            == (str(image), image.stat().st_size, image.stat().st_mtime_ns)
        )
        return {
            "execution": {
                "mode": "slurm" if self.config.scheduler_enabled else "local",
                "allocation": os.getenv(SLURM_JOB_ID_ENV)
                if self.config.allocation
                else None,
            },
            "native": {
                "available": True,
                "hard_memory_limit": False,
                "parallel": False,
            },
            "apptainer": {
                "available": bool(executable and image and image.is_file()),
                "hard_memory_limit": verified and not self.config.allocation,
                "parallel": verified and not self.config.allocation,
                "message": "Uses the dashboard Slurm allocation; no separate task memory cap"
                if self.config.allocation
                else "Verify memory isolation with a batch preview"
                if executable and image and image.is_file()
                else "Install Apptainer and configure PFAS_APPTAINER_IMAGE",
            },
        }

    def image_identity(self) -> str:
        image = self.config.image
        if not image or not image.is_file():
            raise ValueError("Set PFAS_APPTAINER_IMAGE to a built SIF image")
        stat = image.stat()
        key = (str(image), stat.st_size, stat.st_mtime_ns)
        if self.verified and self.verified[0] == key:
            return self.verified[1]
        return file_hash(image)

    def base(self, memory_bytes: int, cpus: int, scheduler: bool = False) -> list[str]:
        executable = shutil.which(self.config.apptainer)
        if not executable:
            raise ValueError("Apptainer executable unavailable")
        args = [
            executable,
            "exec",
            "--cleanenv",
            "--contain",
            "--pid",
        ]
        if scheduler or self.config.allocation:
            return args
        return args + [
            "--memory",
            str(memory_bytes),
            "--memory-swap",
            str(memory_bytes),
            "--cpus",
            str(cpus),
            "--cpuset-cpus",
            ",".join(map(str, compute_cpus())),
        ]

    def verify(self) -> str:
        identity = self.image_identity()
        if self.config.allocation:
            if not shutil.which(self.config.apptainer):
                raise ValueError("Apptainer executable unavailable")
            return identity
        image = self.config.image
        assert image is not None
        stat = image.stat()
        key = (str(image), stat.st_size, stat.st_mtime_ns)
        if self.verified and self.verified[0] == key:
            return identity
        limit = 128 * MIB
        command = [
            *self.base(limit, 1),
            "--bind",
            f"{ROOT}:{ROOT}:ro",
            "--bind",
            "/sys/fs/cgroup:/sys/fs/cgroup:ro",
            str(image),
            "/opt/venv/bin/python",
            str(ROOT / "dashboard/container_runner.py"),
            "--probe",
            str(limit),
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False
        )
        if result.returncode:
            raise ValueError(
                "Apptainer memory-isolation probe failed: " + result.stderr[-1000:]
            )
        try:
            report = json.loads(result.stdout)
            if not report["verified"]:
                raise ValueError("Hard memory and swap limits were not verified")
        except (KeyError, json.JSONDecodeError) as error:
            raise ValueError("Invalid Apptainer memory-isolation probe") from error
        self.verified = (key, identity)
        return identity

    def wrap(
        self,
        command: list[str],
        directory: Path,
        resources: Resources,
        image_hash: str | None,
        scheduler: bool = False,
    ) -> list[str]:
        if self.image_identity() != image_hash:
            raise ValueError("Container image changed; preview and retry explicitly")
        binds = {ROOT, self.config.artifacts, self.config.pseudos}
        args = self.base(resources["memory_bytes"], resources["cpus"], scheduler)
        for path in sorted(binds):
            args += ["--bind", f"{path}:{path}:ro"]
        args += [
            "--bind",
            f"{directory}:{directory}:rw",
            "--bind",
            "/sys/fs/cgroup:/sys/fs/cgroup:ro",
            "--pwd",
            str(directory),
        ]
        # Match the image's MPI with its QE binary; never inject host MPI/Python.
        for key in THREAD_ENV_VARS:
            args += ["--env", f"{key}=1"]
        if scheduler or self.config.allocation:
            return args + [str(self.config.image), *command]
        args += [
            str(self.config.image),
            "/opt/venv/bin/python",
            str(ROOT / "dashboard/container_runner.py"),
            "--metadata",
            str(directory / "resource.json"),
            "--limit",
            str(resources["memory_bytes"]),
            "--",
            *command,
        ]
        return args

    def chemistry_probe(
        self,
        runtime: str,
        code: str,
        args: tuple[str, ...] = (),
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if runtime == "native":
            python = self.config.prepare_python
            command = [python, "-c", code, *args]
            env = {
                **os.environ,
                "PATH": str(Path(python).absolute().parent)
                + os.pathsep
                + os.environ["PATH"],
            }
        else:
            self.verify()
            command = [
                *self.base(512 * MIB, 1),
                str(self.config.image),
                "/opt/venv/bin/python",
                "-c",
                code,
                *args,
            ]
            env = None
        return subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=env,
        )
