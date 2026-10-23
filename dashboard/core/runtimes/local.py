"""Local execution adapters; the job specification is independent of the scheduler."""

import json
import os
import shutil
import subprocess
from pathlib import Path

from dashboard.core import config as configuration
from dashboard.core import execution, models
from dashboard.core.chemistry.preparation import digest as file_hash
from dashboard.core.config import Config
from dashboard.core.runtimes.cpu import compute_cpus

GIB = execution.GIB

__all__ = ["GIB", "Runtimes", "file_hash", "memory_ceiling"]


def slurm_memory_ceiling() -> int | None:
    if (
        not os.getenv(execution.SLURM_JOB_ID_ENV)
        or os.getenv(execution.EXECUTION_ENV) == execution.ExecutionMode.LOCAL
    ):
        return None
    if os.getenv(execution.SLURM_MEM_PER_NODE_ENV):
        return int(os.environ[execution.SLURM_MEM_PER_NODE_ENV]) * execution.MIB
    if os.getenv(execution.SLURM_MEM_PER_CPU_ENV):
        return (
            int(os.environ[execution.SLURM_MEM_PER_CPU_ENV])
            * execution.MIB
            * int(os.getenv(execution.SLURM_CPUS_PER_TASK_ENV, "1"))
        )
    return None


def cgroup_memory_ceiling(ceiling: int) -> int:
    try:
        relative = Path("/proc/self/cgroup").read_text().split("0::", 1)[1].strip()
        path = Path(execution.CGROUP_ROOT) / relative.lstrip("/")
        # An ancestor can impose a tighter ceiling than the immediate cgroup.
        for parent in [path, *path.parents]:
            if not parent.is_relative_to(execution.CGROUP_ROOT):
                break
            value = (parent / "memory.max").read_text().strip()
            if value != "max":
                ceiling = min(ceiling, int(value))
    except (OSError, ValueError, IndexError):
        pass
    return ceiling


def memory_ceiling() -> int:
    import psutil

    ceiling = psutil.virtual_memory().total
    allocation = slurm_memory_ceiling()
    return cgroup_memory_ceiling(
        min(ceiling, allocation) if allocation is not None else ceiling
    )


class Runtimes:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.verified: tuple[tuple[str, int, int], str] | None = None

    def image_verified(self) -> bool:
        image = self.config.image
        return bool(
            self.verified
            and image
            and image.is_file()
            and self.verified[0]
            == (str(image), image.stat().st_size, image.stat().st_mtime_ns)
        )

    def apptainer_availability(self) -> models.RuntimeCapability:
        image = self.config.image
        available = bool(
            shutil.which(self.config.apptainer) and image and image.is_file()
        )
        verified = self.image_verified() and not self.config.allocation
        if self.config.allocation:
            message = "Uses the dashboard Slurm allocation; no separate task memory cap"
        else:
            message = (
                "Verify memory isolation with a batch preview"
                if available
                else "Install Apptainer and configure PFAS_APPTAINER_IMAGE"
            )
        return models.RuntimeCapability(
            available=available,
            hard_memory_limit=verified,
            parallel=verified,
            message=message,
        )

    def availability(self) -> models.RuntimeAvailability:
        return models.RuntimeAvailability(
            execution={
                "mode": "slurm" if self.config.scheduler_enabled else "local",
                "allocation": os.getenv(execution.SLURM_JOB_ID_ENV)
                if self.config.allocation
                else None,
            },
            native={"available": True, "hard_memory_limit": False, "parallel": False},
            apptainer=self.apptainer_availability(),
        )

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

    def isolation_probe_command(self, image: Path) -> list[str]:
        limit = 128 * execution.MIB
        return [
            *self.base(limit, 1),
            "--bind",
            f"{configuration.ROOT}:{configuration.ROOT}:ro",
            "--bind",
            "/sys/fs/cgroup:/sys/fs/cgroup:ro",
            "--env",
            f"PYTHONPATH={configuration.ROOT}",
            str(image),
            "/opt/venv/bin/python",
            str(configuration.ROOT / "dashboard/app/container.py"),
            "--probe",
            str(limit),
        ]

    def validate_isolation_probe(
        self, result: subprocess.CompletedProcess[str]
    ) -> None:
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
        if not self.verified or self.verified[0] != key:
            result = subprocess.run(
                self.isolation_probe_command(image),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.validate_isolation_probe(result)
            self.verified = (key, identity)
        return identity

    def allocation_command(self, command: list[str], directory: Path) -> list[str]:
        """Use Slurm's allocation directly; no container supervisor or cgroups."""
        executable = shutil.which(self.config.apptainer)
        if not executable:
            raise ValueError("Apptainer executable unavailable")
        threads = ",".join(
            [
                *(f"{name}=1" for name in execution.THREAD_ENV_VARS),
                f"PYTHONPATH={configuration.ROOT}",
            ]
        )
        return [
            executable,
            "exec",
            "--cleanenv",
            "--bind",
            f"{directory}:{directory}",
            "--pwd",
            str(directory),
            "--env",
            threads,
            str(self.config.image),
            *command,
        ]

    def workspace_arguments(self, directory: Path) -> list[str]:
        arguments = []
        for path in sorted(
            {configuration.ROOT, self.config.artifacts, self.config.pseudos}
        ):
            arguments += ["--bind", f"{path}:{path}:ro"]
        # Only this attempt may write; resource observations stay read-only.
        arguments += [
            "--bind",
            f"{directory}:{directory}:rw",
            "--bind",
            "/sys/fs/cgroup:/sys/fs/cgroup:ro",
            "--pwd",
            str(directory),
        ]
        return arguments

    def thread_arguments(self) -> list[str]:
        arguments = []
        for name in execution.THREAD_ENV_VARS:
            arguments += ["--env", f"{name}=1"]
        return arguments + ["--env", f"PYTHONPATH={configuration.ROOT}"]

    def supervisor_command(
        self, command: list[str], directory: Path, resources: models.Resources
    ) -> list[str]:
        return [
            str(self.config.image),
            "/opt/venv/bin/python",
            str(configuration.ROOT / "dashboard/app/container.py"),
            "--metadata",
            str(directory / "resource.json"),
            "--limit",
            str(resources["memory_bytes"]),
            "--",
            *command,
        ]

    def wrap(
        self,
        command: list[str],
        directory: Path,
        resources: models.Resources,
        image_hash: str | None,
        scheduler: bool = False,
    ) -> list[str]:
        if self.image_identity() != image_hash:
            raise ValueError("Container image changed; preview and retry explicitly")
        if scheduler or self.config.allocation:
            return self.allocation_command(command, directory)
        # Workstations retain the existing per-task cgroup supervisor.
        return [
            *self.base(resources["memory_bytes"], resources["cpus"]),
            *self.workspace_arguments(directory),
            *self.thread_arguments(),
            *self.supervisor_command(command, directory, resources),
        ]

    def native_probe(
        self, code: str, args: tuple[str, ...]
    ) -> tuple[list[str], dict[str, str]]:
        python = self.config.prepare_python
        environment = {
            **os.environ,
            "PATH": str(Path(python).absolute().parent)
            + os.pathsep
            + os.environ["PATH"],
        }
        return [python, "-c", code, *args], environment

    def container_probe(self, code: str, args: tuple[str, ...]) -> list[str]:
        self.verify()
        return [
            *self.base(512 * execution.MIB, 1),
            str(self.config.image),
            "/opt/venv/bin/python",
            "-c",
            code,
            *args,
        ]

    def chemistry_probe(
        self,
        runtime: str,
        code: str,
        args: tuple[str, ...] = (),
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if runtime == "native":
            command, environment = self.native_probe(code, args)
        else:
            command, environment = self.container_probe(code, args), None
        return subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=environment,
        )
