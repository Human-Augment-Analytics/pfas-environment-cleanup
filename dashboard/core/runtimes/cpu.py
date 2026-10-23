"""Leave one allowed CPU available to the dashboard and other host work."""

import os

from dashboard.core import execution


def compute_cpus() -> list[int]:
    allowed = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1))
    )
    if (
        os.getenv(execution.SLURM_JOB_ID_ENV)
        and os.getenv(execution.EXECUTION_ENV) != execution.ExecutionMode.LOCAL
    ):
        count = int(os.getenv(execution.SLURM_CPUS_PER_TASK_ENV, str(len(allowed))))
        allowed = allowed[: max(1, count)]
    if os.getenv(execution.COMPUTE_WORKER_ENV) == "1":
        return allowed
    return allowed[1:] if len(allowed) > 1 else allowed


def main(*, cpus: str, command: list[str]) -> None:
    # Run in a separate interpreter: preexec_fn is unsafe in a threaded server.
    allowed_cpus = [int(cpu) for cpu in cpus.split(",")]
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, allowed_cpus)
    if hasattr(os, "nice"):
        os.nice(10)
    os.execvpe(command[0], command, os.environ)
