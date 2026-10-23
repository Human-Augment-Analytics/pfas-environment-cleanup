"""Leave one allowed CPU available to the dashboard and other host work."""

import os
import sys

try:
    from .execution import (
        COMPUTE_WORKER_ENV,
        EXECUTION_ENV,
        SLURM_CPUS_PER_TASK_ENV,
        SLURM_JOB_ID_ENV,
        ExecutionMode,
    )
except ImportError:
    from execution import (  # ty: ignore[unresolved-import]
        # Direct file execution resolves the sibling module on sys.path.
        COMPUTE_WORKER_ENV,
        EXECUTION_ENV,
        SLURM_CPUS_PER_TASK_ENV,
        SLURM_JOB_ID_ENV,
        ExecutionMode,
    )


def compute_cpus() -> list[int]:
    allowed = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1))
    )
    if os.getenv(SLURM_JOB_ID_ENV) and os.getenv(EXECUTION_ENV) != ExecutionMode.LOCAL:
        count = int(os.getenv(SLURM_CPUS_PER_TASK_ENV, str(len(allowed))))
        allowed = allowed[: max(1, count)]
    if os.getenv(COMPUTE_WORKER_ENV) == "1":
        return allowed
    return allowed[1:] if len(allowed) > 1 else allowed


def main() -> None:
    # Run in a separate interpreter: preexec_fn is unsafe in a threaded server.
    cpus = [int(cpu) for cpu in sys.argv[1].split(",")]
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)
    if hasattr(os, "nice"):
        os.nice(10)
    os.execvpe(sys.argv[2], sys.argv[2:], os.environ)


if __name__ == "__main__":
    main()
