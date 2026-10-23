"""Names and limits shared by local execution and Slurm workers."""

from enum import StrEnum


class ExecutionMode(StrEnum):
    AUTO = "auto"
    LOCAL = "local"
    SLURM = "slurm"


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    INTERRUPTED = "interrupted"


class TaskKind(StrEnum):
    DIAGRAM = "diagram"
    PREPARE = "prepare"
    QE = "qe"
    ESTIMATE_RAM = "estimate_ram"


class Runtime(StrEnum):
    NATIVE = "native"
    APPTAINER = "apptainer"


class SlurmState(StrEnum):
    SUBMITTING = "SUBMITTING"
    SUBMISSION_UNKNOWN = "SUBMISSION_UNKNOWN"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    NODE_FAIL = "NODE_FAIL"
    BOOT_FAIL = "BOOT_FAIL"
    PREEMPTED = "PREEMPTED"
    DEADLINE = "DEADLINE"
    REVOKED = "REVOKED"


EXECUTION_ENV = "PFAS_EXECUTION"
COMPUTE_WORKER_ENV = "PFAS_COMPUTE_WORKER"
SLURM_JOB_ID_ENV = "SLURM_JOB_ID"
SLURM_CPUS_PER_TASK_ENV = "SLURM_CPUS_PER_TASK"
SLURM_MEM_PER_NODE_ENV = "SLURM_MEM_PER_NODE"
SLURM_MEM_PER_CPU_ENV = "SLURM_MEM_PER_CPU"
SRUN_COMMAND = "srun"
SBATCH_COMMAND = "sbatch"
SQUEUE_COMMAND = "squeue"
SACCT_COMMAND = "sacct"
SCANCEL_COMMAND = "scancel"
SCHEDULER_COMMANDS = (SBATCH_COMMAND, SQUEUE_COMMAND, SACCT_COMMAND, SCANCEL_COMMAND)
APPLICATION_LOGGER_NAME = "uvicorn.error"
SLURM_CONFIG_ENV = ("SLURM_CONF", "SLURM_CONF_SERVER")
PARENT_JOB_ENV_PREFIXES = ("SBATCH_", "SLURM_", "PMI_", "PMIX_", "OMPI_")
DEFAULT_PARTITION = "ice-cpu"
DEFAULT_ACCOUNT = "coc"
DEFAULT_QOS = "coc-ice"
DEFAULT_QE_HOURS = 18
DEFAULT_ESTIMATION_HOURS = 4
DEFAULT_ESTIMATION_RANKS = (16, 4, 1)
DEFAULT_HEADROOM_PERCENT = 50
DEFAULT_ESTIMATION_TIMEOUT = 120
DEFAULT_LOCAL_MEMORY_GIB = 4
MAX_SLURM_HOURS = 168
MAX_SLURM_PROCESSES = 256
MAX_SLURM_MEMORY_GIB = 65536
MAX_ESTIMATION_RANKS = 16
MAX_PREVIEWS = 8192
MAX_SUBMISSIONS_PER_CYCLE = 16
GIB = 1024**3
MIB = 1024**2
MEMORY_BUDGET_NUMERATOR = 9
MEMORY_BUDGET_DENOMINATOR = 10
SLURM_POLL_SECONDS = 15
DISPATCH_POLL_SECONDS = 0.5
SLURM_COMMAND_TIMEOUT = 30
SLURM_JOB_PREFIX = "pfas-"
SLURM_DIRECTORY_PREFIX = "slurm-"
MANIFEST_FILE = "manifest.json"
SLURM_SCRIPT_FILE = "run.sbatch"
SLURM_LOG_FILE = "slurm.log"
SLURM_RESULT_FILE = "slurm-result.json"
SLURM_STARTED_FILE = "slurm-started.json"
ENTRYPOINT_LOG_FILE = "entrypoint.log"
STDOUT_LOG_FILE = "stdout.log"
STDERR_LOG_FILE = "stderr.log"
INPUT_FILE = "input.in"
PSEUDOS_DIRECTORY = "Pseudopotentials"
CONTAINER_PYTHON = "/opt/venv/bin/python"
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
CALCULATION_KINDS = (TaskKind.QE, TaskKind.ESTIMATE_RAM)
QE_INITIALIZATION_EXIT_CODES = (0, 255)
QE_DEFAULT_CALCULATION = "scf"
QE_ENERGY_MARKER = "total energy"
QE_EVIDENCE_MARKERS = (
    QE_ENERGY_MARKER,
    "JOB DONE.",
    "convergence",
    "BFGS Geometry",
    "bfgs converged",
    "Error in routine",
)
QE_MPI_PROCESS_PATTERNS = (
    r"Number of MPI processes:\s*(\d+)",
    r"running on\s+(\d+)\s+processor",
)
ERROR_TAIL_CHARACTERS = 2000
ACTIVE_TASK_STATUSES = (TaskStatus.QUEUED, TaskStatus.RUNNING)
TERMINAL_SLURM_STATES = frozenset(
    {
        SlurmState.COMPLETED,
        SlurmState.FAILED,
        SlurmState.CANCELLED,
        SlurmState.TIMEOUT,
        SlurmState.OUT_OF_MEMORY,
        SlurmState.NODE_FAIL,
        SlurmState.BOOT_FAIL,
        SlurmState.PREEMPTED,
        SlurmState.DEADLINE,
        SlurmState.REVOKED,
    }
)
