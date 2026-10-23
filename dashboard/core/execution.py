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
DEFAULT_HEADROOM_PERCENT = 50
MAX_SLURM_HOURS = 168
MAX_SLURM_PROCESSES = 256
MAX_SLURM_MEMORY_GIB = 65536
MAX_ESTIMATION_RANKS = 16
MAX_PREVIEWS = 8192
MAX_SUBMISSIONS_PER_CYCLE = 16
GIB = 1024**3
MIB = 1024**2
SLURM_POLL_SECONDS = 15
DISPATCH_POLL_SECONDS = 0.5
SLURM_COMMAND_TIMEOUT = 30
SLURM_JOB_PREFIX = "pfas-"
SLURM_DIRECTORY_PREFIX = "slurm-"
SLURM_SCRIPT_FILE = "run.sbatch"
SLURM_LOG_FILE = "slurm.log"
SLURM_RESULT_FILE = "slurm-result.json"
SLURM_STARTED_FILE = "slurm-started.json"
ENTRYPOINT_LOG_FILE = "entrypoint.log"
STDOUT_LOG_FILE = "stdout.log"
STDERR_LOG_FILE = "stderr.log"
INPUT_FILE = "input.in"
PSEUDOS_DIRECTORY = "Pseudopotentials"
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
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

# Direct Slurm jobs record only launch timestamps and exit status. Output is
# interpreted by the dashboard, not by a Python worker on the compute node.
SLURM_STARTED_TEXT_FILE = "slurm-started.txt"
SLURM_FINISHED_FILE = "slurm-finished.txt"
SNAPSHOT_CHECKSUMS_FILE = "inputs.sha256"
IMAGE_CHECKSUMS_FILE = "image.sha256"
SNAPSHOT_CHECK_LOG = "input-check.log"
SNAPSHOT_CHECK_FAILED = 254
TIMEOUT_EXIT_CODE = 124
TIMEOUT_KILL_SECONDS = 5


# Local process monitoring: bounded reads keep Stop and timeout responsive.
PROCESS_POLL_SECONDS = 0.05
MEMORY_SAMPLE_SECONDS = 0.25
TERMINATE_GRACE_SECONDS = 0.5
STREAM_CHUNK_BYTES = 65536
STREAM_READS_PER_POLL = 16
CPU_APP = "dashboard.app.cpu"
RESOURCE_REPORT_FILE = "resource.json"
CGROUP_ROOT = "/sys/fs/cgroup"
CGROUP_EVENTS_FILE = "memory.events"
CGROUP_CURRENT_FILE = "memory.current"
CGROUP_PEAK_FILE = "memory.peak"
CGROUP_OOM_COUNTER = "oom_kill"
CGROUP_OOM_ERROR = "Memory limit exceeded (cgroup OOM kill)"
PROCESS_RSS_MEASUREMENT = "process-tree RSS"
CGROUP_MEASUREMENT = "cgroup memory"
PYTHON_UNBUFFERED_ENV = "PYTHONUNBUFFERED"
PYTHON_PATH_ENV = "PYTHONPATH"
LOG_METADATA_PREFIXES = (
    "Input hash:",
    "Source hash:",
    "Image hash:",
    "Runtime:",
    "Resources:",
)


CALCULATION_KINDS = (TaskKind.QE, TaskKind.ESTIMATE_RAM)
DEFAULT_DIAGRAM_TIMEOUT = 60
PREPARATION_TOOL_PROBE = "import shutil; import rdkit,pymatgen.core,ase,openbabel; assert shutil.which('obabel'); assert shutil.which('cif2cell')"
PERSISTED_REQUEST_FIELDS = (
    "kind",
    "candidate",
    "system",
    "processes",
    "timeout",
    "runtime",
    "target",
    "batch_id",
    "position",
)
LOCAL_ARTIFACT_FILES = (
    "entrypoint.log",
    "stdout.log",
    "stderr.log",
    "diagram.png",
    "candidate.in",
    "complex.in",
    "tfa.in",
    "input.in",
    "manifest.json",
)

DEFAULT_ESTIMATION_TIMEOUT = 120
