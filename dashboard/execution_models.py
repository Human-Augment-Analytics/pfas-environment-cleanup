"""Persisted execution records; existing task JSON remains backwards compatible."""

from typing import Literal, NotRequired, TypedDict


class SchedulerSettings(TypedDict):
    partition: str
    account: str
    qos: str
    walltime_hours: int
    memory_mode: Literal["manual", "estimate"]
    headroom_percent: float


class SchedulerOverrides(TypedDict, total=False):
    partition: str
    account: str
    qos: str
    walltime_hours: int
    memory_mode: Literal["manual", "estimate"]
    headroom_percent: float


class Resources(TypedDict):
    cpus: int
    memory_bytes: int
    timeout: float | None
    hard_memory_limit: bool


class SchedulerRecord(TypedDict):
    token: str
    job_id: str | None
    state: str
    reason: str
    directory: str
    submitted: str
    command: list[str]


class MemoryUsage(TypedDict):
    current_memory_bytes: int
    peak_memory_bytes: int
    measurement: str


class RAMValue(TypedDict):
    value: str
    unit: str
    bytes: float


class RAMReport(TypedDict):
    per_process: RAMValue | None
    total: RAMValue | None


class Evidence(TypedDict):
    job_done: bool
    scf_converged: bool
    relaxation_completed: bool
    confirmed: bool
    energy_ry: float | None
    provisional: bool


class TaskResult(TypedDict, total=False):
    status: str
    exit_code: int | None
    error: str
    started: str
    ended: str
    evidence: Evidence
    ram_estimate: RAMReport
    usage: MemoryUsage
    command: list[str]


class TaskRecord(TypedDict):
    id: str
    kind: str
    candidate: str
    system: str
    processes: int
    status: str
    runtime: str
    resources: Resources
    timeout: float | None
    command: list[str]
    image_hash: str | None
    input_hash: str
    source_hash: str
    pseudopotentials: dict[str, str]
    artifacts: dict[str, str]
    target: NotRequired[str]
    batch_id: NotRequired[str | None]
    scheduler: NotRequired[SchedulerRecord]
    scheduler_settings: NotRequired[SchedulerSettings]
    stop_requested: NotRequired[bool]
    exit_code: NotRequired[int | None]
    error: NotRequired[str]
    ended: NotRequired[str | None]
    started: NotRequired[str | None]
    evidence: NotRequired[Evidence]
    ram_estimate: NotRequired[RAMReport]
    usage: NotRequired[MemoryUsage]


class CandidateRecord(TypedDict):
    id: str
    cid: str
    smiles: str
    fields: dict[str, str]


class WorkerConfig(TypedDict):
    artifacts: str
    pseudos: str
    apptainer: str
    image: str | None


class WorkerManifest(TypedDict):
    tasks: list[TaskRecord]
    config: WorkerConfig


class BatchReviewRequest(TypedDict, total=False):
    candidates: list[str]
    kind: str
    system: str
    runtime: str
    processes: int
    memory_gib: float
    timeout: float | None
    retry: bool
    rerun_completed: bool
    target: str
    issue_tokens: bool
    scheduler: SchedulerOverrides | None
    id: str
    expected: list[dict]
    image_hash: str | None


class RuntimeCapability(TypedDict):
    available: bool
    hard_memory_limit: bool
    parallel: bool
    message: NotRequired[str]


class ExecutionContext(TypedDict):
    mode: str
    allocation: str | None


class RuntimeAvailability(TypedDict):
    native: RuntimeCapability
    apptainer: RuntimeCapability
    execution: ExecutionContext
