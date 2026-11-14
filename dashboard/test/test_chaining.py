"""Exercise durable queue advancement against an in-memory Slurm controller."""

import asyncio
import select
import subprocess
import sys
from dataclasses import dataclass, field

import httpx
import pytest
from fastapi import FastAPI

from dashboard.app import server
from dashboard.core import execution, models
from dashboard.core.jobs import persistence
from dashboard.core.jobs.tasks import Manager
from dashboard.core.slurm import scheduler
from dashboard.test.test_scheduler import manager as manager_fixture  # noqa: F401
from dashboard.test.test_scheduler import queue

JOB_NAME_OPTION = "--job-name="
DEPENDENCY_OPTION = "--dependency=afterany:"
LIVE_STATES = ("PENDING", "RUNNING")
COMMAND_TIMEOUT = 30
FIRST_JOB_ID = 1000


@dataclass
class Job:
    identifier: str
    name: str
    arguments: list[str]
    state: str = "PENDING"

    @property
    def dispatcher(self) -> bool:
        return any(arg.startswith(DEPENDENCY_OPTION) for arg in self.arguments)


@dataclass
class Slurm:
    jobs: dict[str, Job] = field(default_factory=dict)
    calls: list[list[str]] = field(default_factory=list)
    uncertain: str | None = None
    accept_uncertain: bool = True
    offline: bool = False

    def command(self, arguments: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(arguments.copy())
        program = arguments[0]
        if program == execution.SBATCH_COMMAND:
            return self.submit(arguments)
        if self.offline:
            return self.response(stderr="controller unavailable", returncode=1)
        if program == execution.SCANCEL_COMMAND:
            self.jobs[arguments[1]].state = "CANCELLED"
            return self.response()
        assert program in (execution.SQUEUE_COMMAND, execution.SACCT_COMMAND)
        jobs = list(self.jobs.values())
        if program == execution.SQUEUE_COMMAND:
            jobs = [job for job in jobs if job.state in LIVE_STATES]
        if "--jobs" in arguments:
            identifiers = arguments[arguments.index("--jobs") + 1].split(",")
            jobs = [job for job in jobs if job.identifier in identifiers]
        return self.response("".join(self.row(job, program) for job in jobs))

    def submit(self, arguments: list[str]) -> subprocess.CompletedProcess[str]:
        name = next(
            arg.removeprefix(JOB_NAME_OPTION)
            for arg in arguments
            if arg.startswith(JOB_NAME_OPTION)
        )
        identifier = str(FIRST_JOB_ID + len(self.jobs))
        job = Job(identifier, name, arguments.copy())
        kind = "dispatcher" if job.dispatcher else "compute"
        uncertain = self.uncertain == kind
        if not uncertain or self.accept_uncertain:
            self.jobs[identifier] = job
        if uncertain:
            self.uncertain = None
            raise subprocess.TimeoutExpired(arguments, COMMAND_TIMEOUT)
        return self.response(identifier + "\n")

    def row(self, job: Job, program: str) -> str:
        reason = "Resources" if program == execution.SQUEUE_COMMAND else "0:0"
        return f"{job.identifier}|{job.name}|{job.state}|{reason}\n"

    def response(
        self, stdout: str = "", stderr: str = "", returncode: int = 0
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], returncode, stdout, stderr)

    def submissions(self, *, dispatcher: bool) -> list[Job]:
        return [job for job in self.jobs.values() if job.dispatcher == dispatcher]


@pytest.fixture(name="manager")
def chain_manager(manager_fixture: Manager) -> Manager:  # noqa: F811
    return manager_fixture


@pytest.fixture
def slurm(monkeypatch: pytest.MonkeyPatch) -> Slurm:
    controller = Slurm()
    monkeypatch.setattr(scheduler, "command", controller.command)
    return controller


def dispatcher_record(manager: Manager) -> models.SchedulerRecord:
    record = manager.chain.data()["dispatcher"]
    assert record is not None
    return record


def job_identifier(record: models.SchedulerRecord) -> str:
    identifier = record["job_id"]
    assert isinstance(identifier, str)
    return identifier


def task_job(manager: Manager, slurm: Slurm, task_id: str) -> Job:
    record = manager.store.get(task_id)["scheduler"]
    assert record["job_id"] is not None
    return slurm.jobs[job_identifier(record)]


def dispatch(
    manager: Manager,
    slurm: Slurm,
    monkeypatch: pytest.MonkeyPatch,
    task_id: str,
    outcome: str = "COMPLETED",
) -> None:
    task_job(manager, slurm, task_id).state = outcome
    record = dispatcher_record(manager)
    assert record is not None
    dispatcher = slurm.jobs[job_identifier(record)]
    dispatcher.state = "RUNNING"
    with monkeypatch.context() as context:
        context.setenv(execution.SLURM_JOB_ID_ENV, dispatcher.identifier)
        manager.chain.recover(predecessor=task_id)
    dispatcher.state = "COMPLETED"


def test_three_jobs_advance_only_from_matching_dispatcher(
    manager: Manager, slurm: Slurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = [queue(manager) for _ in range(3)]
    manager.chain.configure(enabled=True, paused=False)
    assert manager.chain.data()["active_task_id"] == tasks[0]["id"]
    assert manager.store.ordered_ids() == [task["id"] for task in tasks[1:]]
    manager.scheduler.submit_waiting()
    assert len(slurm.submissions(dispatcher=False)) == 1

    # A stale callback must not consume a newer task's queue position.
    manager.chain.recover(predecessor="stale-task")
    assert len(slurm.jobs) == 2
    for index, task in enumerate(tasks):
        record = dispatcher_record(manager)
        compute = task_job(manager, slurm, task["id"])
        assert f"{DEPENDENCY_OPTION}{compute.identifier}" in record["command"]
        dispatch(manager, slurm, monkeypatch, task["id"])
        expected = tasks[index + 1]["id"] if index + 1 < len(tasks) else None
        assert manager.chain.data()["active_task_id"] == expected
    assert len(slurm.submissions(dispatcher=False)) == 3
    assert len(slurm.submissions(dispatcher=True)) == 3
    assert manager.chain.data()["dispatcher"] is None


def test_live_insert_reorder_and_remove_affect_next_selection(
    manager: Manager, slurm: Slurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second, removed = [queue(manager) for _ in range(3)]
    manager.chain.configure(enabled=True, paused=False)
    inserted = queue(manager)
    original = manager.store.ordered_ids()
    desired = [inserted["id"], removed["id"], second["id"]]
    assert manager.store.reorder(desired, original) == desired
    manager.remove_waiting(removed["id"])
    assert manager.store.ordered_ids() == [inserted["id"], second["id"]]
    dispatch(manager, slurm, monkeypatch, first["id"])
    assert manager.chain.data()["active_task_id"] == inserted["id"]
    assert not manager.store.get(removed["id"]).get("scheduler")
    assert len(slurm.submissions(dispatcher=False)) == 2


@pytest.mark.parametrize(
    "outcome,status",
    [
        ("FAILED", "failed"),
        ("TIMEOUT", "failed"),
        ("CANCELLED", "canceled"),
    ],
)
def test_terminal_compute_outcomes_advance_without_retry(
    manager: Manager,
    slurm: Slurm,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    status: str,
) -> None:
    first, second = queue(manager), queue(manager)
    manager.chain.configure(enabled=True, paused=False)
    if outcome == "CANCELLED":
        manager.cancel_task(first["id"])
        assert [
            execution.SCANCEL_COMMAND,
            task_job(manager, slurm, first["id"]).identifier,
        ] in slurm.calls
    dispatch(manager, slurm, monkeypatch, first["id"], outcome)
    assert manager.store.get(first["id"])["status"] == status
    assert manager.chain.data()["active_task_id"] == second["id"]
    assert len(slurm.submissions(dispatcher=False)) == 2
    assert manager.chain.data()["error"] == ""


def test_pause_survives_callback_and_resume_starts_next_task(
    manager: Manager, slurm: Slurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = queue(manager), queue(manager)
    manager.chain.configure(enabled=True, paused=False)
    manager.chain.configure(paused=True)
    dispatch(manager, slurm, monkeypatch, first["id"])
    manager.chain.recover()
    assert len(slurm.submissions(dispatcher=False)) == 1
    assert manager.store.ordered_ids() == [second["id"]]
    manager.chain.configure(paused=False)
    assert manager.chain.data()["active_task_id"] == second["id"]


def test_failed_dispatcher_is_replaced_without_resubmitting_compute(
    manager: Manager, slurm: Slurm
) -> None:
    task = queue(manager)
    manager.chain.configure(enabled=True, paused=False)
    original = dispatcher_record(manager)
    slurm.jobs[job_identifier(original)].state = "FAILED"
    manager.chain.recover()
    replacement = dispatcher_record(manager)
    assert replacement["job_id"] != original["job_id"]
    assert (
        f"{DEPENDENCY_OPTION}{task_job(manager, slurm, task['id']).identifier}"
        in replacement["command"]
    )
    for _ in range(3):
        manager.chain.recover()
    assert len(slurm.submissions(dispatcher=False)) == 1
    assert len(slurm.submissions(dispatcher=True)) == 2


@pytest.mark.parametrize("kind", ["compute", "dispatcher"])
def test_accepted_before_record_reconnects_by_unique_job_name(
    manager: Manager, slurm: Slurm, kind: str
) -> None:
    task = queue(manager)
    slurm.uncertain = kind
    manager.chain.configure(enabled=True, paused=False)
    unknown = (
        manager.store.get(task["id"])["scheduler"]
        if kind == "compute"
        else dispatcher_record(manager)
    )
    assert unknown["job_id"] is None
    assert unknown["state"] == execution.SlurmState.SUBMISSION_UNKNOWN
    manager.chain.recover()
    recovered = (
        manager.store.get(task["id"])["scheduler"]
        if kind == "compute"
        else dispatcher_record(manager)
    )
    assert recovered["job_id"] is not None
    assert (
        slurm.jobs[job_identifier(recovered)].name
        == execution.SLURM_JOB_PREFIX + unknown["token"]
    )
    for _ in range(3):
        manager.chain.recover()
    assert len(slurm.submissions(dispatcher=False)) == 1
    assert len(slurm.submissions(dispatcher=True)) == 1


@pytest.mark.parametrize("kind", ["compute", "dispatcher"])
def test_uncertain_submission_without_matches_blocks_retries(
    manager: Manager, slurm: Slurm, kind: str
) -> None:
    queue(manager)
    queue(manager)
    slurm.uncertain, slurm.accept_uncertain = kind, False
    manager.chain.configure(enabled=True, paused=False)
    submissions = sum(call[0] == execution.SBATCH_COMMAND for call in slurm.calls)
    for _ in range(3):
        manager.chain.recover()
    assert (
        sum(call[0] == execution.SBATCH_COMMAND for call in slurm.calls) == submissions
    )
    active_id = manager.chain.data()["active_task_id"]
    assert isinstance(active_id, str)
    record = (
        manager.store.get(active_id)["scheduler"]
        if kind == "compute"
        else dispatcher_record(manager)
    )
    assert record["job_id"] is None
    assert record["state"] == execution.SlurmState.SUBMISSION_UNKNOWN


def test_controller_outage_preserves_claims_without_duplicate_submissions(
    manager: Manager, slurm: Slurm
) -> None:
    first, second = queue(manager), queue(manager)
    manager.chain.configure(enabled=True, paused=False)
    before = dispatcher_record(manager)
    slurm.offline = True
    for _ in range(3):
        state = manager.chain.recover()
        assert state["error"]
    assert manager.chain.data()["dispatcher"] == before
    assert manager.chain.data()["active_task_id"] == first["id"]
    assert manager.store.ordered_ids() == [second["id"]]
    slurm.offline = False
    manager.chain.recover()
    assert manager.chain.data()["error"] == ""
    assert len(slurm.jobs) == 2


def test_two_stores_reject_conflicting_queue_order(manager: Manager) -> None:
    tasks = [queue(manager) for _ in range(3)]
    original = [task["id"] for task in tasks]
    other = persistence.Store(manager.config.artifacts / "tasks.sqlite", recover=False)
    try:
        assert other.ordered_ids() == original
        desired = list(reversed(original))
        assert manager.store.reorder(desired, original) == desired
        with pytest.raises(ValueError, match="changed|refresh"):
            other.reorder(original, original)
        assert other.ordered_ids() == desired
        assert other.reorder(original, desired) == original
        assert manager.store.ordered_ids() == original
    finally:
        other.db.close()


@pytest.mark.parametrize("indices", [(0, 0), (0,), (0, 1, 2)])
def test_reorder_rejects_duplicates_missing_and_unknown_ids(
    manager: Manager, indices: tuple[int, ...]
) -> None:
    tasks = [queue(manager) for _ in range(2)]
    original = [task["id"] for task in tasks]
    choices = [*original, "unknown-task"]
    with pytest.raises(ValueError):
        manager.store.reorder([choices[index] for index in indices], original)
    assert manager.store.ordered_ids() == original


def test_stale_removal_never_cancels_claimed_task(
    manager: Manager, slurm: Slurm
) -> None:
    task = queue(manager)
    manager.chain.configure(enabled=True, paused=False)
    before = manager.store.get(task["id"])
    with pytest.raises(ValueError, match="selected|refresh"):
        manager.remove_waiting(task["id"])
    assert manager.store.get(task["id"]) == before
    assert not any(call[0] == execution.SCANCEL_COMMAND for call in slurm.calls)


@pytest.fixture
def api(manager: Manager) -> FastAPI:
    app = FastAPI()
    app.state.manager = manager
    app.include_router(server.get_router())
    return app


def post(
    api: FastAPI, path: str, body: dict[str, object] | None = None
) -> httpx.Response:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=api)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.post(path, json=body)

    return asyncio.run(request())


def test_api_chain_start_recover_reorder_and_stale_removal(
    manager: Manager, slurm: Slurm, api: FastAPI
) -> None:
    tasks = [queue(manager) for _ in range(3)]
    original = [task["id"] for task in tasks]
    desired = list(reversed(original))
    ordered = post(api, "/api/queue/order", {"ids": desired, "expected": original})
    assert ordered.status_code == 200 and ordered.json() == desired
    started = post(api, "/api/queue/chain", {"enabled": True, "paused": False})
    assert started.status_code == 200
    assert started.json()["active_task_id"] == desired[0]
    assert post(api, "/api/queue/recover").status_code == 200
    stale = post(api, f"/api/queue/tasks/{desired[0]}/remove")
    assert stale.status_code == 400
    assert not any(call[0] == execution.SCANCEL_COMMAND for call in slurm.calls)
    assert post(api, f"/api/queue/tasks/{desired[1]}/remove").status_code == 200
    assert manager.store.ordered_ids() == [desired[2]]
    assert len(slurm.jobs) == 2


@pytest.mark.parametrize(
    "path,body,status",
    [
        ("/api/queue/order", {"ids": [], "expected": []}, 400),
        ("/api/queue/order", {"ids": []}, 422),
        ("/api/queue/chain", {"enabled": "invalid"}, 422),
    ],
)
def test_api_validation_preserves_waiting_queue(
    manager: Manager,
    slurm: Slurm,
    api: FastAPI,
    path: str,
    body: dict[str, object],
    status: int,
) -> None:
    task = queue(manager)
    assert post(api, path, body).status_code == status
    assert manager.store.ordered_ids() == [task["id"]]
    assert not slurm.jobs


@pytest.mark.parametrize("kind", ["compute", "dispatcher"])
def test_crash_after_acceptance_keeps_submission_intent(
    manager: Manager, slurm: Slurm, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    task = queue(manager)

    def crash_after_acceptance(
        arguments: list[str],
    ) -> subprocess.CompletedProcess[str]:
        result = slurm.command(arguments)
        if arguments[0] == execution.SBATCH_COMMAND:
            dispatcher = any(arg.startswith(DEPENDENCY_OPTION) for arg in arguments)
            if dispatcher == (kind == "dispatcher"):
                raise SystemExit("process died before recording returned job ID")
        return result

    with monkeypatch.context() as context:
        context.setattr(scheduler, "command", crash_after_acceptance)
        with pytest.raises(SystemExit):
            manager.chain.configure(enabled=True, paused=False)
    record = (
        manager.store.get(task["id"])["scheduler"]
        if kind == "compute"
        else dispatcher_record(manager)
    )
    assert record["state"] == execution.SlurmState.SUBMITTING
    assert record["job_id"] is None
    manager.chain.recover()
    assert len(slurm.submissions(dispatcher=False)) == 1
    assert len(slurm.submissions(dispatcher=True)) == 1
    assert manager.store.get(task["id"])["scheduler"]["job_id"] is not None
    assert job_identifier(dispatcher_record(manager)) is not None


def test_crash_after_active_reservation_recovers_without_losing_task(
    manager: Manager, slurm: Slurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = queue(manager)

    def crash_before_submit(*args: object, **kwargs: object) -> None:
        raise SystemExit("process died after active reservation")

    with monkeypatch.context() as context:
        context.setattr(manager.scheduler, "submit", crash_before_submit)
        with pytest.raises(SystemExit):
            manager.chain.configure(enabled=True, paused=False)
    assert manager.chain.data()["active_task_id"] == task["id"]
    assert not manager.store.get(task["id"]).get("scheduler")
    assert not slurm.jobs
    manager.chain.recover()
    assert manager.chain.data()["active_task_id"] == task["id"]
    assert len(slurm.jobs) == 2


def test_empty_chain_accepts_new_work_after_pause_and_resume(
    manager: Manager, slurm: Slurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager.chain.configure(enabled=True, paused=False)
    assert manager.chain.data()["active_task_id"] is None
    assert not slurm.jobs
    first = queue(manager)
    manager.chain.recover()
    dispatch(manager, slurm, monkeypatch, first["id"])
    assert manager.chain.data()["active_task_id"] is None
    manager.chain.configure(paused=True)
    second = queue(manager)
    manager.chain.recover()
    assert not manager.store.get(second["id"]).get("scheduler")
    manager.chain.configure(paused=False)
    assert manager.chain.data()["active_task_id"] == second["id"]
    assert len(slurm.submissions(dispatcher=False)) == 2


def test_dispatcher_store_does_not_recover_server_local_running_task(
    manager: Manager,
) -> None:
    task = queue(manager)
    manager.store.update(
        task["id"],
        target=execution.ExecutionMode.LOCAL,
        status=execution.TaskStatus.RUNNING,
    )
    other = persistence.Store(manager.config.artifacts / "tasks.sqlite", recover=False)
    try:
        assert other.get(task["id"])["status"] == execution.TaskStatus.RUNNING
        assert manager.store.get(task["id"])["status"] == execution.TaskStatus.RUNNING
    finally:
        other.db.close()


def test_mutation_lock_excludes_a_separate_process(manager: Manager) -> None:
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from dashboard.core.jobs.persistence import Store\n"
        "print('ready', flush=True)\n"
        "store = Store(Path(sys.argv[1]), recover=False)\n"
        "with store.mutation_lock:\n"
        "    print('acquired', flush=True)\n"
        "store.db.close()\n"
    )
    with manager.store.mutation_lock:
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(manager.config.artifacts / "tasks.sqlite"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        # Construction itself cooperates with the shared filesystem lock.
        assert not select.select([child.stdout], [], [], 0.1)[0]
        assert child.poll() is None
    output, error = child.communicate(timeout=1)
    assert child.returncode == 0, error
    assert output.strip() == "acquired"


@pytest.mark.parametrize("replacement", [False, True])
def test_dispatcher_history_restores_metadata_after_crash(
    manager: Manager,
    slurm: Slurm,
    monkeypatch: pytest.MonkeyPatch,
    replacement: bool,
) -> None:
    task = queue(manager)
    if replacement:
        manager.chain.configure(enabled=True, paused=False)
        old = dispatcher_record(manager)
        slurm.jobs[job_identifier(old)].state = "FAILED"
    previous_token = dispatcher_record(manager)["token"] if replacement else None
    save_metadata = manager.store.set_chain

    def crash_before_metadata(values: models.ChainUpdate) -> models.ChainState:
        record = values.get("dispatcher")
        if (
            record
            and record["job_id"] is not None
            and record["state"] == execution.SlurmState.PENDING
            and record["token"] != previous_token
        ):
            raise SystemExit("task history saved; chain metadata not committed")
        return save_metadata(values)

    with monkeypatch.context() as context:
        context.setattr(manager.store, "set_chain", crash_before_metadata)
        with pytest.raises(SystemExit):
            if replacement:
                manager.chain.recover()
            else:
                manager.chain.configure(enabled=True, paused=False)
    latest = manager.store.get(task["id"])["chain_dispatchers"][-1]
    assert latest["job_id"] is not None
    assert manager.chain.data()["dispatcher"] != latest
    submissions = len(slurm.jobs)
    manager.chain.recover()
    assert job_identifier(dispatcher_record(manager)) == latest["job_id"]
    assert len(slurm.jobs) == submissions
    assert len(slurm.submissions(dispatcher=False)) == 1


def test_monitor_surfaces_dispatcher_failure_without_advancing(
    manager: Manager, slurm: Slurm
) -> None:
    queue(manager)
    queue(manager)
    manager.chain.configure(enabled=True, paused=False)
    dispatcher = dispatcher_record(manager)
    slurm.jobs[job_identifier(dispatcher)].state = "FAILED"
    manager.scheduler.poll(force=True)
    state = manager.chain.data()
    assert dispatcher_record(manager)["state"] == "FAILED"
    assert "Dispatcher" in state["error"]
    assert len(slurm.submissions(dispatcher=False)) == 1
    assert len(slurm.submissions(dispatcher=True)) == 1


def test_mode_cannot_change_while_chain_jobs_are_outstanding(
    manager: Manager, slurm: Slurm
) -> None:
    queue(manager)
    manager.chain.configure(enabled=True, paused=False)
    with pytest.raises(ValueError, match="outstanding"):
        manager.chain.configure(enabled=False)
    assert manager.chain.data()["enabled"]
    manager.chain.configure(paused=True)
    with pytest.raises(ValueError, match="outstanding"):
        manager.chain.configure(enabled=False)
    assert len(slurm.jobs) == 2


def test_confirmed_compute_rejection_allows_advancement(
    manager: Manager, slurm: Slurm
) -> None:
    first, second = queue(manager), queue(manager)
    slurm.uncertain, slurm.accept_uncertain = "compute", False
    manager.chain.configure(enabled=True, paused=False)
    manager.scheduler.mark_rejected(manager.store.get(first["id"]))
    assert manager.store.get(first["id"])["scheduler"]["state"] == "FAILED"
    manager.chain.recover()
    assert manager.chain.data()["active_task_id"] == second["id"]
    assert manager.chain.data()["error"] == ""
    assert len(slurm.submissions(dispatcher=False)) == 1


def test_dispatcher_rejection_requires_absence_and_matching_token(
    manager: Manager, slurm: Slurm
) -> None:
    queue(manager)
    slurm.uncertain = "dispatcher"
    manager.chain.configure(enabled=True, paused=False)
    record = dispatcher_record(manager)
    with pytest.raises(ValueError, match="changed"):
        manager.chain.mark_dispatcher_rejected("stale-token")
    with pytest.raises(ValueError, match="matching"):
        manager.chain.mark_dispatcher_rejected(record["token"])
    assert manager.chain.data()["dispatcher"] == record
    manager.chain.recover()
    assert len(slurm.submissions(dispatcher=True)) == 1


def test_confirmed_dispatcher_rejection_recreates_only_dispatcher(
    manager: Manager, slurm: Slurm, api: FastAPI
) -> None:
    task = queue(manager)
    slurm.uncertain, slurm.accept_uncertain = "dispatcher", False
    manager.chain.configure(enabled=True, paused=False)
    record = dispatcher_record(manager)
    result = post(api, "/api/queue/dispatcher/rejected", {"token": record["token"]})
    assert result.status_code == 200
    assert result.json()["dispatcher"]["state"] == "FAILED"
    manager.chain.recover()
    assert manager.chain.data()["error"] == ""
    history = manager.store.get(task["id"])["chain_dispatchers"]
    assert [item["state"] for item in history] == ["FAILED", "PENDING"]
    assert len(slurm.submissions(dispatcher=False)) == 1
    assert len(slurm.submissions(dispatcher=True)) == 1


def test_setup_failure_advances_to_next_task_exactly_once(
    manager: Manager, slurm: Slurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = queue(manager), queue(manager)
    submit = manager.scheduler.submit
    attempts: list[str] = []

    def fail_first_setup(
        tasks: list[models.TaskRecord], *, chain: bool = False
    ) -> None:
        task_id = tasks[0]["id"]
        attempts.append(task_id)
        assert not manager.store.get(task_id).get("scheduler")
        if task_id == first["id"]:
            raise OSError("Unable to write computation script")
        submit(tasks, chain=chain)

    monkeypatch.setattr(manager.scheduler, "submit", fail_first_setup)
    manager.chain.configure(enabled=True, paused=False)
    for _ in range(3):
        manager.chain.recover()
    failed = manager.store.get(first["id"])
    assert failed["status"] == execution.TaskStatus.FAILED
    assert failed["ended"] and "Unable to write" in failed["error"]
    assert not failed.get("scheduler")
    assert attempts == [first["id"], second["id"]]
    assert manager.chain.data()["active_task_id"] == second["id"]
    assert manager.chain.data()["error"] == ""
    assert len(slurm.submissions(dispatcher=False)) == 1
    assert len(slurm.submissions(dispatcher=True)) == 1


def grouped_ram_request() -> models.BatchReviewRequest:
    return {
        "kind": "estimate_ram",
        "candidates": ["500"],
        "target": "slurm",
        "runtime": "native",
        "memory_gib": 64,
    }


@pytest.mark.parametrize("processes", [[16, 4, 1], [1]])
def test_chain_rejects_grouped_ram_review_and_submission(
    manager: Manager, slurm: Slurm, processes: list[int]
) -> None:
    request = grouped_ram_request()
    request["estimate_processes"] = processes
    preview = manager.batches.review(**request)
    manager.chain.configure(enabled=True, paused=True)
    with pytest.raises(ValueError, match="Grouped RAM.*chain mode"):
        manager.batches.review(**request)
    with pytest.raises(ValueError, match="Grouped RAM.*chain mode"):
        manager.batches.submit(
            {
                **request,
                "estimate_processes": processes,
                "expected": preview["entries"],
                "image_hash": None,
            }
        )
    assert not manager.store.ordered_ids()
    assert not slurm.jobs


def test_pending_group_blocks_chain_enable_and_retains_independent_grouping(
    manager: Manager, slurm: Slurm
) -> None:
    processes = [16, 4, 1]
    request = grouped_ram_request()
    request["estimate_processes"] = processes
    preview = manager.batches.review(**request)
    batch = manager.batches.submit(
        {
            **request,
            "estimate_processes": processes,
            "expected": preview["entries"],
            "image_hash": None,
        }
    )
    waiting = manager.store.ordered_ids()
    with pytest.raises(ValueError, match="grouped RAM.*before enabling"):
        manager.chain.configure(enabled=True, paused=False)
    assert not manager.chain.data()["enabled"]
    assert manager.store.ordered_ids() == waiting
    assert not slurm.jobs
    manager.scheduler.submit_waiting()
    records = [manager.store.get(task_id)["scheduler"] for task_id in batch["task_ids"]]
    assert len({job_identifier(record) for record in records}) == 1
    assert len(slurm.submissions(dispatcher=False)) == 1
    assert not slurm.submissions(dispatcher=True)
