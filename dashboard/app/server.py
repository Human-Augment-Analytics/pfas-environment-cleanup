import subprocess
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StrictInt

from dashboard.core import candidates as candidate_data
from dashboard.core import config as configuration
from dashboard.core import models
from dashboard.core.config import Config
from dashboard.core.jobs.processes import tail
from dashboard.core.jobs.tasks import Manager


class Action(BaseModel):
    kind: str
    candidate: str
    system: str = "candidate"
    processes: StrictInt = 1
    target: str = "local"
    timeout: float | None = None
    expected_hash: str | None = None
    input_id: str | None = None
    preview_id: str | None = None
    expected_pseudos: dict[str, str] | None = None
    runtime: str = "native"
    memory_gib: float = 4
    scheduler: models.SchedulerSettings | None = None


class BatchAction(BaseModel):
    target: str = "local"
    candidates: list[str]
    kind: str
    system: str = "candidate"
    runtime: str = "native"
    processes: StrictInt = 1
    memory_gib: float = 4
    scheduler: models.SchedulerSettings | None = None
    estimate_processes: list[StrictInt] | None = None
    timeout: float | None = None
    retry: bool = False
    rerun_completed: bool = False
    id: str | None = None
    image_hash: str | None = None
    expected: list[dict] = Field(default_factory=list)


class QueueSettings(BaseModel):
    paused: bool | None = None
    concurrency: StrictInt | None = None
    cpus: StrictInt | None = None
    memory_bytes: StrictInt | None = None


class SlurmExport(BaseModel):
    candidates: list[str]
    include_tfa: bool = False
    processes: StrictInt = 8
    memory_gib: float = 16
    memory_mode: str = "estimate"
    headroom_percent: float = 50
    walltime_hours: StrictInt = 18
    partition: str = "ice-cpu"
    account: str = "coc"
    qos: str = "coc-ice"
    expected: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    manager = app.state.manager
    manager.start()
    try:
        yield
    finally:
        manager.close()


async def same_origin(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if request.headers.get("sec-fetch-site") == "cross-site" or (
            origin and urlparse(origin).netloc != request.headers.get("host")
        ):
            from fastapi.responses import JSONResponse

            return JSONResponse({"detail": "Same-origin actions only"}, status_code=403)
    return await call_next(request)


def data(request: Request) -> dict:
    manager = manager_for(request)
    return manager.data()


def preview(action: Action, request: Request) -> dict:
    manager = manager_for(request)
    return checked(
        lambda: manager.preview(
            action.candidate,
            action.system,
            action.processes,
            action.target,
            action.runtime,
            action.input_id,
            action.kind,
            scheduler=action.scheduler,
            memory_gib=action.memory_gib,
            timeout=action.timeout,
        )
    )


def queue(action: Action, request: Request) -> dict:
    manager = manager_for(request)
    return checked(lambda: manager.queue(**action.model_dump()))


def batch_preview(action: BatchAction, request: Request) -> dict:
    manager = manager_for(request)
    review = checked(lambda: manager.batches.review(**action.model_dump()))
    # Bulk review has no geometry viewer; keep coordinates in single-job previews.
    review["entries"] = [
        {key: value for key, value in entry.items() if key != "geometry"}
        for entry in review["entries"]
    ]
    return review


def batch_submit(action: BatchAction, request: Request) -> dict:
    manager = manager_for(request)
    return checked(lambda: manager.batches.submit(action.model_dump()))


def slurm_preview(action: SlurmExport, request: Request) -> dict:
    manager = manager_for(request)
    from dashboard.core.slurm.export import review

    return checked(lambda: review(manager, **action.model_dump(exclude={"expected"})))


def slurm_export(action: SlurmExport, request: Request) -> Response:
    manager = manager_for(request)
    from dashboard.core.slurm.export import bundle

    content = checked(lambda: bundle(manager, **action.model_dump()))
    return Response(
        content,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="pfas-slurm-batch.zip"'},
    )


def queue_view(request: Request) -> dict:
    manager = manager_for(request)
    return manager.queue_data()


def queue_settings(settings: QueueSettings, request: Request) -> dict:
    manager = manager_for(request)
    return checked(lambda: manager.configure(settings.model_dump(exclude_none=True)))


def queue_clear(request: Request) -> dict:
    manager = manager_for(request)
    return manager.clear_queue()


def batch_cancel(id: str, request: Request) -> dict[str, str]:
    manager = manager_for(request)
    with manager.guard:
        batch = manager.store.batch(id)
        if not batch:
            raise HTTPException(404, "Unknown batch")
        for task_id in batch["task_ids"]:
            manager.cancel_task(task_id)
    return {"status": "stop requested"}


def batch_retry(id: str, request: Request) -> dict:
    manager = manager_for(request)
    return checked(lambda: manager.batches.retry(id))


def task_logs(id: str, request: Request) -> dict[str, str]:
    manager = manager_for(request)
    return checked(
        lambda: {
            name: tail(
                manager.config.artifacts / manager.store.get(id)["id"] / (name + ".log")
            )
            for name in ("entrypoint", "stdout", "stderr")
        }
    )


def bulk(request: Request) -> list[dict]:
    manager = manager_for(request)
    successful = {
        t["candidate"]
        for t in manager.store.tasks()
        if t["kind"] == "diagram"
        and t["status"] == "succeeded"
        and (diagram_path := manager.store.artifact(t["id"] + "-diagram.png"))
        and Path(diagram_path).exists()
    }
    return [
        manager.queue("diagram", c)
        for c in manager.candidates
        if c != "tfa" and c not in successful
    ]


def stop(id: str, request: Request) -> None:
    manager = manager_for(request)
    return checked(lambda: manager.cancel_task(id))


def mark_rejected(id: str, request: Request) -> None:
    manager = manager_for(request)
    return checked(lambda: manager.scheduler.mark_rejected(manager.store.get(id)))


def artifact(id: str, request: Request) -> FileResponse:
    manager = manager_for(request)
    path = manager.store.artifact(id)
    if not path or not Path(path).is_file():
        raise HTTPException(404)
    return FileResponse(path)


ROUTES = (
    ("GET", "/api/data", data),
    ("POST", "/api/preview", preview),
    ("POST", "/api/tasks", queue),
    ("POST", "/api/batches/preview", batch_preview),
    ("POST", "/api/batches", batch_submit),
    ("POST", "/api/slurm/preview", slurm_preview),
    ("POST", "/api/slurm/export", slurm_export),
    ("GET", "/api/queue", queue_view),
    ("POST", "/api/queue/settings", queue_settings),
    ("POST", "/api/queue/clear", queue_clear),
    ("POST", "/api/batches/{id}/cancel", batch_cancel),
    ("GET", "/api/batches/{id}/retry", batch_retry),
    ("GET", "/api/tasks/{id}/logs", task_logs),
    ("POST", "/api/diagrams", bulk),
    ("POST", "/api/tasks/{id}/stop", stop),
    ("POST", "/api/tasks/{id}/mark-rejected", mark_rejected),
    ("GET", "/api/artifacts/{id}", artifact),
)


def manager_for(request: Request) -> Manager:
    return request.app.state.manager


def get_router() -> APIRouter:
    router = APIRouter()
    for method, path, endpoint in ROUTES:
        router.add_api_route(path, endpoint, methods=[method])
    return router


def create_app(config: Config | None = None) -> FastAPI:
    manager = Manager(
        config or Config(),
        {
            candidate["id"]: candidate
            for candidate in [*candidate_data.load(), candidate_data.TFA]
        },
    )
    app = FastAPI(lifespan=lifespan)
    app.state.manager = manager
    app.middleware("http")(same_origin)
    app.include_router(get_router())
    # The same origin serves UI and API, so forwarding needs only one port.
    static = configuration.ROOT / "web/frontend/dist"
    if static.exists():
        app.mount("/", StaticFiles(directory=static, html=True), name="frontend")
    return app


def checked[T](fn: Callable[[], T]) -> T:
    try:
        return fn()
    except NotImplementedError as error:
        raise HTTPException(501, str(error)) from error
    except (ValueError, OSError, StopIteration, subprocess.SubprocessError) as error:
        raise HTTPException(400, str(error)) from error
