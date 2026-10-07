import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StrictInt

from .candidates import TFA, load
from .config import ROOT, Config
from .tasks import Manager


class Action(BaseModel):
    kind: str
    candidate: str
    system: str = "candidate"
    processes: StrictInt = 1
    target: str = "local"
    timeout: float | None = None
    expected_hash: str | None = None


def create_app(config=None):
    manager = Manager(config or Config(), {c["id"]: c for c in [*load(), TFA]})

    @asynccontextmanager
    async def lifespan(app):
        manager.start()
        yield
        manager.close()

    app = FastAPI(lifespan=lifespan)
    app.state.manager = manager

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if request.headers.get("sec-fetch-site") == "cross-site" or (
                origin and urlparse(origin).netloc != request.headers.get("host")
            ):
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    {"detail": "Same-origin actions only"}, status_code=403
                )
        return await call_next(request)

    @app.get("/api/data")
    def data():
        return manager.data()

    @app.post("/api/preview")
    def preview(action: Action):
        return checked(
            lambda: manager.preview(
                action.candidate, action.system, action.processes, action.target
            )
        )

    @app.post("/api/tasks")
    def queue(action: Action):
        return checked(lambda: manager.queue(**action.model_dump()))

    @app.post("/api/diagrams")
    def bulk():
        successful = {
            t["candidate"]
            for t in manager.store.tasks()
            if t["kind"] == "diagram"
            and t["status"] == "succeeded"
            and manager.store.artifact(t["id"] + "-diagram.png")
            and Path(manager.store.artifact(t["id"] + "-diagram.png")).exists()
        }
        return [
            manager.queue("diagram", c)
            for c in manager.candidates
            if c != "tfa" and c not in successful
        ]

    @app.post("/api/tasks/{id}/stop")
    def stop(id: str):
        return checked(lambda: manager.cancel_task(id))

    @app.get("/api/artifacts/{id}")
    def artifact(id: str):
        path = manager.store.artifact(id)
        if not path or not Path(path).is_file():
            raise HTTPException(404)
        return FileResponse(path)

    static = ROOT / "web/frontend/dist"
    if static.exists():
        app.mount("/", StaticFiles(directory=static, html=True), name="frontend")
    return app


def checked(fn):
    try:
        return fn()
    except NotImplementedError as error:
        raise HTTPException(501, str(error)) from error
    except (ValueError, OSError, StopIteration, subprocess.SubprocessError) as error:
        raise HTTPException(400, str(error)) from error
