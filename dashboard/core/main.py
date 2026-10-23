"""Startup workflows; CLI parsing belongs to dashboard.app.cli."""

import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from dashboard.core import config as configuration
from dashboard.core.config import Config
from dashboard.core.jobs.processes import terminate
from dashboard.core.snapshot import export


def frontend_environment(mode: str) -> dict[str, str]:
    return {**os.environ, "PFAS_DATA_MODE": mode}


def export_destination(output: str | None) -> Path:
    destination = Path(output or configuration.ROOT / "web/public").resolve()
    forbidden = (
        configuration.ROOT / "web/frontend/dist",
        configuration.ROOT / "web/snapshot",
        configuration.ROOT,
    )
    if destination in forbidden or destination in configuration.ROOT.parents:
        raise ValueError("Export destination must be a separate site directory")
    return destination


def export_snapshot(frontend: Path, destination: Path) -> None:
    # Build separately so exporting never overwrites the live frontend.
    with TemporaryDirectory(prefix="pfas-snapshot-build-") as build:
        subprocess.run(
            ["npm", "run", "build", "--", "--outDir", build],
            cwd=frontend,
            check=True,
            env=frontend_environment("snapshot"),
        )
        export(Config(), configuration.ROOT / "web/snapshot")
        shutil.copytree(build, destination, dirs_exist_ok=True)
    # Copy the exported dataset alongside the independently built page.
    shutil.copytree(
        configuration.ROOT / "web/snapshot",
        destination / "snapshot",
        dirs_exist_ok=True,
    )
    print(f"Snapshot site exported to {destination}; no tasks started")


def start_vite(frontend: Path, host: str, port: int) -> subprocess.Popen[bytes]:
    environment = {
        **frontend_environment("live"),
        "PFAS_API_PORT": str(port),
        "PFAS_API_HOST": host,
    }
    return subprocess.Popen(
        ["npm", "run", "dev", "--", "--host", host],
        cwd=frontend,
        start_new_session=True,
        env=environment,
    )


def serve(frontend: Path, host: str, port: int, dev: bool) -> None:
    import uvicorn

    from dashboard.app.server import create_app

    # Development uses Vite; production serves an already built page in FastAPI.
    vite = start_vite(frontend, host, port) if dev else None
    if not dev:
        subprocess.run(
            ["npm", "run", "build"],
            cwd=frontend,
            check=True,
            env=frontend_environment("live"),
        )
    try:
        config = Config()
        print(f"Preparation interpreter: {config.prepare_python}", flush=True)
        uvicorn.run(create_app(config), host=host, port=port)
    finally:
        if vite:
            terminate(vite)


def main(
    *,
    action: str = "serve",
    dev: bool = False,
    host: str = "127.0.0.1",
    port: int = 8000,
    output: str | None = None,
) -> None:
    frontend = configuration.ROOT / "web/frontend"
    subprocess.run(["npm", "ci", "--no-audit", "--no-fund"], cwd=frontend, check=True)
    if action == "export":
        export_snapshot(frontend, export_destination(output))
    else:
        serve(frontend, host, port, dev)
