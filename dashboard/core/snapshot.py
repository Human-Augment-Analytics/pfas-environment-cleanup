import json
import shutil
import sqlite3
from pathlib import Path

from dashboard.core import candidates as candidate_data
from dashboard.core.chemistry.geometry import prepared_geometries
from dashboard.core.config import Config
from dashboard.core.jobs.persistence import now

SNAPSHOT_TASK_FIELDS = (
    "id",
    "kind",
    "candidate",
    "system",
    "processes",
    "status",
    "created",
    "started",
    "ended",
    "exit_code",
    "evidence",
    "elapsed_seconds",
    "runtime",
    "resources",
    "usage",
    "image_hash",
    "version",
    "source_hash",
    "executed_input_hash",
    "pseudopotentials",
    "ram_estimate",
)


def readonly_tasks(database: Path) -> list[dict]:
    if not database.exists():
        return []
    # Export must not construct Store, whose startup recovers pending work.
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        return [
            json.loads(row[0])
            for row in db.execute("SELECT data FROM tasks ORDER BY rowid DESC")
        ]


def copy_diagram(config: Config, task: dict, output: Path) -> dict[str, str]:
    source = config.artifacts / task["id"] / "diagram.png"
    if (
        task["kind"] != "diagram"
        or task["status"] != "succeeded"
        or not source.exists()
    ):
        return {}
    name = task["id"] + ".png"
    shutil.copyfile(source, output / name)
    return {"diagram.png": "snapshot/" + name}


def snapshot_task(config: Config, task: dict, output: Path) -> dict:
    item = {key: task[key] for key in SNAPSHOT_TASK_FIELDS if key in task}
    item["geometries"] = prepared_geometries(
        task, lambda name: config.artifacts / task["id"] / name
    )
    item["artifacts"] = copy_diagram(config, task, output)
    return item


def export(config: Config, output: Path | str) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tasks = [
        snapshot_task(config, task, output)
        for task in readonly_tasks(config.artifacts / "tasks.sqlite")
    ]
    data = {
        "mode": "snapshot",
        "timestamp": now(),
        "candidates": candidate_data.load(),
        "reference": candidate_data.TFA,
        "tasks": tasks,
    }
    (output / "data.json").write_text(json.dumps(data, indent=2))
    return data
