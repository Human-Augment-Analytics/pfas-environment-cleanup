import json
import shutil
from pathlib import Path

from .candidates import TFA, load
from .persistence import now


def export(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tasks = []
    database = config.artifacts / "tasks.sqlite"
    if database.exists():
        # Read-only connection: exporting must not recover or mutate task state.
        import sqlite3

        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
            raw = [
                json.loads(row[0])
                for row in db.execute("SELECT data FROM tasks ORDER BY rowid DESC")
            ]
        for task in raw:
            item = {
                k: task[k]
                for k in (
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
                )
                if k in task
            }
            item["artifacts"] = {}
            if task["kind"] == "diagram" and task["status"] == "succeeded":
                source = config.artifacts / task["id"] / "diagram.png"
                if source.exists():
                    name = task["id"] + ".png"
                    shutil.copyfile(source, output / name)
                    item["artifacts"]["diagram.png"] = "snapshot/" + name
            tasks.append(item)
    data = {
        "mode": "snapshot",
        "timestamp": now(),
        "candidates": load(),
        "reference": TFA,
        "tasks": tasks,
    }
    (output / "data.json").write_text(json.dumps(data, indent=2))
    return data
