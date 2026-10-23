import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from dashboard.core import models


def now() -> str:
    return datetime.now(UTC).isoformat()


class Store:
    def __init__(self, path: Path) -> None:
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.create_schema()
        self.recover_running()
        recovered_tasks = self.tasks()
        for batch in self.batches():
            self.recover_batch(batch, recovered_tasks)

    def create_schema(self) -> None:
        """Keep the schema's linear SQL declarations together; no recovery runs here."""
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, data TEXT NOT NULL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS artifacts (id TEXT PRIMARY KEY, path TEXT NOT NULL)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS task_action ON tasks (json_extract(data, '$.kind'), json_extract(data, '$.candidate'), json_extract(data, '$.system'), json_extract(data, '$.status'))"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS batches (id TEXT PRIMARY KEY, data TEXT NOT NULL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS settings (id TEXT PRIMARY KEY, data TEXT NOT NULL)"
        )
        self.db.commit()

    def recover_running(self) -> None:
        # Local processes died with the server; independent Slurm jobs did not.
        for task in self.tasks():
            if task["status"] == "running" and task.get("target") != "slurm":
                self.update(
                    task["id"],
                    status="interrupted",
                    ended=now(),
                    error="Server restarted; retry explicitly",
                )

    def recovered_entries(self, batch: dict, tasks: list[models.TaskRecord]) -> dict:
        entries = {
            (entry["candidate"], entry.get("processes", 1)): entry
            for entry in batch["entries"]
        }
        # Tasks may have committed before their batch entry was saved.
        for task in tasks:
            if task.get("batch_id") == batch["id"]:
                entries[(task["candidate"], task.get("processes", 1))] = {
                    "candidate": task["candidate"],
                    "processes": task.get("processes", 1),
                    "status": "queued",
                    "task_id": task["id"],
                    "reason": "",
                }
        return entries

    def batch_pairs(self, batch: dict) -> list[tuple[str, int]]:
        ranks = batch.get("estimate_processes") or [
            batch.get("resources", {}).get("cpus", 1)
        ]
        return [
            (candidate, rank) for candidate in batch["candidates"] for rank in ranks
        ]

    def recover_batch(self, batch: dict, tasks: list[models.TaskRecord]) -> None:
        entries = self.recovered_entries(batch, tasks)
        pairs = self.batch_pairs(batch)
        # Preserve accepted work and explicitly flag entries lost during acceptance.
        for candidate, rank in pairs:
            entries.setdefault(
                (candidate, rank),
                {
                    "candidate": candidate,
                    "status": "unavailable",
                    "reason": "Batch acceptance interrupted; review again",
                },
            )
        batch["entries"] = [entries[pair] for pair in pairs]
        batch["accepting"] = False
        known = set(batch["task_ids"]) | {
            task["id"] for task in tasks if task.get("batch_id") == batch["id"]
        }
        batch["task_ids"] = [
            entry["task_id"]
            for entry in batch["entries"]
            if entry.get("task_id") in known
        ]
        self.put_batch(batch)

    def tasks(self) -> list[models.TaskRecord]:
        with self.lock:
            return [
                json.loads(row[0])
                for row in self.db.execute("SELECT data FROM tasks ORDER BY rowid DESC")
            ]

    def get(self, id: str) -> models.TaskRecord:
        with self.lock:
            row = self.db.execute("SELECT data FROM tasks WHERE id=?", (id,)).fetchone()
        if row is None:
            raise ValueError("Unknown task")
        return json.loads(row[0])

    def prepared(self, candidate: str, system: str) -> list[models.TaskRecord]:
        # Resolve defaults without decoding the entire history for every queued job.
        sql = "SELECT data FROM tasks WHERE json_extract(data, '$.kind')='prepare' AND json_extract(data, '$.status')='succeeded'"
        parameters = ()
        if system != "tfa":
            sql += " AND json_extract(data, '$.candidate')=?"
            parameters = (candidate,)
        with self.lock:
            return [
                json.loads(row[0])
                for row in self.db.execute(sql + " ORDER BY rowid DESC", parameters)
            ]

    def active(
        self, kind: str, candidate: str, system: str
    ) -> models.TaskRecord | None:
        with self.lock:
            row = self.db.execute(
                "SELECT data FROM tasks WHERE json_extract(data, '$.kind')=? AND json_extract(data, '$.candidate')=? AND json_extract(data, '$.system')=? AND json_extract(data, '$.status') IN ('queued', 'running') LIMIT 1",
                (kind, candidate, system),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def next_queued(self) -> models.TaskRecord | None:
        with self.lock:
            row = self.db.execute(
                "SELECT data FROM tasks WHERE json_extract(data, '$.status')='queued' ORDER BY rowid LIMIT 1"
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, task: dict | models.TaskRecord) -> models.TaskRecord:
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO tasks VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET data=excluded.data",
                (task["id"], json.dumps(task)),
            )
        return cast(models.TaskRecord, task)

    def update(self, id: str, **values: object) -> models.TaskRecord:
        with self.lock:
            return self.put({**self.get(id), **values})

    def register(self, id: str, path: Path) -> str:
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO artifacts VALUES (?, ?)", (id, str(path))
            )
        return "/api/artifacts/" + id

    def artifact(self, id: str) -> str | None:
        with self.lock:
            row = self.db.execute(
                "SELECT path FROM artifacts WHERE id=?", (id,)
            ).fetchone()
        return row[0] if row else None

    def batches(self) -> list[dict]:
        with self.lock:
            return [
                json.loads(row[0])
                for row in self.db.execute(
                    "SELECT data FROM batches ORDER BY rowid DESC"
                )
            ]

    def batch(self, id: str) -> dict | None:
        with self.lock:
            row = self.db.execute(
                "SELECT data FROM batches WHERE id=?", (id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put_batch(self, batch: dict) -> dict:
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO batches VALUES (?, ?)",
                (batch["id"], json.dumps(batch)),
            )
        return batch

    def settings(self, defaults: dict) -> dict:
        with self.lock:
            row = self.db.execute(
                "SELECT data FROM settings WHERE id='queue'"
            ).fetchone()
        return {**defaults, **(json.loads(row[0]) if row else {})}

    def set_settings(self, settings: dict) -> dict:
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO settings VALUES ('queue', ?)",
                (json.dumps(settings),),
            )
        return settings

    def queued(self) -> list[models.TaskRecord]:
        return [t for t in reversed(self.tasks()) if t["status"] == "queued"]
