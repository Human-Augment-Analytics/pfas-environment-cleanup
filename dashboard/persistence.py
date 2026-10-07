import json
import sqlite3
import threading
from datetime import UTC, datetime


def now():
    return datetime.now(UTC).isoformat()


class Store:
    def __init__(self, path):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, data TEXT NOT NULL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS artifacts (id TEXT PRIMARY KEY, path TEXT NOT NULL)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS task_action ON tasks (json_extract(data, '$.kind'), json_extract(data, '$.candidate'), json_extract(data, '$.system'), json_extract(data, '$.status'))"
        )
        for task in self.tasks():
            if task["status"] in ("queued", "running"):
                self.update(
                    task["id"],
                    status="interrupted",
                    ended=now(),
                    error="Server restarted; retry explicitly",
                )

    def tasks(self):
        with self.lock:
            return [
                json.loads(row[0])
                for row in self.db.execute("SELECT data FROM tasks ORDER BY rowid DESC")
            ]

    def get(self, id):
        with self.lock:
            row = self.db.execute("SELECT data FROM tasks WHERE id=?", (id,)).fetchone()
        if row is None:
            raise ValueError("Unknown task")
        return json.loads(row[0])

    def active(self, kind, candidate, system):
        with self.lock:
            row = self.db.execute(
                "SELECT data FROM tasks WHERE json_extract(data, '$.kind')=? AND json_extract(data, '$.candidate')=? AND json_extract(data, '$.system')=? AND json_extract(data, '$.status') IN ('queued', 'running') LIMIT 1",
                (kind, candidate, system),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def next_queued(self):
        with self.lock:
            row = self.db.execute(
                "SELECT data FROM tasks WHERE json_extract(data, '$.status')='queued' ORDER BY rowid LIMIT 1"
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, task):
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO tasks VALUES (?, ?)",
                (task["id"], json.dumps(task)),
            )
        return task

    def update(self, id, **values):
        with self.lock:
            return self.put({**self.get(id), **values})

    def register(self, id, path):
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO artifacts VALUES (?, ?)", (id, str(path))
            )
        return "/api/artifacts/" + id

    def artifact(self, id):
        with self.lock:
            row = self.db.execute(
                "SELECT path FROM artifacts WHERE id=?", (id,)
            ).fetchone()
        return row[0] if row else None
