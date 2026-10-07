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
        return next(t for t in self.tasks() if t["id"] == id)

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
