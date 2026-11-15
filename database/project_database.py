"""SQLite metadata: projects, processing history, voice assignments."""
from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

from config.settings import data_dir

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects(
  id TEXT PRIMARY KEY, name TEXT, video_path TEXT, project_dir TEXT, status TEXT,
  created REAL, updated REAL);
CREATE TABLE IF NOT EXISTS history(
  id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT, ts REAL, event TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS voice_assignments(
  project_id TEXT, speaker_id TEXT, voice_id TEXT, PRIMARY KEY(project_id, speaker_id));
"""


class ProjectDatabase:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else data_dir() / "geodubber.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._c() as c:
            c.executescript(SCHEMA)

    def _c(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def create_project(self, name: str, video_path: str, project_dir: str) -> str:
        pid = uuid.uuid4().hex[:12]
        now = time.time()
        with self._c() as c:
            c.execute("INSERT INTO projects VALUES(?,?,?,?,?,?,?)", (pid, name, video_path, project_dir, "new", now, now))
        self.log(pid, "created", video_path)
        return pid

    def set_status(self, pid: str, status: str, detail: str = "") -> None:
        with self._c() as c:
            c.execute("UPDATE projects SET status=?, updated=? WHERE id=?", (status, time.time(), pid))
        self.log(pid, status, detail)

    def log(self, pid: str, event: str, detail: str = "") -> None:
        with self._c() as c:
            c.execute("INSERT INTO history(project_id, ts, event, detail) VALUES(?,?,?,?)", (pid, time.time(), event, detail))

    def projects(self) -> list[dict]:
        with self._c() as c:
            return [dict(r) for r in c.execute("SELECT * FROM projects ORDER BY updated DESC")]

    def project(self, pid: str) -> Optional[dict]:
        with self._c() as c:
            r = c.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
            return dict(r) if r else None

    def history(self, pid: Optional[str] = None) -> list[dict]:
        with self._c() as c:
            q = "SELECT * FROM history " + ("WHERE project_id=? " if pid else "") + "ORDER BY ts DESC"
            return [dict(r) for r in c.execute(q, (pid,) if pid else ())]

    def delete_project(self, pid: str) -> None:
        with self._c() as c:
            for t in ("projects", "history", "voice_assignments"):
                c.execute(f"DELETE FROM {t} WHERE " + ("id" if t == "projects" else "project_id") + "=?", (pid,))

    def save_assignments(self, pid: str, mapping: dict[str, str]) -> None:
        with self._c() as c:
            c.execute("DELETE FROM voice_assignments WHERE project_id=?", (pid,))
            c.executemany("INSERT INTO voice_assignments VALUES(?,?,?)", [(pid, k, v) for k, v in mapping.items()])

    def assignments(self, pid: str) -> dict[str, str]:
        with self._c() as c:
            return {r["speaker_id"]: r["voice_id"] for r in
                    c.execute("SELECT speaker_id, voice_id FROM voice_assignments WHERE project_id=?", (pid,))}
