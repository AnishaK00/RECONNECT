"""Replaceable SQLite storage for durable memories."""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMORY_DB_PATH = PROJECT_ROOT / "uploads" / "data" / "memory" / "memories.db"

class MemoryStore(Protocol):
    def create(self, memory: dict) -> None: ...
    def get(self, memory_id: str) -> dict | None: ...
    def update(self, memory: dict) -> None: ...
    def get_recent(self, participant: str, limit: int = 3) -> list[dict]: ...
    def list_all(self) -> list[dict]: ...
    def list_by_category(self, category: str, limit: int | None = None) -> list[dict]: ...

class SQLiteMemoryStore:
    """Simple local implementation of the memory-store interface."""
    def __init__(self, path: str | Path = DEFAULT_MEMORY_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS memories (
              memory_id TEXT PRIMARY KEY, category TEXT NOT NULL, summary TEXT NOT NULL,
              participants TEXT NOT NULL, about TEXT NOT NULL DEFAULT '[]', reported_by TEXT NOT NULL DEFAULT '[]', source_events TEXT NOT NULL,
              earliest_event_time TEXT NOT NULL, latest_event_time TEXT NOT NULL,
              importance TEXT NOT NULL, entities TEXT NOT NULL, topic TEXT,
              emotion TEXT NOT NULL DEFAULT 'neutral', created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL)""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
            if "about" not in columns:
                conn.execute("ALTER TABLE memories ADD COLUMN about TEXT NOT NULL DEFAULT '[]'")
            if "reported_by" not in columns:
                conn.execute("ALTER TABLE memories ADD COLUMN reported_by TEXT NOT NULL DEFAULT '[]'")
    def _connect(self):
        conn = sqlite3.connect(self.path); conn.row_factory = sqlite3.Row; return conn
    @staticmethod
    def _row(row):
        value = dict(row)
        for field in ("participants", "about", "reported_by", "source_events", "entities"): value[field] = json.loads(value[field])
        # Migrate the earlier flat about/reported_by representation in-memory.
        if value["about"] and isinstance(value["about"][0], str):
            reporters = value.get("reported_by", [])
            value["about"] = [{"person": person, "reported_by": reporters[min(index, len(reporters) - 1)] if reporters else "unknown", "what_was_said": "Legacy record; original claim unavailable."} for index, person in enumerate(value["about"])]
        return value
    @staticmethod
    def _values(memory):
        return (memory["memory_id"], memory["category"], memory["summary"], json.dumps(memory["participants"]), json.dumps(memory.get("about", [])), "[]", json.dumps(memory["source_events"]), memory["earliest_event_time"], memory["latest_event_time"], memory["importance"], json.dumps(memory["entities"]), memory.get("topic"), memory.get("emotion", "neutral"), memory["created_at"], memory["updated_at"])
    def create(self, memory):
        with self._connect() as conn:
            conn.execute("""INSERT INTO memories (
                memory_id, category, summary, participants, about, reported_by,
                source_events, earliest_event_time, latest_event_time, importance,
                entities, topic, emotion, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", self._values(memory))
    def get(self, memory_id):
        with self._connect() as conn: row = conn.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
        return self._row(row) if row else None
    def update(self, memory):
        with self._connect() as conn:
            result = conn.execute("""UPDATE memories SET category=?, summary=?, participants=?, about=?, reported_by=?, source_events=?, earliest_event_time=?, latest_event_time=?, importance=?, entities=?, topic=?, emotion=?, created_at=?, updated_at=? WHERE memory_id=?""", self._values(memory)[1:] + (memory["memory_id"],))
            if result.rowcount != 1: raise KeyError(f"Memory does not exist: {memory['memory_id']}")
    def get_recent(self, participant, limit=3):
        with self._connect() as conn: rows = conn.execute("SELECT * FROM memories ORDER BY updated_at DESC").fetchall()
        return [self._row(r) for r in rows if participant in json.loads(r["participants"])][:limit]

    def list_all(self):
        with self._connect() as conn: rows = conn.execute("SELECT * FROM memories ORDER BY updated_at DESC").fetchall()
        return [self._row(row) for row in rows]

    def list_by_category(self, category, limit=None):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE category=? ORDER BY updated_at DESC", (category,)
            ).fetchall()
        result = [self._row(row) for row in rows]
        return result[:limit] if limit is not None else result