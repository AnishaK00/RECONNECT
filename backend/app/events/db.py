"""SQLite storage helpers for grouped audio events."""

import json
import logging
import sqlite3
from pathlib import Path

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_DIR = PROJECT_ROOT / "uploads" / "data" / "events"
DB_PATH = DB_DIR / "events.db"


def init_db() -> None:
    """Create or migrate the event database at its project-root path."""

    DB_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'open',
                participants TEXT NOT NULL,
                start TEXT NOT NULL,
                end TEXT NOT NULL,
                location TEXT,
                recordings TEXT NOT NULL,
                segments TEXT NOT NULL,
                closed_at TEXT,
                llm_status TEXT NOT NULL DEFAULT 'not_eligible'
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_status ON events(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_closed_at ON events(closed_at)")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
        if "llm_status" not in columns:
            conn.execute(
                "ALTER TABLE events ADD COLUMN llm_status TEXT NOT NULL DEFAULT 'not_eligible'"
            )
            conn.execute(
                "UPDATE events SET llm_status = CASE WHEN status = 'closed' THEN 'pending' ELSE 'not_eligible' END"
            )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_llm_status ON events(llm_status)")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS grouping_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT OR IGNORE INTO grouping_meta (key, value) VALUES ('unconsumed_event_count', '0')"
        )
    LOGGER.info("[events] database initialized path=%s", DB_PATH)


def get_connection() -> sqlite3.Connection:
    init_db()
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def _event_from_row(row: sqlite3.Row) -> dict:
    event = dict(row)
    for field in ("participants", "recordings", "segments"):
        event[field] = json.loads(event[field])
    return event


def list_events(status: str | None = None, limit: int | None = None) -> list[dict]:
    query = "SELECT * FROM events"
    params: tuple[str, ...] = ()
    if status is not None:
        query += " WHERE status = ?"
        params = (status,)
    query += " ORDER BY start DESC"
    if limit is not None:
        query += " LIMIT ?"
        params = params + (limit,)
    with get_connection() as conn:
        return [_event_from_row(row) for row in conn.execute(query, params)]


def get_event(event_id: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
    return _event_from_row(row) if row else None
