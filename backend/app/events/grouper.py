"""Group normalized speech segments into conversation events."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from app.events.db import get_connection, init_db

LOGGER = logging.getLogger(__name__)
GAP_THRESHOLD_SECONDS = 90.0


@dataclass
class GrouperSegment:
    parent_recording_id: str
    segment_id: str
    speaker: str
    start: str
    end: str
    text: str


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _segment_data(segment: GrouperSegment) -> dict:
    return {
        "speaker": segment.speaker,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
    }


def _create_event(conn, segment: GrouperSegment) -> None:
    conn.execute(
        """
        INSERT INTO events (
            event_id, status, participants, start, end, location, recordings,
            segments, closed_at, llm_status
        ) VALUES (?, 'open', ?, ?, ?, NULL, ?, ?, NULL, 'not_eligible')
        """,
        (
            str(uuid.uuid4()),
            json.dumps([segment.speaker]),
            segment.start,
            segment.end,
            json.dumps([segment.parent_recording_id]),
            json.dumps([_segment_data(segment)]),
        ),
    )


def _extend_event(conn, event: dict, segment: GrouperSegment) -> None:
    participants = json.loads(event["participants"])
    if segment.speaker not in participants:
        participants.append(segment.speaker)
    recordings = json.loads(event["recordings"])
    if segment.parent_recording_id not in recordings:
        recordings.append(segment.parent_recording_id)
    segments = json.loads(event["segments"])
    segments.append(_segment_data(segment))
    segments.sort(key=lambda item: _parse_iso(item["start"]))
    start = min(_parse_iso(event["start"]), _parse_iso(segment.start)).isoformat()
    end = max(_parse_iso(event["end"]), _parse_iso(segment.end)).isoformat()
    conn.execute(
        """
        UPDATE events SET participants = ?, start = ?, end = ?, recordings = ?, segments = ?
        WHERE event_id = ?
        """,
        (json.dumps(participants), start, end, json.dumps(recordings), json.dumps(segments), event["event_id"]),
    )


def _close_event(conn, event: dict) -> None:
    conn.execute(
        "UPDATE events SET status = 'closed', closed_at = ?, llm_status = 'pending' WHERE event_id = ?",
        (_now_iso(), event["event_id"]),
    )
    conn.execute(
        "UPDATE grouping_meta SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) WHERE key = 'unconsumed_event_count'"
    )


def process_segment(segment: GrouperSegment) -> None:
    """Persist one finalized segment, safely handling out-of-order completion."""

    init_db()
    with get_connection() as conn:
        conn.execute("BEGIN EXCLUSIVE")
        try:
            row = conn.execute("SELECT * FROM events WHERE status = 'open'").fetchone()
            if row is None:
                _create_event(conn, segment)
            else:
                event = dict(row)
                gap = (_parse_iso(segment.start) - _parse_iso(event["end"])).total_seconds()
                if gap > GAP_THRESHOLD_SECONDS:
                    _close_event(conn, event)
                    _create_event(conn, segment)
                else:
                    _extend_event(conn, event, segment)
            conn.commit()
        except Exception:
            conn.rollback()
            LOGGER.exception("[grouper] failed segment=%s", segment.segment_id)
            raise
