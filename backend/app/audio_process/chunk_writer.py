"""Build a wall-clock-anchored chunk.json from a recording's segments.json."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

LOGGER = logging.getLogger(__name__)
_OVERLAP_WARN_THRESHOLD_SEC = -0.5


def _parse_dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


def _recording_id_from_path(recording_dir: Path) -> int:
    match = re.search(r"(\d+)$", recording_dir.name)
    return int(match.group(1)) if match else 0


def build_chunk(segments_path: Path) -> dict:
    """Read segments.json and return its normalized chunk representation."""

    raw = json.loads(segments_path.read_text(encoding="utf-8"))
    segments = raw.get("segments", [])
    if not segments:
        raise ValueError(f"segments.json at {segments_path} contains no segments")

    recording_id = raw.get("recording_id")
    if not recording_id:
        recording_id = _recording_id_from_path(segments_path.parent.parent)

    # New manifests use the pre-roll-corrected base timestamp as recording_id.
    # This lets segment ends be derived from base_time + end_sec without losing
    # precision to the display-oriented, rounded duration_sec field.
    base_time: datetime | None = None
    if isinstance(recording_id, str):
        try:
            base_time = _parse_dt(recording_id)
        except ValueError:
            pass

    output_segments: list[dict] = []
    participants: list[str] = []
    previous_end: datetime | None = None
    for index, segment in enumerate(segments):
        start = str(segment["recorded_at"])
        start_dt = _parse_dt(start)
        if base_time is not None and segment.get("end_sec") is not None:
            end_dt = base_time + timedelta(seconds=float(segment["end_sec"]))
        else:
            end_dt = start_dt + timedelta(seconds=float(segment["duration_sec"]))
        speaker = segment.get("speaker_identity") or segment.get("speaker") or "unknown"

        if previous_end is not None:
            gap_sec = (start_dt - previous_end).total_seconds()
            if gap_sec < _OVERLAP_WARN_THRESHOLD_SEC:
                LOGGER.warning(
                    "[chunk_writer] timestamp anomaly in %s segment=%s gap=%.2fs",
                    segments_path,
                    segment.get("segment_id", f"index_{index}"),
                    gap_sec,
                )
        previous_end = end_dt
        if speaker not in participants:
            participants.append(speaker)
        output_segments.append(
            {
                "segment_id": segment["segment_id"],
                "speaker": speaker,
                "start": start,
                "end": end_dt.isoformat(),
                "text": segment.get("transcript", ""),
            }
        )

    earliest_start = min(_parse_dt(segment["start"]) for segment in output_segments)
    latest_end = max(_parse_dt(segment["end"]) for segment in output_segments)
    return {
        "recording_id": recording_id,
        "recording_start": earliest_start.isoformat(),
        "recording_end": latest_end.isoformat(),
        "participants": participants,
        "segments": output_segments,
    }


def write_chunk(segments_path: Path) -> Path:
    """Build and write chunk.json alongside segments.json."""

    chunk_path = segments_path.parent / "chunk.json"
    chunk_path.write_text(
        json.dumps(build_chunk(segments_path), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    LOGGER.info("[chunk_writer] chunk.json written path=%s", chunk_path)
    return chunk_path
