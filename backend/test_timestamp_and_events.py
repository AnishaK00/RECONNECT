import json
from datetime import datetime, timedelta, timezone

from app.audio_process.chunk_writer import build_chunk
from app.audio_stream.segment_processor import StreamingSegmentProcessor
from app.audio_stream.speech_segment import FinalizedSpeechSegment
from app.events import db
from app.events.grouper import GrouperSegment, process_segment


def test_segment_timestamp_uses_copy2_pre_roll_formula():
    recorded_at = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    recording = FinalizedSpeechSegment(
        segment_id=1,
        data=b"",
        start_sample=0,
        end_sample=0,
        finalized_reason="silence",
        recorded_at=recorded_at,
        pre_roll_samples=16_000,
    )

    # Exact copy-2 formula: base_time = recorded_at - pre_roll_samples / SAMPLE_RATE.
    assert StreamingSegmentProcessor._segment_recorded_at(recording, 2.5) == (
        recorded_at - timedelta(seconds=1) + timedelta(seconds=2.5)
    ).isoformat()


def test_chunk_uses_timestamp_id_and_corrected_segment_times(tmp_path):
    segments_path = tmp_path / "recording_0004" / "segments" / "segments.json"
    segments_path.parent.mkdir(parents=True)
    base_time = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    segments_path.write_text(
        json.dumps(
            {
                "recording_id": base_time.isoformat(),
                "segments": [
                    {
                        "segment_id": "first",
                        "speaker": "alice",
                        "recorded_at": (base_time + timedelta(seconds=0.5)).isoformat(),
                        "duration_sec": 1.0,
                        "end_sec": 1.5,
                        "transcript": "one",
                    },
                    {
                        "segment_id": "second",
                        "speaker": "bob",
                        "recorded_at": (base_time + timedelta(seconds=3)).isoformat(),
                        "duration_sec": 0.001,
                        "end_sec": 5.0,
                        "transcript": "two",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    chunk = build_chunk(segments_path)
    assert chunk["recording_id"] == base_time.isoformat()
    assert chunk["segments"][0]["start"] == (base_time + timedelta(seconds=0.5)).isoformat()
    assert chunk["segments"][1]["end"] == (base_time + timedelta(seconds=5)).isoformat()


def test_chunk_falls_back_to_legacy_recording_folder_id(tmp_path):
    segments_path = tmp_path / "recording_0004" / "segments" / "segments.json"
    segments_path.parent.mkdir(parents=True)
    segments_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "segment_id": "first",
                        "recorded_at": "2026-09-09T12:00:00+00:00",
                        "duration_sec": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert build_chunk(segments_path)["recording_id"] == 4


def test_grouper_keeps_true_range_when_jobs_finish_out_of_order(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_DIR", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "events.db")
    process_segment(
        GrouperSegment("later", "later", "alice", "2026-09-09T12:00:10+00:00", "2026-09-09T12:00:15+00:00", "later")
    )
    process_segment(
        GrouperSegment("earlier", "earlier", "bob", "2026-09-09T12:00:00+00:00", "2026-09-09T12:00:05+00:00", "earlier")
    )

    event = db.list_events()[0]
    assert event["start"] == "2026-09-09T12:00:00+00:00"
    assert event["end"] == "2026-09-09T12:00:15+00:00"
    assert [segment["start"] for segment in event["segments"]] == [
        "2026-09-09T12:00:00+00:00",
        "2026-09-09T12:00:10+00:00",
    ]
