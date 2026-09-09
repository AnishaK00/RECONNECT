"""Focused tests for live stream normalization, recording, and handoff."""

import asyncio
from collections import deque
from datetime import datetime, timezone
import json
import logging
import threading

import numpy as np
from fastapi.testclient import TestClient

from app.audio_stream.format_adapter import (
    StreamAudioFormat,
    adapt_pcm16_stream_chunk,
)
from app.audio_stream.segment_processor import StreamingSegmentProcessor
from app.audio_stream.noise_profile import SessionNoiseProfile
from app.audio_process.segmentation import fallback_vad_segments
from app.audio_process.vad import SpeechRegion
from app.audio_stream.speech_segment import FinalizedSpeechSegment
from app.audio_stream.streaming_recorder import StreamingSpeechRecorder
from app.audio_stream.streaming_vad import VADEvent, VADEventType, VADState
from app.main import app
from app.schemas.audio import (
    AudioInfo,
    AudioPerceptionResult,
    AudioSegment,
    PipelineInfo,
    PreprocessingInfo,
)
from live_microphone_client import chunk_frames, format_metadata


def event(event_type: VADEventType, state: VADState) -> VADEvent:
    return VADEvent(
        event_type=event_type,
        state=state,
        is_speech=state == VADState.SPEAKING,
        speech_probability=0.9 if state == VADState.SPEAKING else 0.1,
        silence_duration_ms=0.0,
    )


class FakeVAD:
    def __init__(self, events: list[list[VADEvent]]):
        self._events = deque(events)
        self.state = VADState.IDLE

    def process(self, data: bytes) -> list[VADEvent]:
        result = self._events.popleft() if self._events else []
        if result:
            self.state = result[-1].state
        return result

    def reset(self) -> None:
        self.state = VADState.IDLE


def pcm(samples: int) -> bytes:
    return b"\x00\x00" * samples


def test_pcm16_stream_adapter_passthrough_and_stereo_conversion():
    canonical = StreamAudioFormat(16000, 1, "pcm_s16le")
    source = np.array([100, -100], dtype="<i2").tobytes()
    assert adapt_pcm16_stream_chunk(source, canonical) == source

    stereo = StreamAudioFormat(16000, 2, "pcm_s16le")
    stereo_data = np.array([[1000, -1000], [3000, 1000]], dtype="<i2").tobytes()
    converted = np.frombuffer(adapt_pcm16_stream_chunk(stereo_data, stereo), dtype="<i2")
    assert converted.shape == (2,)
    assert converted[0] == 0
    assert abs(int(converted[1]) - 2000) <= 1


def test_pcm16_stream_adapter_resamples_and_rejects_bad_alignment():
    source_format = StreamAudioFormat(8000, 1, "pcm_s16le")
    source = (np.arange(80, dtype="<i2") * 100).tobytes()
    converted = adapt_pcm16_stream_chunk(source, source_format)
    assert len(converted) > len(source)

    try:
        adapt_pcm16_stream_chunk(b"\x00", source_format)
        raise AssertionError("Expected invalid PCM alignment to fail")
    except ValueError:
        pass


def test_recorder_uses_preroll_retains_silence_and_restarts_buffer():
    finalized: list[FinalizedSpeechSegment] = []
    vad = FakeVAD(
        [
            [event(VADEventType.SPEECH_STARTED, VADState.SPEAKING)],
            [],
            [event(VADEventType.SPEECH_ENDED, VADState.IDLE)],
        ]
    )
    recorder = StreamingSpeechRecorder(
        finalized.append,
        vad=vad,
        pre_roll_seconds=0.5,
    )

    recorder.process(pcm(8000))
    assert recorder.is_recording
    assert recorder.active_segment.pre_roll_samples == 8000

    started_at = recorder.active_segment.recorded_at
    assert started_at is not None
    recorder.process(pcm(320))
    recorder.process(pcm(320))
    assert len(finalized) == 1
    assert finalized[0].finalized_reason == "silence"
    assert finalized[0].num_samples == 8640
    assert finalized[0].recorded_at == started_at
    assert recorder.temp_buffer.buffered_samples == 0

    recorder.process(pcm(320))
    assert recorder.temp_buffer.start_sample == 8640
    assert recorder.temp_buffer.end_sample == 8960


def test_recorder_emits_lifecycle_logs_and_status_snapshot(caplog):
    recorder = StreamingSpeechRecorder(
        lambda segment: None,
        vad=FakeVAD([[event(VADEventType.SPEECH_STARTED, VADState.SPEAKING)]]),
        pre_roll_seconds=0.01,
    )
    with caplog.at_level(logging.INFO):
        recorder.process(pcm(320))

    assert "[BUFFER] started" in caplog.text
    assert "[RECORDING] started" in caplog.text
    snapshot = recorder.status_snapshot()
    assert snapshot["vad_frames"] == 1
    assert snapshot["active_segment_id"] == 1


def test_recorder_splits_continuous_speech_at_main_duration_limit():
    finalized: list[FinalizedSpeechSegment] = []
    vad = FakeVAD(
        [
            [event(VADEventType.SPEECH_STARTED, VADState.SPEAKING)],
            [],
        ]
    )
    recorder = StreamingSpeechRecorder(
        finalized.append,
        vad=vad,
        pre_roll_seconds=0.01,
        max_main_duration_sec=0.1,
    )
    recorder.process(pcm(160))
    recorder.process(pcm(1600))

    assert len(finalized) == 1
    assert finalized[0].finalized_reason == "max_duration"
    assert finalized[0].num_samples == 1760
    assert recorder.active_segment is not None
    assert recorder.active_segment.segment_id == 2


def test_stream_processor_saves_wav_and_passes_stream_metadata(tmp_path, caplog):
    received = []

    def fake_pipeline(**kwargs):
        received.append(kwargs)
        kwargs["cleaned_path"].write_bytes(b"cleaned")
        kwargs["segments_output_dir"].mkdir(parents=True)
        (kwargs["segments_output_dir"] / "speaker_001.wav").write_bytes(b"segment")
        kwargs["progress_callback"]("preprocessing", "started", {})
        kwargs["progress_callback"]("export", "completed", {"final_segment_count": 1})
        return None

    processor = StreamingSegmentProcessor("test-session", pipeline_runner=fake_pipeline)
    processor.session_dir = tmp_path
    segment = FinalizedSpeechSegment(1, pcm(320), 0, 320, "silence")

    async def run():
        processor.submit(segment)
        await processor.drain()

    with caplog.at_level(logging.INFO):
        asyncio.run(run())
    recording_dir = tmp_path / "recording_0001"
    assert (recording_dir / "raw.wav").exists()
    assert (recording_dir / "cleaned.wav").exists()
    assert (recording_dir / "segments" / "speaker_001.wav").exists()
    manifest = json.loads((recording_dir / "recording.json").read_text())
    assert manifest["status"] == "completed"
    assert "stage=preprocessing status=started" in caplog.text
    assert "stage=export status=completed" in caplog.text
    assert "recording_manifest=saved" in caplog.text
    assert received[0]["stream_segment"].finalized_reason == "silence"
    assert received[0]["session_id"] == "test-session"


def test_stream_processor_isolates_pipeline_failures(tmp_path):
    def failing_pipeline(**kwargs):
        raise RuntimeError("offline processing unavailable")

    processor = StreamingSegmentProcessor("failure-session", pipeline_runner=failing_pipeline)
    processor.session_dir = tmp_path

    async def run():
        processor.submit(FinalizedSpeechSegment(1, pcm(320), 0, 320, "silence"))
        await processor.drain()

    # The worker logs the failure but does not leak it to the live recorder.
    asyncio.run(run())
    recording_dir = tmp_path / "recording_0001"
    assert (recording_dir / "raw.wav").exists()
    assert json.loads((recording_dir / "recording.json").read_text())["status"] == "failed"


def test_session_noise_profile_requires_minimum_idle_audio():
    profile = SessionNoiseProfile(min_seconds=1.5, max_seconds=2.0)
    profile.add_pcm16(pcm(16000))
    assert profile.snapshot() is None
    profile.add_pcm16(pcm(8000))
    snapshot = profile.snapshot()
    assert snapshot is not None
    assert len(snapshot) == 24000


def test_vad_fallback_preserves_short_speech_as_unknown():
    regions = [
        SpeechRegion("seg_001", 0.1, 1.2, 1600, 19200),
        SpeechRegion("seg_002", 1.3, 1.5, 20800, 24000),
    ]
    fallback = fallback_vad_segments(regions, min_duration_sec=0.5)
    assert len(fallback) == 1
    assert fallback[0].speaker_label == "SPEAKER_UNKNOWN"
    assert fallback[0].speaker_source == "vad_fallback"


def test_microphone_client_helpers_use_actual_device_rate():
    assert chunk_frames(44100) == 882
    assert format_metadata(44100, 1) == {
        "sample_rate": 44100,
        "channels": 1,
        "encoding": "pcm_s16le",
    }


def test_websocket_requires_metadata_then_acknowledges_audio():
    client = TestClient(app)
    with client.websocket_connect("/ws/audio") as websocket:
        websocket.send_text('{"sample_rate": 16000, "channels": 1, "encoding": "pcm_s16le"}')
        connected = websocket.receive_json()
        assert connected["received"] is True
        assert connected["session_id"]

        websocket.send_bytes(pcm(320))
        acknowledgement = websocket.receive_json()
        assert acknowledgement == {"received": True, "chunk_number": 1, "chunk_size": 640}


def test_websocket_rejects_audio_before_metadata_and_bad_metadata():
    client = TestClient(app)
    with client.websocket_connect("/ws/audio") as websocket:
        websocket.send_bytes(pcm(320))
        assert websocket.receive_json()["error"].startswith("First WebSocket message")

    with client.websocket_connect("/ws/audio") as websocket:
        websocket.send_text('{"sample_rate": 16000, "channels": 1, "encoding": "float32"}')
        assert "Only pcm_s16le" in websocket.receive_json()["error"]


EXISTING_SEGMENT_KEYS = {
    "segment_id",
    "speaker_label",
    "speaker",
    "speaker_source",
    "start_sec",
    "end_sec",
    "duration_sec",
    "vad_confidence",
    "audio_path",
}


def _perception_result(segments: list[AudioSegment]) -> AudioPerceptionResult:
    return AudioPerceptionResult(
        saved=True,
        session_id="test-session",
        recording_started_at="should-not-be-used",
        duration_sec=1.0,
        audio=AudioInfo(
            original_path="raw.wav",
            cleaned_path="cleaned.wav",
            sample_rate=16000,
        ),
        preprocessing=PreprocessingInfo(
            original_sample_rate=16000,
            peak_normalized=0.95,
            preprocessing_completed=True,
        ),
        segments=segments,
        pipeline=PipelineInfo(
            vad_model="test",
            diarization_model="test",
            processing_time_sec=0.1,
        ),
        speaker_count=len({segment.speaker_label for segment in segments}),
        segment_count=len(segments),
    )


def test_stream_processor_writes_transcripts_and_session_file_in_order(tmp_path):
    hold_first = threading.Event()
    first_waiting = threading.Event()
    second_done = threading.Event()

    def fake_pipeline(**kwargs):
        recording_id = kwargs["stream_segment"].main_segment_id
        if recording_id == 1:
            first_waiting.set()
            assert hold_first.wait(timeout=5)
        kwargs["cleaned_path"].write_bytes(b"cleaned")
        kwargs["segments_output_dir"].mkdir(parents=True, exist_ok=True)
        wav_path = kwargs["segments_output_dir"] / f"speaker_01_{recording_id:03d}.wav"
        wav_path.write_bytes(b"segment")
        result = _perception_result(
            [
                AudioSegment(
                    segment_id=f"seg_{recording_id:03d}",
                    # Pipeline times are relative to the recording WAV.
                    start_sec=0.5,
                    end_sec=1.5,
                    speaker_label="SPEAKER_00",
                    audio_path=str(wav_path),
                    speaker_source="diarization",
                    vad_confidence=0.9,
                )
            ]
        )
        if recording_id == 2:
            second_done.set()
        return result

    processor = StreamingSegmentProcessor(
        "session-transcript",
        pipeline_runner=fake_pipeline,
        transcriber=lambda path: f"text-from-{path}",
    )
    processor.session_dir = tmp_path
    earlier = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    later = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)

    async def wait_set(event: threading.Event, timeout: float = 5.0) -> None:
        deadline = asyncio.get_event_loop().time() + timeout
        while not event.is_set():
            if asyncio.get_event_loop().time() > deadline:
                raise AssertionError("timed out waiting for pipeline event")
            await asyncio.sleep(0.01)

    async def run():
        processor.submit(
            FinalizedSpeechSegment(1, pcm(320), 0, 16000, "silence", earlier)
        )
        await wait_set(first_waiting)
        processor.submit(
            FinalizedSpeechSegment(2, pcm(320), 16000, 32000, "silence", later)
        )
        await wait_set(second_done)
        hold_first.set()
        await processor.drain()

    asyncio.run(run())

    segments_manifest = json.loads(
        (tmp_path / "recording_0001" / "segments" / "segments.json").read_text()
    )
    segment = segments_manifest["segments"][0]
    assert EXISTING_SEGMENT_KEYS <= segment.keys()
    assert segment["transcript"].startswith("text-from-")
    assert segment["recorded_at"] == datetime(
        2026, 1, 1, 12, 0, 0, 500000, tzinfo=timezone.utc
    ).isoformat()

    session_transcript = json.loads((tmp_path / "transcript.json").read_text())
    assert session_transcript["session_id"] == "session-transcript"
    assert [row["recording_id"] for row in session_transcript["segments"]] == [1, 2]
    assert session_transcript["segments"][0]["transcript"].startswith("text-from-")
    assert session_transcript["segments"][1]["recorded_at"] == datetime(
        2026, 1, 1, 12, 0, 10, 500000, tzinfo=timezone.utc
    ).isoformat()
