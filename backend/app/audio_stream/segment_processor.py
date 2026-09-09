"""Asynchronous handoff from live recordings to the offline audio pipeline."""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
import time

import numpy as np
import soundfile as sf

from app.audio_process.pipeline import run_audio_pipeline
from app.audio_process.chunk_writer import write_chunk
from app.audio_process.transcribe import transcribe_wav
from app.events.grouper import GrouperSegment, process_segment
from app.audio_stream.protocol import SAMPLE_RATE
from app.audio_stream.noise_profile import SessionNoiseProfile
from app.audio_stream.speech_segment import FinalizedSpeechSegment
from app.schemas.audio import AudioPerceptionResult, AudioSegment, StreamSegmentInfo
from app.utils.storage import get_stream_session_folder


LOGGER = logging.getLogger(__name__)

PipelineRunner = Callable[..., AudioPerceptionResult]
Transcriber = Callable[[str | Path], str]
IdentityCallback = Callable[[dict[str, object]], None]
CameraCaptureCallback = Callable[[dict[str, object]], None]

_active_video_proc: subprocess.Popen | None = None
_active_video_lock = Lock()


class StreamingSegmentProcessor:
    """Persist finalized stream segments and process them without blocking VAD."""

    def __init__(
        self,
        session_id: str,
        pipeline_runner: PipelineRunner = run_audio_pipeline,
        transcriber: Transcriber | None = None,
        identity_callback: IdentityCallback | None = None,
        camera_capture_callback: CameraCaptureCallback | None = None,
    ):
        self.session_id = session_id
        self.session_dir = get_stream_session_folder(session_id)
        self._pipeline_runner = pipeline_runner
        self._transcriber = transcriber if transcriber is not None else transcribe_wav
        self._identity_callback = identity_callback
        self._camera_capture_callback = camera_capture_callback
        self._tasks: set[asyncio.Task] = set()
        self.noise_profile = SessionNoiseProfile()
        self._session_transcript_lock = Lock()
        self._session_transcript_by_recording: dict[int, list[dict[str, object]]] = {}
        self._last_video_trigger_ts = 0.0
        self._video_cooldown_sec = float(os.getenv("VIDEO_TRIGGER_COOLDOWN_SEC", "60.0"))
        self._unknown_count = 0

    def add_confirmed_idle_audio(self, data: bytes) -> None:
        """Learn current background sound from live VAD-confirmed silence."""

        self.noise_profile.add_pcm16(data)
        LOGGER.debug(
            "[PIPELINE] noise_profile=learning session=%s duration=%.3fs ready=%s",
            self.session_id,
            self.noise_profile.duration_sec,
            self.noise_profile.is_ready,
        )

    def submit(self, segment: FinalizedSpeechSegment) -> None:
        """Schedule isolated processing for one finalized main recording."""

        profile_snapshot = self.noise_profile.snapshot()
        task = asyncio.create_task(self._process(segment, profile_snapshot))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        LOGGER.debug(
            "----- [PIPELINE] queued session=%s recording=%s -----",
            self.session_id,
            segment.segment_id,
        )

    async def drain(self) -> None:
        """Wait for queued work; useful during graceful shutdown and tests."""

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _process(
        self,
        segment: FinalizedSpeechSegment,
        session_noise_profile: np.ndarray | None,
    ) -> None:
        started_at = time.perf_counter()
        recording_dir = self._recording_dir(segment)
        try:
            original_path, cleaned_path, segments_dir = await asyncio.to_thread(
                self._save_segment,
                segment,
            )
            LOGGER.debug(
                "[PIPELINE] raw_saved session=%s recording=%s path=%s",
                self.session_id,
                segment.segment_id,
                original_path,
            )
            stream_info = StreamSegmentInfo(
                main_segment_id=segment.segment_id,
                stream_start_sec=segment.start_sec,
                stream_end_sec=segment.end_sec,
                finalized_reason=segment.finalized_reason,
            )
            result = await asyncio.to_thread(
                self._pipeline_runner,
                original_path=original_path,
                cleaned_path=cleaned_path,
                segments_output_dir=segments_dir,
                session_id=self.session_id,
                original_filename=original_path.name,
                stream_segment=stream_info,
                session_noise_profile=session_noise_profile,
                progress_callback=self._pipeline_progress(segment, started_at),
                identity_callback=self._identity_callback,
            )

            # Check for unknown speakers that need video capture (no face linked yet)
            unknown_needing_video = [
                s for s in (result.segments or [])
                if s.match_status in ("unenrolled", "new_unenrolled") and s.speaker_identity
            ]
            if unknown_needing_video:
                self._unknown_count += len(unknown_needing_video)
                target_ident = unknown_needing_video[0].speaker_identity
                target_un_id = None
                if target_ident and target_ident.startswith("unenrolled_"):
                    try:
                        target_un_id = int(target_ident.split("_")[1])
                    except ValueError:
                        pass

                LOGGER.info(
                    "[PIPELINE] session=%s recording=%s unknown=%s (needs face enrollment)",
                    self.session_id,
                    segment.segment_id,
                    target_ident,
                )
                if self._camera_capture_callback is not None and target_un_id is not None:
                    self._request_unknown_voice_capture(target_un_id)
                else:
                    # Retain the local webcam path for non-Flutter callers and
                    # for malformed legacy identities that have no numeric ID.
                    self._maybe_trigger_video_capture(target_unenrolled_id=target_un_id)

            # Recognition labels speakers first; transcription runs on those clips.
            transcript_rows = await asyncio.to_thread(
                self._transcribe_labeled_segments,
                segment,
                result,
                stream_info,
            )
            segments_manifest_path = segments_dir / "segments.json"
            await asyncio.to_thread(
                self._write_segments_manifest,
                segments_manifest_path,
                stream_info,
                transcript_rows,
            )
            if transcript_rows:
                chunk_path = await asyncio.to_thread(write_chunk, segments_manifest_path)
                await asyncio.to_thread(self._group_chunk_segments, chunk_path)
            await asyncio.to_thread(
                self._merge_session_transcript,
                segment.segment_id,
                transcript_rows,
            )
            LOGGER.debug(
                "[PIPELINE] segments_manifest=saved session=%s recording=%s path=%s",
                self.session_id,
                segment.segment_id,
                segments_manifest_path,
            )
            manifest = {
                "status": "completed",
                "session_id": self.session_id,
                "recording_id": segment.segment_id,
                "raw_audio_path": str(original_path),
                "cleaned_audio_path": str(cleaned_path),
                "segments_directory": str(segments_dir),
                "segments_manifest_path": str(segments_manifest_path),
                "stream": stream_info.model_dump(mode="json"),
                "pipeline_result": (
                    result.model_dump(mode="json")
                    if hasattr(result, "model_dump")
                    else None
                ),
                "total_processing_time_sec": round(
                    time.perf_counter() - started_at, 3
                ),
            }
            await asyncio.to_thread(
                self._write_manifest,
                recording_dir / "recording.json",
                manifest,
            )
            LOGGER.debug(
                "[PIPELINE] recording_manifest=saved session=%s recording=%s path=%s",
                self.session_id,
                segment.segment_id,
                recording_dir / "recording.json",
            )
            LOGGER.info(
                "[PIPELINE] Recording #%d processed in %.2fs (segments=%d)",
                segment.segment_id,
                time.perf_counter() - started_at,
                len(result.segments or []),
            )
        except Exception as error:
            await asyncio.to_thread(
                self._write_manifest,
                recording_dir / "recording.json",
                {
                    "status": "failed",
                    "session_id": self.session_id,
                    "recording_id": segment.segment_id,
                    "finalized_reason": segment.finalized_reason,
                    "error": str(error),
                    "total_processing_time_sec": round(
                        time.perf_counter() - started_at, 3
                    ),
                },
            )
            # A failed offline job must never interrupt a live stream.
            LOGGER.exception(
                "----- [PIPELINE] failed session=%s recording=%s elapsed=%.3fs -----",
                self.session_id,
                segment.segment_id,
                time.perf_counter() - started_at,
            )

    def _maybe_trigger_video_capture(self, target_unenrolled_id: int | None = None) -> None:
        """Trigger video capture if cooldown has elapsed and no capture is in flight."""
        global _active_video_proc
        with _active_video_lock:
            now = time.time()
            if now - self._last_video_trigger_ts < self._video_cooldown_sec:
                LOGGER.debug(
                    "[PIPELINE] video_trigger=suppressed_cooldown remaining=%.1fs",
                    self._video_cooldown_sec - (now - self._last_video_trigger_ts),
                )
                return

            if _active_video_proc is not None and _active_video_proc.poll() is None:
                LOGGER.info(
                    "[PIPELINE] trigger=unknown_speaker session=%s target=%s action=dropped_in_flight",
                    self.session_id,
                    target_unenrolled_id,
                )
                return

            self._last_video_trigger_ts = now
            backend_root = Path(__file__).resolve().parents[2]
            LOGGER.info(
                "[PIPELINE] trigger=unknown_speaker session=%s target=%s action=start_subprocess",
                self.session_id,
                target_unenrolled_id,
            )
            sub_env = dict(os.environ)
            sub_env["PYTHONIOENCODING"] = "utf-8"
            sub_env["OMP_NUM_THREADS"] = "2"
            sub_env["MKL_NUM_THREADS"] = "2"
            sub_env["OPENBLAS_NUM_THREADS"] = "2"
            sub_env["VECLIB_MAXIMUM_THREADS"] = "2"
            sub_env["NUMEXPR_NUM_THREADS"] = "2"

            creation_flags = 0
            if sys.platform == "win32":
                creation_flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)

            cmd = [sys.executable, str(backend_root / "main.py"), "--take-a-video"]
            if target_unenrolled_id is not None:
                cmd.extend(["--unenrolled-id", str(target_unenrolled_id)])

            _active_video_proc = subprocess.Popen(
                cmd,
                cwd=str(backend_root),
                stdin=subprocess.DEVNULL,
                stdout=None,
                stderr=None,
                env=sub_env,
                creationflags=creation_flags,
            )

    def _request_unknown_voice_capture(self, unenrolled_id: int) -> None:
        """Ask the connected phone to record the separate unknown-voice video."""

        now = time.time()
        if now - self._last_video_trigger_ts < self._video_cooldown_sec:
            LOGGER.debug(
                "[PIPELINE] unknown_voice_capture=suppressed_cooldown remaining=%.1fs",
                self._video_cooldown_sec - (now - self._last_video_trigger_ts),
            )
            return
        self._last_video_trigger_ts = now
        LOGGER.info(
            "[PIPELINE] unknown_voice_capture=requested session=%s target=unenrolled_%s",
            self.session_id,
            unenrolled_id,
        )
        self._camera_capture_callback({
            "event": "unknown_voice_capture_requested",
            "unenrolled_id": unenrolled_id,
            "capture_seconds": 10,
            "target_fps": 10,
        })

    def _save_segment(
        self,
        segment: FinalizedSpeechSegment,
    ) -> tuple[Path, Path, Path]:
        recording_dir = self._recording_dir(segment)
        recording_dir.mkdir(parents=True, exist_ok=True)

        raw_path = recording_dir / "raw.wav"
        pcm16 = np.frombuffer(segment.data, dtype="<i2")
        audio = pcm16.astype(np.float32) / 32768.0
        sf.write(raw_path, audio, SAMPLE_RATE, subtype="PCM_16")

        return (
            raw_path,
            recording_dir / "cleaned.wav",
            recording_dir / "segments",
        )

    def _recording_dir(self, segment: FinalizedSpeechSegment) -> Path:
        return self.session_dir / f"recording_{segment.segment_id:04d}"

    def _pipeline_progress(
        self,
        segment: FinalizedSpeechSegment,
        started_at: float,
    ) -> Callable[[str, str, dict[str, object]], None]:
        def report(stage: str, status: str, details: dict[str, object]) -> None:
            detail_text = " ".join(
                f"{key}={value}" for key, value in details.items()
            )
            LOGGER.debug(
                "[PIPELINE] stage=%s status=%s session=%s recording=%s elapsed=%.3fs %s",
                stage,
                status,
                self.session_id,
                segment.segment_id,
                time.perf_counter() - started_at,
                detail_text,
            )

        return report

    def _transcribe_labeled_segments(
        self,
        recording: FinalizedSpeechSegment,
        result: AudioPerceptionResult | None,
        stream_info: StreamSegmentInfo,
    ) -> list[dict[str, object]]:
        """
        Transcribe each recognition-labeled speaker clip.

        Recognition must finish first so speaker identity / diarization labels
        are attached to the same rows that receive Whisper text. Multiple clips
        within one recording are transcribed in parallel.
        """

        final_segments = result.segments if result is not None else []
        speaker_numbers = {
            label: index
            for index, label in enumerate(
                sorted({segment.speaker_label for segment in final_segments}),
                start=1,
            )
        }

        def speaker_name(segment: AudioSegment) -> str:
            if segment.speaker_identity:
                return segment.speaker_identity
            if segment.speaker_label == "SPEAKER_UNKNOWN":
                return "speaker_unknown"
            return f"speaker_{speaker_numbers[segment.speaker_label]:02d}"

        paths = [segment.audio_path for segment in final_segments]
        if paths:
            workers = min(4, len(paths))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                transcripts = list(pool.map(self._transcriber, paths))
        else:
            transcripts = []

        if recording.recorded_at is not None:
            pre_roll_sec = recording.pre_roll_samples / SAMPLE_RATE
            base_time = recording.recorded_at - timedelta(seconds=pre_roll_sec)
            recording_id_val = base_time.isoformat()
        else:
            recording_id_val = str(stream_info.main_segment_id)

        rows: list[dict[str, object]] = []
        who_is_this_triggered = False
        trigger_phrases = ["who is this", "who's this", "who is that", "who's that"]

        for segment, transcript in zip(final_segments, transcripts):
            text = (transcript or "").lower()
            spk_display = speaker_name(segment)
            if transcript:
                LOGGER.info("[TRANSCRIPT] %s: \"%s\"", spk_display, transcript.strip())
                print(f"\n>>> [TRANSCRIPT] {spk_display}: \"{transcript.strip()}\" <<<\n")
            if not who_is_this_triggered:
                for trigger in trigger_phrases:
                    if trigger in text:
                        who_is_this_triggered = True
                        break

            rows.append(
                {
                    "recording_id": recording_id_val,
                    "segment_id": segment.segment_id,
                    "speaker_label": segment.speaker_label,
                    "speaker": speaker_name(segment),
                    "speaker_source": segment.speaker_source,
                    "start_sec": segment.start_sec,
                    "end_sec": segment.end_sec,
                    "duration_sec": round(segment.end_sec - segment.start_sec, 3),
                    "vad_confidence": segment.vad_confidence,
                    "audio_path": segment.audio_path,
                    "speaker_identity": segment.speaker_identity,
                    "match_status": segment.match_status,
                    "transcript": transcript or "",
                    "recorded_at": self._segment_recorded_at(
                        recording,
                        segment.start_sec,
                    ),
                }
            )

        if who_is_this_triggered:
            LOGGER.info("[PIPELINE] trigger=who_is_this action=request_phone_camera session=%s", stream_info.main_segment_id)
            if self._camera_capture_callback is not None:
                self._camera_capture_callback({
                    "event": "camera_capture_requested",
                    "capture_seconds": 1.5,
                    "target_fps": 10,
                })

        return rows

    @staticmethod
    def _segment_recorded_at(
        recording: FinalizedSpeechSegment,
        segment_start_sec: float,
    ) -> str | None:
        """
        Wall-clock time when this speaker clip began.

        Pipeline segment times are relative to the recording WAV (0 = start of
        that cut), and SpeechSegment.recorded_at is the system time when that
        cut's buffer started. So spoken start ≈ recording start + local start.
        """

        if recording.recorded_at is None:
            return None
        pre_roll_sec = recording.pre_roll_samples / SAMPLE_RATE
        base_time = recording.recorded_at - timedelta(seconds=pre_roll_sec)
        offset_sec = max(0.0, float(segment_start_sec))
        stamp: datetime = base_time + timedelta(seconds=offset_sec)
        return stamp.isoformat()

    def _merge_session_transcript(
        self,
        recording_id: int,
        rows: list[dict[str, object]],
    ) -> None:
        """Rewrite session transcript.json ordered by recording id (handles out-of-order jobs)."""

        with self._session_transcript_lock:
            self._session_transcript_by_recording[recording_id] = rows
            ordered: list[dict[str, object]] = []
            for rid in sorted(self._session_transcript_by_recording):
                ordered.extend(self._session_transcript_by_recording[rid])
            self._write_manifest(
                self.session_dir / "transcript.json",
                {
                    "session_id": self.session_id,
                    "segments": ordered,
                },
            )
            LOGGER.debug(
                "[TRANSCRIPT] session_file=updated session=%s recordings=%s segments=%s",
                self.session_id,
                len(self._session_transcript_by_recording),
                len(ordered),
            )

    @staticmethod
    def _write_manifest(path: Path, data: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _write_segments_manifest(
        path: Path,
        stream_info: StreamSegmentInfo,
        transcript_rows: list[dict[str, object]],
    ) -> None:
        """Persist final speaker segments with recognition labels and transcripts."""

        # Drop recording_id from per-recording manifest rows (kept on session file).
        segments = [
            {key: value for key, value in row.items() if key != "recording_id"}
            for row in transcript_rows
        ]
        recording_id = (
            transcript_rows[0].get("recording_id")
            if transcript_rows
            else str(stream_info.main_segment_id)
        )
        data = {
            "recording_id": recording_id,
            "stream_start_sec": stream_info.stream_start_sec,
            "stream_end_sec": stream_info.stream_end_sec,
            "segments": segments,
        }
        StreamingSegmentProcessor._write_manifest(path, data)

    @staticmethod
    def _group_chunk_segments(chunk_path: Path) -> None:
        """Submit normalized, wall-clock-anchored segments to event grouping."""

        chunk = json.loads(chunk_path.read_text(encoding="utf-8"))
        recording_id = str(chunk["recording_id"])
        for segment in chunk.get("segments", []):
            process_segment(
                GrouperSegment(
                    parent_recording_id=recording_id,
                    segment_id=str(segment["segment_id"]),
                    speaker=str(segment["speaker"]),
                    start=str(segment["start"]),
                    end=str(segment["end"]),
                    text=str(segment.get("text", "")),
                )
            )
