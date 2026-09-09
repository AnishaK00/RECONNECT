"""Stateful live speech recording built on the streaming VAD."""

from collections.abc import Callable
import logging

from app.audio_stream.speech_segment import FinalizedSpeechSegment, SpeechSegment
from app.audio_stream.streaming_vad import VADEvent, VADEventType, VADState, StreamingVAD
from app.audio_stream.temp_buffer import StreamingBuffer


LOGGER = logging.getLogger(__name__)

TEMP_BUFFER_SECONDS = 5.0
PRE_ROLL_SECONDS = 0.5
MAIN_RECORDING_SECONDS = 30.0


class StreamingSpeechRecorder:
    """Coordinate temporary history, continuous VAD, and main recordings."""

    def __init__(
        self,
        on_finalized: Callable[[FinalizedSpeechSegment], None],
        on_confirmed_idle_audio: Callable[[bytes], None] | None = None,
        *,
        vad: StreamingVAD | None = None,
        temp_buffer_seconds: float = TEMP_BUFFER_SECONDS,
        pre_roll_seconds: float = PRE_ROLL_SECONDS,
        max_main_duration_sec: float = MAIN_RECORDING_SECONDS,
    ):
        self.vad = vad or StreamingVAD()
        self.temp_buffer = StreamingBuffer(temp_buffer_seconds)
        self.pre_roll_seconds = pre_roll_seconds
        self.max_main_duration_sec = max_main_duration_sec
        self._on_finalized = on_finalized
        self._on_confirmed_idle_audio = on_confirmed_idle_audio
        self._active: SpeechSegment | None = None
        self._next_segment_id = 1
        self._last_finalized_end_sample = 0
        self._vad_frames_processed = 0
        self._last_speech_probability: float | None = None
        self._temporary_buffer_active = False

    @property
    def active_segment(self) -> SpeechSegment | None:
        return self._active

    @property
    def is_recording(self) -> bool:
        return self._active is not None

    def process(self, data: bytes) -> list[VADEvent]:
        """Accept one canonical chunk and return the VAD events it produced."""

        if not data:
            return []

        if self._active is None:
            if not self._temporary_buffer_active:
                LOGGER.debug("[BUFFER] started")
                self._temporary_buffer_active = True
            prior_start = self.temp_buffer.start_sample
            self.temp_buffer.add(data)
            events = self.vad.process(data)
            self._log_vad_events(events)
            if (
                self._on_confirmed_idle_audio is not None
                and self.vad.state == VADState.IDLE
                and all(event.event_type == VADEventType.NONE for event in events)
            ):
                self._on_confirmed_idle_audio(data)
            if self.temp_buffer.start_sample != prior_start:
                LOGGER.debug(
                    "[BUFFER] evicted_oldest duration=%.3fs",
                    self.temp_buffer.buffered_duration,
                )
            LOGGER.debug(
                "[BUFFER] stored bytes=%s duration=%.3fs",
                len(data),
                self.temp_buffer.buffered_duration,
            )
            if any(event.event_type == VADEventType.SPEECH_STARTED for event in events):
                self._start_recording_from_temp_buffer()
            return events

        events = self.vad.process(data)
        self._log_vad_events(events)
        # The temporary buffer is paused, but its absolute timeline must not
        # fall behind the active recording.
        self.temp_buffer.advance(data)
        LOGGER.debug("[BUFFER] paused bytes=%s", len(data))
        overflow = self._active.add(data)

        if self._active.is_max_duration_reached:
            self._finalize_active("max_duration")
            if self.vad.state == VADState.SPEAKING:
                self._start_continuation(overflow)
            else:
                self.temp_buffer.reset()
                LOGGER.debug("[BUFFER] cleared reason=max_duration")
            return events

        if any(event.event_type == VADEventType.SPEECH_ENDED for event in events):
            self._finalize_active("silence")
            self.temp_buffer.reset()
            LOGGER.debug("[BUFFER] cleared reason=speech_ended")

        return events

    def finish(self) -> None:
        """Finalize an active recording when its WebSocket closes."""

        if self._active is not None:
            self._finalize_active("stream_end")
        self.temp_buffer.reset()
        self._temporary_buffer_active = False
        LOGGER.info("[BUFFER] cleared reason=stream_end")
        self.vad.reset()

    def _start_recording_from_temp_buffer(self) -> None:
        pre_roll = self.temp_buffer.get_recent(self.pre_roll_seconds)
        if pre_roll is None:
            return

        segment = SpeechSegment(
            segment_id=self._next_segment_id,
            start_sample=pre_roll.start_sample,
            max_duration_sec=self.max_main_duration_sec,
            pre_roll_samples=pre_roll.num_samples,
        )
        segment.add(pre_roll.data)
        self._active = segment
        self._next_segment_id += 1
        self.temp_buffer.reset()
        self._temporary_buffer_active = False
        LOGGER.info(
            "----- [RECORDING] started segment=%s pre_roll=%.3fs -----",
            segment.segment_id,
            pre_roll.duration_sec,
        )

    def _start_continuation(self, overflow: bytes) -> None:
        """Continue an utterance immediately after a 30-second split."""

        self._active = SpeechSegment(
            segment_id=self._next_segment_id,
            start_sample=self._last_finalized_end_sample,
            max_duration_sec=self.max_main_duration_sec,
        )
        self._next_segment_id += 1

        remaining = overflow
        while remaining:
            remaining = self._active.add(remaining)
            if not self._active.is_max_duration_reached:
                break
            self._finalize_active("max_duration")
            self._active = SpeechSegment(
                segment_id=self._next_segment_id,
                start_sample=self._last_finalized_end_sample,
                max_duration_sec=self.max_main_duration_sec,
            )
            self._next_segment_id += 1

        LOGGER.info(
            "----- [RECORDING] continued segment=%s reason=max_duration -----",
            self._active.segment_id,
        )

    def _finalize_active(self, reason: str) -> None:
        if self._active is None:
            return

        finalized = self._active.finalize(reason)
        self._last_finalized_end_sample = finalized.end_sample
        self._active = None
        if finalized.num_samples == 0:
            LOGGER.info("discarded empty speech continuation reason=%s", reason)
            return
        LOGGER.info(
            "----- [RECORDING] finalized segment=%s reason=%s duration=%.3fs -----",
            finalized.segment_id,
            reason,
            finalized.duration_sec,
        )
        self._on_finalized(finalized)

    def status_snapshot(self) -> dict[str, object]:
        """Return lightweight state for the server's periodic developer log."""

        return {
            "temp_buffer_state": (
                "paused" if self._active else (
                    "active" if self._temporary_buffer_active else "waiting"
                )
            ),
            "temp_buffer_duration_sec": round(
                self.temp_buffer.buffered_duration, 3
            ),
            "vad_state": self.vad.state.value,
            "vad_frames": self._vad_frames_processed,
            "last_probability": self._last_speech_probability,
            "active_segment_id": (
                self._active.segment_id if self._active else None
            ),
            "main_duration_sec": round(
                self._active.main_duration_sec, 3
            ) if self._active else 0.0,
        }

    def _log_vad_events(self, events: list[VADEvent]) -> None:
        for event in events:
            self._vad_frames_processed += 1
            self._last_speech_probability = event.speech_probability
            # State transitions are visible at INFO; individual inference
            # frames remain available to developers at DEBUG level.
            LOGGER.debug(
                "[VAD] frame event=%s probability=%.3f state=%s",
                event.event_type.value,
                event.speech_probability or 0.0,
                event.state.value,
            )
            if event.event_type != VADEventType.NONE:
                LOGGER.debug(
                    "[VAD] transition=%s probability=%.3f state=%s",
                    event.event_type.value,
                    event.speech_probability or 0.0,
                    event.state.value,
                )
