
from dataclasses import dataclass
from datetime import datetime

from app.audio_stream.protocol import (
    SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
)


MAX_SEGMENT_DURATION_SEC = 30.0


@dataclass
class FinalizedSpeechSegment:
    """
    A completed speech segment.

    The audio data is raw PCM16 mono bytes.
    """

    segment_id: int

    data: bytes

    start_sample: int
    end_sample: int

    finalized_reason: str

    recorded_at: datetime | None = None
    pre_roll_samples: int = 0

    @property
    def num_samples(self) -> int:
        return self.end_sample - self.start_sample

    @property
    def start_sec(self) -> float:
        return self.start_sample / SAMPLE_RATE

    @property
    def end_sec(self) -> float:
        return self.end_sample / SAMPLE_RATE

    @property
    def duration_sec(self) -> float:
        return self.num_samples / SAMPLE_RATE


class SpeechSegment:
    """
    Main recording buffer for one speech segment.

    A SpeechSegment begins when VAD detects speech.

    It can optionally start with pre-roll audio copied
    from the temporary rolling buffer.

    Incoming audio is appended until the segment is
    finalized.

    Maximum duration:
        30 seconds

    This class does NOT:
        - perform VAD
        - detect speech start/end
        - manage silence duration
        - manage the temporary rolling buffer
        - perform downstream processing
    """

    def __init__(
        self,
        segment_id: int,
        start_sample: int,
        max_duration_sec: float = MAX_SEGMENT_DURATION_SEC,
        pre_roll_samples: int = 0,
        recorded_at: datetime | None = None,
    ):
        if max_duration_sec <= 0:
            raise ValueError(
                "max_duration_sec must be greater than 0"
            )
        if pre_roll_samples < 0:
            raise ValueError("pre_roll_samples must not be negative")

        self.segment_id = segment_id

        self.start_sample = start_sample

        self.max_duration_sec = max_duration_sec

        self.pre_roll_samples = pre_roll_samples

        self.recorded_at = (
            recorded_at
            if recorded_at is not None
            else datetime.now().astimezone()
        )

        self.max_samples = int(
            SAMPLE_RATE * max_duration_sec
        ) + pre_roll_samples

        self.max_main_samples = int(
            SAMPLE_RATE * max_duration_sec
        )

        self._buffer = bytearray()

        self._finalized = False

    def add(self, data: bytes) -> bytes:
        """
        Add PCM16 mono audio to the current speech segment.

        The segment never exceeds its configured maximum
        duration.

        If the incoming audio does not completely fit,
         the remaining overflow audio is returned so it can
        be added to the next segment.

        Returns:
        bytes: PCM16 overflow audio that did not fit
        into the current segment.
        """

        if self._finalized:
            raise RuntimeError(
                "Cannot add audio to a finalized segment"
            )

        if not data:
            return b""

        if len(data) % SAMPLE_WIDTH_BYTES != 0:
            raise ValueError(
                "PCM16 audio data must contain an even "
                "number of bytes"
            )

        remaining_samples = (
            self.max_samples - self.num_samples
        )

        # Segment is already full.
        if remaining_samples <= 0:
            return data

        incoming_samples = (
            len(data) // SAMPLE_WIDTH_BYTES
        )

        # Entire incoming audio fits.
        if incoming_samples <= remaining_samples:
            self._buffer.extend(data)
            return b""

        # Only part of the incoming audio fits.
        bytes_that_fit = (
            remaining_samples * SAMPLE_WIDTH_BYTES
        )

        fitting_audio = data[:bytes_that_fit]
        overflow_audio = data[bytes_that_fit:]

        self._buffer.extend(fitting_audio)

        return overflow_audio

    @property
    def num_samples(self) -> int:
        """
        Number of samples currently stored in the segment.
        """

        return (
            len(self._buffer) // SAMPLE_WIDTH_BYTES
        )

    @property
    def duration_sec(self) -> float:
        """
        Current duration of the speech segment.
        """

        return self.num_samples / SAMPLE_RATE

    @property
    def end_sample(self) -> int:
        """
        Absolute sample position immediately after the
        newest sample in the segment.
        """

        return (
            self.start_sample
            + self.num_samples
        )

    @property
    def is_max_duration_reached(self) -> bool:
        """
        True when the segment has reached or exceeded
        its maximum allowed duration.
        """

        return self.main_num_samples >= self.max_main_samples

    @property
    def main_num_samples(self) -> int:
        """Samples captured after copied temporary-buffer pre-roll."""

        return max(0, self.num_samples - self.pre_roll_samples)

    @property
    def main_duration_sec(self) -> float:
        """Duration of the main recording, excluding pre-roll."""

        return self.main_num_samples / SAMPLE_RATE

    @property
    def is_finalized(self) -> bool:
        """
        True if the segment has been finalized.
        """

        return self._finalized

    def finalize(
        self,
        reason: str,
    ) -> FinalizedSpeechSegment:
        """
        Finalize the current speech segment.

        Valid examples of reasons:

            "silence"
            "max_duration"
            "stream_end"
        """

        if self._finalized:
            raise RuntimeError(
                "Segment has already been finalized"
            )

        self._finalized = True

        return FinalizedSpeechSegment(
            segment_id=self.segment_id,
            data=bytes(self._buffer),
            start_sample=self.start_sample,
            end_sample=self.end_sample,
            finalized_reason=reason,
            recorded_at=self.recorded_at,
            pre_roll_samples=self.pre_roll_samples,
        )
