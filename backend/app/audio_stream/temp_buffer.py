
from dataclasses import dataclass

from app.audio_stream.protocol import (
    SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
)


@dataclass
class AudioWindow:
    """
    A portion of audio stored in the rolling temporary buffer.

    Audio data is raw PCM16 mono bytes.
    """

    data: bytes

    start_sample: int
    end_sample: int

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


class StreamingBuffer:
    """
    Rolling temporary audio buffer.

    Keeps only the most recent `buffer_sec` seconds
    of PCM16 mono audio.

    This buffer is used as short-term audio history
    for VAD pre-roll.

    It does NOT:
        - perform VAD
        - detect speech
        - create speech segments
        - enforce maximum recording duration
        - perform audio processing
    """

    def __init__(self, buffer_sec: float = 5.0):
        if buffer_sec <= 0:
            raise ValueError("buffer_sec must be greater than 0")

        self.buffer_sec = buffer_sec

        self.buffer_samples = int(
            SAMPLE_RATE * buffer_sec
        )

        self.buffer_bytes = (
            self.buffer_samples * SAMPLE_WIDTH_BYTES
        )

        self._buffer = bytearray()

        # Absolute sample position of the first
        # sample currently stored in the buffer.
        self._buffer_start_sample = 0

        # Absolute sample position immediately
        # after the newest sample received.
        self._total_samples_received = 0

    def add(self, data: bytes) -> None:
        """
        Add incoming PCM16 audio to the rolling buffer.

        If the buffer becomes larger than the configured
        duration, the oldest audio is removed.
        """

        if not data:
            return

        # PCM16 samples must always contain complete
        # 16-bit samples.
        if len(data) % SAMPLE_WIDTH_BYTES != 0:
            raise ValueError(
                "PCM16 audio data must contain an even "
                "number of bytes"
            )

        num_samples = (
            len(data) // SAMPLE_WIDTH_BYTES
        )

        # Append new audio.
        self._buffer.extend(data)

        # Update absolute timeline.
        self._total_samples_received += num_samples

        # Remove oldest audio if the buffer exceeds
        # the configured maximum duration.
        if len(self._buffer) > self.buffer_bytes:
            excess_bytes = (
                len(self._buffer) - self.buffer_bytes
            )

            # Keep removal aligned to complete PCM16 samples.
            excess_bytes -= (
                excess_bytes % SAMPLE_WIDTH_BYTES
            )

            if excess_bytes > 0:
                del self._buffer[:excess_bytes]

                removed_samples = (
                    excess_bytes // SAMPLE_WIDTH_BYTES
                )

                self._buffer_start_sample += removed_samples

    def get_recent(
        self,
        duration_sec: float,
    ) -> AudioWindow | None:
        """
        Return the most recent `duration_sec` of audio.

        Used primarily for VAD pre-roll.

        If the requested duration is longer than the
        currently buffered audio, all available audio
        is returned.
        """

        if not self._buffer:
            return None

        if duration_sec <= 0:
            raise ValueError(
                "duration_sec must be greater than 0"
            )

        requested_samples = int(
            SAMPLE_RATE * duration_sec
        )

        available_samples = self.buffered_samples

        samples_to_return = min(
            requested_samples,
            available_samples,
        )

        byte_count = (
            samples_to_return * SAMPLE_WIDTH_BYTES
        )

        start_byte = len(self._buffer) - byte_count

        data = bytes(
            self._buffer[start_byte:]
        )

        start_sample = (
            self._buffer_start_sample
            + available_samples
            - samples_to_return
        )

        end_sample = (
            self._buffer_start_sample
            + available_samples
        )

        return AudioWindow(
            data=data,
            start_sample=start_sample,
            end_sample=end_sample,
        )

    def get_all(self) -> AudioWindow | None:
        """
        Return all audio currently stored in the
        rolling buffer.
        """

        if not self._buffer:
            return None

        data = bytes(self._buffer)

        start_sample = self._buffer_start_sample

        end_sample = (
            start_sample + self.buffered_samples
        )

        return AudioWindow(
            data=data,
            start_sample=start_sample,
            end_sample=end_sample,
        )

    def advance(self, data: bytes) -> None:
        """Advance the absolute timeline while temporary buffering is paused."""

        if not data:
            return
        if len(data) % SAMPLE_WIDTH_BYTES != 0:
            raise ValueError(
                "PCM16 audio data must contain an even number of bytes"
            )

        self._total_samples_received += len(data) // SAMPLE_WIDTH_BYTES
        if not self._buffer:
            self._buffer_start_sample = self._total_samples_received

    @property
    def buffered_samples(self) -> int:
        """
        Number of PCM16 samples currently stored.
        """

        return (
            len(self._buffer) // SAMPLE_WIDTH_BYTES
        )

    @property
    def buffered_duration(self) -> float:
        """
        Duration of audio currently stored in seconds.
        """

        return self.buffered_samples / SAMPLE_RATE

    @property
    def start_sample(self) -> int:
        """
        Absolute sample position of the oldest
        audio currently stored.
        """

        return self._buffer_start_sample

    @property
    def end_sample(self) -> int:
        """
        Absolute sample position immediately after
        the newest audio currently stored.
        """

        return (
            self._buffer_start_sample
            + self.buffered_samples
        )

    def reset(self) -> None:
        """
        Clear the rolling buffer while preserving the stream timeline.
        """

        self._buffer.clear()
        self._buffer_start_sample = self._total_samples_received

        self._buffer_start_sample = (
            self._total_samples_received
        )

    def clear(self) -> None:
        """
        Alias for reset().
        """

        self.reset()
