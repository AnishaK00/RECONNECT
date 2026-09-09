"""In-memory background-noise learning for one live audio session."""

import numpy as np

from app.audio_stream.protocol import SAMPLE_RATE, SAMPLE_WIDTH_BYTES


MIN_PROFILE_SECONDS = 1.5
MAX_PROFILE_SECONDS = 10.0


class SessionNoiseProfile:
    """Keep the newest confirmed-idle PCM16 audio without persisting it."""

    def __init__(
        self,
        min_seconds: float = MIN_PROFILE_SECONDS,
        max_seconds: float = MAX_PROFILE_SECONDS,
    ):
        self.min_samples = int(min_seconds * SAMPLE_RATE)
        self.max_samples = int(max_seconds * SAMPLE_RATE)
        self._samples = np.empty(0, dtype=np.float32)

    def add_pcm16(self, data: bytes) -> None:
        """Add an idle canonical PCM16 chunk to the rolling profile."""

        if not data:
            return
        if len(data) % SAMPLE_WIDTH_BYTES:
            raise ValueError("PCM16 noise-profile data must contain whole samples")
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        self._samples = np.concatenate((self._samples, samples))[-self.max_samples:]

    @property
    def duration_sec(self) -> float:
        return len(self._samples) / SAMPLE_RATE

    @property
    def is_ready(self) -> bool:
        return len(self._samples) >= self.min_samples

    def snapshot(self) -> np.ndarray | None:
        """Return a copy only after enough background has been learned."""

        if not self.is_ready:
            return None
        return self._samples.copy()
