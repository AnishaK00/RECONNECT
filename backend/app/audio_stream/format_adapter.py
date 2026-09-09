from dataclasses import dataclass

import librosa
import numpy as np

from app.audio_stream.protocol import SAMPLE_RATE, CHANNELS


PCM_S16LE = "pcm_s16le"


@dataclass(frozen=True)
class StreamAudioFormat:
    """Declared format of binary audio sent over one WebSocket."""

    sample_rate: int
    channels: int
    encoding: str

    @classmethod
    def from_message(cls, message: dict) -> "StreamAudioFormat":
        """Validate the stream's initial JSON format message."""

        try:
            sample_rate = int(message["sample_rate"])
            channels = int(message["channels"])
            encoding = str(message["encoding"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Format metadata must contain integer sample_rate, integer "
                "channels, and string encoding"
            ) from error

        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than 0")
        if channels <= 0 or channels > 8:
            raise ValueError("channels must be between 1 and 8")
        if encoding != PCM_S16LE:
            raise ValueError(
                "Only pcm_s16le input is currently supported"
            )

        return cls(
            sample_rate=sample_rate,
            channels=channels,
            encoding=encoding,
        )

    @property
    def is_canonical(self) -> bool:
        return (
            self.sample_rate == SAMPLE_RATE
            and self.channels == CHANNELS
            and self.encoding == PCM_S16LE
        )


def adapt_pcm16_stream_chunk(
    data: bytes,
    source_format: StreamAudioFormat,
) -> bytes:
    """Convert a declared PCM16 stream chunk into canonical PCM16 bytes."""

    bytes_per_interleaved_frame = 2 * source_format.channels
    if len(data) % bytes_per_interleaved_frame != 0:
        raise ValueError(
            "PCM16 chunk length must align to the declared channel count"
        )

    if source_format.is_canonical:
        return data

    pcm16 = np.frombuffer(data, dtype="<i2")
    frames = pcm16.reshape(-1, source_format.channels)
    mono = frames.mean(axis=1).astype(np.float32) / 32768.0
    normalized = resample_audio(mono, source_format.sample_rate)
    return float_to_pcm16(normalized).astype("<i2", copy=False).tobytes()


def to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert audio to mono.

    Input:
        mono:  (samples,)
        stereo: (samples, channels)

    Output:
        mono: (samples,)
    """

    if audio.ndim == 1:
        return audio

    if audio.ndim == 2:
        return np.mean(audio, axis=1)

    raise ValueError(
        f"Unsupported audio shape: {audio.shape}"
    )


def resample_audio(
    audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """
    Resample audio to the canonical sample rate.
    """

    if sample_rate == SAMPLE_RATE:
        return audio

    return librosa.resample(
        audio,
        orig_sr=sample_rate,
        target_sr=SAMPLE_RATE,
    )


def float_to_pcm16(audio: np.ndarray) -> np.ndarray:
    """
    Convert normalized floating-point audio
    to PCM16 samples.
    """

    audio = np.clip(audio, -1.0, 1.0)

    return (
        audio * 32767
    ).astype(np.int16)


def adapt_audio(
    audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """
    Convert incoming audio into the canonical
    RECONNECT audio format:

        16 kHz
        mono
        PCM16
    """

    # 1. Convert to mono
    audio = to_mono(audio)

    # 2. Resample to 16 kHz if necessary
    audio = resample_audio(
        audio,
        sample_rate,
    )

    # 3. Convert float audio to PCM16
    audio = float_to_pcm16(audio)

    return audio
