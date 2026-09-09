import noisereduce as nr
import numpy as np


def build_noise_profile(
    audio: np.ndarray,
    speech_segments: list[dict],
):
    """
    Build a noise profile from regions not identified
    as candidate speech by VAD.

    Returns:
        Noise waveform or None if no suitable non-speech
        region exists.
    """

    if audio.size == 0:
        return None

    speech_mask = np.zeros(
        len(audio),
        dtype=bool,
    )

    for segment in speech_segments:
        start = segment.start_sample
        end = segment.end_sample

        speech_mask[start:end] = True

    non_speech_audio = audio[
        ~speech_mask
    ]

    if non_speech_audio.size == 0:
        return None

    return non_speech_audio.astype(
        np.float32
    )


def reduce_noise(
    audio: np.ndarray,
    sample_rate: int,
    noise_profile: np.ndarray | None = None,
):
    """
    Reduce background noise using spectral subtraction.

    If a noise profile is available, it is supplied to
    noisereduce for noise estimation.

    If no noise profile is available, the audio is returned
    unchanged for now.
    """

    if audio.size == 0:
        return audio

    if noise_profile is None:
        return audio

    reduced_audio = nr.reduce_noise(
        y=audio,
        sr=sample_rate,
        y_noise=noise_profile,
    )

    return reduced_audio.astype(
        np.float32
    )