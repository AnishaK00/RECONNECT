from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


TARGET_SAMPLE_RATE = 16000
TARGET_PEAK = 0.95


def load_audio(input_path: Path):
    """Load an audio file and return mono audio data and sample rate."""

    try:
        audio_data, sample_rate = librosa.load(
            input_path,
            sr=None,
            mono=True,
        )

        return audio_data.astype(np.float32), int(sample_rate)

    except Exception as librosa_error:

        # Fallback for formats that librosa cannot decode.
        try:
            import torchaudio

            waveform, sample_rate = torchaudio.load(
                str(input_path)
            )

            # Convert multi-channel audio to mono.
            if waveform.shape[0] > 1:
                waveform = waveform.mean(
                    dim=0,
                    keepdim=True,
                )

            audio_data = waveform.squeeze().numpy()

            return (
                audio_data.astype(np.float32),
                int(sample_rate),
            )

        except Exception as torchaudio_error:
            raise RuntimeError(
                f"Unable to load audio file: {input_path}\n"
                f"librosa error: {librosa_error}\n"
                f"torchaudio error: {torchaudio_error}"
            )


def resample_audio(
    audio: np.ndarray,
    sample_rate: int,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
):
    """
    Resample audio to the target sample rate.

    If the audio is already at the target sample rate,
    no resampling is performed.
    """

    if sample_rate == target_sample_rate:
        return audio, sample_rate

    resampled_audio = librosa.resample(
        audio,
        orig_sr=sample_rate,
        target_sr=target_sample_rate,
    )

    return (
        resampled_audio.astype(np.float32),
        target_sample_rate,
    )


def normalize_audio(
    audio: np.ndarray,
    target_peak: float = TARGET_PEAK,
):
    """
    Normalize audio so its maximum absolute amplitude
    reaches the target peak value.
    """

    if audio.size == 0:
        return audio

    max_amplitude = np.max(
        np.abs(audio)
    )

    # Prevent division by zero for silent audio.
    if max_amplitude == 0:
        return audio

    normalized_audio = (
        audio * (target_peak / max_amplitude)
    )

    return normalized_audio.astype(np.float32)


def save_audio(
    output_path: Path,
    audio: np.ndarray,
    sample_rate: int,
):
    """
    Save processed audio as a 16-bit PCM WAV file.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        str(output_path),
        audio,
        sample_rate,
        subtype="PCM_16",
    )