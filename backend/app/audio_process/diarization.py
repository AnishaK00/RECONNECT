"""Speaker diarization using pyannote-audio."""

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import torch
from pyannote.audio import Pipeline

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DIARIZATION_MODEL = (
    "pyannote/speaker-diarization-community-1"
)

TARGET_SAMPLE_RATE = 16000

# Cache the loaded pipeline so the model is not loaded
# again for every audio request.
_pipeline = None


# ---------------------------------------------------------
# Result structure
# ---------------------------------------------------------

@dataclass
class DiarizationTurn:
    """
    Represents one speaker turn.

    Times are always relative to the ORIGINAL
    recording timeline.
    """

    start_sec: float
    end_sec: float
    speaker_label: str


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

def _get_device() -> torch.device:
    """
    Select the best available PyTorch device.

    Priority:
        1. Apple MPS
        2. CUDA
        3. CPU
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ---------------------------------------------------------
# Speaker label normalization
# ---------------------------------------------------------

def _normalize_speaker_label(
    label: str,
) -> str:
    """
    Normalize pyannote speaker labels into:

        SPEAKER_00
        SPEAKER_01
        SPEAKER_02
        ...

    This keeps the format consistent for downstream
    processing.
    """

    if label.upper().startswith("SPEAKER"):

        parts = (
            label.upper()
            .replace("-", "_")
            .split("_")
        )

        if (
            len(parts) >= 2
            and parts[-1].isdigit()
        ):
            return (
                f"SPEAKER_"
                f"{int(parts[-1]):02d}"
            )

    return label.upper()


# ---------------------------------------------------------
# Load diarization pipeline
# ---------------------------------------------------------

def _get_diarization_pipeline():
    """
    Load and cache the pyannote diarization pipeline.

    The pipeline is loaded only once and reused for
    subsequent audio-processing requests.
    """

    global _pipeline

    if _pipeline is not None:
        return _pipeline

    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get(
            "HUGGINGFACE_TOKEN"
        )
    )

    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN or HUGGINGFACE_TOKEN "
            "environment variable is required "
            "for speaker diarization."
        )

    _pipeline = Pipeline.from_pretrained(
        DIARIZATION_MODEL,
        token=hf_token,
    )

    device = _get_device()

    _pipeline.to(device)

    return _pipeline


# ---------------------------------------------------------
# Reconstruct full-length cleaned audio
# ---------------------------------------------------------

def reconstruct_clean_audio(
    cleaned_regions: list[dict[str, Any]],
    total_samples: int,
    sample_rate: int,
) -> np.ndarray:
    """
    Reconstruct a full-length cleaned waveform.

    Each cleaned VAD region is placed back at its
    original sample position.

    Non-speech regions remain silent.


    Args:
        cleaned_regions:
            Regions produced by VAD + noise reduction.

            Each region must contain:

                start_sample
                end_sample
                audio
                segment_id

        total_samples:
            Number of samples in the complete
            processed recording.

        sample_rate:
            Expected to be 16000 Hz.

    Returns:
        Full-length cleaned audio as float32 NumPy array.
    """

    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"Expected {TARGET_SAMPLE_RATE} Hz audio, "
            f"got {sample_rate} Hz."
        )

    if total_samples <= 0:
        raise ValueError(
            "total_samples must be greater than zero."
        )

    cleaned_full_audio = np.zeros(
        total_samples,
        dtype=np.float32,
    )

    for region in cleaned_regions:

        start = region["start_sample"]
        end = region["end_sample"]

        cleaned_audio = np.asarray(
            region["audio"],
            dtype=np.float32,
        )

        # Validate boundaries.
        if start < 0 or end > total_samples:
            raise ValueError(
                f"Invalid sample boundaries for "
                f"{region['segment_id']}: "
                f"{start} → {end}"
            )

        expected_length = end - start

        # Noise reduction should preserve the
        # region duration.
        if len(cleaned_audio) != expected_length:
            raise ValueError(
                f"Cleaned audio length mismatch for "
                f"{region['segment_id']}: "
                f"expected {expected_length}, "
                f"got {len(cleaned_audio)}."
            )

        cleaned_full_audio[start:end] = (
            cleaned_audio
        )

    return cleaned_full_audio


# ---------------------------------------------------------
# Run whole-recording diarization
# ---------------------------------------------------------

def diarize_audio(
    cleaned_audio: np.ndarray,
    sample_rate: int,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[DiarizationTurn]:
    """
    Run speaker diarization on the reconstructed
    full-length cleaned recording.

    Args:
        cleaned_audio:
            Full-length cleaned waveform.

        sample_rate:
            Expected to be 16000 Hz.

        min_speakers:
            Optional minimum number of speakers.

        max_speakers:
            Optional maximum number of speakers.

    Returns:
        Ordered list of DiarizationTurn objects.
    """

    if cleaned_audio.size == 0:
        return []

    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"Expected {TARGET_SAMPLE_RATE} Hz audio, "
            f"got {sample_rate} Hz."
        )

    pipeline = _get_diarization_pipeline()

    waveform = torch.from_numpy(
        cleaned_audio.astype(np.float32)
    )

    # Convert:
    #
    # [samples]
    #
    # into:
    #
    # [channels, samples]
    #
    # because our audio is mono.
    waveform = waveform.unsqueeze(0)

    diarization_kwargs = {}

    if min_speakers is not None:
        diarization_kwargs[
            "min_speakers"
        ] = min_speakers

    if max_speakers is not None:
        diarization_kwargs[
            "max_speakers"
        ] = max_speakers

    output = pipeline(
        {
            "waveform": waveform,
            "sample_rate": sample_rate,
        },
        **diarization_kwargs,
    )

    turns: list[DiarizationTurn] = []

    for turn, speaker in (
        output.speaker_diarization
    ):

        turns.append(
            DiarizationTurn(
                start_sec=round(
                    turn.start,
                    3,
                ),
                end_sec=round(
                    turn.end,
                    3,
                ),
                speaker_label=(
                    _normalize_speaker_label(
                        speaker
                    )
                ),
            )
        )

    # Ensure chronological order.
    turns.sort(
        key=lambda turn: turn.start_sec
    )

    return turns


# ---------------------------------------------------------
# Development fallback
# ---------------------------------------------------------

def diarize_from_vad_fallback(
    speech_regions: list,
) -> list[DiarizationTurn]:
    """
    Development fallback when pyannote is unavailable.

    Each VAD region is temporarily assigned
    SPEAKER_00.

    IMPORTANT:
        This is NOT actual speaker diarization.
        It should only be used for development/testing.
    """

    return [
        DiarizationTurn(
            start_sec=region.start_sec,
            end_sec=region.end_sec,
            speaker_label="SPEAKER_00",
        )
        for region in speech_regions
    ]