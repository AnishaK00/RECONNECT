from dataclasses import dataclass

import numpy as np
import torch

from silero_vad import (
    get_speech_timestamps,
    load_silero_vad,
)


# ---------------------------------------------------------
# VAD Configuration
# ---------------------------------------------------------

VAD_THRESHOLD = 0.5
MIN_SPEECH_DURATION_MS = 250
MIN_SILENCE_DURATION_MS = 100
SPEECH_PADDING_MS = 150


# ---------------------------------------------------------
# Speech Region
# ---------------------------------------------------------

@dataclass
class SpeechRegion:
    """
    Represents a candidate speech region detected by VAD.

    These are NOT final speaker segments.
    """

    segment_id: str

    start_sec: float
    end_sec: float

    start_sample: int
    end_sample: int

    confidence: float | None = None


# ---------------------------------------------------------
# Load Silero VAD model once
# ---------------------------------------------------------

VAD_MODEL = load_silero_vad()


# ---------------------------------------------------------
# Merge overlapping regions
# ---------------------------------------------------------

def merge_overlapping_regions(regions):
    """
    Merge overlapping speech regions.

    Internal regions use sample positions.
    """

    if not regions:
        return []

    sorted_regions = sorted(
        regions,
        key=lambda region: region["start_sample"],
    )

    merged = [
        sorted_regions[0].copy()
    ]

    for current in sorted_regions[1:]:

        previous = merged[-1]

        if (
            current["start_sample"]
            <= previous["end_sample"]
        ):

            previous["end_sample"] = max(
                previous["end_sample"],
                current["end_sample"],
            )

        else:

            merged.append(
                current.copy()
            )

    return merged


# ---------------------------------------------------------
# Detect speech regions
# ---------------------------------------------------------

def detect_speech_regions(
    audio: np.ndarray,
    sample_rate: int,
) -> list[SpeechRegion]:
    """
    Detect candidate speech regions using Silero VAD.

    Processing:

        Audio
          ↓
        Silero VAD
          ↓
        Raw speech timestamps
          ↓
        Add speech padding
          ↓
        Merge overlapping regions
          ↓
        SpeechRegion objects

    These are candidate speech regions.

    They are NOT final speaker segments.
    """

    # -----------------------------------------------------
    # Convert NumPy → PyTorch
    # -----------------------------------------------------

    waveform = torch.from_numpy(
        audio
    ).float()

    # -----------------------------------------------------
    # Run Silero VAD
    # -----------------------------------------------------

    speech_timestamps = get_speech_timestamps(
        waveform,
        VAD_MODEL,
        threshold=VAD_THRESHOLD,
        sampling_rate=sample_rate,
        min_speech_duration_ms=MIN_SPEECH_DURATION_MS,
        min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
    )

    # -----------------------------------------------------
    # Calculate padding
    # -----------------------------------------------------

    padding_samples = int(
        SPEECH_PADDING_MS
        / 1000
        * sample_rate
    )

    total_samples = len(audio)

    # -----------------------------------------------------
    # Add padding
    # -----------------------------------------------------

    padded_regions = []

    for timestamp in speech_timestamps:

        start_sample = max(
            0,
            timestamp["start"]
            - padding_samples,
        )

        end_sample = min(
            total_samples,
            timestamp["end"]
            + padding_samples,
        )

        padded_regions.append(
            {
                "start_sample": start_sample,
                "end_sample": end_sample,
            }
        )

    # -----------------------------------------------------
    # Merge overlapping padded regions
    # -----------------------------------------------------

    merged_regions = merge_overlapping_regions(
        padded_regions
    )

    # -----------------------------------------------------
    # Create SpeechRegion objects
    # -----------------------------------------------------

    speech_regions: list[SpeechRegion] = []

    for index, region in enumerate(
        merged_regions,
        start=1,
    ):

        start_sample = region[
            "start_sample"
        ]

        end_sample = region[
            "end_sample"
        ]

        speech_regions.append(
            SpeechRegion(
                segment_id=(
                    f"seg_{index:03d}"
                ),

                start_sec=round(
                    start_sample
                    / sample_rate,
                    3,
                ),

                end_sec=round(
                    end_sample
                    / sample_rate,
                    3,
                ),

                start_sample=start_sample,

                end_sample=end_sample,

                confidence=None,
            )
        )

    return speech_regions