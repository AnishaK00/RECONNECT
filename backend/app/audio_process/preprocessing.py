"""Offline preprocessing: VAD-guided denoising followed by normalization."""

from pathlib import Path

import numpy as np

from .audio_utils import TARGET_PEAK, TARGET_SAMPLE_RATE, load_audio, normalize_audio, resample_audio, save_audio
from .noise_reduction import build_noise_profile, reduce_noise
from .vad import detect_speech_regions


def preprocess_audio(
    input_path: Path,
    output_path: Path,
    segments_output_dir: Path | None = None,
    session_noise_profile: np.ndarray | None = None,
):
    """Clean speech regions using session/recording noise, then normalize."""

    audio, original_sample_rate = load_audio(input_path)
    audio, processed_sample_rate = resample_audio(
        audio, original_sample_rate, TARGET_SAMPLE_RATE
    )
    raw_rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0

    # VAD sees original audio. The output is normalized only after reduction,
    # preventing residual fan/AC noise from being amplified beforehand.
    speech_regions = detect_speech_regions(audio, processed_sample_rate)
    if session_noise_profile is not None and session_noise_profile.size:
        noise_profile = np.asarray(session_noise_profile, dtype=np.float32)
        noise_profile_source = "session"
    else:
        noise_profile = build_noise_profile(audio, speech_regions)
        noise_profile_source = "recording" if noise_profile is not None else "unavailable"

    # IMPORTANT: Start with a copy of the original audio so we don't accidentally mute
    # real speech if the VAD model incorrectly misses a region.
    cleaned_audio = np.copy(audio)
    if noise_profile is not None:
        for region in speech_regions:
            start, end = region.start_sample, region.end_sample
            cleaned_audio[start:end] = reduce_noise(
                audio[start:end], processed_sample_rate, noise_profile
            )

    cleaned_audio = normalize_audio(cleaned_audio, TARGET_PEAK)
    cleaned_rms = (
        float(np.sqrt(np.mean(np.square(cleaned_audio))))
        if cleaned_audio.size else 0.0
    )
    save_audio(output_path, cleaned_audio, processed_sample_rate)

    return {
        "audio": cleaned_audio,
        "original_sample_rate": original_sample_rate,
        "processed_sample_rate": processed_sample_rate,
        "duration": len(cleaned_audio) / processed_sample_rate,
        "speech_region_count": len(speech_regions),
        "speech_regions": speech_regions,
        "noise_profile_available": noise_profile is not None,
        "noise_profile_source": noise_profile_source,
        "noise_profile_duration_sec": len(noise_profile) / processed_sample_rate if noise_profile is not None else 0.0,
        "noise_reduction_applied": bool(noise_profile is not None and speech_regions),
        "raw_rms": raw_rms,
        "cleaned_rms": cleaned_rms,
        "target_peak": TARGET_PEAK,
        "cleaned_audio_path": str(output_path),
        "preprocessing_completed": True,
    }
