"""Local Whisper transcription for exported speaker clips."""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

_MODEL = None


def _device_and_compute_type() -> tuple[str, str]:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def get_whisper_model():
    """Load faster-whisper once per process."""

    global _MODEL
    if _MODEL is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise ImportError(
                "faster-whisper is not installed. Run: pip install faster-whisper"
            ) from error

        device, compute_type = _device_and_compute_type()
        LOGGER.info(
            "[WHISPER] loading model=base device=%s compute_type=%s",
            device,
            compute_type,
        )
        _MODEL = WhisperModel("base", device=device, compute_type=compute_type)
    return _MODEL


def transcribe_wav(audio_path: str | Path) -> str:
    """Return the transcript for one speaker WAV, or empty string on failure."""

    path = Path(audio_path)
    if not path.exists() or path.stat().st_size == 0:
        LOGGER.warning("[WHISPER] skipped missing/empty path=%s", path)
        return ""
    try:
        model = get_whisper_model()
        segments, _info = model.transcribe(str(path), language=None)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        if text:
            LOGGER.info("[WHISPER] \"%s\" (%s)", text, path.name)
        return text
    except ImportError:
        LOGGER.exception(
            "[WHISPER] faster-whisper missing; install with: pip install faster-whisper"
        )
        return ""
    except Exception:
        LOGGER.exception("[WHISPER] failed path=%s", path)
        return ""
