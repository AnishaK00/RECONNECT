"""Merge VAD + diarization and export per-segment WAV files."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from app.audio_process.diarization import DiarizationTurn
from app.audio_process.vad import SpeechRegion


@dataclass
class ExportedSegment:
    segment_id: str
    start_sec: float
    end_sec: float
    speaker_label: str
    audio_path: str
    vad_confidence: float | None
    speaker_source: str = "diarization"


def fallback_vad_segments(
    vad_regions: list[SpeechRegion],
    min_duration_sec: float = 0.5,
) -> list[ExportedSegment]:
    """Export valid speech even when diarization has insufficient context."""

    return [
        ExportedSegment(
            segment_id=f"seg_{index:03d}",
            start_sec=region.start_sec,
            end_sec=region.end_sec,
            speaker_label="SPEAKER_UNKNOWN",
            audio_path="",
            vad_confidence=region.confidence,
            speaker_source="vad_fallback",
        )
        for index, region in enumerate(vad_regions, start=1)
        if region.end_sec - region.start_sec >= min_duration_sec
    ]


def _overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    """Return the duration of overlap between two time intervals."""

    return max(
        0.0,
        min(end_a, end_b)
        - max(start_a, start_b),
    )


def _subtract_intervals(
    start_sec: float,
    end_sec: float,
    covered: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return pieces of [start_sec, end_sec] not covered by any covered interval."""

    pieces = [(start_sec, end_sec)]
    for cover_start, cover_end in sorted(covered):
        next_pieces: list[tuple[float, float]] = []
        for piece_start, piece_end in pieces:
            if cover_end <= piece_start or cover_start >= piece_end:
                next_pieces.append((piece_start, piece_end))
                continue
            if cover_start > piece_start:
                next_pieces.append((piece_start, cover_start))
            if cover_end < piece_end:
                next_pieces.append((cover_end, piece_end))
        pieces = next_pieces
    return pieces


def _clamp_segment(
    start_sec: float,
    end_sec: float,
    total_duration_sec: float | None,
) -> tuple[float, float]:
    if total_duration_sec is not None:
        start_sec = max(0.0, start_sec)
        end_sec = min(total_duration_sec, end_sec)
    return start_sec, end_sec


def merge_vad_and_diarization(
    diarization_turns: list[DiarizationTurn],
    vad_regions: list[SpeechRegion],
    min_duration_sec: float = 0.5,
    pad_sec: float = 0.0,
    total_duration_sec: float | None = None,
) -> list[ExportedSegment]:
    """
    Merge diarization turns with VAD speech regions.

    VAD defines the valid speech area.
    Diarization defines the speaker identity.

    Each diarization turn is intersected with each overlapping VAD region
    separately (pairwise). That preserves speech when one turn spans a short
    VAD gap, or when multiple VAD bursts belong to the same speaker.

    VAD speech that diarization never labeled is still exported so a missed
    turn cannot drop most of an utterance.
    """

    if not vad_regions:
        return []

    merged: list[ExportedSegment] = []
    seg_index = 1

    if diarization_turns:
        for turn in diarization_turns:
            for vad in vad_regions:
                if (
                    _overlap(
                        turn.start_sec,
                        turn.end_sec,
                        vad.start_sec,
                        vad.end_sec,
                    )
                    <= 0
                ):
                    continue

                start_sec = max(turn.start_sec, vad.start_sec)
                end_sec = min(turn.end_sec, vad.end_sec)

                # Optional padding stays inside this VAD region only.
                start_sec = max(vad.start_sec, start_sec - pad_sec)
                end_sec = min(vad.end_sec, end_sec + pad_sec)
                start_sec, end_sec = _clamp_segment(
                    start_sec, end_sec, total_duration_sec
                )

                duration = end_sec - start_sec
                if duration < min_duration_sec:
                    continue

                vad_confidence = (
                    round(float(vad.confidence), 3)
                    if vad.confidence is not None
                    else None
                )
                merged.append(
                    ExportedSegment(
                        segment_id=f"seg_{seg_index:03d}",
                        start_sec=round(start_sec, 3),
                        end_sec=round(end_sec, 3),
                        speaker_label=turn.speaker_label,
                        audio_path="",
                        vad_confidence=vad_confidence,
                        speaker_source="diarization",
                    )
                )
                seg_index += 1

    # Keep VAD speech that no diarization turn covered. If the recording only
    # has one diarized speaker, inherit that label (same-speaker continuation).
    speaker_labels = {turn.speaker_label for turn in diarization_turns}
    if len(speaker_labels) == 1:
        gap_label = next(iter(speaker_labels))
        gap_source = "diarization_gapfill"
    else:
        gap_label = "SPEAKER_UNKNOWN"
        gap_source = "vad_uncovered"

    covered = [(segment.start_sec, segment.end_sec) for segment in merged]
    for vad in vad_regions:
        for start_sec, end_sec in _subtract_intervals(
            vad.start_sec, vad.end_sec, covered
        ):
            start_sec, end_sec = _clamp_segment(
                start_sec, end_sec, total_duration_sec
            )
            if end_sec - start_sec < min_duration_sec:
                continue
            vad_confidence = (
                round(float(vad.confidence), 3)
                if vad.confidence is not None
                else None
            )
            merged.append(
                ExportedSegment(
                    segment_id=f"seg_{seg_index:03d}",
                    start_sec=round(start_sec, 3),
                    end_sec=round(end_sec, 3),
                    speaker_label=gap_label,
                    audio_path="",
                    vad_confidence=vad_confidence,
                    speaker_source=gap_source,
                )
            )
            seg_index += 1

    merged.sort(key=lambda segment: (segment.start_sec, segment.end_sec))
    for index, segment in enumerate(merged, start=1):
        segment.segment_id = f"seg_{index:03d}"
    return merged


def export_segment_wavs(
    audio: np.ndarray,
    sample_rate: int,
    segments: list[ExportedSegment],
    output_dir: Path,
) -> list[ExportedSegment]:
    """
    Export each merged segment as an individual WAV file.

    The audio passed here should be the reconstructed
    full-length CLEANED audio.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    exported: list[ExportedSegment] = []
    speaker_numbers = {
        label: index
        for index, label in enumerate(
            sorted({segment.speaker_label for segment in segments}),
            start=1,
        )
    }
    speaker_occurrences: dict[str, int] = {}

    for segment in segments:

        start_sample = int(
            segment.start_sec
            * sample_rate
        )

        end_sample = int(
            segment.end_sec
            * sample_rate
        )

        start_sample = max(
            0,
            start_sample,
        )

        end_sample = min(
            end_sample,
            len(audio),
        )

        if end_sample <= start_sample:
            continue

        clip = audio[
            start_sample:end_sample
        ]

        speaker_number = speaker_numbers[segment.speaker_label]
        speaker_occurrences[segment.speaker_label] = (
            speaker_occurrences.get(segment.speaker_label, 0) + 1
        )
        out_path = output_dir / (
            (
                f"speaker_unknown_{speaker_occurrences[segment.speaker_label]:03d}.wav"
                if segment.speaker_label == "SPEAKER_UNKNOWN"
                else f"speaker_{speaker_number:02d}_{speaker_occurrences[segment.speaker_label]:03d}.wav"
            )
        )

        sf.write(
            out_path,
            clip,
            sample_rate,
            subtype="PCM_16",
        )

        exported.append(
            ExportedSegment(
                segment_id=segment.segment_id,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                speaker_label=segment.speaker_label,
                audio_path=str(out_path),
                vad_confidence=segment.vad_confidence,
                speaker_source=segment.speaker_source,
            )
        )

    return exported
