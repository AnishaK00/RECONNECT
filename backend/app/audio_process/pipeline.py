"""End-to-end RECONNECT audio perception pipeline."""

from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
from collections.abc import Callable
import torch
import torchaudio

# Patch torchaudio for compatibility with speechbrain (required by Pyannote/ECAPA-TDNN)
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
if not hasattr(torchaudio, 'io'):
    class DummyIO:
        class StreamReader:
            pass
    torchaudio.io = DummyIO()

from app.audio_process.preprocessing import preprocess_audio
from app.audio_process.diarization import (
    DIARIZATION_MODEL,
    diarize_audio,
)
from app.audio_process.segmentation import (
    fallback_vad_segments,
    merge_vad_and_diarization,
    export_segment_wavs,
)
from app.audio_process.vad import VAD_MODEL

from app.schemas.audio import (
    AudioInfo,
    AudioPerceptionResult,
    AudioSegment,
    PipelineInfo,
    PreprocessingInfo,
    StreamSegmentInfo,
)
from database.db import get_identity_by_name


PipelineProgressCallback = Callable[[str, str, dict[str, object]], None]
IdentityCallback = Callable[[dict[str, object]], None]


def _report(
    callback: PipelineProgressCallback | None,
    stage: str,
    status: str,
    **details: object,
) -> None:
    """Emit optional progress without coupling the offline pipeline to logs."""

    if callback is not None:
        callback(stage, status, details)


def run_audio_pipeline(
    original_path: Path,
    cleaned_path: Path,
    segments_output_dir: Path,
    session_id: str,
    original_filename: str | None = None,
    stream_segment: StreamSegmentInfo | None = None,
    session_noise_profile = None,
    progress_callback: PipelineProgressCallback | None = None,
    identity_callback: IdentityCallback | None = None,
) -> AudioPerceptionResult:
    """
    Run the complete RECONNECT audio perception pipeline.

    Pipeline:

        Original Audio
              ↓
        Load / Resample / Normalize
              ↓
        Silero VAD
              ↓
        Noise Profile
              ↓
        Noise Reduction
              ↓
        Full-Length Clean Audio
              ↓
        Pyannote Diarization
              ↓
        VAD + Diarization Merge
              ↓
        Segment WAV Export
              ↓
        AudioPerceptionResult
    """

    pipeline_start = time.perf_counter()

    # =========================================================
    # 1. PREPROCESSING
    # =========================================================

    _report(progress_callback, "preprocessing", "started")
    preprocessing = preprocess_audio(
        input_path=original_path,
        output_path=cleaned_path,
        session_noise_profile=session_noise_profile,
    )

    audio = preprocessing["audio"]

    sample_rate = preprocessing[
        "processed_sample_rate"
    ]

    duration_sec = preprocessing[
        "duration"
    ]

    vad_regions = preprocessing[
        "speech_regions"
    ]
    _report(
        progress_callback,
        "preprocessing",
        "completed",
        duration_sec=round(duration_sec, 3),
        speech_region_count=len(vad_regions),
        noise_profile_source=preprocessing["noise_profile_source"],
        noise_profile_duration_sec=round(preprocessing["noise_profile_duration_sec"], 3),
        noise_reduction_applied=preprocessing["noise_reduction_applied"],
    )

    # =========================================================
    # 2. DIARIZATION
    # =========================================================

    diarization_turns = []
    if duration_sec >= 10.0:
        _report(progress_callback, "diarization", "started")
        diarization_turns = diarize_audio(cleaned_audio=audio, sample_rate=sample_rate)
        _report(progress_callback, "diarization", "completed", turn_count=len(diarization_turns))
    else:
        _report(progress_callback, "diarization", "skipped", reason="recording_under_10_seconds")

    # =========================================================
    # 3. MERGE VAD + DIARIZATION
    # =========================================================

    _report(progress_callback, "segmentation", "started")
    merged_segments = merge_vad_and_diarization(
        diarization_turns=diarization_turns, vad_regions=vad_regions,
        min_duration_sec=0.5, pad_sec=0.0, total_duration_sec=duration_sec,
    )
    fallback_reason = None
    if vad_regions and not merged_segments:
        fallback_reason = "recording_under_10_seconds" if duration_sec < 10.0 else "no_diarization_turns"
        merged_segments = fallback_vad_segments(vad_regions, min_duration_sec=0.5)
    _report(
        progress_callback,
        "segmentation",
        "completed",
        merged_segment_count=len(merged_segments),
        fallback_reason=fallback_reason,
    )

    # =========================================================
    # 4. EXPORT FINAL SEGMENT WAV FILES
    # =========================================================

    _report(progress_callback, "export", "started")
    exported_segments = export_segment_wavs(
        audio=audio,
        sample_rate=sample_rate,
        segments=merged_segments,
        output_dir=segments_output_dir,
    )
    _report(
        progress_callback,
        "export",
        "completed",
        final_segment_count=len(exported_segments),
    )

    # =========================================================
    # 5. RECOGNITION & SCHEMAS
    # =========================================================
    
    from app.audio_process.recognition import (
        AudioEmbedder,
        AudioMatcher,
        AudioAggregator,
        SessionSpeakerMemory,
        load_audio_tensor,
    )
    from database.db import (
        create_unenrolled_identity,
        update_unenrolled_voice,
        find_matching_unenrolled_voice,
    )
    global _embedder, _matcher, _aggregator, _session_memory, _triggered_unknowns
    if '_embedder' not in globals():
        _embedder = AudioEmbedder()
        _matcher = AudioMatcher()
        _aggregator = AudioAggregator()
        _triggered_unknowns = set()
    if '_session_memory' not in globals():
        _session_memory = SessionSpeakerMemory()

    segments = []
    _report(progress_callback, "recognition", "started")
    
    # 1. Group segments by speaker
    speaker_groups = {}
    for segment in exported_segments:
        if segment.speaker_label not in speaker_groups:
            speaker_groups[segment.speaker_label] = []
        speaker_groups[segment.speaker_label].append(segment)
        
    # 2. Process each speaker (concat clips, match gallery, then session continuity)
    speaker_identities = {}
    
    log_file_path = "data/match_logs.txt"
    from datetime import datetime
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    for speaker_label, spk_segments in speaker_groups.items():
        # Sort segments chronologically
        spk_segments = sorted(spk_segments, key=lambda s: s.start_sec)
        
        try:
            # Concatenate all audio segments for this speaker
            wavs = []
            duration = 0.0
            for seg in spk_segments:
                signal, fs = load_audio_tensor(seg.audio_path)
                if fs != 16000:
                    resampler = torchaudio.transforms.Resample(fs, 16000)
                    signal = resampler(signal)
                wavs.append(signal)
                duration += (seg.end_sec - seg.start_sec)
                
            if not wavs:
                continue
                
            concatenated_signal = torch.cat(wavs, dim=1)
            concat_path = os.path.join(segments_output_dir, f"concat_{speaker_label}.wav")
            import soundfile as sf
            sf.write(
                concat_path,
                concatenated_signal.squeeze(0).cpu().numpy(),
                16000,
            )
            
            # Extract embedding from the glued audio
            embedding = _embedder.extract_embedding(concat_path)
            matched_id, score = _matcher.match(embedding)
            match_reason = "gallery"
            
            if matched_id:
                identity = matched_id
                status = "known"
                print(f"Known identity: {identity}")
                if identity_callback is not None:
                    identity_row = get_identity_by_name(identity)
                    identity_callback({
                        "known": True,
                        "name": identity,
                        "relation": identity_row["relation"] if identity_row else None,
                    })
                _session_memory.remember(
                    session_id, identity, embedding, confirmed=True
                )
                log_msg = (
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"[MATCH]: '{identity}' (Score: {score:.3f} | "
                    f"Speaker: {speaker_label} | Duration: {duration:.2f}s | "
                    f"Via: {match_reason})"
                )
                print(f"\n{log_msg}")
            else:
                gallery_best_id, gallery_best_score = _matcher.best_match(embedding)
                continuity_id, continuity_score, continuity_reason = (
                    _session_memory.resolve(
                        session_id,
                        embedding,
                        gallery_best_id,
                        gallery_best_score,
                    )
                )
                if continuity_id is not None:
                    identity = continuity_id
                    is_unenrolled = identity.startswith("unenrolled_")
                    status = "unenrolled" if is_unenrolled else "known"
                    score = continuity_score
                    match_reason = continuity_reason or "session_continuity"
                    _session_memory.remember(
                        session_id, identity, embedding, confirmed=not is_unenrolled
                    )
                    prefix = "[MATCH]" if not is_unenrolled else "[SESSION UNKNOWN]"
                    log_msg = (
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"{prefix}: '{identity}' (Score: {score:.3f} | "
                        f"Speaker: {speaker_label} | Duration: {duration:.2f}s | "
                        f"Via: {match_reason})"
                    )
                    print(f"\n{log_msg}")
                    if identity_callback is not None:
                        if is_unenrolled:
                            identity_callback({
                                "known": False,
                                "speaker": identity,
                                "has_face": False,
                            })
                        else:
                            identity_row = get_identity_by_name(identity)
                            identity_callback({
                                "known": True,
                                "name": identity,
                                "relation": identity_row["relation"] if identity_row else None,
                            })
                else:
                    # Save concatenated audio for unknowns (longest representation)
                    # Attempt to match against unenrolled voice embeddings
                    match_row, match_score = find_matching_unenrolled_voice(embedding, threshold=0.52)
                    if match_row is not None:
                        identity = f"unenrolled_{match_row['id']}"
                        has_face = match_row.get("face_embedding") is not None
                        status = "unenrolled_with_face" if has_face else "unenrolled"
                        # Update stored voice embedding
                        update_unenrolled_voice(match_row['id'], embedding)
                    else:
                        # Create a new unenrolled identity entry
                        new_id = create_unenrolled_identity(voice_embedding=embedding)
                        identity = f"unenrolled_{new_id}"
                        status = "new_unenrolled"

                    _session_memory.remember(session_id, identity, embedding, confirmed=False)
                    score = gallery_best_score
                    face_note = " (face linked)" if status == "unenrolled_with_face" else " (no face)"
                    log_msg = (
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"[NO MATCH]: {identity}{face_note} (Highest Score: {score:.3f} | "
                        f"Speaker: {speaker_label} | Duration: {duration:.2f}s)"
                    )
                    print(f"\n{log_msg}")
                    if identity_callback is not None:
                        identity_callback({
                            "known": False,
                            "speaker": identity,
                            "has_face": status == "unenrolled_with_face",
                        })

            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(log_msg + "\n")
                
            speaker_identities[speaker_label] = {"identity": identity, "status": status}
            
        except Exception as e:
            print(f"Error during recognition for {speaker_label}: {e}")
            speaker_identities[speaker_label] = {"identity": None, "status": None}
            
    # 3. Apply to all segments
    for segment in exported_segments:
        identity_info = speaker_identities.get(segment.speaker_label, {"identity": None, "status": None})
        segments.append(
            AudioSegment(
                segment_id=segment.segment_id,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                speaker_label=segment.speaker_label,
                speaker_source=segment.speaker_source,
                audio_path=segment.audio_path,
                vad_confidence=segment.vad_confidence,
                speaker_identity=identity_info["identity"],
                match_status=identity_info["status"],
            )
        )
        
    _report(progress_callback, "recognition", "completed")

    # =========================================================
    # 6. COUNT SPEAKERS
    # =========================================================

    speaker_labels = {
        segment.speaker_label
        for segment in segments
    }

    # =========================================================
    # 7. PROCESSING TIME
    # =========================================================

    processing_time_sec = round(
        time.perf_counter() - pipeline_start,
        3,
    )

    # =========================================================
    # 8. BUILD FINAL SCHEMA
    # =========================================================

    result = AudioPerceptionResult(
        saved=True,

        session_id=session_id,

        recording_started_at=(
            datetime.now().astimezone().isoformat()
        ),

        duration_sec=round(
            duration_sec,
            3,
        ),

        original_filename=original_filename,

        stream_segment=stream_segment,

        audio=AudioInfo(
            original_path=str(
                original_path
            ),

            cleaned_path=str(
                cleaned_path
            ),

            sample_rate=sample_rate,
        ),

        preprocessing=PreprocessingInfo(
            original_sample_rate=(
                preprocessing[
                    "original_sample_rate"
                ]
            ),

            noise_profile_sec=preprocessing["noise_profile_duration_sec"],
            noise_profile_source=preprocessing["noise_profile_source"],
            noise_reduction_applied=preprocessing["noise_reduction_applied"],
            raw_rms=preprocessing["raw_rms"],
            cleaned_rms=preprocessing["cleaned_rms"],

            peak_normalized=(
                preprocessing[
                    "target_peak"
                ]
            ),

            noise_reduction_mode="vad_regions",

            preprocessing_completed=(
                preprocessing[
                    "preprocessing_completed"
                ]
            ),
        ),

        segments=segments,

        pipeline=PipelineInfo(
            vad_model="Silero VAD",

            diarization_model=(
                DIARIZATION_MODEL
            ),

            processing_time_sec=(
                processing_time_sec
            ),
        ),

        speaker_count=len(
            speaker_labels
        ),

        segment_count=len(
            segments
        ),
    )

    _report(
        progress_callback,
        "pipeline",
        "completed",
        processing_time_sec=processing_time_sec,
    )
    return result
