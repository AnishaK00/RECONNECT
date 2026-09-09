"""Schemas for RECONNECT audio perception output."""

from pydantic import BaseModel, Field


class AudioSegment(BaseModel):
    """
    One finalized speech segment.

    This is the output that can be passed to
    downstream processing such as Whisper.
    """

    segment_id: str

    start_sec: float = Field(ge=0)

    end_sec: float = Field(gt=0)

    speaker_label: str

    speaker_source: str = "diarization"

    audio_path: str

    vad_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    speaker_identity: str | None = None

    match_status: str | None = None



class AudioInfo(BaseModel):
    """Information about the audio files."""

    original_path: str

    cleaned_path: str

    sample_rate: int


class PreprocessingInfo(BaseModel):
    """Information about audio preprocessing."""

    original_sample_rate: int

    noise_profile_sec: float | None = None

    noise_profile_source: str = "unavailable"

    noise_reduction_applied: bool = False

    raw_rms: float | None = None

    cleaned_rms: float | None = None

    peak_normalized: float

    noise_reduction_mode: str = "vad_regions"

    preprocessing_completed: bool


class PipelineInfo(BaseModel):
    """Models and processing information."""

    vad_model: str

    diarization_model: str

    processing_time_sec: float | None = None


class StreamSegmentInfo(BaseModel):
    """Live-stream context for an offline pipeline invocation."""

    main_segment_id: int = Field(ge=1)

    stream_start_sec: float = Field(ge=0)

    stream_end_sec: float = Field(gt=0)

    finalized_reason: str


class AudioPerceptionResult(BaseModel):
    """
    Final output of the RECONNECT audio perception stage.

    Contains:
        - audio information
        - preprocessing information
        - finalized speaker segments
        - pipeline information

    Whisper can consume the segments list.
    """

    saved: bool

    session_id: str

    recording_started_at: str

    duration_sec: float

    original_filename: str | None = None

    stream_segment: StreamSegmentInfo | None = None

    audio: AudioInfo

    preprocessing: PreprocessingInfo

    segments: list[AudioSegment]

    pipeline: PipelineInfo

    speaker_count: int

    segment_count: int
