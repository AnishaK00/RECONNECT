from collections import deque

import pytest

from app.audio_stream.streaming_vad import (
    SEGMENT_SILENCE_DURATION_MS,
    VAD_FRAME_SAMPLES,
    VADEventType,
    VADState,
    StreamingVAD,
)
from app.audio_stream.protocol import (
    SAMPLE_WIDTH_BYTES,
)


# ---------------------------------------------------------
# Test constants
# ---------------------------------------------------------

SPEECH_PROBABILITY = 0.9
SILENCE_PROBABILITY = 0.1


def make_pcm16_audio(num_samples: int) -> bytes:
    """
    Create dummy PCM16 mono audio.

    The actual values do not matter because speech
    probabilities are mocked in these tests.
    """

    return b"\x00\x00" * num_samples


class MockStreamingVAD(StreamingVAD):
    """
    StreamingVAD with mocked speech probabilities.

    This allows us to test buffering and state-machine
    behavior without depending on real Silero predictions.
    """

    def __init__(
        self,
        probabilities: list[float],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.probabilities = deque(probabilities)

    def _get_speech_probability(
        self,
        data: bytes,
    ) -> float:
        """
        Return predefined probabilities instead of running
        the actual Silero model.
        """

        if not self.probabilities:
            raise RuntimeError(
                "No mocked speech probabilities remaining"
            )

        return self.probabilities.popleft()


# ---------------------------------------------------------
# TEST 1
# Empty / initial state
# ---------------------------------------------------------

def test_initial_state():
    vad = MockStreamingVAD(
        probabilities=[],
    )

    assert vad.state == VADState.IDLE
    assert vad.buffered_samples == 0
    assert vad.silence_duration_ms == 0.0

    print("\nTEST 1 PASSED")
    print("Initial state: IDLE")
    print("Initial buffer: 0 samples")


# ---------------------------------------------------------
# TEST 2
# 320 samples is not enough for Silero
# ---------------------------------------------------------

def test_buffering_with_incomplete_frame():
    vad = MockStreamingVAD(
        probabilities=[],
    )

    audio = make_pcm16_audio(320)

    events = vad.process(audio)

    assert events == []

    assert vad.buffered_samples == 320

    print("\nTEST 2 PASSED")
    print("Input: 320 samples")
    print("Events:", len(events))
    print("Buffered samples:", vad.buffered_samples)


# ---------------------------------------------------------
# TEST 3
# Two 320-sample chunks produce one frame
# and preserve the remaining samples
# ---------------------------------------------------------

def test_buffering_preserves_remaining_audio():
    vad = MockStreamingVAD(
        probabilities=[SILENCE_PROBABILITY],
    )

    first_chunk = make_pcm16_audio(320)
    second_chunk = make_pcm16_audio(320)

    first_events = vad.process(first_chunk)

    assert first_events == []
    assert vad.buffered_samples == 320

    second_events = vad.process(second_chunk)

    assert len(second_events) == 1

    assert (
        vad.buffered_samples
        == 640 - VAD_FRAME_SAMPLES
    )

    assert vad.buffered_samples == 128

    print("\nTEST 3 PASSED")
    print("First chunk: 320 samples")
    print("Second chunk: 320 samples")
    print("Silero frame processed:", VAD_FRAME_SAMPLES)
    print("Remaining samples:", vad.buffered_samples)


# ---------------------------------------------------------
# TEST 4
# IDLE -> SPEAKING
# ---------------------------------------------------------

def test_speech_started():
    vad = MockStreamingVAD(
        probabilities=[SPEECH_PROBABILITY],
    )

    audio = make_pcm16_audio(
        VAD_FRAME_SAMPLES
    )

    events = vad.process(audio)

    assert len(events) == 1

    event = events[0]

    assert (
        event.event_type
        == VADEventType.SPEECH_STARTED
    )

    assert event.state == VADState.SPEAKING

    assert event.is_speech is True

    assert vad.state == VADState.SPEAKING

    print("\nTEST 4 PASSED")
    print("Event:", event.event_type.value)
    print("State:", event.state.value)
    print("Probability:", event.speech_probability)


# ---------------------------------------------------------
# TEST 5
# Speech continues
# ---------------------------------------------------------

def test_speech_continuing():
    vad = MockStreamingVAD(
        probabilities=[
            SPEECH_PROBABILITY,
            SPEECH_PROBABILITY,
        ],
    )

    audio = make_pcm16_audio(
        VAD_FRAME_SAMPLES * 2
    )

    events = vad.process(audio)

    assert len(events) == 2

    assert (
        events[0].event_type
        == VADEventType.SPEECH_STARTED
    )

    assert (
        events[1].event_type
        == VADEventType.SPEECH_CONTINUING
    )

    assert vad.state == VADState.SPEAKING

    assert vad.silence_duration_ms == 0.0

    print("\nTEST 5 PASSED")
    print("First event:", events[0].event_type.value)
    print("Second event:", events[1].event_type.value)
    print("Current state:", vad.state.value)


# ---------------------------------------------------------
# TEST 6
# Short silence does not end speech
# ---------------------------------------------------------

def test_short_silence_keeps_speaking():
    vad = MockStreamingVAD(
        probabilities=[
            SPEECH_PROBABILITY,
            SILENCE_PROBABILITY,
        ],
    )

    audio = make_pcm16_audio(
        VAD_FRAME_SAMPLES * 2
    )

    events = vad.process(audio)

    assert len(events) == 2

    assert (
        events[0].event_type
        == VADEventType.SPEECH_STARTED
    )

    assert (
        events[1].event_type
        == VADEventType.NONE
    )

    assert vad.state == VADState.SPEAKING

    assert vad.silence_duration_ms > 0

    assert (
        vad.silence_duration_ms
        < SEGMENT_SILENCE_DURATION_MS
    )

    print("\nTEST 6 PASSED")
    print("Short silence duration:")
    print(vad.silence_duration_ms, "ms")
    print("State remains:", vad.state.value)


# ---------------------------------------------------------
# TEST 7
# Speech resumes after short silence
# ---------------------------------------------------------

def test_speech_resumes_after_short_silence():
    vad = MockStreamingVAD(
        probabilities=[
            SPEECH_PROBABILITY,
            SILENCE_PROBABILITY,
            SILENCE_PROBABILITY,
            SPEECH_PROBABILITY,
        ],
    )

    audio = make_pcm16_audio(
        VAD_FRAME_SAMPLES * 4
    )

    events = vad.process(audio)

    assert len(events) == 4

    assert (
        events[0].event_type
        == VADEventType.SPEECH_STARTED
    )

    assert (
        events[1].event_type
        == VADEventType.NONE
    )

    assert (
        events[2].event_type
        == VADEventType.NONE
    )

    assert (
        events[3].event_type
        == VADEventType.SPEECH_CONTINUING
    )

    assert vad.state == VADState.SPEAKING

    assert vad.silence_duration_ms == 0.0

    print("\nTEST 7 PASSED")
    print("Speech resumed successfully")
    print("Silence counter reset:", vad.silence_duration_ms)


# ---------------------------------------------------------
# TEST 8
# Long silence ends speech
# ---------------------------------------------------------

def test_long_silence_ends_speech():
    """
    Start with speech.

    Then provide enough non-speech frames to exceed
    the configured silence duration limit.
    """

    vad = MockStreamingVAD(
        probabilities=[
            SPEECH_PROBABILITY,
        ]
        + [
            SILENCE_PROBABILITY
        ] * 20,
    )

    audio = make_pcm16_audio(
        VAD_FRAME_SAMPLES * 21
    )

    events = vad.process(audio)

    event_types = [
        event.event_type
        for event in events
    ]

    assert (
        VADEventType.SPEECH_STARTED
        in event_types
    )

    assert (
        VADEventType.SPEECH_ENDED
        in event_types
    )

    speech_end_events = [
        event
        for event in events
        if event.event_type
        == VADEventType.SPEECH_ENDED
    ]

    assert len(speech_end_events) == 1

    assert vad.state == VADState.IDLE

    print("\nTEST 8 PASSED")
    print("Speech ended after sufficient silence")
    print(
        "Final state:",
        vad.state.value,
    )


# ---------------------------------------------------------
# TEST 9
# Multiple inference frames from one input
# ---------------------------------------------------------

def test_multiple_frames_from_single_input():
    vad = MockStreamingVAD(
        probabilities=[
            SPEECH_PROBABILITY,
            SPEECH_PROBABILITY,
            SILENCE_PROBABILITY,
        ],
    )

    audio = make_pcm16_audio(
        VAD_FRAME_SAMPLES * 3
    )

    events = vad.process(audio)

    assert len(events) == 3

    assert (
        events[0].event_type
        == VADEventType.SPEECH_STARTED
    )

    assert (
        events[1].event_type
        == VADEventType.SPEECH_CONTINUING
    )

    assert (
        events[2].event_type
        == VADEventType.NONE
    )

    assert vad.buffered_samples == 0

    assert vad.state == VADState.SPEAKING

    print("\nTEST 9 PASSED")
    print("Frames processed:", len(events))
    print(
        "Final buffered samples:",
        vad.buffered_samples,
    )


# ---------------------------------------------------------
# TEST 10
# Reset behavior
# ---------------------------------------------------------

def test_reset():
    vad = MockStreamingVAD(
        probabilities=[
            SPEECH_PROBABILITY,
        ],
    )

    # Add exactly one frame to enter SPEAKING.
    audio = make_pcm16_audio(
        VAD_FRAME_SAMPLES
    )

    vad.process(audio)

    assert vad.state == VADState.SPEAKING

    # Add incomplete audio to the internal buffer.
    incomplete_audio = make_pcm16_audio(320)

    vad.process(incomplete_audio)

    assert vad.buffered_samples == 320

    # Reset everything.
    vad.reset()

    assert vad.state == VADState.IDLE

    assert vad.silence_duration_ms == 0.0

    assert vad.buffered_samples == 0

    print("\nTEST 10 PASSED")
    print("State after reset:", vad.state.value)
    print(
        "Buffered samples after reset:",
        vad.buffered_samples,
    )