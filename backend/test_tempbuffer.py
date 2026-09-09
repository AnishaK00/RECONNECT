
import pytest
from app.audio_stream.protocol import (
    SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
)
from app.audio_stream.temp_buffer import (
    StreamingBuffer,
)


def create_pcm16_audio(num_samples: int) -> bytes:
    """
    Create dummy PCM16 audio with the requested
    number of samples.
    """
    return bytes(
        num_samples * SAMPLE_WIDTH_BYTES
    )


def test_empty_buffer():
    buffer = StreamingBuffer(
        buffer_sec=5.0
    )

    assert buffer.buffered_samples == 0
    assert buffer.buffered_duration == 0.0
    assert buffer.get_all() is None

    print("\nTEST 1 PASSED")
    print("Buffer is initially empty")


def test_add_audio():
    buffer = StreamingBuffer(
        buffer_sec=5.0
    )

    # Add 1 second of audio.
    buffer.add(
        create_pcm16_audio(
            SAMPLE_RATE
        )
    )

    assert buffer.buffered_samples == SAMPLE_RATE
    assert buffer.buffered_duration == 1.0

    print("\nTEST 2 PASSED")
    print(
        f"Buffered samples: "
        f"{buffer.buffered_samples}"
    )
    print(
        f"Buffered duration: "
        f"{buffer.buffered_duration}s"
    )


def test_get_recent_audio():
    buffer = StreamingBuffer(
        buffer_sec=5.0
    )

    # Add 2 seconds of audio.
    buffer.add(
        create_pcm16_audio(
            SAMPLE_RATE * 2
        )
    )

    # Request the most recent 0.5 seconds.
    recent = buffer.get_recent(
        duration_sec=0.5
    )

    assert recent is not None

    assert recent.num_samples == (
        int(SAMPLE_RATE * 0.5)
    )

    assert recent.duration_sec == 0.5

    assert recent.end_sample == (
        SAMPLE_RATE * 2
    )

    print("\nTEST 3 PASSED")
    print(
        f"Recent duration: "
        f"{recent.duration_sec}s"
    )


def test_request_more_than_available():
    buffer = StreamingBuffer(
        buffer_sec=5.0
    )

    # Only 2 seconds are available.
    buffer.add(
        create_pcm16_audio(
            SAMPLE_RATE * 2
        )
    )

    # Request 5 seconds.
    recent = buffer.get_recent(
        duration_sec=5.0
    )

    assert recent is not None

    # Should return all available audio.
    assert recent.num_samples == (
        SAMPLE_RATE * 2
    )

    assert recent.duration_sec == 2.0

    assert recent.start_sample == 0
    assert recent.end_sample == (
        SAMPLE_RATE * 2
    )

    print("\nTEST 4 PASSED")
    print(
        "Requested 5 seconds, "
        "received all available 2 seconds"
    )


def test_rolling_buffer_discards_old_audio():
    buffer = StreamingBuffer(
        buffer_sec=5.0
    )

    # Add 6 seconds of audio.
    buffer.add(
        create_pcm16_audio(
            SAMPLE_RATE * 6
        )
    )

    # Buffer must keep only the newest 5 seconds.
    assert buffer.buffered_samples == (
        SAMPLE_RATE * 5
    )

    assert buffer.buffered_duration == 5.0

    # The first second should have been discarded.
    assert buffer.start_sample == SAMPLE_RATE

    assert buffer.end_sample == (
        SAMPLE_RATE * 6
    )

    print("\nTEST 5 PASSED")
    print(
        "Added: 6 seconds"
    )
    print(
        "Buffered: 5 seconds"
    )
    print(
        f"Buffer starts at sample: "
        f"{buffer.start_sample}"
    )
    print(
        f"Buffer ends at sample: "
        f"{buffer.end_sample}"
    )


def test_get_recent_after_rolling():
    buffer = StreamingBuffer(
        buffer_sec=5.0
    )

    # Add 7 seconds.
    buffer.add(
        create_pcm16_audio(
            SAMPLE_RATE * 7
        )
    )

    # Buffer contains seconds 2 through 7.
    assert buffer.start_sample == (
        SAMPLE_RATE * 2
    )

    assert buffer.end_sample == (
        SAMPLE_RATE * 7
    )

    # Request most recent 1 second.
    recent = buffer.get_recent(
        duration_sec=1.0
    )

    assert recent is not None

    assert recent.num_samples == SAMPLE_RATE

    # Should represent seconds 6 to 7.
    assert recent.start_sample == (
        SAMPLE_RATE * 6
    )

    assert recent.end_sample == (
        SAMPLE_RATE * 7
    )

    print("\nTEST 6 PASSED")
    print(
        f"Recent start sample: "
        f"{recent.start_sample}"
    )
    print(
        f"Recent end sample: "
        f"{recent.end_sample}"
    )


def test_get_all_audio():
    buffer = StreamingBuffer(
        buffer_sec=5.0
    )

    # Add 3 seconds.
    buffer.add(
        create_pcm16_audio(
            SAMPLE_RATE * 3
        )
    )

    window = buffer.get_all()

    assert window is not None

    assert window.num_samples == (
        SAMPLE_RATE * 3
    )

    assert window.start_sample == 0

    assert window.end_sample == (
        SAMPLE_RATE * 3
    )

    print("\nTEST 7 PASSED")
    print(
        f"Available duration: "
        f"{window.duration_sec}s"
    )


def test_reset_buffer():
    buffer = StreamingBuffer(
        buffer_sec=5.0
    )

    # Add 3 seconds.
    buffer.add(
        create_pcm16_audio(
            SAMPLE_RATE * 3
        )
    )

    assert buffer.buffered_samples == (
        SAMPLE_RATE * 3
    )

    buffer.reset()

    assert buffer.buffered_samples == 0
    assert buffer.buffered_duration == 0.0

    print("\nTEST 8 PASSED")
    print("Buffer successfully reset")


def test_invalid_duration_request():
    buffer = StreamingBuffer(
        buffer_sec=5.0
    )

    buffer.add(
        create_pcm16_audio(
            SAMPLE_RATE
        )
    )

    with pytest.raises(ValueError):
        buffer.get_recent(
            duration_sec=0
        )

    print("\nTEST 9 PASSED")
    print(
        "Invalid duration correctly rejected"
    )


def test_invalid_pcm16_data():
    buffer = StreamingBuffer(
        buffer_sec=5.0
    )

    # Odd number of bytes is invalid for PCM16.
    invalid_audio = b"\x00"

    with pytest.raises(ValueError):
        buffer.add(
            invalid_audio
        )

    print("\nTEST 10 PASSED")
    print(
        "Invalid PCM16 data correctly rejected"
    )
