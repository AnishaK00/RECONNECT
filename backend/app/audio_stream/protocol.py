"""Audio streaming protocol constants."""

SAMPLE_RATE = 16_000
CHANNELS = 1

# PCM16 = signed 16-bit integer = 2 bytes per sample
SAMPLE_WIDTH_BYTES = 2

# Target streaming chunk duration
CHUNK_DURATION_MS = 20

# Number of samples in one 20 ms chunk
SAMPLES_PER_CHUNK = (
    SAMPLE_RATE * CHUNK_DURATION_MS // 1000
)

# Number of bytes in one 20 ms mono PCM16 chunk
BYTES_PER_CHUNK = (
    SAMPLES_PER_CHUNK
    * CHANNELS
    * SAMPLE_WIDTH_BYTES
)


"""
SAMPLE_RATE          = 16000
CHANNELS             = 1
SAMPLE_WIDTH_BYTES   = 2
CHUNK_DURATION_MS    = 20
SAMPLES_PER_CHUNK    = 320
BYTES_PER_CHUNK      = 640
"""