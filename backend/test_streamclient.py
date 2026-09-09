import asyncio

import numpy as np
import soundfile as sf
import websockets

from app.audio_stream.protocol import (
    SAMPLE_RATE,
    SAMPLES_PER_CHUNK,
    CHUNK_DURATION_MS,
)


WS_URL = "ws://127.0.0.1:8000/ws/audio"

AUDIO_FILE = "uploads/cleaned/test_preprocessing.wav"


def load_audio():
    """Load audio and convert it to canonical streaming format."""

    audio, sample_rate = sf.read(
        AUDIO_FILE,
        dtype="float32",
    )

    print("=" * 60)
    print("AUDIO FILE")
    print("=" * 60)

    print(f"File              : {AUDIO_FILE}")
    print(f"Original rate     : {sample_rate} Hz")
    print(f"Original shape    : {audio.shape}")
    print(f"Original dtype    : {audio.dtype}")

    # Convert stereo → mono
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
        print("Converted         : stereo → mono")

    # Resample if necessary
    if sample_rate != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE,
        )

        sample_rate = SAMPLE_RATE

        print(f"Resampled         : {SAMPLE_RATE} Hz")

    print(f"Final shape       : {audio.shape}")
    print(f"Final rate        : {sample_rate} Hz")
    print(f"Duration          : {len(audio) / sample_rate:.2f} sec")

    return audio


async def stream_audio(audio):
    """Stream audio to FastAPI in 20 ms canonical PCM16 chunks."""

    async with websockets.connect(WS_URL) as websocket:

        # The backend must know how to interpret following binary bytes.
        await websocket.send(
            """{
                "sample_rate": 16000,
                "channels": 1,
                "encoding": "pcm_s16le"
            }"""
        )
        print("Server:", await websocket.recv())

        print("\n" + "=" * 60)
        print("WEBSOCKET STREAM")
        print("=" * 60)

        print(f"Connected to      : {WS_URL}")
        print(f"Chunk size        : {SAMPLES_PER_CHUNK} samples")
        print(f"Chunk duration    : {CHUNK_DURATION_MS} ms")
        print()

        chunk_number = 0

        for start in range(
            0,
            len(audio),
            SAMPLES_PER_CHUNK,
        ):

            chunk = audio[
                start:start + SAMPLES_PER_CHUNK
            ]

            # Ignore incomplete final chunk for now
            if len(chunk) < SAMPLES_PER_CHUNK:
                break

            # Convert float [-1, 1] → PCM16
            pcm16 = (
                np.clip(chunk, -1.0, 1.0)
                * 32767
            ).astype(np.int16)

            data = pcm16.tobytes()

            chunk_number += 1

            print(
                f"Sending chunk {chunk_number:04d} | "
                f"{len(data)} bytes"
            )

            await websocket.send(data)

            response = await websocket.recv()

            print(f"Server: {response}")

            # Simulate real-time streaming
            await asyncio.sleep(
                CHUNK_DURATION_MS / 1000
            )

        print("\nFinished streaming audio.")


async def main():
    audio = load_audio()

    await stream_audio(audio)


if __name__ == "__main__":
    asyncio.run(main())
