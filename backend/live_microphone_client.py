"""Capture a local microphone and stream it to the RECONNECT backend."""

import argparse
import asyncio
import json
import logging
import time

import sounddevice as sd
import websockets
from websockets.exceptions import ConnectionClosed


LOGGER = logging.getLogger(__name__)
DEFAULT_URL = "ws://127.0.0.1:8000/ws/audio"
CHUNK_DURATION_SEC = 0.020


def chunk_frames(sample_rate: int) -> int:
    """Return the closest whole-sample frame count for a 20 ms chunk."""

    return max(1, round(sample_rate * CHUNK_DURATION_SEC))


def format_metadata(sample_rate: int, channels: int) -> dict[str, object]:
    """Build the metadata required by the audio WebSocket endpoint."""

    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "encoding": "pcm_s16le",
    }


def list_input_devices() -> None:
    """Print microphones that can be used with --device."""

    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            print(
                f"[{index}] {device['name']} | "
                f"inputs={device['max_input_channels']} | "
                f"default_rate={device['default_samplerate']:.0f} Hz"
            )


async def stream_microphone(args: argparse.Namespace) -> None:
    """Capture PCM16 microphone chunks and deliver them over WebSocket.

    This function now automatically reconnects if the server closes the
    connection (e.g., after reporting a known identity). It continues streaming
    indefinitely until the user stops the process with Ctrl+C.
    """

    device = sd.query_devices(args.device, "input")
    sample_rate = args.sample_rate or int(round(device["default_samplerate"]))
    if args.channels > device["max_input_channels"]:
        raise ValueError(
            f"Device supports only {device['max_input_channels']} input channels"
        )

    frames = chunk_frames(sample_rate)
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=500)  # increased to reduce dropped chunks
    loop = asyncio.get_running_loop()
    dropped_chunks = 0

    def enqueue(data: bytes) -> None:
        nonlocal dropped_chunks
        try:
            queue.put_nowait(data)
        except asyncio.QueueFull:
            dropped_chunks += 1
            LOGGER.warning("microphone queue full; dropped_chunks=%s", dropped_chunks)

    def callback(indata, frames_count, time_info, status) -> None:
        if status:
            LOGGER.warning("microphone status=%s", status)
        loop.call_soon_threadsafe(enqueue, bytes(indata))

    print(
        f"Microphone: {device['name']} | {sample_rate} Hz | "
        f"{args.channels} channel(s) | {frames} samples/chunk"
    )

    while True:  # Reconnect loop
        try:
            async with websockets.connect(args.url) as websocket:
                await websocket.send(json.dumps(format_metadata(sample_rate, args.channels)))
                connected = json.loads(await websocket.recv())
                if not connected.get("received"):
                    raise RuntimeError(connected.get("error", "Server rejected stream format"))

                print(f"Connected to {args.url} | session={connected.get('session_id')}")
                sent_chunks = sent_bytes = 0
                last_report_at = time.monotonic()

                with sd.RawInputStream(
                    samplerate=sample_rate,
                    blocksize=frames,
                    device=args.device,
                    channels=args.channels,
                    dtype="int16",
                    callback=callback,
                ):
                    print("Recording live microphone audio. Press Ctrl+C to stop.")
                    while True:
                        data = await queue.get()
                        await websocket.send(data)
                        acknowledgement = json.loads(await websocket.recv())
                        if not acknowledgement.get("received"):
                            raise RuntimeError(
                                acknowledgement.get("error", "Server rejected audio")
                            )

                        identity_result = acknowledgement.get("identity_result")
                        if identity_result is not None:
                            if identity_result.get("known"):
                                print(
                                    f"Known identity: {identity_result['name']} "
                                    f"({identity_result.get('relation') or 'relation unavailable'})"
                                )
                            else:
                                print(
                                    f"Unknown speaker detected: {identity_result.get('speaker', 'unenrolled')}"
                                )

                        sent_chunks += 1
                        sent_bytes += len(data)
                        now = time.monotonic()
                        if now - last_report_at >= 15.0:
                            print(
                                f"Sent chunks={sent_chunks} bytes={sent_bytes} "
                                f"queued={queue.qsize()} dropped={dropped_chunks}"
                            )
                            last_report_at = now
        except ConnectionClosed as err:
            # Server closed the connection (normally after a known identity).
            # Reconnect after a short pause.
            print(f"\nMicrophone connection closed ({err.code}); reconnecting...")
            await asyncio.sleep(1)
            continue
        except Exception as exc:
            # Any other error should abort the stream.
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream a local microphone to RECONNECT"
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", help="Input device ID or name")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--sample-rate",
        type=int,
        help="Optional capture-rate override; defaults to the device rate",
    )
    parser.add_argument("--channels", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if args.list_devices:
        list_input_devices()
        return
    try:
        asyncio.run(stream_microphone(args))
    except KeyboardInterrupt:
        print("\nMicrophone stream stopped; server will finalize active audio.")
    except ConnectionClosed as error:
        if error.code == 1012:
            print("\nServer restarted while streaming. Start Uvicorn without --reload.")
        else:
            print(f"\nMicrophone connection closed: {error}")
    except ConnectionRefusedError:
        print("\nCould not connect to the audio server. Start Uvicorn first.")


if __name__ == "__main__":
    main()
