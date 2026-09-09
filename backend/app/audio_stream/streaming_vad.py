from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

from silero_vad import load_silero_vad

from app.audio_stream.protocol import (
    SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
)


# ---------------------------------------------------------
# Silero VAD configuration
# ---------------------------------------------------------

VAD_THRESHOLD = 0.5

# Silero VAD requires exactly 512 samples at 16 kHz.
VAD_FRAME_SAMPLES = 512

# 512 samples at 16 kHz = 32 ms.
VAD_FRAME_DURATION_MS = (
    VAD_FRAME_SAMPLES
    / SAMPLE_RATE
    * 1000
)

# How long silence must continue before we consider 
# the current speech segment complete.
SEGMENT_SILENCE_DURATION_MS = 1500


# ---------------------------------------------------------
# VAD states
# ---------------------------------------------------------

class VADState(str, Enum):
    IDLE = "idle"
    SPEAKING = "speaking"


# ---------------------------------------------------------
# VAD event types
# ---------------------------------------------------------

class VADEventType(str, Enum):
    NONE = "none"
    SPEECH_STARTED = "speech_started"
    SPEECH_CONTINUING = "speech_continuing"
    SPEECH_ENDED = "speech_ended"


# ---------------------------------------------------------
# VAD event
# ---------------------------------------------------------

@dataclass
class VADEvent:
    """
    Result returned after processing audio.

    An event is produced only when enough audio is
    available for one or more Silero inference frames.
    """

    event_type: VADEventType

    state: VADState

    is_speech: bool

    speech_probability: float | None

    silence_duration_ms: float


# ---------------------------------------------------------
# Streaming VAD
# ---------------------------------------------------------

class StreamingVAD:
    """
    Stateful streaming wrapper around Silero VAD.

    Input:
        PCM16 mono audio at 16 kHz.

    Incoming audio may arrive in 20 ms / 320-sample
    chunks, but Silero requires exactly 512 samples
    per inference frame.

    This class therefore maintains a small internal
    inference buffer.
    """

    def __init__(
        self,
        threshold: float = VAD_THRESHOLD,
        silence_duration_ms: float = (
            SEGMENT_SILENCE_DURATION_MS
        ),
    ):
        self.threshold = threshold

        self.silence_duration_limit_ms = (
            silence_duration_ms
        )

        # Load the model once per StreamingVAD instance.
        self.model = load_silero_vad()

        # Internal PCM16 inference buffer.
        self._buffer = bytearray()

        # Current speech state.
        self.state = VADState.IDLE

        # Consecutive non-speech duration while speaking.
        self.silence_duration_ms = 0.0

    # -----------------------------------------------------
    # Public processing method
    # -----------------------------------------------------

    def process(
        self,
        data: bytes,
    ) -> list[VADEvent]:
        """
        Process incoming PCM16 audio.

        The input can be any valid number of complete
        PCM16 samples.

        Returns a list because one incoming chunk may
        cause multiple 512-sample inference frames to
        be processed.
        """

        if not data:
            return []

        if (
            len(data)
            % SAMPLE_WIDTH_BYTES
            != 0
        ):
            raise ValueError(
                "PCM16 audio data must contain an "
                "even number of bytes"
            )

        self._buffer.extend(data)

        events: list[VADEvent] = []

        frame_bytes = (
            VAD_FRAME_SAMPLES
            * SAMPLE_WIDTH_BYTES
        )

        while len(self._buffer) >= frame_bytes:

            # Extract exactly one Silero frame.
            frame_data = bytes(
                self._buffer[:frame_bytes]
            )

            # Remove processed audio.
            del self._buffer[:frame_bytes]

            probability = self._get_speech_probability(
                frame_data
            )

            event = self._update_state(
                probability
            )

            events.append(event)

        return events

    # -----------------------------------------------------
    # Silero inference
    # -----------------------------------------------------

    def _get_speech_probability(
        self,
        data: bytes,
    ) -> float:
        """
        Convert PCM16 bytes into float32 audio and run
        Silero VAD.
        """

        # PCM16 bytes -> int16 samples.
        pcm16 = np.frombuffer(
            data,
            dtype=np.int16,
        )

        # int16 -> normalized float32.
        audio = (
            pcm16.astype(np.float32)
            / 32768.0
        )

        # NumPy -> PyTorch tensor.
        waveform = torch.from_numpy(audio)

        with torch.no_grad():

            result = self.model(
                waveform,
                SAMPLE_RATE,
            )

        return float(
            result.item()
        )

    # -----------------------------------------------------
    # State machine
    # -----------------------------------------------------

    def _update_state(
        self,
        probability: float,
    ) -> VADEvent:
        """
        Update the VAD state machine using one Silero
        speech probability.
        """

        is_speech = (
            probability >= self.threshold
        )

        # -------------------------------------------------
        # IDLE
        # -------------------------------------------------

        if self.state == VADState.IDLE:

            if is_speech:

                self.state = (
                    VADState.SPEAKING
                )

                self.silence_duration_ms = 0.0

                return VADEvent(
                    event_type=(
                        VADEventType.SPEECH_STARTED
                    ),
                    state=self.state,
                    is_speech=True,
                    speech_probability=probability,
                    silence_duration_ms=0.0,
                )

            return VADEvent(
                event_type=VADEventType.NONE,
                state=self.state,
                is_speech=False,
                speech_probability=probability,
                silence_duration_ms=0.0,
            )

        # -------------------------------------------------
        # SPEAKING
        # -------------------------------------------------

        if is_speech:

            # Speech resumed or continued.
            self.silence_duration_ms = 0.0

            return VADEvent(
                event_type=(
                    VADEventType.SPEECH_CONTINUING
                ),
                state=self.state,
                is_speech=True,
                speech_probability=probability,
                silence_duration_ms=0.0,
            )

        # Non-speech while currently speaking.
        self.silence_duration_ms += (
            VAD_FRAME_DURATION_MS
        )

        # Has silence lasted long enough?
        if (
            self.silence_duration_ms
            >= self.silence_duration_limit_ms
        ):

            self.state = VADState.IDLE

            return VADEvent(
                event_type=(
                    VADEventType.SPEECH_ENDED
                ),
                state=self.state,
                is_speech=False,
                speech_probability=probability,
                silence_duration_ms=(
                    self.silence_duration_ms
                ),
            )

        # Short silence: keep current speech segment alive.
        return VADEvent(
            event_type=VADEventType.NONE,
            state=self.state,
            is_speech=False,
            speech_probability=probability,
            silence_duration_ms=(
                self.silence_duration_ms
            ),
        )

    # -----------------------------------------------------
    # Reset
    # -----------------------------------------------------

    def reset(self) -> None:
        """
        Reset streaming VAD state for a new audio stream.
        """
        # Clear incoming VAD state for a new audio stream
        self._buffer.clear()

        #reset our state machine
        self.state = VADState.IDLE
        
        #reset accumulated silence
        self.silence_duration_ms = 0.0

        # Reset Silero's internal recurrent state.

        self.model.reset_states()

    # -----------------------------------------------------
    # Properties
    # -----------------------------------------------------

    @property
    def buffered_samples(self) -> int:
        """
        Number of samples currently waiting for a full
        Silero inference frame.
        """

        return (
            len(self._buffer)
            // SAMPLE_WIDTH_BYTES
        )

    @property
    def buffered_duration_ms(self) -> float:
        """
        Duration of audio currently waiting in the
        internal VAD inference buffer.
        """

        return (
            self.buffered_samples
            / SAMPLE_RATE
            * 1000
        )