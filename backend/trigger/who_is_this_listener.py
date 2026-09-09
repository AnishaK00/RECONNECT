import speech_recognition as sr
import os
import time


class WhoIsThisListener:

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.non_speaking_duration = 0.3
        self.recognizer.phrase_threshold = 0.3

        self.microphone = self._build_microphone()

        self.trigger_phrases = [
            "who is this",
            "who's this",
            "who is that",
            "who's that",
        ]

        # Circuit breaker configuration (default: 3 failures within 60s)
        self._failure_count = 0
        self._first_failure_ts = None
        self._circuit_breaker_tripped = False
        self._circuit_failure_limit = int(os.getenv("CB_FAILURE_LIMIT", "3"))
        self._circuit_window_secs = int(os.getenv("CB_WINDOW_SECS", "60"))

        # Simple post‑trigger cooldown to give audio buffer time (seconds)
        self._post_trigger_cooldown = float(os.getenv("POST_TRIGGER_COOLDOWN", "2"))
        self._last_success_ts = 0.0

        print("Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Microphone ready.")
        print(f"Energy threshold: {self.recognizer.energy_threshold:.0f}")

    def _build_microphone(self):
        """
        Select microphone in a robust order:
        1) MIC_DEVICE_INDEX env var (explicit index)
        2) MIC_DEVICE_NAME env var (substring match)
        3) System default microphone
        """
        names = sr.Microphone.list_microphone_names()

        if names:
            print("Detected microphone devices:")
            for idx, name in enumerate(names):
                print(f"  [{idx}] {name}")
        else:
            print("No microphone devices reported by PyAudio.")

        mic_index = os.getenv("MIC_DEVICE_INDEX")
        mic_name_hint = (os.getenv("MIC_DEVICE_NAME") or "").strip().lower()

        if mic_index is not None:
            try:
                idx = int(mic_index)
                print(f"Using MIC_DEVICE_INDEX={idx}")
                return sr.Microphone(device_index=idx)
            except Exception as error:
                print(f"Invalid MIC_DEVICE_INDEX '{mic_index}': {error}")

        if mic_name_hint:
            for idx, name in enumerate(names):
                if mic_name_hint in name.lower():
                    print(f"Using MIC_DEVICE_NAME match: [{idx}] {name}")
                    return sr.Microphone(device_index=idx)
            print(f"No microphone matched MIC_DEVICE_NAME='{mic_name_hint}'.")

        print("Using system default microphone.")
        return sr.Microphone()

    # ------------------------------------------
    # Continuous trigger listener
    # ------------------------------------------

    def listen_for_trigger(self):

        print()
        print("Listening...")
        print("Say: 'Who is this?' or 'Take a video'")

        while True:
            # If circuit breaker is active, skip listening and wait for reset.
            if self._circuit_breaker_tripped:
                print("[CIRCUIT_BREAKER] Trigger suppression active. Waiting for reset.")
                time.sleep(1)
                continue

            try:
                # --------------------------------
                # Wait for a speech segment
                # --------------------------------
                with self.microphone as source:
                    # Short re‑calibration helps when room noise changes.
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    audio = self.recognizer.listen(
                        source,
                        timeout=None,
                        phrase_time_limit=3,
                    )

                print("Speech captured. Recognizing...")
                text = self.recognizer.recognize_google(audio)
                text = text.lower().strip()
                print(f"Heard: '{text}'")

                # --------------------------------
                # Check trigger
                # --------------------------------

                for trigger in self.trigger_phrases:
                    if trigger in text:
                        print("\n" + "=" * 50)
                        print("WHO IS THIS TRIGGER DETECTED")
                        print("=" * 50)
                        self._reset_failure_counters()
                        # Enforce a short cooldown after a successful trigger
                        self._last_success_ts = time.time()
                        time.sleep(self._post_trigger_cooldown)
                        return "who_is_this"

                print("Not a trigger.")
                print("Continuing to listen...")

            # ------------------------------------
            # Speech was not understandable
            # ------------------------------------

            except sr.UnknownValueError:
                print("Could not understand speech.")
                self._record_failure()
                continue

            # ------------------------------------
            # Google STT / network error
            # ------------------------------------

            except sr.RequestError as error:
                print(f"Speech recognition error: {error}")
                self._record_failure()
                continue

            # ------------------------------------
            # Ctrl+C
            # ------------------------------------

            except KeyboardInterrupt:
                print()
                print("Stopping audio listener...")
                raise