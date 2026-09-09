import speech_recognition as sr


MIC_INDEX = 0


def main():

    recognizer = sr.Recognizer()

    microphone = sr.Microphone(
        device_index=MIC_INDEX
    )

    print("=" * 50)
    print("MICROPHONE TEST")
    print("=" * 50)

    print(
        f"Using microphone device: {MIC_INDEX}"
    )

    print(
        "Name:",
        sr.Microphone.list_microphone_names()[
            MIC_INDEX
        ]
    )

    print()
    print("Calibrating...")

    with microphone as source:

        recognizer.adjust_for_ambient_noise(
            source,
            duration=1
        )

    print(
        f"Energy threshold: "
        f"{recognizer.energy_threshold:.0f}"
    )

    print()
    print("Recording for 10 seconds.")
    print("SPEAK NOW.")
    print()

    with microphone as source:

        print("🎤 RECORDING...")

        audio = recognizer.record(
            source,
            duration=10
        )

        print("✅ Recording finished.")

    # Save recording
    with open(
        "debug_audio.wav",
        "wb"
    ) as file:

        file.write(
            audio.get_wav_data()
        )

    print()
    print(
        "Saved:"
        " debug_audio.wav"
    )

    print()
    print(
        "Now play debug_audio.wav."
    )

    print(
        "Can you hear your voice?"
    )

    # Try recognition
    print()
    print(
        "Sending to Google Speech Recognition..."
    )

    try:

        text = recognizer.recognize_google(
            audio
        )

        print()
        print(
            "Google heard:"
        )

        print(text)

    except sr.UnknownValueError:

        print(
            "Google could not understand "
            "the recording."
        )

    except sr.RequestError as error:

        print(
            "Google Speech Recognition error:"
        )

        print(error)


if __name__ == "__main__":
    main()