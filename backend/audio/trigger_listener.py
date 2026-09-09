import speech_recognition as sr


class TriggerListener:

    def __init__(self):

        self.recognizer = sr.Recognizer()

        self.microphone = sr.Microphone()

        self.trigger_phrases = [

            "who is this",

            "who's this",

            "who is that",

            "who's that"
        ]


    def listen_for_trigger(self):

        with self.microphone as source:

            print(
                "\nListening for trigger..."
            )

            self.recognizer.adjust_for_ambient_noise(
                source,
                duration=0.5
            )

            try:

                audio = self.recognizer.listen(

                    source,

                    timeout=None,

                    phrase_time_limit=5

                )


                text = (

                    self.recognizer
                    .recognize_google(
                        audio
                    )
                    .lower()

                )


                print(
                    f"Heard: {text}"
                )


                for trigger in (
                    self.trigger_phrases
                ):

                    if trigger in text:

                        print(
                            f"Trigger detected: {trigger}"
                        )

                        return True


                return False


            except sr.UnknownValueError:

                return False


            except sr.RequestError as error:

                print(
                    f"Speech recognition error: {error}"
                )

                return False