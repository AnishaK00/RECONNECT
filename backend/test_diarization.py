# from pathlib import Path

# import torch
# from pyannote.audio import Pipeline


# AUDIO_PATH = Path(
#     "uploads/cleaned/test_vad_cleaned.wav"
# )


# print("=" * 60)
# print("STEP 1: CHECK DEVICE")
# print("=" * 60)

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")

# print(f"Using device: {device}")


# print("\n" + "=" * 60)
# print("STEP 2: LOAD DIARIZATION PIPELINE")
# print("=" * 60)

# pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization-community-1"
# )

# pipeline.to(device)

# print("Diarization pipeline loaded successfully.")


# print("\n" + "=" * 60)
# print("STEP 3: RUN DIARIZATION")
# print("=" * 60)

# print(f"Input audio: {AUDIO_PATH}")

# output = pipeline(
#     str(AUDIO_PATH)
# )


# print("\n" + "=" * 60)
# print("STEP 4: DIARIZATION RESULTS")
# print("=" * 60)

# # VAD region used to create test_vad_cleaned.wav
# vad_start_sec = 11.564

# segments = []

# for turn, speaker in output.speaker_diarization:

#     start_sec = turn.start
#     end_sec = turn.end

#     original_start_sec = (
#         vad_start_sec + start_sec
#     )

#     original_end_sec = (
#         vad_start_sec + end_sec
#     )

#     segment = {
#         "speaker_label": speaker,
#         "start_sec": round(
#             original_start_sec,
#             3,
#         ),
#         "end_sec": round(
#             original_end_sec,
#             3,
#         ),
#     }

#     segments.append(segment)

#     print(
#         f"{speaker}: "
#         f"{original_start_sec:.3f} → "
#         f"{original_end_sec:.3f} sec"
#     )


# print("\nStructured segments:")

# for segment in segments:
#     print(segment)


# print("\n" + "=" * 60)
# print("DIARIZATION TEST COMPLETE")
# print("=" * 60)


from pathlib import Path

import soundfile as sf

from app.audio_process.diarization import (
    diarize_audio,
)


AUDIO_PATH = Path(
    "uploads/cleaned/test_vad_cleaned.wav"
)


print("=" * 60)
print("STEP 1: LOAD AUDIO")
print("=" * 60)

audio, sample_rate = sf.read(
    AUDIO_PATH,
    dtype="float32",
)

print(f"Sample rate : {sample_rate} Hz")
print(f"Samples     : {len(audio)}")
print(
    f"Duration    : "
    f"{len(audio) / sample_rate:.3f} sec"
)


print("\n" + "=" * 60)
print("STEP 2: RUN DIARIZATION")
print("=" * 60)

turns = diarize_audio(
    audio,
    sample_rate,
)


print("\n" + "=" * 60)
print("STEP 3: DIARIZATION RESULTS")
print("=" * 60)

if not turns:

    print("No speaker turns detected.")

else:

    for turn in turns:

        print(
            f"{turn.speaker_label}: "
            f"{turn.start_sec:.3f} → "
            f"{turn.end_sec:.3f} sec"
        )


print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)