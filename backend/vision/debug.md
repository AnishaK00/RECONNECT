 closed session=session_001 chunks=9122 source_bytes=16091208 canonical_bytes=5838080
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [3144]
(venv) PS C:\Users\ASUS\OneDrive\Desktop\binu\reconnect\backend> uvicorn app.main:app --host 127.0.0.1 --port 8000
INFO:     Started server process [29276]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:55642 - "WebSocket /ws/audio" [accepted]
INFO:     connection open
2026-09-08 13:14:02,728 | [STREAM] format=accepted session=session_002 source=44100Hz/1ch/pcm_s16le normalization=enabled
2026-09-08 13:14:04,353 | ----- [RECORDING] started segment=1 pre_roll=0.500s -----
2026-09-08 13:14:11,393 | ----- [RECORDING] finalized segment=1 reason=silence duration=7.560s -----
Initializing AudioEmbedder (SpeechBrain)...
2026-09-08 13:14:13,035 | ----- [RECORDING] started segment=2 pre_roll=0.500s -----
2026-09-08 13:14:19,792 | ----- [RECORDING] finalized segment=2 reason=silence duration=7.260s -----
2026-09-08 13:14:21,392 | ----- [RECORDING] started segment=3 pre_roll=0.500s -----
2026-09-08 13:14:28,034 | ----- [RECORDING] finalized segment=3 reason=silence duration=7.140s -----
2026-09-08 13:14:28,752 | ----- [RECORDING] started segment=4 pre_roll=0.500s -----
2026-09-08 13:14:43,993 | ----- [RECORDING] finalized segment=4 reason=silence duration=15.720s -----
2026-09-08 13:14:44,793 | ----- [RECORDING] started segment=5 pre_roll=0.500s -----
Error during recognition for SPEAKER_UNKNOWN: database is locked
2026-09-08 13:14:45,777 | [WHISPER] loading model=base device=cpu compute_type=int8
2026-09-08 13:14:45,778 | [WHISPER] loading model=base device=cpu compute_type=int8
2026-09-08 13:14:49,660 | [WHISPER] "patient" (speaker_unknown_002.wav)
2026-09-08 13:14:54,752 | ----- [RECORDING] finalized segment=5 reason=silence duration=10.460s -----
2026-09-08 13:14:55,153 | ----- [RECORDING] started segment=6 pre_roll=0.420s -----
2026-09-08 13:14:59,473 | ----- [RECORDING] finalized segment=6 reason=silence duration=4.740s -----
2026-09-08 13:14:59,596 | ----- [RECORDING] started segment=7 pre_roll=0.120s -----
2026-09-08 13:15:07,952 | ----- [RECORDING] finalized segment=7 reason=silence duration=8.480s -----
2026-09-08 13:15:08,046 | ----- [RECORDING] started segment=8 pre_roll=0.060s -----
2026-09-08 13:15:11,405 | [WHISPER] "TRS now" (speaker_unknown_001.wav)
2026-09-08 13:15:11,425 | [TRANSCRIPT] speaker_unknown: "TRS now"

>>> [TRANSCRIPT] speaker_unknown: "TRS now" <<<

2026-09-08 13:15:11,425 | [TRANSCRIPT] speaker_unknown: "patient"

>>> [TRANSCRIPT] speaker_unknown: "patient" <<<

2026-09-08 13:15:11,434 | [PIPELINE] Recording #1 processed in 60.04s (segments=2)
2026-09-08 13:15:15,870 | ----- [RECORDING] finalized segment=8 reason=silence duration=7.900s -----
2026-09-08 13:15:16,313 | ----- [RECORDING] started segment=9 pre_roll=0.440s -----
Error during recognition for SPEAKER_UNKNOWN: database is locked
2026-09-08 13:15:21,539 | [WHISPER] "Oh" (speaker_unknown_002.wav)
2026-09-08 13:15:32,528 | [WHISPER] "elom gotak mu kardena nena user kuta ho mission ko" (speaker_unknown_001.wav)
2026-09-08 13:15:32,529 | [TRANSCRIPT] speaker_unknown: "elom gotak mu kardena nena user kuta ho mission ko"

>>> [TRANSCRIPT] speaker_unknown: "elom gotak mu kardena nena user kuta ho mission ko" <<<   

2026-09-08 13:15:32,529 | [TRANSCRIPT] speaker_unknown: "Oh"

>>> [TRANSCRIPT] speaker_unknown: "Oh" <<<

2026-09-08 13:15:32,534 | [PIPELINE] Recording #2 processed in 72.74s (segments=2)
2026-09-08 13:15:33,672 | ----- [RECORDING] finalized segment=9 reason=silence duration=17.820s -----
2026-09-08 13:15:34,913 | ----- [RECORDING] started segment=10 pre_roll=0.500s -----
2026-09-08 13:15:36,830 | ----- [RECORDING] finalized segment=10 reason=silence duration=2.400s -----
2026-09-08 13:15:36,874 | ----- [RECORDING] started segment=11 pre_roll=0.060s -----
2026-09-08 13:15:38,393 | ----- [RECORDING] finalized segment=11 reason=silence duration=1.560s -----
2026-09-08 13:15:38,529 | [PIPELINE] Recording #11 processed in 0.14s (segments=0)
2026-09-08 13:15:39,231 | ----- [RECORDING] started segment=12 pre_roll=0.500s -----
2026-09-08 13:15:40,828 | ----- [RECORDING] finalized segment=12 reason=silence duration=2.100s -----
2026-09-08 13:15:40,905 | [PIPELINE] Recording #12 processed in 0.07s (segments=0)


>>> [TRANSCRIPT] speaker_unknown: "patient" <<<

2026-09-08 13:15:11,434 | [PIPELINE] Recording #1 processed in 60.04s (segments=2)
2026-09-08 13:15:15,870 | ----- [RECORDING] finalized segment=8 reason=silence duration=7.900s -----
2026-09-08 13:15:16,313 | ----- [RECORDING] started segment=9 pre_roll=0.440s -----
Error during recognition for SPEAKER_UNKNOWN: database is locked
2026-09-08 13:15:21,539 | [WHISPER] "Oh" (speaker_unknown_002.wav)
2026-09-08 13:15:32,528 | [WHISPER] "elom gotak mu kardena nena user kuta ho mission ko" (speaker_unknown_001.wav)
2026-09-08 13:15:32,529 | [TRANSCRIPT] speaker_unknown: "elom gotak mu kardena nena user kuta ho mission ko"

>>> [TRANSCRIPT] speaker_unknown: "elom gotak mu kardena nena user kuta ho mission ko" <<<   

2026-09-08 13:15:32,529 | [TRANSCRIPT] speaker_unknown: "Oh"

>>> [TRANSCRIPT] speaker_unknown: "Oh" <<<

2026-09-08 13:15:32,534 | [PIPELINE] Recording #2 processed in 72.74s (segments=2)
2026-09-08 13:15:33,672 | ----- [RECORDING] finalized segment=9 reason=silence duration=17.820s -----
2026-09-08 13:15:34,913 | ----- [RECORDING] started segment=10 pre_roll=0.500s -----
2026-09-08 13:15:36,830 | ----- [RECORDING] finalized segment=10 reason=silence duration=2.400s -----
2026-09-08 13:15:36,874 | ----- [RECORDING] started segment=11 pre_roll=0.060s -----
2026-09-08 13:15:38,393 | ----- [RECORDING] finalized segment=11 reason=silence duration=1.560s -----
2026-09-08 13:15:38,529 | [PIPELINE] Recording #11 processed in 0.14s (segments=0)
2026-09-08 13:15:39,231 | ----- [RECORDING] started segment=12 pre_roll=0.500s -----
2026-09-08 13:15:40,828 | ----- [RECORDING] finalized segment=12 reason=silence duration=2.100s -----
2026-09-08 13:15:40,905 | [PIPELINE] Recording #12 processed in 0.07s (segments=0)
Error during recognition for SPEAKER_UNKNOWN: database is locked
2026-09-08 13:16:02,110 | ----- [RECORDING] started segment=13 pre_roll=0.500s -----
2026-09-08 13:16:12,475 | [WHISPER] "पर लाwin लाелен गले कमें गयी आ� slotfilm." (speaker_unkn  nown_001.wav)
2026-09-08 13:16:12,488 | [TRANSCRIPT] speaker_unknown: "पर लाwin लाелен गले कमें गयी आ� slot  tfilm."

>>> [TRANSCRIPT] speaker_unknown: "पर लाwin लाелен गले कमें गयी आ� slotfilm." <<<

2026-09-08 13:16:12,493 | [PIPELINE] Recording #3 processed in 104.46s (segments=1)
Error during recognition for SPEAKER_01: database is locked
2026-09-08 13:16:32,110 | ----- [RECORDING] finalized segment=13 reason=max_duration duration=30.500s -----
2026-09-08 13:16:32,110 | ----- [RECORDING] continued segment=14 reason=max_duration -----
2026-09-08 13:16:34,832 | ----- [RECORDING] finalized segment=14 reason=silence duration=2.740s -----
2026-09-08 13:16:37,111 | ----- [RECORDING] started segment=15 pre_roll=0.500s -----
2026-09-08 13:16:38,673 | ----- [RECORDING] finalized segment=15 reason=silence duration=2.080s -----
2026-09-08 13:16:38,773 | [PIPELINE] Recording #15 processed in 0.10s (segments=0)
2026-09-08 13:16:40,591 | ----- [RECORDING] started segment=16 pre_roll=0.500s -----
2026-09-08 13:16:42,273 | ----- [RECORDING] finalized segment=16 reason=silence duration=2.180s -----
2026-09-08 13:16:42,356 | [PIPELINE] Recording #16 processed in 0.08s (segments=0)
2026-09-08 13:16:45,150 | ----- [RECORDING] started segment=17 pre_roll=0.500s -----
