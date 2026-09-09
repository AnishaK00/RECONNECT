"""
RECONNECT — Active Speaker Detection Pipeline
=============================================
Step 5 + Step 6 of the Perception Agent

What this does:
  1. Takes a video file as input
  2. Detects all faces in each frame (InsightFace/RetinaFace), clustering
     unenrolled faces persistently across the whole video via PersonRegistry
  3. Separates audio into speaker turns (PyAnnote diarisation)
  4. Recognises each turn's voice, trusting PyAnnote's own diarisation
     clustering across turns rather than re-clustering short segments
  5. Determines which face is speaking at each moment (TalkNet), reusing
     the identity Step 2 already assigned rather than re-deriving it
  6. Links voice-identity + face-identity + ASD result into one person via
     PersonRegistry, handling enrolled/unenrolled combinations and
     off-screen-voice cases
  7. Outputs final confirmed identity per speaking turn

Install requirements:
  pip install insightface onnxruntime
  pip install pyannote.audio
  pip install opencv-python
  pip install torch torchaudio
  pip install numpy scipy
  pip install talknet-asd
  pip install python_speech_features

  # For PyAnnote you need a HuggingFace token:
  # Sign up at huggingface.co
  # Accept terms at:
  #   hf.co/pyannote/speaker-diarization-3.1
  #   hf.co/pyannote/segmentation-3.0
  # Then set your token below

Usage:
  python active_speaker_detection.py --video test_video.mp4
"""

import os
import sys
import math
import cv2
import torch
import numpy as np
import torchaudio
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from collections import Counter
from scipy.io import wavfile
from python_speech_features import mfcc
from insightface.app import FaceAnalysis

from pyannote.audio import Pipeline

import soundfile as sf

try:
    from .identity_registry import PersonRegistry, ASD_LINK_THRESHOLD
except ImportError:
    # Allow running this file directly from the activity folder.
    from identity_registry import PersonRegistry, ASD_LINK_THRESHOLD

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

TALKNET_DIR = Path(__file__).resolve().parents[2] / "TalkNet-ASD"
sys.path.insert(0, str(TALKNET_DIR))

from talkNet import talkNet

# ── HuggingFace token for PyAnnote ──────────────────────────
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


def extract_audio_from_video(video_path: str, output_audio: str = None) -> str:
    if output_audio is None:
        fd, output_audio = tempfile.mkstemp(prefix="extracted_audio_", suffix=".wav")
        os.close(fd)

    print(f"[Step 1] Extracting audio from {video_path}...")

    cmd = [
        "ffmpeg", "-y",         # y to run ffmpeg program
        "-i", video_path,
        "-ac", "1",          # mono
        "-ar", "16000",      # 16kHz — required by ECAPA-TDNN / MFCC extraction
        "-vn",                # no video
        output_audio
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if os.path.exists(output_audio):
            try:
                os.unlink(output_audio)
            except Exception:
                pass
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    print(f"  Audio extracted → {output_audio}")
    return output_audio


def extract_frames(video_path: str, fps: int = 1) -> list:
    """
    Extract frames from video at specified FPS.
    Returns list of (timestamp_seconds, frame_numpy_array).
    """
    print(f"[Step 1] Extracting frames at {fps}fps...")
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()     #ret is a boolean indicating if the frame was read successfully
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps
            frames.append((timestamp, frame))
        frame_idx += 1

    cap.release()
    print(f"  Extracted {len(frames)} frames from {video_path}")
    return frames


def extract_frames_in_range(video_path: str, start: float, end: float) -> list:
    """
    Re-extract frames at the video's NATIVE fps for a short time window —
    used only for TalkNet scoring. Step 2's low-fps sampling is fine for
    whole-video face tracking, but TalkNet's lip-sync scoring needs real
    frame rate: subsampling loses the fine mouth-movement detail it
    depends on, and forcing a 4:1 audio:video ratio onto artificially
    thinned frames misaligns audio and video in time even when the
    frame COUNTS happen to match.
    """
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = int(start * video_fps)
    end_frame = int(end * video_fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)       #jump directly to start_frame to avoid reading all frames before it
    frames = []
    frame_idx = start_frame
    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((frame_idx / video_fps, frame))  # frame_idx / video_fps works like a timestamp for the frame
        frame_idx += 1

    cap.release()
    return frames

# Intersection over Union: solves the problem of tracking the same person across video frames without repeatedly doing speaker/face identification. cheaper
def bb_iou(boxA, boxB):
    """
    Standard IoU between two (x1, y1, x2, y2) boxes — same math as the
    original TalkNet repo's bb_intersection_over_union, used here to
    match dense-frame detections against Step 2's already-known identity
    positions without re-deriving identity via a fresh embedding lookup
    on every native-fps frame.
    """
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = areaA + areaB - inter
    return inter / float(denom) if denom > 0 else 0.0


def check_diarization_speaker_count(turns: list, registry) -> dict:
    """
    Quick sanity check: compares how many unique voices PyAnnote found
    against how many faces were detected in the video. A mismatch
    (e.g. 1 diarized voice but 2+ faces on screen) is a strong signal
    diarization may have merged two distinct speakers into one cluster —
    worth flagging rather than silently trusting downstream.

    IMPORTANT: this must run AFTER Step 2 (face detection) so
    registry.people is populated, but it intentionally does NOT require
    face_only_sightings/speaking_turns/is_confirmed — those only get set
    during Step 6 (resolve_turn), which hasn't run yet at the point this
    check is called. Requiring them here would make confirmed_faces
    always empty and the check would never fire.
    """
    unique_speakers = set(t["speaker_label"] for t in turns)
    detected_faces = [rec for rec in registry.people.values() if rec.face_samples > 0]

    result = {
        "diarized_speaker_count": len(unique_speakers),
        "detected_face_count": len(detected_faces),
        "mismatch": len(unique_speakers) < len(detected_faces),
    }

    if result["mismatch"]:
        print(f"\n  ⚠️  WARNING: diarization found {len(unique_speakers)} unique voice(s), "
              f"but {len(detected_faces)} face(s) were detected on screen.")
        print(f"      This may mean PyAnnote merged two distinct speakers into one voice "
              f"cluster — treat per-speaker face associations with extra caution for this clip.")

    return result


#  STEP 2 — Face detection + recognition (InsightFace)    

TRACK_MAX_GAP_FRAMES    = 8     # how many sampled frames a track survives with no detection
TRACK_DIST_FRACTION     = 0.15  # max center-to-center movement (as fraction of frame diagonal)
                                 # to count as "still the same person" between frames
MIN_FACE_AREA_FRACTION  = 0.004 # face must cover at least 0.4% of frame area — calibrated
                                 # against test footage where real in-person faces measured
                                 # ~0.009-0.011 and inset-video-thumbnail faces measured
                                 # ~0.0007. Filters out faces from inset videos/thumbnails
                                 # playing within frame, which are much smaller than a real
                                 # in-person participant at normal camera distance.


class FaceDetector:
    """
    Uses InsightFace (RetinaFace + ArcFace) to detect faces, then identifies
    them via TWO layers:
      1. Spatial/temporal tracking — if a face this frame sits close to
         where a known identity was last seen a couple of frames ago,
         it's trusted as the same person WITHOUT re-matching the
         embedding. This is what actually solves fragmentation: a
         person's embedding wobbles frame to frame (talking, turning,
         lighting) far more than their screen position does.
      2. Embedding matching via the registry — only used when no track
         is nearby (a new person, or one re-entering after an occlusion/
         gap longer than TRACK_MAX_GAP_FRAMES).

    Also maintains a second, lightweight detection-only InsightFace
    instance (self.dense_app) used only for TalkNet's dense native-fps
    re-detection pass (Step 5/6). That pass only needs bboxes to match
    against Step 2's already-known identities via IoU — it never needs
    the recognition/landmark/genderage models, so skipping them there
    saves significant compute on every native-fps frame.
    """

    def __init__(self, registry: PersonRegistry):
        

        print("[Step 2] Loading InsightFace models...")
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))         #detects face features, area of face coverage, embeddings, etc everything 


        print("[Step 2] Loading lightweight detection-only InsightFace instance...")
        self.dense_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            allowed_modules=["detection"]
        )       #only regardung detection module, not recognition/landmark/genderage modules specifically for bbox
        self.dense_app.prepare(ctx_id=0, det_size=(320, 320))

        self.registry = registry
        self.active_tracks = {}   # identity -> {"center": (x, y), "last_frame_idx": int}
        self._frame_idx = 0
        print(f"  InsightFace loaded. Known identities so far: {list(self.registry.people.keys())}")

    def enroll_face(self, name: str, image_path: str):
        """
        Enroll a new person by extracting their face embedding from an image.
        Call this during the enrollment phase before detection.
        """
        img = cv2.imread(image_path)        #converts image to numpy array consisting of pixel values
        faces = self.app.get(img)           #InsightFace: detects face, calculates bounding box, calculates embedding, calculates landmarks, calculates detection confidence, potentially other attributes and stores all
        if not faces:
            print(f"  WARNING: No face found in {image_path}")
            return
        face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])      #if multiple faces in image, select the largest one (by area) as the primary face to enroll
        self.registry.enroll_face(name, face.embedding)     # while enrolling, storing only the name and calculate embedding
        print(f"  Enrolled face: {name}")

    def detect_and_recognize(self, frame: np.ndarray) -> list:
        """
        Detect all faces in frame and identify them — spatial tracking
        first, registry embedding matching as fallback.
        "face_crop": np.array  ← cropped face image for TalkNet
          
        """
        faces_raw = self.app.get(frame)         #getting all the faces detected in the frame, with their bounding boxes, embeddings, landmarks, etc.
        results = []

        frame_h, frame_w = frame.shape[:2]      # each frame  here is (360,360,3) so only talking first 2 , 3 is for rgb/ color channels
        diagonal = (frame_w ** 2 + frame_h ** 2) ** 0.5 #pythagorean theorem to calculate the diagonal length of the frame for tracking
        track_dist_threshold = TRACK_DIST_FRACTION * diagonal  #0.15* diagonal to dtermine if a face is close enough to a previous detetcion  whli=ile comparing from center of bbox to be considered the same person
        used_identities_this_frame = set()      #empty set to keep track of which identities have already been assigned to a face in this frame
        frame_area = frame_w * frame_h

        for face in faces_raw:
            bbox = face.bbox.astype(int).tolist()   #decimal to int and nparray to list
            x1, y1, x2, y2 = bbox

            face_area = (x2 - x1) * (y2 - y1)
            if face_area / frame_area < MIN_FACE_AREA_FRACTION:     #(if face_area/frame_area< 0.004)
                continue  # skip — too small to be a real in-frame participant
                          # (e.g. a face inside an inset video/thumbnail)

            embedding = face.embedding
            center = ((x1 + x2) / 2, (y1 + y2) / 2)

            # ── 1. spatial/temporal continuity check ──
            best_track_id, best_dist = None, track_dist_threshold
            for tid, track in self.active_tracks.items():       #active tracks is dict which stores the last known center and frame index of each identity eg: {"Unknown_1": {"center": (100, 100), "last_frame_idx": 5}, 2: {"center": (200, 200), "last_frame_idx": 10}}
                if tid in used_identities_this_frame:
                    continue
                if self._frame_idx - track["last_frame_idx"] > TRACK_MAX_GAP_FRAMES:            
                    continue
                dx = center[0] - track["center"][0]
                dy = center[1] - track["center"][1]
                dist = (dx * dx + dy * dy) ** 0.5       #distance between the current face's center and the last known center of the track
                if dist < best_dist:
                    best_dist, best_track_id = dist, tid

            if best_track_id is not None:
                identity = best_track_id
                self.registry.reinforce_face(identity, embedding)
                face_confidence = 1.0   # trusted via tracking, not re-matched
            else:
                # ── 2. no nearby track — fall back to embedding matching ──
                identity, best_sim, is_new = self.registry.identify_face(embedding)
                face_confidence = 0.0 if is_new else float(best_sim)

            self.active_tracks[identity] = {"center": center, "last_frame_idx": self._frame_idx}
            used_identities_this_frame.add(identity)

            x1c, y1c = max(0, x1), max(0, y1)       # fix negative coordinates to avoid indexing errors
            face_crop = frame[y1c:y2, x1c:x2]       # for talknet later face cropped just to take portion of face from the frame

            results.append({
                "bbox":            bbox,
                "embedding":       embedding,
                "identity":        identity,
                "face_confidence": face_confidence,
                "face_crop":       face_crop
            })

        self._frame_idx += 1
        return results

    def process_all_frames(self, frames: list) -> dict:
        """
        Run face detection on all extracted frames.

        Returns dict: {timestamp: [face_results]}
        """
        print(f"[Step 2] Running face detection on {len(frames)} frames...")
        frame_faces = {}

        for i, (timestamp, frame) in enumerate(frames):     #enumerate adds a counter to an iterable and returns it in a form of enumerate object. This enumerate object can then be used directly in for loops or be converted into a list of tuples using list() method
                                                            #(timestamp, frame) to unpack each tuple into its components
            detections = self.detect_and_recognize(frame)
            frame_faces[timestamp] = detections
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(frames)} frames")

        all_identities = set()
        for dets in frame_faces.values():
            for d in dets:
                all_identities.add(d["identity"])
        print(f"  Identities found across video: {all_identities}")
        return frame_faces


# STEP 3 — Speaker diarisation (PyAnnote)     

class SpeakerDiariser:
    " to determine who is speaking when, and to split the audio into speaker turns. Each turn gets a temporary label (SPEAKER_00, SPEAKER_01 etc.) that stays consistent for the same voice across the whole video — this labeling is trusted downstream instead of re-clustering turns."
    """
    Uses PyAnnote 3.0 to split audio into speaker turns.
    Each turn gets a temporary label (SPEAKER_00, SPEAKER_01 etc.) that
    stays consistent for the same voice across the whole video — this
    labeling is trusted downstream instead of re-clustering turns.
    """

    def __init__(self, hf_token: str):
        print("[Step 3] Loading PyAnnote speaker diarisation model...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )

        if torch.cuda.is_available():
            self.pipeline = self.pipeline.to(torch.device("cuda"))
        print("  PyAnnote loaded.")

    def diarise(self, audio_path: str) -> list:
        """
        Run speaker diarisation on audio file.

        Returns list of dicts:
        [
          {"speaker_label": "SPEAKER_00", "start": 0.0, "end": 3.2},
          ...
        ]
        """

        print(f"[Step 3] Running speaker diarisation on {audio_path}...")

        waveform, sample_rate = sf.read(audio_path, dtype="float32")

        if waveform.ndim == 1:
            waveform = waveform[None, :]
        else:
            waveform = waveform.T

        waveform = torch.from_numpy(waveform)

        audio = {
            "waveform": waveform,           # audio waveform is a 2D tensor where each row represents a channel and each column represents a sample. It is used as input to the PyAnnote pipeline for speaker diarisation.
            "sample_rate": sample_rate,     # sample_rate is the number of samples per second in the audio file, used to convert time in seconds to sample indices to determine how many samples correspond to a given time duration in the audio file to know when or how quickly waveform occurs
        }

        turns = []
        diarisation = self.pipeline(audio)

        for turn, _, speaker in diarisation.speaker_diarization.itertracks(yield_label=True):
            turns.append({
                "speaker_label": speaker,
                "start": turn.start,
                "end": turn.end
            })

        print(f"  Found {len(turns)} speaker turns")
        unique_speakers = set(t["speaker_label"] for t in turns)
        print(f"  Unique speakers: {unique_speakers}")

        return turns


# STEP 4 — Speaker recognition (ECAPA-TDNN)            

class SpeakerRecogniser:
    """
    Uses SpeechBrain's ECAPA-TDNN model to extract speaker embeddings
    from diarised speech segments.

    PyAnnote first determines when each speaker is talking and assigns
    temporary labels such as SPEAKER_00 and SPEAKER_01. This class then
    extracts a numerical voice embedding for each sufficiently long
    speaker turn.

    The embeddings can later be compared with enrolled voice embeddings
    in PersonRegistry to identify the speaker's actual identity.
    """

    def __init__(self, registry: PersonRegistry):
        from speechbrain.inference.classifiers import EncoderClassifier

        print("[Step 4] Loading ECAPA-TDNN speaker recognition model...")
        savedir = BACKEND_ROOT / "tmp_speechbrain"
        if not savedir.exists():
            savedir = BACKEND_ROOT.parent / "pretrained_models" / "spkrec-ecapa-voxceleb"
        self.model = EncoderClassifier.from_hparams(
            source=str(savedir),
            savedir=str(savedir),
        )
        self.registry = registry
        print(f"  ECAPA-TDNN loaded. Known identities so far: {list(self.registry.people.keys())}")

    def enroll_voice(self, name: str, audio_path: str):
        """
        Enroll a person's voice from a 30-second audio sample.
        """
        embedding = self._extract_embedding(audio_path)
        if embedding is not None:
            self.registry.enroll_voice(name, embedding)
            print(f"  Enrolled voice: {name}")

    def _extract_embedding(self, audio_path: str):
        """Extract speaker embedding from audio file."""
        try:
            fs, audio_array = wavfile.read(audio_path)      #sample rate and audio data as numpy array

            if audio_array.dtype == np.int16:
                audio_array = audio_array.astype(np.float32) / 32768.0      #signed 16-bit PCM audio is in the range [-32768, 32767], so dividing by 32768.0 normalizes it to [-1.0, 1.0]
            elif audio_array.dtype == np.int32:
                audio_array = audio_array.astype(np.float32) / 2147483648.0     #signed 32-bit PCM audio is in the range [-2147483648, 2147483647], so dividing by 2147483648.0 normalizes it to [-1.0, 1.0]
            else:               
                audio_array = audio_array.astype(np.float32)

            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)      #for only one channel

            signal = torch.from_numpy(audio_array).unsqueeze(0)     #nparray to  tensor and ass a batch dimension: eg: before -> (samples, ) after -> (1, samples)

            if fs != 16000:     #sampling to 16000 because ecpa excpects 16khz audio
                resampler = torchaudio.transforms.Resample(fs, 16000)
                signal = resampler(signal)
            embedding = self.model.encode_batch(signal)
            return embedding.squeeze()  #mebedding may have extra dimensions, squeeze removes them to get a 1D vector eg: before -> (1, 192) after -> (192, )
        except Exception as e:  
            print(f"  WARNING: Could not extract embedding from {audio_path}: {e}")
            return None

    def _extract_embedding_from_segment(self, audio_array: np.ndarray, sr: int = 16000):
        """Extract speaker embedding from numpy audio array."""
        signal = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            signal = resampler(signal)
        embedding = self.model.encode_batch(signal)
        return embedding.squeeze()

    def recognise_turns(self, audio_path: str, turns: list) -> list:
        """
        For each speaker turn, extract embedding and identify it via
        registry.identify_voice_by_turn — trusting speaker_label for
        cross-turn clustering, ECAPA only for matching enrolled voices.

        Returns turns with added identity fields.
        """
        print(f"[Step 4] Recognising speakers in {len(turns)} turns...")
        sr, audio_data = wavfile.read(audio_path)
        audio_float = audio_data.astype(np.float32) / 32768.0

        recognised_turns = []

        for turn in turns:
            start_sample = int(turn["start"] * sr)
            end_sample = int(turn["end"] * sr)
            segment = audio_float[start_sample:end_sample]

            if len(segment) < sr * 0.5:
                # Skip segments shorter than 0.5 seconds
                continue

            try:
                embedding = self._extract_embedding_from_segment(segment, sr)
            except Exception as e:
                print(f"  WARNING: Embedding extraction failed: {e}")
                continue

            recognised_turns.append({
                **turn,                       #unpack the original turn dictionary to include all its fields i.e. Take all key-value pairs from the turn dictionary and put them into this new dictionary.
                "voice_embedding":  embedding
            })

        print(f"  Speaker recognition complete.")
        return recognised_turns


# STEP 5 — Active Speaker Detection (TalkNet) 


class ActiveSpeakerDetector:
    """
    Uses TalkNet to determine which face is the active speaker
    at each moment by analysing lip movement sync with audio.

    Every return path here produces the same shape:
      {"candidates": [(identity_label, normalized_score), ...], "method": str}
    identity_label is always one Step 2 already assigned — this class
    never invents or re-derives face identity from embeddings, it only
    scores which of Step 2's existing identities is speaking.

    TalkNet paper: https://arxiv.org/abs/2107.08806
    Trained on AVA-ActiveSpeaker dataset
    """

    def __init__(self):
        try:
            print("[Step 5] Loading TalkNet ASD model...")
            self.model = talkNet()
            model_path = Path(__file__).resolve().parents[2] / "TalkNet-ASD" / "pretrain_TalkSet.model"
            self.model.loadParameters(str(model_path))      #load pretrained weights/ paramters
            self.model.eval()                               #set the model to evaluation mode
            print("  TalkNet loaded.")
        except ImportError:
            print("  WARNING: TalkNet not installed.")
            print("  Install: pip install talknet-asd")
            print("  Falling back to heuristic ASD.")
            self.model = None

    def detect_active_speaker(
        self,
        frames: list,
        frame_faces: dict,
        audio_path: str,
        turn: dict,
        video_path: str,
        face_app,
        registry
    ) -> dict:
        turn_frames = [
            (ts, frame) for ts, frame in frames
            if turn["start"] <= ts <= turn["end"]
        ]
        faces_in_turn = []
        for ts, _ in turn_frames:
            if ts in frame_faces:
                for face in frame_faces[ts]:
                    faces_in_turn.append((ts, face))

        if not faces_in_turn:
            return {"candidates": [], "method": "no_faces"}

        if self.model is not None:
            return self._run_talknet(turn, video_path, face_app, registry, audio_path, frame_faces)
        else:
            identity_counts = Counter(f["identity"] for _, f in faces_in_turn)
            most_common = identity_counts.most_common(1)[0][0]
            return {"candidates": [(most_common, 0.5)], "method": "frequency_heuristic"}

    def _prep_face_crop(self, frame, bbox, pad_scale=0.40):
        """
        Reproduce TalkNet's original preprocessing: pad the face bbox by
        pad_scale (matching their crop_video cropScale=0.40), resize the
        padded square to 224x224, then center-crop to 112x112 — NOT a
        direct resize to 112x112. TalkNet was trained on this exact framing;
        skipping the padding/center-crop step changes the effective zoom
        and aspect ratio enough to suppress real lip-sync signal.
        """
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        s = max(w, h) / 2               #here take bigger dimention for square
        bsi = int(s * (1 + 2 * pad_scale))

        padded = cv2.copyMakeBorder(frame, bsi, bsi, bsi, bsi,
                                     cv2.BORDER_CONSTANT, value=(110, 110, 110))
        mx, my = cx + bsi, cy + bsi

        #actual crop for TalkNet: padded square around the face, with padding and scaling applied
        face = padded[int(my - s):int(my + s * (1 + 2 * pad_scale)),
                      int(mx - s * (1 + pad_scale)):int(mx + s * (1 + pad_scale))]

        if face.size == 0:
            return None

        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (224, 224))
        face = face[56:168, 56:168]  # center crop 224 -> 112, matches original
        return face

    def _run_talknet(self, turn: dict, video_path: str, face_app, registry, audio_path: str, frame_faces: dict) -> dict:
        sr, audio_data = wavfile.read(audio_path)
        start_sample = int(turn["start"] * sr)
        end_sample = int(turn["end"] * sr)
        audio_segment = audio_data[start_sample:end_sample]

        temp_audio = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wavfile.write(tmp.name, sr, audio_segment)
                temp_audio = tmp.name
            return self._run_talknet_compute(turn, video_path, face_app, registry, temp_audio, frame_faces)
        finally:
            if temp_audio and os.path.exists(temp_audio):
                try:
                    os.unlink(temp_audio)
                except Exception:
                    pass

    def _run_talknet_compute(self, turn: dict, video_path: str, face_app, registry, temp_audio: str, frame_faces: dict) -> dict:
        # ── dense, native-fps re-detection for THIS turn only ──
        dense_frames = extract_frames_in_range(video_path, turn["start"], turn["end"])
        n_dense = len(dense_frames)

        # frame-index-aligned: identity -> {frame_idx: crop}, NOT a flat list.
        # A flat list loses timing — if identity X is only matched on frames
        # 2, 5, 9... the flat list looks like 3 consecutive frames to
        # TalkNet, which destroys the audio/video sync it depends on.
        #
        # Identity for each dense-frame face is resolved via IoU against
        # Step 2's nearest 8fps detection (not a fresh embedding lookup) —
        # cheaper, and sufficient since native-fps frames barely move
        # position between Step 2's sampled frames.
        crops_by_frame_idx = {}
        for i, (ts, frame) in enumerate(dense_frames):
            # detection-only pass — face_app here is the lightweight
            # dense_app instance (detection module only), not the full
            # recognition-capable app used in Step 2.
            faces = face_app.get(frame)

            # nearest Step 2 (8fps) detection to this native-fps frame's
            # timestamp — computed once per FRAME, reused for every face
            # detected in that frame (all faces in a frame share the same
            # timestamp, so this doesn't need to be recomputed per face).
            nearest = min(frame_faces.keys(), key=lambda t: abs(t - ts), default=None)
            nearest_dets = frame_faces.get(nearest, []) if nearest is not None else []

            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int).tolist()

                best_identity, best_iou = None, 0.3  # min IoU threshold
                for det in nearest_dets:
                    bx1, by1, bx2, by2 = det["bbox"]
                    iou = bb_iou((x1, y1, x2, y2), (bx1, by1, bx2, by2))
                    if iou > best_iou:
                        best_iou, best_identity = iou, det["identity"]

                if best_identity is None:
                    continue  # don't score a face we can't confidently attribute

                prepped = self._prep_face_crop(frame, (x1, y1, x2, y2))
                if prepped is None:
                    continue
                crops_by_frame_idx.setdefault(best_identity, {})[i] = prepped

        # Build a full-length, gap-filled sequence per identity so the
        # video stream stays continuous and time-aligned with audio.
        # Gaps (missed detections) are forward-filled from the last known
        # crop; leading gaps are back-filled from the first known crop.
        MIN_COVERAGE = 0.4  # skip identities detected in <40% of frames —
                             # too sparse to trust a lip-sync score from
        face_crops_by_identity = {}
        for identity, frame_map in crops_by_frame_idx.items():
            coverage = len(frame_map) / n_dense if n_dense else 0.0
            print(f"      {identity}: detected in {len(frame_map)}/{n_dense} "
                  f"frames ({coverage:.0%} coverage)")
            if coverage < MIN_COVERAGE:             #talkNet requires continuously reasonable sequence of mouth movements to detect so
                print(f"      WARNING: skipping {identity}, coverage too sparse for reliable ASD")
                continue

            filled = []
            last_crop = None
            for i in range(n_dense):
                if i in frame_map:
                    last_crop = frame_map[i]
                filled.append(last_crop)
            # back-fill any leading None entries (identity's first
            # detection happened partway through the turn)
            first_valid = next((c for c in filled if c is not None), None)
            filled = [c if c is not None else first_valid for c in filled]
            filled = [c for c in filled if c is not None]  # identity never detected at all
            if filled:
                face_crops_by_identity[identity] = filled

        best_identity = "Unknown"
        best_score = -float("inf")
        all_scores = []

        for identity, crops in face_crops_by_identity.items():
            if not crops:
                continue
            try:
                # crops are already 112x112 grayscale from _prep_face_crop —
                # just filter out any that came back malformed.
                resized = [
                    c for c in crops
                    if c is not None and c.shape[0] > 0 and c.shape[1] > 0
                ]

                if not resized:
                    continue

                video_array = np.asarray(resized, dtype=np.float32)  # [Timestamp, Height, Width]

                # Load audio and convert to MFCC — TalkNet's audio frontend
                # expects (B, T, 13) MFCC features, not raw waveform samples.
                sample_rate, audio_array = wavfile.read(temp_audio)

                if audio_array.dtype == np.int16:
                    audio_array = audio_array.astype(np.float32) / 32768.0
                elif audio_array.dtype == np.int32:
                    audio_array = audio_array.astype(np.float32) / 2147483648.0
                else:
                    audio_array = audio_array.astype(np.float32)

                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=1)

                if sample_rate != 16000:
                    signal = torch.from_numpy(audio_array).unsqueeze(0)
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    audio_array = resampler(signal).squeeze(0).numpy()

                #talknet doesnt take raw audio, it takes MFCC features as input, so converting audio waveform to mfcc
                #mfcc extracts 13 MFCC features from the audio signal, with a window length of 25ms and a step size of 10ms. The output is a 2D array where each row corresponds to a frame and each column corresponds to an MFCC coefficient such as characteristics of the  audio signal
                
                mfcc_feat = mfcc(
                    audio_array,
                    samplerate=16000,
                    numcep=13,          #13 MFCC features to extract
                    winlen=0.025,          #winlien=0.025 means 25ms window length for each frame means instead of analyzing entire audio once breaks then inti small chunks of 25ms and extracts features from each chunk
                    winstep=0.010,          #winstep=0.01 means 10ms step size between consecutive frames, so each frame overlaps with the previous one by 15ms (25ms - 10ms)
                )

                # TalkNet's audio encoder downsamples MFCC frames by 4x
                # internally, so video_frames must equal mfcc_frames // 4.
                # Trim both streams to the longest length that satisfies
                # that ratio.
                length = min(mfcc_feat.shape[0] // 4, video_array.shape[0])
                if length < 1:
                    print(f"  WARNING: turn too short for TalkNet ({identity}), skipping")
                    continue


                #align audio and video lengths
                mfcc_feat = mfcc_feat[:length * 4, :]           #4 audio MFCC frames ↔ 1 video frame required by talknet, so trimming audio to be 4x the length of video frames
                video_array = video_array[:length, :]

                video_tensor = torch.from_numpy(video_array).unsqueeze(0)          # [B, T, H, W]
                audio_tensor = torch.from_numpy(mfcc_feat).float().unsqueeze(0)    # [B, T_audio, 13]

                print(
                    f"  TalkNet input shapes: "
                    f"video={tuple(video_tensor.shape)}, "
                    f"audio={tuple(audio_tensor.shape)}"
                )

                with torch.no_grad():
                    talknet_model = self.model.model
                    device = next(talknet_model.parameters()).device
                    video_tensor = video_tensor.to(device)
                    audio_tensor = audio_tensor.to(device)

                    visual_embed = talknet_model.forward_visual_frontend(video_tensor)
                    audio_embed = talknet_model.forward_audio_frontend(audio_tensor)
                    audio_embed, visual_embed = talknet_model.forward_cross_attention(audio_embed, visual_embed)
                    outs_av = talknet_model.forward_audio_visual_backend(audio_embed, visual_embed)

                    # lossAV.FC projects the 256-dim fused embedding to 2 raw logits/frame
                    # and returns logit[:,1] (speaking class) as a numpy array — this is
                    # the real scoring step; outs_av alone is just an intermediate embedding.
                    pred_score = self.model.lossAV.forward(outs_av, labels=None)
                    score = float(pred_score.mean())

                print(f"      RAW logit score for {identity}: {score!r}")

                # raw logit, unbounded — sigmoid maps it to a 0-1 "confidence"
                # without destroying relative ranking the way clamping would
                normalized = 1 / (1 + math.exp(-score))

                all_scores.append((identity, normalized))

                if score > best_score:
                    best_score = score
                    best_identity = identity

            except Exception as e:
                print(f"  WARNING: TalkNet failed for {identity}: {e}")
                continue

        return {
            "candidates": sorted(all_scores, key=lambda x: x[1], reverse=True),
            "method": "talknet"
        }


#  STEP 6 — Final matching + output                       

class SpeakerFaceMatcher:
    """
    Hands each turn's voice embedding + ASD face candidates to the shared
    PersonRegistry, which owns all the cross-modal linking logic.
    """

    def match(self, recognised_turns, frame_faces, frames, audio_path, asd, registry, video_path, face_app):
        print(f"[Step 6] Matching {len(recognised_turns)} speaker turns...")
        matched_speakers = []

        turn_asd_results = []
        
        # ── Pass 1: Run ASD for all turns ──
        for turn in recognised_turns:
            asd_result = asd.detect_active_speaker(frames, frame_faces, audio_path, turn, video_path, face_app, registry)
            turn_asd_results.append({
                "turn": turn,
                "face_candidates": asd_result.get("candidates", [])
            })
           
        # ── Splitting PyAnnote Clusters ──
        # Group turns by original speaker_label
        speaker_to_turns = {}
        for res in turn_asd_results:
            sl = res["turn"]["speaker_label"]
            speaker_to_turns.setdefault(sl, []).append(res)
            
        for sl, res_list in speaker_to_turns.items():
            top_faces_in_cluster = set()
            face_wins = {}
            for res in res_list:
                candidates = res["face_candidates"]
                if candidates:
                    top_face, top_score = max(candidates, key=lambda x: x[1])
                    top_face_resolved = registry.resolve(top_face)
                    top_faces_in_cluster.add(top_face_resolved)
                    face_wins[top_face_resolved] = face_wins.get(top_face_resolved, 0) + 1
                        
            if len(top_faces_in_cluster) > 1:
                print(f"  [Cluster Split] Diarisation cluster {sl} contains {len(top_faces_in_cluster)} distinct active faces: {top_faces_in_cluster}. Splitting cluster based on relative TalkNet scores.")
                best_overall_face = max(face_wins.items(), key=lambda x: x[1])[0]
                
                for res in res_list:
                    candidates = res["face_candidates"]
                    assigned_face = None
                    if candidates:
                        top_face, top_score = max(candidates, key=lambda x: x[1])
                        assigned_face = registry.resolve(top_face)
                    
                    if assigned_face is None:
                        assigned_face = best_overall_face
                        
                    res["turn"]["speaker_label"] = f"{sl}_split_{assigned_face}"

        # ── Pass 2: Resolve identities ──
        for res in turn_asd_results:
            turn = res["turn"]
            face_candidates = res["face_candidates"]

            result = registry.resolve_turn(
                voice_embedding=turn.get("voice_embedding"),
                voice_speaker_label=turn.get("speaker_label"),
                face_candidates=face_candidates
            )

            matched_speakers.append({
                **result,
                "start_time":    round(turn["start"], 2),
                "end_time":      round(turn["end"], 2),
                "speaker_label": turn["speaker_label"],
            })

            icon = {
                "DUAL_CONFIRMED":       "✓✓",
                "LINKED_UNCONFIRMED":   "🔗",
                "VOICE_ONLY_OFFSCREEN": "🎙 ",
                "FACE_ONLY":            "👁 ",
                "UNCONFIRMED":          "? ",
            }.get(result["confirmation"], "? ")
            print(f"  {icon} [{turn['start']:.1f}s-{turn['end']:.1f}s] "
                  f"{result['label']} ({result['confirmation']}) conf={result['confidence']:.2f}")

            association = result.get("speaker_face_association")
            if association:
                print(
                    f"      ↳ PyAnnote {turn['speaker_label']} "
                    f"→ {association['face_id']} "
                    f"(avg ASD={association['confidence']:.2f}, "
                    f"observations={association['observations']})"
                )

            candidates = result.get("face_candidates", [])
            if candidates:
                print("      Face candidates:")
                for candidate in candidates:
                    print(
                        f"        {candidate['face_id']}: "
                        f"{candidate['asd_confidence']:.2f}"
                    )

        return matched_speakers


def render_debug_video(video_path: str, frame_faces: dict, matched_results: list, output_path: str = "debug_overlay.mp4"):
    """
    Burns identity + ASD score onto every 8fps sampled frame so you can
    visually confirm whether the highest-scoring candidate in each turn
    is actually the person whose mouth is moving — a direct sanity check
    on TalkNet's output, independent of the confidence-threshold logic.

    cv2.VideoWriter never carries audio, so after writing the silent
    boxed overlay, the original video's audio track is muxed back in via
    ffmpeg to produce a second, watchable-with-sound file.
    """
    abs_output_path = os.path.abspath(output_path)
    print(f"[Debug Overlay] Writing to: {abs_output_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Debug Overlay] ERROR: could not open input video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Debug Overlay] Input video: {w}x{h} @ {fps:.2f}fps, {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not out.isOpened():
        print(f"[Debug Overlay] ERROR: VideoWriter failed to open with mp4v codec for {output_path}")
        print(f"[Debug Overlay] Retrying with XVID/.avi container...")
        cap.release()

        output_path = output_path.rsplit(".", 1)[0] + ".avi"
        abs_output_path = os.path.abspath(output_path)
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        if not out.isOpened():
            print(f"[Debug Overlay] ERROR: VideoWriter also failed with XVID/.avi. Aborting.")
            cap.release()
            return
        print(f"[Debug Overlay] XVID/.avi VideoWriter opened successfully: {abs_output_path}")

    timestamps = sorted(frame_faces.keys())
    if not timestamps:
        print("[Debug Overlay] WARNING: frame_faces is empty — no detections to overlay. "
              "Video will be written with no boxes.")

    frame_idx = 0
    ts_idx = 0
    frames_written = 0
    frames_with_boxes = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps

        while ts_idx + 1 < len(timestamps) and timestamps[ts_idx + 1] <= t:
            ts_idx += 1
        current_ts = timestamps[ts_idx] if timestamps else None
        detections = frame_faces.get(current_ts, []) if current_ts is not None else []

        active_turn = next(
            (r for r in matched_results if r["start_time"] <= t <= r["end_time"]),
            None
        )
        score_by_identity = {}
        confirmed_face_id = None
        if active_turn:
            for c in active_turn.get("face_candidates", []):
                score_by_identity[c["face_id"]] = c["asd_confidence"]
            if active_turn["confirmation"] == "DUAL_CONFIRMED":
                confirmed_face_id = active_turn.get("face_id")

        if detections:
            frames_with_boxes += 1

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            identity = det["identity"]
            score = score_by_identity.get(identity)
            is_confirmed = identity == confirmed_face_id

            color = (0, 255, 0) if is_confirmed else (0, 165, 255)
            label = identity
            if score is not None:
                label += f" {score:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)
        frames_written += 1
        frame_idx += 1

    cap.release()
    out.release()

    print(f"[Debug Overlay] Frames written: {frames_written}/{total_frames}")
    print(f"[Debug Overlay] Frames with at least one box drawn: {frames_with_boxes}")

    if frames_written == 0:
        print("[Debug Overlay] ERROR: 0 frames written — input video likely failed to decode any frames.")
        return
    elif not os.path.exists(abs_output_path):
        print(f"[Debug Overlay] ERROR: file does not exist after writing.")
        return

    size_kb = os.path.getsize(abs_output_path) / 1024
    print(f"[Debug Overlay] SUCCESS (video-only): {abs_output_path} ({size_kb:.1f} KB)")

    # ── mux original audio back in — cv2.VideoWriter never carries audio ──
    silent_path = abs_output_path
    final_path = silent_path.rsplit(".", 1)[0] + "_with_audio.mp4"

    cmd = [
        "ffmpeg", "-y",
        "-i", silent_path,      # video (no audio) from OpenCV
        "-i", video_path,       # original video (has audio)
        "-c:v", "copy",         # don't re-encode video, just copy it
        "-c:a", "aac",          # encode audio track as aac for mp4 compatibility
        "-map", "0:v:0",        # take video from the silent overlay
        "-map", "1:a:0",        # take audio from the original video
        "-shortest",
        final_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[Debug Overlay] WARNING: audio mux failed: {result.stderr}")
        print(f"[Debug Overlay] Silent video still available at: {silent_path}")
    else:
        print(f"[Debug Overlay] SUCCESS (with audio): {os.path.abspath(final_path)}")


#  MAIN — Full pipeline           

def run_pipeline(
    video_path: str,
    enrolled_faces: dict = None,
    enrolled_voices: dict = None,
    output_path: str = "asd_output.json"
):
    """
    Run the complete Active Speaker Detection pipeline.
    Ensures extracted audio is safely cleaned up even on failure.
    """
    audio_path = extract_audio_from_video(video_path)
    try:
        return _run_pipeline_inner(
            video_path=video_path,
            audio_path=audio_path,
            enrolled_faces=enrolled_faces,
            enrolled_voices=enrolled_voices,
            output_path=output_path,
        )
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except Exception:
                pass


def _run_pipeline_inner(
    video_path: str,
    audio_path: str,
    enrolled_faces: dict = None,
    enrolled_voices: dict = None,
    output_path: str = "asd_output.json"
):
    print("\n" + "="*60)
    print("RECONNECT — Active Speaker Detection Pipeline")
    print("="*60 + "\n")

    # ── Step 1: Extract video frames ─────────────────────────
    frames = extract_frames(video_path, fps=2)
    registry = PersonRegistry(enrolled_faces, enrolled_voices)

    # ── Step 2: Face detection + recognition ────────────────
    face_detector = FaceDetector(registry)
    frame_faces = face_detector.process_all_frames(frames)

    # ── Step 3: Speaker diarisation ──────────────────────────
    diariser = SpeakerDiariser(HF_TOKEN)
    turns = diariser.diarise(audio_path)

    # ── Sanity check: does diarized speaker count match face count? ──
    # Must run AFTER Step 2 (needs registry.people populated) and can run
    # here since it only depends on face_samples, not on Step 6 fields.
    diarization_check = check_diarization_speaker_count(turns, registry)

    # ── Step 4: Speaker recognition ──────────────────────────
    speaker_recogniser = SpeakerRecogniser(registry)
    recognised_turns = speaker_recogniser.recognise_turns(audio_path, turns)

    # ── Step 5 + 6: ASD + Final matching ────────────────────
    registry.set_total_video_turns(len(recognised_turns))
    asd = ActiveSpeakerDetector()
    matcher = SpeakerFaceMatcher()
    results = matcher.match(
        recognised_turns, frame_faces, frames, audio_path, asd, registry,
        video_path, face_detector.dense_app   # lightweight, detection-only app
    )

    # ── Final determination — accumulated evidence, not per-turn spikes ──
    final_links = registry.finalize_speaker_face_links(
        min_turns=1,
        min_avg_confidence=0.05,
        min_win_ratio=0.6,
    )
    print("\nFinal voice=face determinations:")
    for speaker_label, person_id in final_links.items():
        if person_id:
            rec = registry.people[person_id]
            print(f"  {speaker_label} → {rec.display_label()} (person_id={person_id}) CONFIRMED")
        else:
            print(f"  {speaker_label} → insufficient evidence, not linked")

    render_debug_video(video_path, frame_faces, results, output_path="debug_overlay.mp4")

    # ── Save output ──────────────────────────────────────────
    output = {
        "video":             video_path,
        "total_turns":       len(results),
        "diarization_check": diarization_check,
        "matched_speakers":  results
    }

    def _json_default(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total speaker turns:   {len(results)}")

    conf_counts = Counter(r["confirmation"] for r in results)
    for conf, count in conf_counts.items():
        print(f"  {conf}: {count}")

    unique_ids = set(r["label"] for r in results)
    print(f"\nUnique speakers identified: {unique_ids}")

    # Cross-modal association summary: PyAnnote speaker -> persistent face ID
    if hasattr(registry, "speaker_face_associations"):
        associations = registry.speaker_face_associations()
        print("\nPyAnnote speaker -> face associations:")
        if associations:
            for speaker_label, info in associations.items():
                print(
                    f"  {speaker_label} -> {info['face_id']} "
                    f"(confidence={info['confidence']:.2f}, "
                    f"evidence={info['observations']})"
                )
        else:
            print("  No reliable speaker/face associations yet.")

    print(f"\nFull output saved to: {output_path}")

    return results


#  QUICK TEST 

def quick_test():
    """
    Quick test with your own video.

    HOW TO TEST:
    1. Record a 30-60 second video of 1-2 people talking
       (use your phone, save as test_video.mp4)
    2. Put test_video.mp4 in same folder as this script
    3. Run: python active_speaker_detection.py
    4. Check asd_output.json for results

    For enrollment testing — add face images:
    5. Take 3-5 photos of a person (clear face, different angles)
    6. Call face_detector.enroll_face("YourName", "photo.jpg")
    7. Record 30s voice sample and call
       speaker_recogniser.enroll_voice("YourName", "voice.wav")
    """

    video_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_video.mp4"
    )

    if not os.path.exists(video_file):
        print("="*60)
        print("TEST MODE — No video found")
        print("="*60)
        print(f"\nPlease provide a test video file: {video_file}")
        print("\nHow to create a test video:")
        print("  1. Record yourself (or 2 people) talking for 30-60s")
        print("  2. Save as 'test_video.mp4' in this folder")
        print("  3. Run this script again")
        print("\nFor enrollment testing:")
        print("  Add face images and voice samples as shown in quick_test()")
        return

    results = run_pipeline(
        video_path      = video_file,
        enrolled_faces  = {},    # add {name: embedding} here after enrollment
        enrolled_voices = {},    # add {name: embedding} here after enrollment
        output_path     = "asd_output.json"
    )

    print("\n" + "="*60)
    print("READABLE OUTPUT")
    print("="*60)
    for r in results:
        print(
            f"\n[{r['start_time']}s → {r['end_time']}s]"
            f"\n  Speaker:        {r['label']}"
            f"\n  Confirmation:   {r['confirmation']}"
            f"\n  Confidence:     {r['confidence']:.0%}"
            f"\n  Voice id:       {r['voice_id']}"
            f"\n  Face id:        {r['face_id']}"
            f"\n  ASD confidence: {r['asd_confidence']:.0%}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RECONNECT Active Speaker Detection")
    parser.add_argument("--video", type=str, default="test_video.mp4", help="Path to input video file")
    parser.add_argument("--output", type=str, default="asd_output.json", help="Path to save JSON output")
    args = parser.parse_args()

    if os.path.exists(args.video):
        run_pipeline(args.video, output_path=args.output)
    else:
        quick_test()