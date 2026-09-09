"""Process a phone video for an already detected unenrolled voice.

This is intentionally separate from the manual ``/who-is-this/capture``
flow.  It mirrors the post-recording stages of ``main.py --take-a-video``:
face detection on the first second, active-speaker detection on the complete
video, then voice-to-face binding for an existing unenrolled identity.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch

from activity.test_realtime import run_pipeline as run_activity_pipeline
from config import TARGET_FPS
from database.db import (
    find_matching_enrolled_voice,
    find_matching_unenrolled_voice,
    get_all_identities,
    get_unenrolled_identity,
    update_unenrolled_face,
    voice_blob_to_embeddings,
)
from vision.pipeline import WhoIsThisPipeline, save_results


def _extract_first_second_frames(video_path: Path) -> list[tuple[float, object]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Could not open the uploaded video.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    next_sample_time = 0.0
    frames: list[tuple[float, object]] = []
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_number / fps
            if timestamp >= 1.0:
                break
            if timestamp >= next_sample_time:
                frames.append((timestamp, frame))
                next_sample_time += 1.0 / TARGET_FPS
            frame_number += 1
    finally:
        capture.release()
    if not frames:
        raise ValueError("The uploaded video contains no usable frames.")
    return frames


def _face_for_turn(registry, turn_face_id: str | None):
    """Find ASD-confirmed face embedding, retaining local-capture semantics."""

    if registry and hasattr(registry, "people") and turn_face_id in registry.people:
        person = registry.people[turn_face_id]
        embedding = (
            person.face_embedding
            if person.face_embedding is not None
            else (person.face_gallery[0] if person.face_gallery else None)
        )
        return embedding, turn_face_id
    if registry and hasattr(registry, "speaker_face_associations"):
        for association in registry.speaker_face_associations().values():
            face_id = association.get("face_id")
            if face_id and face_id in registry.people:
                person = registry.people[face_id]
                embedding = (
                    person.face_embedding
                    if person.face_embedding is not None
                    else (person.face_gallery[0] if person.face_gallery else None)
                )
                if embedding is not None:
                    return embedding, face_id
    return None, turn_face_id


def _best_face_image(vision_people: list[dict], turn_face_id: str | None) -> bytes | None:
    candidates = [
        person for person in vision_people
        if person.get("entity_id") == turn_face_id
        or (isinstance(person.get("entity_id"), str) and person["entity_id"].startswith("unknown_"))
    ] + vision_people
    for person in candidates:
        image_path = person.get("best_face_image")
        if image_path and os.path.exists(image_path):
            try:
                return Path(image_path).read_bytes()
            except OSError:
                continue
    return None


def _target_voice_match(turn_voice: np.ndarray, target_unenrolled_id: int) -> float:
    target = get_unenrolled_identity(target_unenrolled_id)
    if not target or not target.get("voice_embedding"):
        return -1.0
    best = -1.0
    for stored in voice_blob_to_embeddings(target["voice_embedding"]):
        candidate = np.asarray(stored, dtype=np.float32).reshape(-1)
        if candidate.shape != turn_voice.shape or np.linalg.norm(candidate) == 0:
            continue
        score = float(
            np.dot(turn_voice, candidate) /
            (np.linalg.norm(turn_voice) * np.linalg.norm(candidate))
        )
        best = max(best, score)
    return best


def process_unknown_voice_video(
    video_path: Path,
    target_unenrolled_id: int,
    patient_id: int,
) -> dict[str, object]:
    """Run the local unknown-voice video pipeline against an uploaded phone video."""

    if get_unenrolled_identity(target_unenrolled_id) is None:
        raise ValueError(f"Unenrolled identity {target_unenrolled_id} does not exist.")

    vision_frames = _extract_first_second_frames(video_path)
    vision_results = WhoIsThisPipeline(patient_id=patient_id).process_frames(
        vision_frames,
        capture_seconds=1.0,
    )
    save_results(vision_results)
    vision_people = vision_results.get("people", [])

    enrolled_voices: dict[str, torch.Tensor] = {}
    for identity in get_all_identities():
        if identity.get("voice_embedding"):
            embeddings = voice_blob_to_embeddings(identity["voice_embedding"])
            if embeddings:
                enrolled_voices[identity["name"]] = torch.tensor(
                    embeddings[-1], dtype=torch.float32
                )

    results = run_activity_pipeline(
        video_path=str(video_path),
        enrolled_faces={},
        enrolled_voices=enrolled_voices,
        output_path="asd_output.json",
    )
    registry = getattr(results, "registry", None)
    if not isinstance(results, list):
        return {
            "status": "no_face_link",
            "unenrolled_id": target_unenrolled_id,
            "reason": "No active speaker turn was available for voice-face binding.",
        }

    for turn in results:
        raw_voice = turn.get("voice_embedding")
        if raw_voice is None:
            continue
        if hasattr(raw_voice, "detach"):
            voice = raw_voice.detach().cpu().numpy().flatten().astype(np.float32)
        else:
            voice = np.asarray(raw_voice, dtype=np.float32).flatten()
        if np.linalg.norm(voice) == 0:
            continue

        enrolled_match, _ = find_matching_enrolled_voice(voice, threshold=0.50)
        if enrolled_match is not None:
            continue
        target_score = _target_voice_match(voice, target_unenrolled_id)
        matched_id = target_unenrolled_id if target_score >= 0.40 else None
        if matched_id is None:
            match, _ = find_matching_unenrolled_voice(voice, threshold=0.42)
            matched_id = match["id"] if match is not None else None
        if matched_id != target_unenrolled_id:
            continue

        face_embedding, face_id = _face_for_turn(
            registry,
            turn.get("face_id") or turn.get("person_id"),
        )
        if face_embedding is None:
            continue
        image = _best_face_image(vision_people, face_id)
        update_unenrolled_face(target_unenrolled_id, face_embedding, face_image=image)
        return {
            "status": "bound",
            "unenrolled_id": target_unenrolled_id,
            "face_id": face_id,
            "face_image_saved": image is not None,
        }

    return {
        "status": "no_face_link",
        "unenrolled_id": target_unenrolled_id,
        "reason": "No confirmed active face was linked to the target voice.",
    }
