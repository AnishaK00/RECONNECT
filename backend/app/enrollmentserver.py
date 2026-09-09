"""HTTP enrollment endpoint — reuses the same building blocks as
enroll_user.py / enroll_audio.py, so behavior stays identical to the
CLI flow (duplicate-face short-circuit, 3-sample voice concatenation, etc.)
"""

import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from database.db import (
    append_voice_embedding, caregiver_owns_patient, create_identity,
    delete_identity, find_matching_face, get_patient_enrolled_people, resolve_pending_unknown,
)
from app.authserver import current_actor, require_caregiver
from enrollment.enroll import ENROLLMENT_IMAGE_PATH
from enroll_audio import create_voice_embedding, ENROLLMENT_AUDIO_PATH
from app.audio_process.preprocessing import preprocess_audio

router = APIRouter(prefix="/enroll", tags=["enrollment"])

# ---- Lazy singletons, same pattern already used in app/audio_process/pipeline.py ----
_face_engine = None
_audio_embedder = None


def _get_face_engine():
    global _face_engine
    if _face_engine is None:
        from vision.face_engine import FaceEngine
        _face_engine = FaceEngine()
    return _face_engine


# def _get_audio_embedder():
#     # create_voice_embedding() already builds its own AudioEmbedder() per call
#     # (see enroll_audio.py) — kept here only if you later want to cache it
#     # across requests instead. Not used directly below.
#     pass


def _decode_image(raw_bytes: bytes):
    array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail="Could not decode one of the uploaded photos.")
    return image


@router.post("/person")
async def enroll_person(
    patient_id: int = Form(...),
    name: str = Form(...),
    relation: str = Form(...),
    photos: list[UploadFile] = File(...),
    voice_samples: list[UploadFile] = File(...),
    pending_unknown_id: int | None = Form(default=None),
    caregiver=Depends(require_caregiver),
):
    if not name.strip() or not relation.strip():
        raise HTTPException(status_code=422, detail="Name and relation are required.")
    if len(voice_samples) != 3:
        raise HTTPException(status_code=422, detail="Exactly 3 voice samples are required.")
    if not caregiver_owns_patient(caregiver["actor_id"], patient_id):
        raise HTTPException(status_code=403, detail="That patient is not linked to this caregiver.")

    face_engine = _get_face_engine()

    # --------------------------------------------------------------
    # 1. Find the best usable face across ALL submitted photos
    #    (same "largest bbox area" convention as enroll.py / enroll_user.py)
    # --------------------------------------------------------------
    best_face = None
    best_area = -1
    best_image = None

    for photo in photos:
        raw = await photo.read()
        image = _decode_image(raw)
        faces = face_engine.detect_faces(image)
        usable = [f for f in faces if f.get("embedding") is not None]
        for face in usable:
            x1, y1, x2, y2 = face["bbox"]
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_face = face
                best_image = image

    if best_face is None:
        raise HTTPException(
            status_code=422,
            detail="No usable face was detected in any of the submitted photos.",
        )

    embedding = best_face["embedding"]

    # --------------------------------------------------------------
    # 2. Duplicate check — mirrors enroll_user.py exactly:
    #    if the face already exists, stop with no DB/file changes.
    # --------------------------------------------------------------
    existing, similarity = find_matching_face(embedding, patient_id=patient_id)
    if existing is not None:
        return {
            "status": "duplicate",
            "identity_id": existing["id"],
            "name": existing["name"],
            "similarity": similarity,
        }

    # --------------------------------------------------------------
    # 3. New identity — create SQLite row + save the chosen photo to disk
    # --------------------------------------------------------------
    identity_id = create_identity(name, relation, embedding, face_image=best_image, patient_id=patient_id)

    image_dir = Path(ENROLLMENT_IMAGE_PATH) / str(identity_id)
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / (datetime.now().strftime("%Y%m%d_%H%M%S.jpg"))
    cv2.imwrite(str(image_path), best_image)

    # --------------------------------------------------------------
    # 4. Voice — same 3-sample loop as enroll_user.py, driven by
    #    uploaded files instead of the local microphone.
    # --------------------------------------------------------------
    audio_dir = ENROLLMENT_AUDIO_PATH / str(identity_id)
    audio_dir.mkdir(parents=True, exist_ok=True)
    try:
        for index, sample in enumerate(voice_samples, start=1):
            raw_path = audio_dir / f"enrollment_voice_{index:02d}.wav"
            cleaned_path = audio_dir / f"enrollment_voice_{index:02d}_cleaned.wav"

            raw_bytes = await sample.read()
            if not raw_bytes:
                raise ValueError("An uploaded voice sample was empty.")
            raw_path.write_bytes(raw_bytes)
            preprocess_audio(raw_path, cleaned_path)
            voice_embedding = create_voice_embedding(cleaned_path)
            append_voice_embedding(identity_id, voice_embedding)
    except Exception as exc:
        delete_identity(identity_id)
        shutil.rmtree(image_dir, ignore_errors=True)
        shutil.rmtree(audio_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=f"Could not process enrollment audio: {exc}") from exc
    if pending_unknown_id is not None:
        resolve_pending_unknown(pending_unknown_id, patient_id, identity_id)

    return {
        "status": "enrolled",
        "identity_id": identity_id,
        "patient_id": patient_id,
        "name": name,
        "relation": relation,
        "photos_submitted": len(photos),
        "voice_samples_processed": len(voice_samples),
    }


@router.get("/patients/{patient_id}/people")
def enrolled_people(patient_id: int, actor=Depends(current_actor)):
    allowed = actor["actor_type"] == "patient" and actor["actor_id"] == patient_id
    allowed = allowed or (actor["actor_type"] == "caregiver" and caregiver_owns_patient(actor["actor_id"], patient_id))
    if not allowed:
        raise HTTPException(status_code=403, detail="That patient is not linked to this caregiver.")
    people = get_patient_enrolled_people(patient_id)
    for person in people:
        person["photo_url"] = f"/who-is-this/people/{person['id']}/photo?patient_id={patient_id}"
    return {"people": people}
