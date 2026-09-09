"""Phone-camera adapters for manual and unknown-voice vision flows."""
import asyncio
import shutil
import tempfile
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.authserver import current_actor, require_caregiver
from config import CAPTURE_SECONDS, TARGET_FPS
from database.db import (
    caregiver_owns_patient,
    create_pending_unknown,
    get_patient_identity_by_name,
    get_patient_identity_image,
    get_pending_unknown_image,
    get_pending_unknowns,
)

router = APIRouter(prefix="/who-is-this", tags=["who-is-this"])
unknown_voice_router = APIRouter(prefix="/unknown-voice", tags=["unknown-voice"])


def _can_access_patient(actor, patient_id: int) -> bool:
    return (
        actor["actor_type"] == "patient" and actor["actor_id"] == patient_id
    ) or (
        actor["actor_type"] == "caregiver"
        and caregiver_owns_patient(actor["actor_id"], patient_id)
    )


def _extract_phone_frames(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Could not read the phone camera video.")
    frames = []
    frame_interval = 1.0 / TARGET_FPS
    next_timestamp = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp > CAPTURE_SECONDS:
                break
            if timestamp + 0.001 >= next_timestamp:
                frames.append((timestamp, frame))
                next_timestamp += frame_interval
    finally:
        capture.release()
    if not frames:
        raise ValueError("The phone camera video contained no usable frames.")
    return frames


@router.post("/capture")
async def process_phone_capture(
    patient_id: int = Form(...),
    video: UploadFile = File(...),
    actor=Depends(current_actor),
):
    if not _can_access_patient(actor, patient_id):
        raise HTTPException(status_code=403, detail="You cannot process captures for this patient.")

    temp_dir = Path(tempfile.mkdtemp(prefix="reconnect_who_"))
    video_path = temp_dir / "phone_capture.mp4"
    try:
        video_bytes = await video.read()
        if not video_bytes:
            raise HTTPException(status_code=422, detail="The phone camera video was empty.")
        video_path.write_bytes(video_bytes)
        frames = _extract_phone_frames(video_path)
        # The pipeline's recognition/quality/unknown logic is unchanged; only
        # the camera source is supplied by the phone rather than cv2 webcam.
        from vision.pipeline import WhoIsThisPipeline
        result = WhoIsThisPipeline(patient_id=patient_id).process_frames(
            frames, capture_seconds=CAPTURE_SECONDS
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    people = result.get("people", [])
    known = next((person for person in people if person.get("status") == "known"), None)
    if known and known.get("identity"):
        identity = get_patient_identity_by_name(known["identity"], patient_id)
        if identity:
            return {
                "status": "known",
                "identity_id": identity["id"],
                "name": identity["name"],
                "relation": identity["relation"],
                "photo_url": f"/who-is-this/people/{identity['id']}/photo?patient_id={patient_id}",
            }

    unknown = next((person for person in people if person.get("status") in {"new_unknown", "existing_unknown"}), None)
    if unknown:
        image_path = unknown.get("best_face_image")
        pending_id = create_pending_unknown(patient_id, image_path)
        if pending_id:
            return {"status": "unknown", "pending_unknown_id": pending_id}

    raise HTTPException(
        status_code=422,
        detail="No usable face was detected. Keep the face centered, close, and in focus.",
    )


@unknown_voice_router.post("/capture")
async def process_unknown_voice_capture(
    patient_id: int = Form(...),
    unenrolled_id: int = Form(...),
    video: UploadFile = File(...),
    actor=Depends(current_actor),
):
    """Bind an uploaded phone-video face to an existing unknown voice profile.

    This intentionally does not share the manual ``/who-is-this/capture``
    endpoint or create a pending face-enrollment record.
    """

    if not _can_access_patient(actor, patient_id):
        raise HTTPException(status_code=403, detail="You cannot process captures for this patient.")
    if video.content_type and not video.content_type.startswith("video/"):
        raise HTTPException(status_code=422, detail="The unknown-voice capture must be a video file.")

    temp_dir = Path(tempfile.mkdtemp(prefix="reconnect_unknown_voice_"))
    video_path = temp_dir / "unknown_voice_capture.mp4"
    try:
        video_bytes = await video.read()
        if not video_bytes:
            raise HTTPException(status_code=422, detail="The unknown-voice video was empty.")
        video_path.write_bytes(video_bytes)
        from app.unknown_voice_capture import process_unknown_voice_video

        return await asyncio.to_thread(
            process_unknown_voice_video,
            video_path,
            unenrolled_id,
            patient_id,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("/people/{identity_id}/photo")
def enrolled_photo(identity_id: int, patient_id: int, actor=Depends(current_actor)):
    if not _can_access_patient(actor, patient_id):
        raise HTTPException(status_code=403, detail="You cannot access this image.")
    image = get_patient_identity_image(identity_id, patient_id)
    if not image:
        raise HTTPException(status_code=404, detail="Enrollment photo not found.")
    return Response(content=image, media_type="image/jpeg")


@router.get("/patients/{patient_id}/pending")
def pending_unknowns(patient_id: int, caregiver=Depends(require_caregiver)):
    if not caregiver_owns_patient(caregiver["actor_id"], patient_id):
        raise HTTPException(status_code=403, detail="That patient is not linked to this caregiver.")
    rows = get_pending_unknowns(patient_id)
    for row in rows:
        row["image_url"] = f"/who-is-this/pending/{row['id']}/photo?patient_id={patient_id}"
    return {"pending": rows}


@router.get("/pending/{pending_id}/photo")
def pending_photo(pending_id: int, patient_id: int, caregiver=Depends(require_caregiver)):
    if not caregiver_owns_patient(caregiver["actor_id"], patient_id):
        raise HTTPException(status_code=403, detail="You cannot access this image.")
    image = get_pending_unknown_image(pending_id, patient_id)
    if not image:
        raise HTTPException(status_code=404, detail="Pending unknown image not found.")
    return Response(content=image, media_type="image/jpeg")
