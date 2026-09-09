import sys
from datetime import datetime
from pathlib import Path

import cv2


BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from database.db import (
    create_identity,
    find_matching_face,
    init_database,
    append_voice_embedding,
)
from enrollment.enroll import ENROLLMENT_IMAGE_PATH, FaceEngine
from enroll_audio import record_voice_embedding


def _capture_image():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    print("Look at the camera. Press SPACE to capture or ESC to cancel.")
    captured = None

    while True:
        success, frame = camera.read()
        if not success:
            continue

        cv2.imshow("Enroll user", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            captured = frame.copy()
            break
        if key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()

    if captured is None:
        raise RuntimeError("Face capture cancelled.")
    return captured


def _get_face_image():
    image_path = input(
        "Face image path (leave blank to use camera): "
    ).strip().strip('"')

    if image_path:
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        return image

    return _capture_image()


def enroll_user():
    init_database()

    name = input("Name: ").strip()
    relation = input("Relation: ").strip()
    if not name or not relation:
        raise ValueError("Name and relation are required.")

    image = _get_face_image()
    face_engine = FaceEngine()
    faces = face_engine.detect_faces(image)
    usable_faces = [face for face in faces if face.get("embedding") is not None]
    if not usable_faces:
        raise RuntimeError("No usable face embedding was detected.")

    face = max(
        usable_faces,
        key=lambda item: (
            item["bbox"][2] - item["bbox"][0]
        ) * (
            item["bbox"][3] - item["bbox"][1]
        ),
    )
    embedding = face["embedding"]
    existing, similarity = find_matching_face(embedding)

    if existing is not None:
        identity_id = existing["id"]
        print(
            f"Face already enrolled as {existing['name']} "
            f"(id={identity_id}, similarity={similarity:.3f})."
        )
        print("Enrollment stopped; no database or image changes were made.")
        return identity_id
    else:
        identity_id = create_identity(name, relation, embedding, face_image=image)
        print(f"Face enrolled in SQLite with id={identity_id} (including image BLOB).")

    image_directory = Path(ENROLLMENT_IMAGE_PATH) / name
    image_directory.mkdir(parents=True, exist_ok=True)
    image_path = image_directory / (
        datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
    )
    cv2.imwrite(str(image_path), image)
    print(f"Face image saved to {image_path}.")

    wants_voice = input("Enroll voice now? (y/n): ").strip().lower()
    if wants_voice in ("y", "yes"):
        for sample_number in range(1, 4):
            print(f"Voice sample {sample_number} of 3")
            voice_embedding = record_voice_embedding(
                identity=name,
                sample_number=sample_number,
            )
            append_voice_embedding(identity_id, voice_embedding)
        print("Three voice embeddings saved in SQLite.")
    else:
        print("Voice enrollment skipped; voice_embedding remains NULL.")

    return identity_id


if __name__ == "__main__":
    try:
        enroll_user()
    except (RuntimeError, ValueError) as error:
        print(f"Enrollment failed: {error}")