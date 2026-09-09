import os
import json
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow this file to be run directly as well as from the backend package.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from vision.face_engine import FaceEngine
from config import GALLERY_PATH


ENROLLMENT_IMAGE_PATH = str(BACKEND_ROOT / "data" / "enrollment_images")


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)


def get_file_hash(file_path):
    """
    Creates a hash of the exact image file.

    This lets us determine whether the exact
    same image has already been enrolled.
    """

    hasher = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            chunk = file.read(8192)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def load_metadata(metadata_path):

    if not os.path.exists(metadata_path):

        return {
            "images": {}
        }

    try:

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except (
        json.JSONDecodeError,
        OSError
    ):

        return {
            "images": {}
        }


def save_metadata(
    metadata_path,
    metadata
):

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )


def get_next_embedding_index(
    gallery_path
):

    indices = []

    if not os.path.exists(
        gallery_path
    ):

        return 1

    for filename in os.listdir(
        gallery_path
    ):

        if not filename.endswith(".npy"):
            continue

        if not filename.startswith(
            "embedding_"
        ):
            continue

        number = filename[
            len("embedding_"):
            -len(".npy")
        ]

        try:

            indices.append(
                int(number)
            )

        except ValueError:

            continue

    if not indices:

        return 1

    return max(indices) + 1


def process_person_folder(
    identity,
    face_engine
):

    # --------------------------------------
    # Source folder
    # --------------------------------------

    source_folder = os.path.join(
        ENROLLMENT_IMAGE_PATH,
        identity
    )

    # --------------------------------------
    # Automatically create/use
    # same-named gallery folder
    # --------------------------------------

    gallery_folder = os.path.join(
        GALLERY_PATH,
        identity
    )

    os.makedirs(
        gallery_folder,
        exist_ok=True
    )

    metadata_path = os.path.join(
        gallery_folder,
        "metadata.json"
    )

    metadata = load_metadata(
        metadata_path
    )

    if "images" not in metadata:

        metadata["images"] = {}

    # --------------------------------------
    # Find images
    # --------------------------------------

    image_files = [

        filename

        for filename in sorted(
            os.listdir(source_folder)
        )

        if filename.lower().endswith(
            IMAGE_EXTENSIONS
        )
    ]

    if not image_files:

        print(
            f"[{identity}] No images found."
        )

        return {
            "new": 0,
            "skipped": 0,
            "failed": 0
        }

    # --------------------------------------
    # Find next embedding number
    # --------------------------------------

    next_index = (
        get_next_embedding_index(
            gallery_folder
        )
    )

    new_count = 0
    skipped_count = 0
    failed_count = 0

    print()
    print(
        f"Processing: {identity}"
    )

    print(
        f"Images found: {len(image_files)}"
    )

    for filename in image_files:

        image_path = os.path.join(
            source_folder,
            filename
        )

        print(
            f"  → {filename}",
            end=" "
        )

        # ==================================
        # CHECK WHETHER IMAGE ALREADY EXISTS
        # ==================================

        file_hash = get_file_hash(
            image_path
        )

        if file_hash in metadata["images"]:

            old_embedding = (
                metadata["images"][
                    file_hash
                ]["embedding_file"]
            )

            print(
                f"[SKIP - already enrolled]"
            )

            print(
                f"     embedding: "
                f"{old_embedding}"
            )

            skipped_count += 1

            continue

        # ==================================
        # READ IMAGE
        # ==================================

        image = cv2.imread(
            image_path
        )

        if image is None:

            print(
                "[FAILED - cannot read image]"
            )

            failed_count += 1

            continue

        # ==================================
        # DETECT FACE
        # ==================================

        faces = (
            face_engine.detect_faces(
                image
            )
        )

        if not faces:

            print(
                "[FAILED - no face detected]"
            )

            failed_count += 1

            continue

        # ==================================
        # CHOOSE LARGEST FACE
        # ==================================

        face = max(

            faces,

            key=lambda item:

                (
                    item["bbox"][2]
                    - item["bbox"][0]
                )
                *
                (
                    item["bbox"][3]
                    - item["bbox"][1]
                )
        )

        embedding = face[
            "embedding"
        ]

        if embedding is None:

            print(
                "[FAILED - face quality too low for embedding]"
            )

            failed_count += 1

            continue

        # ==================================
        # SAVE EMBEDDING
        # ==================================

        embedding_filename = (
            f"embedding_{next_index:03d}.npy"
        )

        embedding_path = os.path.join(
            gallery_folder,
            embedding_filename
        )

        np.save(
            embedding_path,
            embedding
        )

        # ==================================
        # STORE IMAGE → EMBEDDING MAPPING
        # ==================================

        metadata["images"][
            file_hash
        ] = {

            "source_image":
                filename,

            "source_path":
                os.path.abspath(
                    image_path
                ),

            "embedding_file":
                embedding_filename
        }

        print(
            f"[ENROLLED → "
            f"{embedding_filename}]"
        )

        next_index += 1
        new_count += 1

    # ======================================
    # SAVE METADATA
    # ======================================

    metadata["identity"] = identity

    save_metadata(
        metadata_path,
        metadata
    )

    return {
        "new": new_count,
        "skipped": skipped_count,
        "failed": failed_count
    }


def main():

    print()
    print("=" * 60)
    print("MEMORYLENS AUTOMATIC ENROLLMENT")
    print("=" * 60)

    # --------------------------------------
    # Check enrollment directory
    # --------------------------------------

    if not os.path.exists(
        ENROLLMENT_IMAGE_PATH
    ):

        print()
        print(
            "ERROR:"
        )

        print(
            f"Enrollment directory does not exist:"
        )

        print(
            os.path.abspath(
                ENROLLMENT_IMAGE_PATH
            )
        )

        return

    # --------------------------------------
    # Find ALL person folders
    # --------------------------------------

    person_folders = [

        folder

        for folder in sorted(
            os.listdir(
                ENROLLMENT_IMAGE_PATH
            )
        )

        if os.path.isdir(
            os.path.join(
                ENROLLMENT_IMAGE_PATH,
                folder
            )
        )
    ]

    if not person_folders:

        print()
        print(
            "No person folders found."
        )

        print()
        print(
            "Create folders like:"
        )

        print(
            "data/enrollment_images/mom/"
        )

        print(
            "data/enrollment_images/dad/"
        )

        return

    # --------------------------------------
    # Automatically create gallery root
    # --------------------------------------

    os.makedirs(
        GALLERY_PATH,
        exist_ok=True
    )

    # --------------------------------------
    # Load face model ONCE
    # --------------------------------------

    print()
    print(
        "Loading face recognition model..."
    )

    face_engine = FaceEngine()

    # --------------------------------------
    # Process every person folder
    # --------------------------------------

    total_new = 0
    total_skipped = 0
    total_failed = 0

    for identity in person_folders:

        result = process_person_folder(
            identity,
            face_engine
        )

        total_new += result["new"]
        total_skipped += result["skipped"]
        total_failed += result["failed"]

    # --------------------------------------
    # Final summary
    # --------------------------------------

    print()
    print("=" * 60)
    print("ENROLLMENT FINISHED")
    print("=" * 60)

    print(
        f"People processed: "
        f"{len(person_folders)}"
    )

    print(
        f"New embeddings: "
        f"{total_new}"
    )

    print(
        f"Already enrolled: "
        f"{total_skipped}"
    )

    print(
        f"Failed images: "
        f"{total_failed}"
    )

    print()
    print(
        "Gallery location:"
    )

    print(
        os.path.abspath(
            GALLERY_PATH
        )
    )


if __name__ == "__main__":

    main()