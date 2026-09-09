import os
import json
import shutil

import numpy as np

from config import (
    UNKNOWN_PATH,
    GALLERY_PATH
)


def label_unknown(
    entity_id
):

    unknown_path = os.path.join(
        UNKNOWN_PATH,
        entity_id
    )

    if not os.path.exists(
        unknown_path
    ):

        print(
            f"Unknown entity not found: "
            f"{entity_id}"
        )

        return

    metadata_path = os.path.join(
        unknown_path,
        "metadata.json"
    )

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(
            file
        )

    print()

    print("NEW PERSON DETECTED")

    print(
        f"Entity ID: {entity_id}"
    )

    print(
        f"Face image: "
        f'{metadata["best_face_image"]}'
    )

    identity = input(
        "\nWho is this? "
        "Enter name: "
    ).strip().lower()

    if not identity:

        print(
            "No label entered."
        )

        return

    identity_path = os.path.join(
        GALLERY_PATH,
        identity
    )

    os.makedirs(
        identity_path,
        exist_ok=True
    )

    source_embeddings = os.path.join(
        unknown_path,
        "face_embeddings.npy"
    )

    embeddings = np.load(
        source_embeddings
    )

    existing_files = [

        filename

        for filename in os.listdir(
            identity_path
        )

        if filename.endswith(".npy")
    ]

    start_index = (
        len(existing_files)
        + 1
    )

    for index, embedding in enumerate(

        embeddings,

        start=start_index

    ):

        output_path = os.path.join(

            identity_path,

            f"embedding_{index:03d}.npy"
        )

        np.save(
            output_path,
            embedding
        )

    identity_metadata = {

        "identity": identity,

        "source_unknown": entity_id,

        "face_embeddings_added":
            len(embeddings)
    }

    identity_metadata_path = os.path.join(

        identity_path,

        "metadata.json"
    )

    with open(
        identity_metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            identity_metadata,
            file,
            indent=4
        )

    # Save labeled identity into SQLite enrolled_identities
    best_face_file = metadata.get("best_face_image")
    face_img_bytes = None
    if best_face_file and os.path.exists(best_face_file):
        try:
            with open(best_face_file, "rb") as f:
                face_img_bytes = f.read()
        except Exception:
            pass

    try:
        from database.db import create_identity
        first_emb = embeddings[0] if len(embeddings) > 0 else None
        if first_emb is not None:
            db_id = create_identity(identity, "unspecified", first_emb, face_image=face_img_bytes)
            print(f"Stored {identity} in SQLite enrolled_identities (id={db_id}, with image BLOB).")
    except Exception as db_err:
        print(f"Warning: Could not save {identity} to SQLite: {db_err}")

    # For this prototype, remove the
    # unknown folder after successful labeling.
    shutil.rmtree(
        unknown_path
    )

    print()

    print(
        f"{entity_id} successfully "
        f"labeled as {identity}."
    )