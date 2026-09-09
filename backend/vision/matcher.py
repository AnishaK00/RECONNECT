import os
import numpy as np

from database.db import blob_to_embedding, get_all_identities

from config import (
    GALLERY_PATH,
    KNOWN_MATCH_THRESHOLD
)


class FaceMatcher:

    def __init__(self, patient_id=None):

        self.gallery = {}
        self.patient_id = patient_id

        self.load_gallery()

    def load_gallery(self):

        database_rows = get_all_identities(patient_id=self.patient_id)

        for row in database_rows:
            self.gallery[row["name"]] = [
                blob_to_embedding(row["face_embedding"])
            ]

        if database_rows:
            return

        if not os.path.exists(
            GALLERY_PATH
        ):
            return

        for identity in os.listdir(
            GALLERY_PATH
        ):

            identity_path = os.path.join(
                GALLERY_PATH,
                identity
            )

            if not os.path.isdir(
                identity_path
            ):
                continue

            embeddings = []

            for filename in os.listdir(
                identity_path
            ):

                if filename.endswith(".npy"):

                    embedding_path = os.path.join(
                        identity_path,
                        filename
                    )

                    try:

                        embedding = np.load(
                            embedding_path,
                            allow_pickle=False
                        )

                    except (
                        OSError,
                        ValueError
                    ):

                        continue

                    if (
                        embedding.ndim != 1
                        or not np.issubdtype(
                            embedding.dtype,
                            np.number
                        )
                    ):

                        continue

                    embeddings.append(
                        embedding
                    )

            if embeddings:

                self.gallery[identity] = (
                    np.array(
                        embeddings,
                        dtype=np.float32
                    )
                )

    def match(self, embedding):

        best_identity = None

        best_similarity = -1.0

        for identity, embeddings in (
            self.gallery.items()
        ):

            for stored_embedding in embeddings:

                similarity = float(
                    np.dot(
                        embedding,
                        stored_embedding
                    )
                )

                if similarity > best_similarity:

                    best_similarity = similarity

                    best_identity = identity

        if (
            best_identity is not None
            and best_similarity
            >= KNOWN_MATCH_THRESHOLD
        ):

            return {

                "matched": True,

                "identity": best_identity,

                "similarity": best_similarity

            }

        return {

            "matched": False,

            "identity": None,

            "similarity": best_similarity
        }
