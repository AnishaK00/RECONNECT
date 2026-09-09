import os
import json
import uuid

import cv2
import numpy as np

from config import (
    UNKNOWN_PATH,
    UNKNOWN_CLUSTER_THRESHOLD
)


class UnknownManager:

    def __init__(self):

        self.clusters = {}

        self.persistent_unknowns = {}

        self.load_persistent_unknowns()

    def load_persistent_unknowns(self):

        if not os.path.exists(
            UNKNOWN_PATH
        ):
            return

        for entity_id in os.listdir(
            UNKNOWN_PATH
        ):

            entity_path = os.path.join(
                UNKNOWN_PATH,
                entity_id
            )

            if not os.path.isdir(
                entity_path
            ):
                continue

            embedding_path = os.path.join(
                entity_path,
                "face_embeddings.npy"
            )

            metadata_path = os.path.join(
                entity_path,
                "metadata.json"
            )

            if not os.path.exists(
                embedding_path
            ):
                continue

            embeddings = np.load(
                embedding_path
            )

            metadata = {}

            if os.path.exists(
                metadata_path
            ):

                with open(
                    metadata_path,
                    "r",
                    encoding="utf-8"
                ) as file:

                    metadata = json.load(
                        file
                    )

            self.persistent_unknowns[
                entity_id
            ] = {

                "entity_id": entity_id,

                "embeddings": embeddings,

                "metadata": metadata
            }

    def cosine_similarity(
        self,
        embedding_a,
        embedding_b
    ):

        return float(
            np.dot(
                embedding_a,
                embedding_b
            )
        )

    def find_best_match(
        self,
        embedding,
        collection
    ):

        best_entity_id = None

        best_similarity = -1.0

        for entity_id, entity in (
            collection.items()
        ):

            embeddings = entity[
                "embeddings"
            ]

            for stored_embedding in embeddings:

                similarity = (
                    self.cosine_similarity(
                        embedding,
                        stored_embedding
                    )
                )

                if similarity > best_similarity:

                    best_similarity = similarity

                    best_entity_id = entity_id

        return (
            best_entity_id,
            best_similarity
        )

    def create_cluster(
        self,
        entity_id,
        is_new
    ):

        self.clusters[entity_id] = {

            "entity_id": entity_id,

            "embeddings": [],

            "observations": [],

            "is_new": is_new
        }

    def add_observation(
        self,
        embedding,
        observation,
        face_crop
    ):

        # -----------------------------------
        # STEP 1
        # Match with unknowns in this session
        # -----------------------------------

        entity_id, similarity = (
            self.find_best_match(
                embedding,
                self.clusters
            )
        )

        if (
            entity_id is not None
            and similarity
            >= UNKNOWN_CLUSTER_THRESHOLD
        ):

            self.clusters[
                entity_id
            ]["embeddings"].append(
                embedding
            )

            self.clusters[
                entity_id
            ]["observations"].append({

                "observation": observation,

                "face_crop": face_crop
            })

            return {

                "entity_id": entity_id,

                "status": (
                    "new_unknown"
                    if self.clusters[
                        entity_id
                    ]["is_new"]
                    else "existing_unknown"
                ),

                "is_new": self.clusters[
                    entity_id
                ]["is_new"]
            }

        # -----------------------------------
        # STEP 2
        # Match persistent unknowns
        # -----------------------------------

        entity_id, similarity = (
            self.find_best_match(
                embedding,
                self.persistent_unknowns
            )
        )

        if (
            entity_id is not None
            and similarity
            >= UNKNOWN_CLUSTER_THRESHOLD
        ):

            self.create_cluster(
                entity_id,
                is_new=False
            )

            self.clusters[
                entity_id
            ]["embeddings"].append(
                embedding
            )

            self.clusters[
                entity_id
            ]["observations"].append({

                "observation": observation,

                "face_crop": face_crop
            })

            return {

                "entity_id": entity_id,

                "status": "existing_unknown",

                "is_new": False
            }

        # -----------------------------------
        # STEP 3
        # Completely new unknown
        # -----------------------------------

        entity_id = (
            "unknown_"
            + uuid.uuid4().hex[:8]
        )

        self.create_cluster(
            entity_id,
            is_new=True
        )

        self.clusters[
            entity_id
        ]["embeddings"].append(
            embedding
        )

        self.clusters[
            entity_id
        ]["observations"].append({

            "observation": observation,

            "face_crop": face_crop
        })

        return {

            "entity_id": entity_id,

            "status": "new_unknown",

            "is_new": True
        }

    def get_face_quality(
        self,
        stored_observation
    ):

        observation = stored_observation[
            "observation"
        ]

        detection_score = observation[
            "detection_score"
        ]

        x1, y1, x2, y2 = observation[
            "bbox"
        ]

        width = x2 - x1
        height = y2 - y1

        face_area = width * height

        return (
            detection_score
            * face_area
        )

    def save_unknowns(self):

        os.makedirs(
            UNKNOWN_PATH,
            exist_ok=True
        )

        results = []

        for entity_id, cluster in (
            self.clusters.items()
        ):

            entity_path = os.path.join(
                UNKNOWN_PATH,
                entity_id
            )

            os.makedirs(
                entity_path,
                exist_ok=True
            )

            embeddings = np.array(
                cluster["embeddings"],
                dtype=np.float32
            )

            np.save(
                os.path.join(
                    entity_path,
                    "face_embeddings.npy"
                ),
                embeddings
            )

            best_observation = max(
                cluster["observations"],
                key=self.get_face_quality
            )

            best_face = best_observation[
                "face_crop"
            ]

            best_face_path = os.path.join(
                entity_path,
                "best_face.jpg"
            )

            if (
                best_face is not None
                and best_face.size > 0
            ):

                cv2.imwrite(
                    best_face_path,
                    best_face
                )

            status = (
                "new_unknown"
                if cluster["is_new"]
                else "existing_unknown"
            )

            unenrolled_db_id = None
            if cluster["is_new"]:
                try:
                    from database.db import create_unenrolled_identity
                    face_emb = embeddings[0] if len(embeddings) > 0 else None
                    unenrolled_db_id = create_unenrolled_identity(face_embedding=face_emb, face_image=best_face)
                    print(f"[{entity_id}] Saved new unknown face to unenrolled_identities (id={unenrolled_db_id}) in SQLite.")
                except Exception as db_err:
                    print(f"[{entity_id}] Warning: could not store unknown face in SQLite: {db_err}")

            metadata = {

                "entity_id": entity_id,

                "status": status,

                "unenrolled_db_id": unenrolled_db_id,

                "observation_count": len(
                    cluster["observations"]
                ),

                "best_face_image": best_face_path,

                "label_request": (
                    cluster["is_new"]
                ),

                "modality": {

                    "face": True,

                    "voice": False
                }
            }

            with open(
                os.path.join(
                    entity_path,
                    "metadata.json"
                ),
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    metadata,
                    file,
                    indent=4
                )

            results.append(
                metadata
            )

        return results