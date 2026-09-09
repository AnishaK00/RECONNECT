import cv2
import numpy as np

from insightface.app import FaceAnalysis

from config import (
    DETECTION_THRESHOLD,
    MIN_FACE_SIZE,
    BLUR_THRESHOLD
)


class FaceEngine:

    def __init__(self):

        self.app = FaceAnalysis(
            name="buffalo_l"
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

    # ==================================================
    # CALCULATE FACE SHARPNESS
    # ==================================================

    def calculate_sharpness(
        self,
        face_crop
    ):

        if (
            face_crop is None
            or face_crop.size == 0
        ):

            return 0.0

        gray = cv2.cvtColor(
            face_crop,
            cv2.COLOR_BGR2GRAY
        )

        # Variance of Laplacian.
        #
        # Higher value:
        #     sharper image
        #
        # Lower value:
        #     blurrier image

        sharpness = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        return float(
            sharpness
        )

    # ==================================================
    # CROP FACE SAFELY
    # ==================================================

    def crop_face(
        self,
        frame,
        bbox
    ):

        x1, y1, x2, y2 = bbox

        height, width = frame.shape[:2]

        x1 = max(
            0,
            x1
        )

        y1 = max(
            0,
            y1
        )

        x2 = min(
            width,
            x2
        )

        y2 = min(
            height,
            y2
        )

        if (
            x2 <= x1
            or y2 <= y1
        ):

            return None

        return frame[
            y1:y2,
            x1:x2
        ].copy()

    # ==================================================
    # DETECT FACES
    # ==================================================

    def detect_faces(
        self,
        frame
    ):

        faces = self.app.get(
            frame
        )

        results = []

        # ==================================================
        # PROCESS EVERY DETECTED FACE
        # ==================================================

        for face in faces:

            # ------------------------------------------
            # Detection confidence
            # ------------------------------------------

            detection_score = float(
                face.det_score
            )

            if (
                detection_score
                < DETECTION_THRESHOLD
            ):

                continue

            # ------------------------------------------
            # Bounding box
            # ------------------------------------------

            bbox = face.bbox.astype(
                int
            ).tolist()

            x1, y1, x2, y2 = bbox

            face_width = (
                x2 - x1
            )

            face_height = (
                y2 - y1
            )

            # ------------------------------------------
            # Crop face
            # ------------------------------------------

            face_crop = (
                self.crop_face(
                    frame,
                    bbox
                )
            )

            if (
                face_crop is None
                or face_crop.size == 0
            ):

                continue

            # ------------------------------------------
            # Calculate sharpness
            # ------------------------------------------

            sharpness = (
                self.calculate_sharpness(
                    face_crop
                )
            )

            # ==================================================
            # QUALITY DECISION
            # ==================================================

            quality_reasons = []

            # Too small
            if (
                face_width
                < MIN_FACE_SIZE
                or
                face_height
                < MIN_FACE_SIZE
            ):

                quality_reasons.append(
                    "face_too_small"
                )

            # Too blurry
            if (
                sharpness
                < BLUR_THRESHOLD
            ):

                quality_reasons.append(
                    "blur"
                )

            # ==================================================
            # GOOD / POOR
            # ==================================================

            if quality_reasons:

                quality = "poor"

                recognition_allowed = False

            else:

                quality = "good"

                recognition_allowed = True

            # ==================================================
            # GOOD FACE
            # ==================================================

            if recognition_allowed:

                embedding = (
                    face.normed_embedding
                    .astype(
                        np.float32
                    )
                )

            # ==================================================
            # POOR FACE
            #
            # IMPORTANT:
            #
            # We DO NOT throw away the detection.
            #
            # We keep the bbox and quality information.
            #
            # But we don't trust the embedding for identity.
            # ==================================================

            else:

                embedding = None

            # ==================================================
            # RESULT
            # ==================================================

            results.append({

                "bbox":
                    bbox,

                "embedding":
                    embedding,

                "detection_score":
                    detection_score,

                "sharpness":
                    sharpness,

                "face_width":
                    face_width,

                "face_height":
                    face_height,

                "quality":
                    quality,

                "quality_reasons":
                    quality_reasons,

                "recognition_allowed":
                    recognition_allowed
            })

        return results