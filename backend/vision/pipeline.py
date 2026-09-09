import os
import json
import time

import cv2

from config import (
    CAMERA_INDEX,
    CAPTURE_SECONDS,
    TARGET_FPS,
    OUTPUT_FILE
)

from vision.face_engine import (
    FaceEngine
)

from vision.matcher import (
    FaceMatcher
)

from vision.unknown_manager import (
    UnknownManager
)

from vision.aggregator import (
    ObservationAggregator
)

from vision.position import (
    PositionEstimator
)


class WhoIsThisPipeline:

    def __init__(self, patient_id=None):

        self.face_engine = FaceEngine()

        self.matcher = FaceMatcher(patient_id=patient_id)

        self.unknown_manager = (
            UnknownManager()
        )

        self.aggregator = (
            ObservationAggregator()
        )

        self.position_estimator = (
            PositionEstimator()
        )

    # ==================================================
    # CROP FACE
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
            int(x1)
        )

        y1 = max(
            0,
            int(y1)
        )

        x2 = min(
            width,
            int(x2)
        )

        y2 = min(
            height,
            int(y2)
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
    # RUN
    # ==================================================

    def run(self):

        cap = cv2.VideoCapture(
            CAMERA_INDEX
        )

        if not cap.isOpened():

            raise RuntimeError(
                "Could not open camera."
            )

        frames = []
        start_time = time.time()
        frame_interval = 1 / TARGET_FPS
        last_capture_time = 0

        print(
            "Capturing faces..."
        )

        while (
            time.time() - start_time
            < CAPTURE_SECONDS
        ):

            success, frame = cap.read()

            current_time = time.time()

            if (
                success
                and current_time - last_capture_time
                >= frame_interval
            ):

                frames.append(
                    (current_time - start_time, frame)
                )

                last_capture_time = current_time

        cap.release()

        return self.process_frames(
            frames,
            capture_seconds=CAPTURE_SECONDS
        )

    def process_frames(
        self,
        frames,
        capture_seconds=CAPTURE_SECONDS
    ):

        observations = []

        frame_number = 0

        last_frame_width = None

        # ==================================================
        # PROCESS FRAMES
        # ==================================================

        for frame_item in frames:

            if isinstance(frame_item, tuple):

                timestamp = frame_item[0]
                frame = frame_item[-1]

            else:

                timestamp = time.time()
                frame = frame_item

            frame_number += 1

            last_frame_width = (
                frame.shape[1]
            )

            # ==================================================
            # DETECT ALL FACES
            # ==================================================

            faces = (
                self.face_engine
                .detect_faces(
                    frame
                )
            )

            # ==================================================
            # PROCESS EACH FACE
            # ==================================================

            for face in faces:

                bbox = face[
                    "bbox"
                ]

                detection_score = (
                    face[
                        "detection_score"
                    ]
                )

                sharpness = (
                    face[
                        "sharpness"
                    ]
                )

                quality = (
                    face[
                        "quality"
                    ]
                )

                quality_reasons = (
                    face[
                        "quality_reasons"
                    ]
                )

                recognition_allowed = (
                    face[
                        "recognition_allowed"
                    ]
                )

                embedding = (
                    face[
                        "embedding"
                    ]
                )

                # ==================================================
                # BASE OBSERVATION
                # ==================================================

                observation = {

                    "frame_number":
                        frame_number,

                    "bbox":
                        bbox,

                    "detection_score":
                        detection_score,

                    "sharpness":
                        sharpness,

                    "quality":
                        quality,

                    "quality_reasons":
                        quality_reasons,

                    "recognition_allowed":
                        recognition_allowed,

                    "timestamp":
                        timestamp
                }

                # ==================================================
                # POOR FACE
                # ==================================================
                #
                # IMPORTANT:
                #
                # The person WAS detected.
                #
                # We don't say "absent".
                #
                # We simply don't attempt recognition.
                # ==================================================

                if not recognition_allowed:

                    observation[
                        "status"
                    ] = (
                        "detected_low_quality"
                    )

                    observation[
                        "identity"
                    ] = None

                    observation[
                        "entity_id"
                    ] = None

                    observation[
                        "similarity"
                    ] = None

                    observations.append(
                        observation
                    )

                    print(
                        "Face detected "
                        f"but quality is poor "
                        f"(sharpness="
                        f"{sharpness:.1f}, "
                        f"reasons="
                        f"{quality_reasons})"
                    )

                    continue

                # ==================================================
                # GOOD FACE
                # ==================================================

                # At this point we have a usable
                # ArcFace embedding.

                match_result = (
                    self.matcher.match(
                        embedding
                    )
                )

                # ==================================================
                # KNOWN
                # ==================================================

                if match_result[
                    "matched"
                ]:

                    observation[
                        "entity_id"
                    ] = (
                        match_result[
                            "identity"
                        ]
                    )

                    observation[
                        "status"
                    ] = "known"

                    observation[
                        "identity"
                    ] = (
                        match_result[
                            "identity"
                        ]
                    )

                    observation[
                        "similarity"
                    ] = (
                        match_result[
                            "similarity"
                        ]
                    )

                # ==================================================
                # UNKNOWN
                # ==================================================

                else:

                    face_crop = (
                        self.crop_face(
                            frame,
                            bbox
                        )
                    )

                    unknown_result = (
                        self.unknown_manager
                        .add_observation(
                            embedding,
                            observation,
                            face_crop
                        )
                    )

                    observation[
                        "entity_id"
                    ] = (
                        unknown_result[
                            "entity_id"
                        ]
                    )

                    observation[
                        "status"
                    ] = (
                        unknown_result[
                            "status"
                        ]
                    )

                    observation[
                        "identity"
                    ] = None

                    observation[
                        "similarity"
                    ] = (
                        match_result.get(
                            "similarity"
                        )
                    )

                # ==================================================
                # STORE RECOGNIZED OBSERVATION
                # ==================================================

                observations.append(
                    observation
                )

        print(
            f"Captured "
            f"{len(observations)} "
            f"face observations."
        )

        # ==================================================
        # AGGREGATE RECOGNIZED OBSERVATIONS
        # ==================================================

        recognized_observations = [

            observation

            for observation in observations

            if observation.get(
                "entity_id"
            ) is not None
        ]

        people = (
            self.aggregator.aggregate(
                recognized_observations
            )
        )

        # ==================================================
        # POSITION
        # ==================================================

        if last_frame_width:

            people = (
                self.position_estimator
                .assign_positions(
                    people,
                    last_frame_width
                )
            )

        # ==================================================
        # SAVE UNKNOWN PEOPLE
        # ==================================================

        unknown_results = (
            self.unknown_manager
            .save_unknowns()
        )

        unknown_map = {

            item["entity_id"]: item

            for item in unknown_results
        }

        # ==================================================
        # ADD LABEL INFORMATION
        # ==================================================

        for person in people:

            entity_id = person[
                "entity_id"
            ]

            if entity_id in unknown_map:

                unknown_info = unknown_map[
                    entity_id
                ]

                person["status"] = (
                    unknown_info["status"]
                )

                person["label_request"] = (
                    unknown_info["label_request"]
                )

                person["best_face_image"] = (
                    unknown_info["best_face_image"]
                )

        # ==================================================
        # LOW-QUALITY DETECTIONS
        # ==================================================
        #
        # These are useful later for voice correlation.
        #
        # They are NOT treated as people/entities yet.
        # ==================================================

        low_quality_observations = [

            observation

            for observation in observations

            if observation.get(
                "status"
            )
            == "detected_low_quality"
        ]

        # ==================================================
        # FINAL RESULT
        # ==================================================

        return {

            "people":
                people,

            "capture_seconds":
                capture_seconds,

            "observation_count":
                len(observations),

            "low_quality_detection_count":
                len(
                    low_quality_observations
                ),

            "low_quality_observations":
                low_quality_observations,

            "timestamp":
                time.time()
        }


# ==================================================
# SAVE RESULTS
# ==================================================

def save_results(
    results
):

    output_directory = (
        os.path.dirname(
            OUTPUT_FILE
        )
    )

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )

    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                results
            )
            + "\n"
        )
