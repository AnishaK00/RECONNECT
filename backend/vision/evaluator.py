# vision/evaluator.py

from collections import defaultdict
from config import (
    MIN_GOOD_OBSERVATIONS,
    MIN_IDENTITY_VOTES,
    MIN_AVERAGE_SIMILARITY
)


class FaceEvidence:

    def __init__(self, track_id):

        self.track_id = track_id

        self.observations = []

        self.last_bbox = None


    def add_observation(self, observation):

        self.observations.append(observation)

        self.last_bbox = observation["bbox"]


    def evaluate(self):

        known_observations = [
            obs
            for obs in self.observations
            if obs["status"] == "known"
        ]

        # Not enough evidence yet
        if len(known_observations) < MIN_GOOD_OBSERVATIONS:

            return {
                "ready": False,
                "track_id": self.track_id
            }

        identity_scores = defaultdict(list)

        for obs in known_observations:

            identity = obs["identity"]

            similarity = obs["similarity"]

            identity_scores[identity].append(
                similarity
            )

        best_identity = None
        best_scores = []

        for identity, scores in identity_scores.items():

            if len(scores) > len(best_scores):

                best_identity = identity

                best_scores = scores

        vote_count = len(best_scores)

        average_similarity = (
            sum(best_scores)
            / len(best_scores)
        )

        if (
            vote_count >= MIN_IDENTITY_VOTES
            and
            average_similarity >= MIN_AVERAGE_SIMILARITY
        ):

            return {
                "ready": True,
                "track_id": self.track_id,
                "identity": best_identity,
                "status": "known",
                "supporting_observations": vote_count,
                "average_similarity": average_similarity
            }

        return {
            "ready": False,
            "track_id": self.track_id
        }