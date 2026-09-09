from collections import defaultdict, Counter

from config import (
    MIN_UNKNOWN_OBSERVATIONS
)


class ObservationAggregator:

    def aggregate(
        self,
        observations
    ):

        groups = defaultdict(list)

        for observation in observations:

            entity_id = observation[
                "entity_id"
            ]

            groups[
                entity_id
            ].append(
                observation
            )

        results = []

        for entity_id, group in (
            groups.items()
        ):

            statuses = Counter(
                item["status"]
                for item in group
            )

            final_status = (
                statuses.most_common(1)[0][0]
            )

            observation_count = len(group)

            if final_status != "known":

                if observation_count < MIN_UNKNOWN_OBSERVATIONS:

                    continue

            identities = [

                item.get("identity")

                for item in group

                if item.get("identity")
            ]

            identity = None

            if identities:

                identity = Counter(
                    identities
                ).most_common(1)[0][0]

            latest_observation = max(
                group,
                key=lambda item:
                    item["timestamp"]
            )

            results.append({

                "entity_id":
                    entity_id,

                "status":
                    final_status,

                "identity":
                    identity,

                "observation_count":
                    len(group),

                "bbox":
                    latest_observation[
                        "bbox"
                    ],

                "timestamp":
                    latest_observation[
                        "timestamp"
                    ]
            })

        return results