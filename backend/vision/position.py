class PositionEstimator:

    def assign_positions(
        self,
        people,
        frame_width
    ):

        # ----------------------------------------------
        # Safety check
        # ----------------------------------------------

        if not frame_width:

            return people

        for person in people:

            bbox = person.get(
                "bbox"
            )

            if not bbox:

                person[
                    "position"
                ] = "unknown"

                continue

            x1, y1, x2, y2 = bbox

            # ------------------------------------------
            # Center of person's face
            # ------------------------------------------

            center_x = (
                x1 + x2
            ) / 2.0

            # ------------------------------------------
            # Convert to 0.0 - 1.0
            # ------------------------------------------

            relative_position = (
                center_x
                / float(frame_width)
            )

            # ------------------------------------------
            # LEFT
            # ------------------------------------------

            if relative_position < 0.33:

                position = "left"

            # ------------------------------------------
            # RIGHT
            # ------------------------------------------

            elif relative_position > 0.66:

                position = "right"

            # ------------------------------------------
            # CENTER
            # ------------------------------------------

            else:

                position = "center"

            person[
                "position"
            ] = position

            # Useful for debugging
            person[
                "relative_position"
            ] = relative_position

        return people