"""
Canberra distance relation metric.
"""

from __future__ import annotations

from .base_numeric_relation_builder import (
    BaseNumericRelationBuilder,
)


class CanberraDistance(
    BaseNumericRelationBuilder,
):
    """
    Canberra distance metric.

    Canberra distance is a weighted version of Manhattan
    distance that normalizes each dimension by the sum
    of absolute values.

    Formula:

        distance =
            Σ |x_i - y_i| / (|x_i| + |y_i|)

    Raw score range

        0 <= score <= dimensions
    """

    @property
    def name(
        self,
    ) -> str:
        return "canberra"

    def score(
        self,
        source: list[float],
        target: list[float],
    ) -> float:
        """
        Compute Canberra distance.
        """

        self._validate_shapes(
            source,
            target,
        )

        distance = 0.0

        for x, y in zip(
            source,
            target,
        ):
            denominator = (
                abs(x)
                +
                abs(y)
            )

            # If both values are zero, this dimension
            # contributes nothing to the distance.
            if denominator == 0:
                continue

            distance += (
                abs(x - y)
                /
                denominator
            )

        return distance

    def affinity(
        self,
        raw_score: float,
    ) -> float:
        """
        Convert Canberra distance into graph affinity.

        Uses:

            affinity = 1 / (1 + distance)

        Returns
        -------

        float

            0 < affinity <= 1
        """

        return 1.0 / (
            1.0 + raw_score
        )