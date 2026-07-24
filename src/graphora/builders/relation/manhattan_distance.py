"""
Manhattan distance relation metric.
"""

from __future__ import annotations

from .base_numeric_relation_builder import (
    BaseNumericRelationBuilder,
)


class ManhattanDistance(
    BaseNumericRelationBuilder,
):
    """
    Manhattan distance metric.

    Raw score range

        0 <= score < +inf
    """

    @property
    def name(
        self,
    ) -> str:
        return "manhattan"

    def score(
        self,
        source: list[float],
        target: list[float],
    ) -> float:
        """
        Compute Manhattan distance.

        Manhattan distance is the sum of absolute
        differences between vector dimensions.
        """

        self._validate_shapes(
            source,
            target,
        )

        return sum(
            abs(x - y)
            for x, y in zip(
                source,
                target,
            )
        )

    def affinity(
        self,
        raw_score: float,
    ) -> float:
        """
        Convert Manhattan distance into graph affinity.

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