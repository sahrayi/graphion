"""
Chebyshev distance relation metric.
"""

from __future__ import annotations

from .base_numeric_relation_builder import (
    BaseNumericRelationBuilder,
)


class ChebyshevDistance(
    BaseNumericRelationBuilder,
):
    """
    Chebyshev distance metric.

    Also known as L-infinity distance.

    Raw score range

        0 <= score < +inf
    """

    @property
    def name(
        self,
    ) -> str:
        return "chebyshev"

    def score(
        self,
        source: list[float],
        target: list[float],
    ) -> float:
        """
        Compute Chebyshev distance.

        Formula:

            max(|x1-y1|, |x2-y2|, ..., |xn-yn|)
        """

        self._validate_shapes(
            source,
            target,
        )

        if not source:
            return 0.0

        return max(
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
        Convert Chebyshev distance into graph affinity.

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