"""
Bray-Curtis distance relation metric.
"""

from __future__ import annotations

from .base_numeric_relation_builder import (
    BaseNumericRelationBuilder,
)


class BrayCurtisDistance(
    BaseNumericRelationBuilder,
):
    """
    Bray-Curtis distance metric.

    Measures dissimilarity between two numeric vectors.
    Commonly used for abundance, frequency, and weighted
    feature representations.

    Formula:

        distance =
            Σ |x_i - y_i|
            -------------
            Σ |x_i + y_i|

    Raw score range

        0 <= score <= 1
    """

    @property
    def name(
        self,
    ) -> str:
        return "bray_curtis"

    def score(
        self,
        source: list[float],
        target: list[float],
    ) -> float:
        """
        Compute Bray-Curtis distance.
        """

        self._validate_shapes(
            source,
            target,
        )

        numerator = sum(
            abs(x - y)
            for x, y in zip(
                source,
                target,
            )
        )

        denominator = sum(
            abs(x + y)
            for x, y in zip(
                source,
                target,
            )
        )

        # Both vectors are zero vectors.
        # Distance is undefined, treat as no relation.
        if denominator == 0:
            return 0.0

        return numerator / denominator

    def affinity(
        self,
        raw_score: float,
    ) -> float:
        """
        Convert Bray-Curtis distance into graph affinity.

        Uses:

            affinity = 1 / (1 + distance)

        Returns
        -------

        float

            0.5 <= affinity <= 1
        """

        return 1.0 / (
            1.0 + raw_score
        )