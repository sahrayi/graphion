"""
Hamming distance relation metric.
"""

from __future__ import annotations

from .base_numeric_relation_builder import (
    BaseNumericRelationBuilder,
)


class HammingDistance(
    BaseNumericRelationBuilder,
):
    """
    Hamming distance metric.

    Measures the number of positions where two vectors
    differ.

    Designed for:

    - binary vectors
    - categorical encoded features
    - discrete feature representations

    Raw score range

        0 <= score <= dimensions
    """

    @property
    def name(
        self,
    ) -> str:
        return "hamming"

    def score(
        self,
        source: list[float],
        target: list[float],
    ) -> float:
        """
        Compute Hamming distance.

        Formula:

            count(x_i != y_i)

        Returns the number of dimensions where
        the two vectors differ.
        """

        self._validate_shapes(
            source,
            target,
        )

        return float(
            sum(
                x != y
                for x, y in zip(
                    source,
                    target,
                )
            )
        )

    def affinity(
        self,
        raw_score: float,
    ) -> float:
        """
        Convert Hamming distance into graph affinity.

        Uses normalized distance:

            affinity = 1 / (1 + distance)

        Returns
        -------

        float

            0 < affinity <= 1
        """

        return 1.0 / (
            1.0 + raw_score
        )