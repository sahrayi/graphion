"""
Dot product relation metric.
"""

from __future__ import annotations

from math import exp

from .base_numeric_relation_builder import (
    BaseNumericRelationBuilder,
)


class DotProduct(
    BaseNumericRelationBuilder,
):
    """
    Dot product metric.

    Raw score range

        -inf < score < +inf
    """

    @property
    def name(
        self,
    ) -> str:
        return "dot_product"

    def score(
        self,
        source: list[float],
        target: list[float],
    ) -> float:
        """
        Compute dot product.
        """

        self._validate_shapes(
            source,
            target,
        )

        return sum(
            x * y
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
        Convert dot product into graph affinity.

        Uses sigmoid transformation:

            affinity = 1 / (1 + exp(-score))

        Returns
        -------
        float

            0 < affinity < 1
        """

        return 1.0 / (
            1.0 + exp(-raw_score)
        )