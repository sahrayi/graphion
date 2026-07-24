"""
Minkowski distance relation metric.
"""

from __future__ import annotations

from math import pow

from .base_numeric_relation_builder import (
    BaseNumericRelationBuilder,
)


class MinkowskiDistance(
    BaseNumericRelationBuilder,
):
    """
    Minkowski distance metric.

    Generalized distance metric that includes:

    - Manhattan distance (p=1)
    - Euclidean distance (p=2)
    - Chebyshev distance (p=inf)

    Formula:

        distance =
            ( Σ |x_i - y_i|^p )^(1/p)

    Raw score range

        0 <= score < +inf
    """

    def __init__(
        self,
        *,
        p: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if p <= 0:
            raise ValueError(
                "Minkowski parameter p must be greater than zero."
            )

        self.p = p

    @property
    def name(
        self,
    ) -> str:
        return "minkowski"

    def score(
        self,
        source: list[float],
        target: list[float],
    ) -> float:
        """
        Compute Minkowski distance.
        """

        self._validate_shapes(
            source,
            target,
        )

        if not source:
            return 0.0

        return pow(
            sum(
                pow(
                    abs(x - y),
                    self.p,
                )
                for x, y in zip(
                    source,
                    target,
                )
            ),
            1.0 / self.p,
        )

    def affinity(
        self,
        raw_score: float,
    ) -> float:
        """
        Convert Minkowski distance into graph affinity.

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