"""
RBF (Gaussian) similarity relation metric.
"""

from __future__ import annotations

from math import exp

from .base_numeric_relation_builder import (
    BaseNumericRelationBuilder,
)


class RBFSimilarity(
    BaseNumericRelationBuilder,
):
    """
    Radial Basis Function (Gaussian) similarity metric.

    Converts Euclidean distance into a similarity score.

    Formula:

        similarity =
            exp(-gamma * ||x - y||²)

    Parameters
    ----------

    gamma:
        Controls the decay rate.

        Higher gamma:
            - similarity decreases faster
            - graph becomes sparser

        Lower gamma:
            - similarity decreases slower
            - more edges survive


    Raw score range

        0 < score <= 1
    """

    def __init__(
        self,
        *,
        gamma: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if gamma <= 0:
            raise ValueError(
                "RBF gamma must be greater than zero."
            )

        self.gamma = gamma

    @property
    def name(
        self,
    ) -> str:
        return "rbf"

    def score(
        self,
        source: list[float],
        target: list[float],
    ) -> float:
        """
        Compute RBF similarity.

        Formula:

            exp(-gamma * squared_distance)
        """

        self._validate_shapes(
            source,
            target,
        )

        squared_distance = sum(
            (x - y) ** 2
            for x, y in zip(
                source,
                target,
            )
        )

        return exp(
            -self.gamma
            *
            squared_distance
        )

    def affinity(
        self,
        raw_score: float,
    ) -> float:
        """
        Convert RBF similarity into graph affinity.

        RBF output is already normalized.

        Returns
        -------

        float

            0 < affinity <= 1
        """

        return max(
            0.0,
            min(
                1.0,
                raw_score,
            ),
        )