"""
Cosine similarity relation builder.
"""

from __future__ import annotations

from .base_numeric_relation_builder import (
    BaseNumericRelationBuilder,
)


class CosineSimilarity(
    BaseNumericRelationBuilder,
):
    """
    Build relations using cosine similarity.

    Raw score range:

        -1 <= score <= 1
    """

    @property
    def name(
        self,
    ) -> str:
        return "cosine"

    def score(
        self,
        source: list[float],
        target: list[float],
    ) -> float:
        """
        Compute cosine similarity.
        """

        self._validate_shapes(
            source,
            target,
        )

        denominator = (
            self._norm(source)
            * self._norm(target)
        )

        if denominator == 0:
            return 0.0

        numerator = sum(
            x * y
            for x, y in zip(
                source,
                target,
            )
        )

        return numerator / denominator

    def affinity(
        self,
        raw_score: float,
    ) -> float:
        """
        Convert cosine similarity into graph affinity.

        Negative similarities are discarded.

        Returns
        -------
        float

            0 <= affinity <= 1
        """

        return max(
            0.0,
            raw_score,
        )