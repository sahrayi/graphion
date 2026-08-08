"""
Cosine similarity relation builder.
"""

from __future__ import annotations

from collections.abc import Sequence

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

    Graph affinity:

        0 <= affinity <= 1

    Negative cosine similarities are mapped to
    zero affinity.
    """

    EPSILON = 1e-300

    # ==================================================
    # Metadata
    # ==================================================

    @property
    def name(
        self,
    ) -> str:
        """
        Relation metric name.
        """

        return "cosine"

    # ==================================================
    # Raw metric
    # ==================================================

    def score(
        self,
        source: Sequence[float],
        target: Sequence[float],
    ) -> float:
        """
        Compute cosine similarity.

        Returns
        -------
        float
            Raw cosine similarity in the range:

                -1 <= score <= 1
        """

        self._validate_shapes(
            source,
            target,
        )

        denominator = (
            self._norm(source)
            * self._norm(target)
        )

        if denominator < self.EPSILON:

            return 0.0

        numerator = sum(
            x * y
            for x, y in zip(
                source,
                target,
            )
        )

        return numerator / denominator

    # ==================================================
    # Affinity conversion
    # ==================================================

    def affinity(
        self,
        raw_score: float,
    ) -> float:
        """
        Convert cosine similarity into graph affinity.

        Negative similarities are mapped to zero.

        Examples
        --------
        score =  1.0 -> affinity = 1.0
        score =  0.5 -> affinity = 0.5
        score =  0.0 -> affinity = 0.0
        score = -0.5 -> affinity = 0.0
        """

        return max(
            0.0,
            raw_score,
        )