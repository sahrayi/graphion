"""
Tanimoto similarity relation metric.
"""

from __future__ import annotations

from .base_numeric_relation_builder import (
    BaseNumericRelationBuilder,
)


class TanimotoSimilarity(
    BaseNumericRelationBuilder,
):
    """
    Tanimoto similarity metric.

    Also known as extended Jaccard similarity.

    Designed for numeric vectors and sparse feature
    representations.

    Formula:

        similarity =
            x · y
            -------------
            ||x||² + ||y||² - x · y

    Raw score range

        0 <= score <= 1
    """

    @property
    def name(
        self,
    ) -> str:
        return "tanimoto"

    def score(
        self,
        source: list[float],
        target: list[float],
    ) -> float:
        """
        Compute Tanimoto similarity.
        """

        self._validate_shapes(
            source,
            target,
        )

        dot_product = sum(
            x * y
            for x, y in zip(
                source,
                target,
            )
        )

        source_norm = sum(
            x * x
            for x in source
        )

        target_norm = sum(
            y * y
            for y in target
        )

        denominator = (
            source_norm
            +
            target_norm
            -
            dot_product
        )

        # Both vectors are zero vectors.
        # Similarity is undefined, treat as no relation.
        if denominator == 0:
            return 0.0

        return max(
            0.0,
            dot_product / denominator,
        )

    def affinity(
        self,
        raw_score: float,
    ) -> float:
        """
        Convert Tanimoto similarity into graph affinity.

        Tanimoto is already normalized.

        Returns
        -------

        float

            0 <= affinity <= 1
        """

        return max(
            0.0,
            min(
                1.0,
                raw_score,
            ),
        )