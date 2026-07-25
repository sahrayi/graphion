"""
Angular similarity relation metric.
"""

from __future__ import annotations

from math import acos, pi

from .base_numeric_relation_builder import (
    BaseNumericRelationBuilder,
)


class AngularSimilarity(
    BaseNumericRelationBuilder,
):
    """
    Angular similarity metric.

    Measures similarity based on the angle
    between two vectors.

    Raw score range

        0 <= score <= 1
    """

    @property
    def name(
        self,
    ) -> str:
        return "angular"

    def score(
        self,
        source: list[float],
        target: list[float],
    ) -> float:
        """
        Compute angular similarity.

        Formula:

            similarity = 1 - angle / pi
        """

        self._validate_shapes(
            source,
            target,
        )

        source_norm = self._norm(source)
        target_norm = self._norm(target)

        if (
            source_norm == 0
            or target_norm == 0
        ):
            return 0.0

        cosine = (
            sum(
                x * y
                for x, y in zip(
                    source,
                    target,
                )
            )
            /
            (
                source_norm
                *
                target_norm
            )
        )

        # Floating point errors can slightly
        # exceed the valid acos range.
        cosine = max(
            -1.0,
            min(
                1.0,
                cosine,
            ),
        )

        angle = acos(cosine)

        return 1.0 - (
            angle / pi
        )

    def affinity(
        self,
        raw_score: float,
    ) -> float:
        """
        Angular similarity is already normalized.

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