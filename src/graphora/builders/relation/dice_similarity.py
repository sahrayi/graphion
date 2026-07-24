"""
Dice similarity relation metric.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any

from .base_relation_builder import (
    BaseRelationBuilder,
)


class DiceSimilarity(
    BaseRelationBuilder,
):
    """
    Dice similarity metric.

    Designed for set-based feature representations:

    - keywords
    - entities
    - tags
    - categories
    - categorical attributes

    Formula:

        similarity =
            2 * |A ∩ B|
            -------------
              |A| + |B|

    Raw score range:

        0 <= score <= 1
    """

    @property
    def name(
        self,
    ) -> str:
        return "dice"

    # --------------------------------------------------
    # Vector preparation
    # --------------------------------------------------

    def prepare_vector(
        self,
        vector: list[Any],
    ) -> set[Hashable]:
        """
        Convert feature representation into a set.

        Duplicate values are removed because Dice
        similarity operates on sets.

        Examples
        --------

        Input:

            [
                "ai",
                "news",
                "iran",
            ]

        Output:

            {
                "ai",
                "news",
                "iran",
            }
        """

        try:
            return set(vector)

        except TypeError as exc:
            raise TypeError(
                "Dice features must contain only hashable values."
            ) from exc

    # --------------------------------------------------
    # Metric
    # --------------------------------------------------

    def score(
        self,
        source: set[Hashable],
        target: set[Hashable],
    ) -> float:
        """
        Compute Dice similarity.

        Formula:

            2 * |A ∩ B|
            -------------
              |A| + |B|
        """

        denominator = (
            len(source)
            +
            len(target)
        )

        # Both sets are empty.
        # Similarity is undefined.
        #
        # We treat it as no relation.
        if denominator == 0:
            return 0.0

        return (
            2.0
            *
            len(source & target)
            /
            denominator
        )

    # --------------------------------------------------
    # Graph affinity
    # --------------------------------------------------

    def affinity(
        self,
        raw_score: float,
    ) -> float:
        """
        Convert raw Dice similarity into graph affinity.

        Dice similarity is already normalized.

        Returns:

            0 <= affinity <= 1
        """

        return max(
            0.0,
            min(
                1.0,
                raw_score,
            ),
        )