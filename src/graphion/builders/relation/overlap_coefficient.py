"""
Overlap coefficient relation metric.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

from .base_relation_builder import (
    BaseRelationBuilder,
)


class OverlapCoefficient(
    BaseRelationBuilder,
):
    """
    Overlap coefficient metric.

    Also known as Szymkiewicz-Simpson coefficient.

    Designed for set-based features such as:

    - keywords
    - entities
    - tags
    - categories

    Formula:

        similarity =
            |A ∩ B|
            ----------
            min(|A|, |B|)

    Raw score range

        0 <= score <= 1
    """

    @property
    def name(
        self,
    ) -> str:
        return "overlap"

    def prepare_vector(
        self,
        vector: list[Any],
    ) -> set[Hashable]:
        """
        Convert feature collection into a set.

        Duplicate values are removed because
        overlap coefficient operates on sets.
        """

        try:
            return set(vector)

        except TypeError as exc:
            raise TypeError(
                "Overlap features must contain "
                "only hashable values."
            ) from exc

    def score(
        self,
        source: set[Hashable],
        target: set[Hashable],
    ) -> float:
        """
        Compute overlap coefficient.

        Formula:

            |A ∩ B|
            ----------
            min(|A|, |B|)
        """

        denominator = min(
            len(source),
            len(target),
        )

        # Empty sets have undefined similarity.
        # Treat them as no relation.
        if denominator == 0:
            return 0.0

        intersection = source & target

        return (
            len(intersection)
            /
            denominator
        )

    def affinity(
        self,
        raw_score: float,
    ) -> float:
        """
        Convert overlap coefficient into graph affinity.

        Overlap coefficient is already normalized.

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