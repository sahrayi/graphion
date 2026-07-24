"""
Jaccard similarity relation metric.
"""

from __future__ import annotations

from collections.abc import (
    Hashable,
    Iterable,
)

from graphora.core.models import Relation

from .base_relation_builder import (
    BaseRelationBuilder,
)


class JaccardSimilarity(
    BaseRelationBuilder,
):
    """
    Jaccard similarity metric.

    Designed for set-based features such as:

    - keywords
    - entities
    - tags
    - categories

    Raw score range

        0 <= score <= 1
    """

    @property
    def name(
        self,
    ) -> str:
        return "jaccard"

    def prepare_vector(
        self,
        vector: Iterable[Hashable],
    ) -> set[Hashable]:
        """
        Convert feature collection into a set.

        Duplicate values are removed because
        Jaccard similarity operates on sets.
        """

        try:
            return set(vector)

        except TypeError as exc:
            raise TypeError(
                "Jaccard features must contain "
                "only hashable values."
            ) from exc

    def score(
        self,
        source: set[Hashable],
        target: set[Hashable],
    ) -> float:
        """
        Compute Jaccard similarity.

        Formula:

            |A ∩ B|
            -------
            |A ∪ B|
        """

        union = source | target

        if not union:
            # Empty sets have undefined Jaccard similarity.
            # Treat them as no relation.
            return 0.0

        intersection = source & target

        return (
            len(intersection)
            /
            len(union)
        )

    def affinity(
        self,
        raw_score: float,
    ) -> float:
        """
        Convert Jaccard similarity into graph affinity.

        Jaccard is already normalized.

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