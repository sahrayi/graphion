"""
Jaccard similarity relation metric.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable

from .base_relation_builder import BaseRelationBuilder


class JaccardSimilarity(BaseRelationBuilder):
    """
    Jaccard similarity metric.
    """

    @property
    def name(self) -> str:
        return "jaccard"

    def prepare_vector(
        self,
        vector: Iterable[Hashable],
    ) -> set[Hashable]:
        """Convert feature collection into a set."""
        try:
            return set(vector)
        except TypeError as exc:
            raise TypeError(
                "Jaccard features must contain only hashable values."
            ) from exc

    def score(
        self,
        source: set[Hashable],
        target: set[Hashable],
    ) -> float:
        """Compute Jaccard similarity."""
        union = source | target
        if not union:
            return 0.0

        intersection = source & target
        return len(intersection) / len(union)

    def affinity(self, raw_score: float) -> float:
        """Convert Jaccard similarity into graph affinity."""
        return max(0.0, min(1.0, raw_score))