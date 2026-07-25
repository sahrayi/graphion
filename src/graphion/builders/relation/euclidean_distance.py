"""
Euclidean distance relation metric.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

from .base_numeric_relation_builder import BaseNumericRelationBuilder


class EuclideanDistance(BaseNumericRelationBuilder):
    """
    Euclidean distance metric.

    Raw score range:
        0 <= score < +inf
    """

    @property
    def name(self) -> str:
        return "euclidean"

    def score(
        self,
        source: Sequence[float],
        target: Sequence[float],
    ) -> float:
        """Compute Euclidean distance."""
        self._validate_shapes(source, target)
        return sqrt(sum((x - y) ** 2 for x, y in zip(source, target)))

    def affinity(self, raw_score: float) -> float:
        """Convert Euclidean distance into graph affinity."""
        return 1.0 / (1.0 + raw_score)