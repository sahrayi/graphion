"""
RBF (Gaussian) similarity relation metric.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import exp, isfinite

from .base_numeric_relation_builder import BaseNumericRelationBuilder


class RBFSimilarity(BaseNumericRelationBuilder):
    """
    Radial Basis Function (Gaussian) similarity metric.
    """

    def __init__(
        self,
        *,
        gamma: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not isfinite(gamma) or gamma <= 0:
            raise ValueError("RBF gamma must be a finite positive value.")
        self.gamma = gamma

    @property
    def name(self) -> str:
        return "rbf"

    def score(
        self,
        source: Sequence[float],
        target: Sequence[float],
    ) -> float:
        """Compute RBF similarity."""
        self._validate_shapes(source, target)
        squared_distance = sum((x - y) ** 2 for x, y in zip(source, target))
        return exp(-self.gamma * squared_distance)

    def affinity(self, raw_score: float) -> float:
        """Convert RBF score into graph affinity."""
        return max(0.0, min(1.0, raw_score))