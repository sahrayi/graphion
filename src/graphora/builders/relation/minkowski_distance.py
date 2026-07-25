"""
Minkowski distance relation metric.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import inf, isfinite, pow

from .base_numeric_relation_builder import BaseNumericRelationBuilder


class MinkowskiDistance(BaseNumericRelationBuilder):
    """
    Minkowski distance metric.
    """

    def __init__(
        self,
        *,
        p: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if p <= 0 or (not isfinite(p) and p != inf):
            raise ValueError(
                "Minkowski parameter p must be a positive finite value or infinity."
            )
        self.p = p

    @property
    def name(self) -> str:
        return "minkowski"

    def score(
        self,
        source: Sequence[float],
        target: Sequence[float],
    ) -> float:
        """Compute Minkowski distance."""
        self._validate_shapes(source, target)

        if self.p == inf:
            return max(abs(x - y) for x, y in zip(source, target))

        return pow(
            sum(pow(abs(x - y), self.p) for x, y in zip(source, target)),
            1.0 / self.p,
        )

    def affinity(self, raw_score: float) -> float:
        """Convert Minkowski distance into graph affinity."""
        return 1.0 / (1.0 + raw_score)