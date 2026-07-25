"""
Base class for numeric relation builders.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from typing import Any

from .base_relation_builder import BaseRelationBuilder


class BaseNumericRelationBuilder(BaseRelationBuilder):
    """
    Base class for relation builders operating on numeric feature vectors.
    """

    ZERO_TOLERANCE = 1e-12

    def __init__(self, *, normalize: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.normalize = normalize

    def prepare_vector(self, vector: Sequence[Any]) -> list[float]:
        """Prepare one numeric vector."""
        self._validate_vector(vector)
        numeric_vector = [float(value) for value in vector]
        if self.normalize:
            numeric_vector = self._normalize(numeric_vector)
        return numeric_vector

    @staticmethod
    def _validate_vector(vector: Sequence[Any]) -> None:
        """Validate one numeric vector."""
        if len(vector) == 0:
            raise ValueError("Feature vector cannot be empty.")

        for value in vector:
            try:
                float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "Feature vectors must contain only numeric values."
                ) from exc

    @staticmethod
    def _validate_shapes(source: Sequence[Any], target: Sequence[Any]) -> None:
        """Validate equal vector dimensions."""
        if len(source) != len(target):
            raise ValueError("Feature vectors must have equal length.")

    @staticmethod
    def _norm(vector: Sequence[float]) -> float:
        """Compute L2 norm."""
        return sqrt(sum(value * value for value in vector))

    @classmethod
    def _normalize(cls, vector: list[float]) -> list[float]:
        """Apply L2 normalization."""
        norm = cls._norm(vector)
        if norm < cls.ZERO_TOLERANCE:
            return vector
        return [value / norm for value in vector]