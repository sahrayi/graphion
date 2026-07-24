"""
Base class for numeric relation builders.
"""

from __future__ import annotations

from math import sqrt
from typing import Any

from .base_relation_builder import (
    BaseRelationBuilder,
)


class BaseNumericRelationBuilder(
    BaseRelationBuilder,
):
    """
    Base class for relation builders operating on
    numeric feature vectors.

    Responsibilities
    ----------------
    - validate numeric vectors
    - convert values to float
    - optionally apply L2 normalization
    - provide common numeric utilities

    Subclasses implement:

    - score()
    - affinity()
    """

    def __init__(
        self,
        *,
        normalize: bool = False,
        **kwargs,
    ) -> None:

        super().__init__(
            **kwargs,
        )

        self.normalize = normalize

    # --------------------------------------------------
    # Vector preparation
    # --------------------------------------------------

    def prepare_vector(
        self,
        vector: list[Any],
    ) -> list[float]:
        """
        Prepare one numeric vector.

        Steps:

        1. Validate values are numeric.
        2. Convert values to float.
        3. Optionally normalize using L2 norm.
        """

        self._validate_vector(
            vector,
        )

        numeric_vector = [
            float(value)
            for value in vector
        ]

        if self.normalize:
            numeric_vector = self._normalize(
                numeric_vector,
            )

        return numeric_vector

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    @staticmethod
    def _validate_vector(
        vector: list[Any],
    ) -> None:
        """
        Validate one numeric vector.

        Any value accepted by float()
        is considered numeric.
        """

        if not vector:
            raise ValueError(
                "Feature vector cannot be empty."
            )

        for value in vector:

            try:
                float(value)

            except (
                TypeError,
                ValueError,
            ) as exc:

                raise TypeError(
                    "Feature vectors must contain "
                    "only numeric values."
                ) from exc

    @staticmethod
    def _validate_shapes(
        source: list[Any],
        target: list[Any],
    ) -> None:
        """
        Validate equal vector dimensions.
        """

        if len(source) != len(target):
            raise ValueError(
                "Feature vectors must have equal length."
            )

    # --------------------------------------------------
    # Numeric utilities
    # --------------------------------------------------

    @staticmethod
    def _norm(
        vector: list[float],
    ) -> float:
        """
        Compute L2 norm.
        """

        return sqrt(
            sum(
                value * value
                for value in vector
            )
        )

    @classmethod
    def _normalize(
        cls,
        vector: list[float],
    ) -> list[float]:
        """
        Apply L2 normalization.

        Zero vectors are returned unchanged.
        """

        norm = cls._norm(
            vector,
        )

        if norm == 0:
            return vector

        return [
            value / norm
            for value in vector
        ]