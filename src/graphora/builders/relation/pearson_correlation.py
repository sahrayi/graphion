"""
Pearson correlation relation metric.
"""

from __future__ import annotations

from math import sqrt

from .base_numeric_relation_builder import (
    BaseNumericRelationBuilder,
)


class PearsonCorrelation(
    BaseNumericRelationBuilder,
):
    """
    Pearson correlation coefficient metric.

    Raw score range

        -1 <= score <= 1
    """

    @property
    def name(
        self,
    ) -> str:
        return "pearson"

    def score(
        self,
        source: list[float],
        target: list[float],
    ) -> float:
        """
        Compute Pearson correlation coefficient.
        """

        self._validate_shapes(
            source,
            target,
        )

        if not source:
            return 0.0

        source_mean = sum(source) / len(source)
        target_mean = sum(target) / len(target)

        source_centered = [
            value - source_mean
            for value in source
        ]

        target_centered = [
            value - target_mean
            for value in target
        ]

        numerator = sum(
            x * y
            for x, y in zip(
                source_centered,
                target_centered,
            )
        )

        source_norm = sqrt(
            sum(
                x * x
                for x in source_centered
            )
        )

        target_norm = sqrt(
            sum(
                y * y
                for y in target_centered
            )
        )

        denominator = (
            source_norm
            * target_norm
        )

        # Constant vectors have undefined correlation.
        # Treat them as no relation.
        if denominator == 0:
            return 0.0

        return numerator / denominator

    def affinity(
        self,
        raw_score: float,
    ) -> float:
        """
        Convert Pearson correlation into graph affinity.

        Mapping:

            [-1, 1] -> [0, 1]

        using:

            affinity = (score + 1) / 2

        Returns
        -------

        float

            0 <= affinity <= 1
        """

        return (
            raw_score + 1.0
        ) / 2.0