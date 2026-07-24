"""
Reducer stage interface.
"""

from __future__ import annotations

from typing import Protocol

from graphora.core.models import FeatureSet
from graphora.core.results import StageResult

from .stage import Stage


class Reducer(Stage, Protocol):
    """
    Interface for feature reduction algorithms.
    """

    def reduce(
        self,
        features: FeatureSet,
    ) -> FeatureSet:
        """
        Reduce feature dimensionality.
        """
        ...

    def execute(
        self,
        input_data: FeatureSet,
    ) -> StageResult[FeatureSet]:
        """
        Execute reducer stage.
        """
        return StageResult(
            output=self.reduce(input_data),
        )