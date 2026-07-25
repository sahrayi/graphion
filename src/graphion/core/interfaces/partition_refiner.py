"""
Partition refinement interface.
"""

from __future__ import annotations

from typing import Protocol

from graphion.core.models import PartitionSet
from graphion.core.results import StageResult

from .stage import Stage


class PartitionRefiner(Stage, Protocol):
    """
    Interface for partition refinement algorithms.
    """

    def refine(
        self,
        partitions: PartitionSet,
    ) -> PartitionSet:
        """
        Refine partitions.
        """
        ...

    def execute(
        self,
        input_data: PartitionSet,
    ) -> StageResult[PartitionSet]:
        """
        Execute partition refinement.
        """
        return StageResult(
            output=self.refine(input_data),
        )