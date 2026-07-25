"""
Partition detector interface.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import (
    Generic,
    Protocol,
)

from graphion.core.types import TId

from graphion.core.models import (
    Graph,
    PartitionSet,
)
from graphion.core.results import StageResult

from .stage import Stage




class PartitionDetector(
    Stage,
    Protocol,
    Generic[TId],
):
    """
    Interface for community detection algorithms.

    A PartitionDetector transforms a Graph
    into a set of graph partitions.

    Implementations define:

    - community detection algorithm
    - partition extraction strategy
    - clustering logic
    """

    def detect(
        self,
        graph: Graph[TId],
    ) -> PartitionSet[TId]:
        """
        Detect partitions from graph.
        """
        ...

    def execute(
        self,
        input_data: Graph[TId],
    ) -> StageResult[PartitionSet[TId]]:
        """
        Execute partition detection stage.
        """

        return StageResult(
            output=self.detect(
                input_data,
            ),
        )