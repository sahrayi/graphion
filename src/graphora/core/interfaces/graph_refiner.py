"""
Graph refinement interface.
"""

from __future__ import annotations

from typing import Protocol, Generic

from graphora.core.models import Graph
from graphora.core.results import StageResult
from graphora.core.types import TId

from .stage import Stage


class GraphRefiner(
    Stage,
    Protocol,
    Generic[TId],
):
    """
    Interface for graph refinement algorithms.
    """

    def refine(
        self,
        graph: Graph[TId],
    ) -> Graph[TId]:
        """
        Refine graph.
        """
        ...

    def execute(
        self,
        input_data: Graph[TId],
    ) -> StageResult[Graph[TId]]:
        """
        Execute graph refinement.
        """
        return StageResult(
            output=self.refine(input_data),
        )