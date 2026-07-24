"""
Graph refinement interface.
"""

from __future__ import annotations

from typing import Protocol

from graphora.core.models import Graph
from graphora.core.results import StageResult

from .stage import Stage


class GraphRefiner(Stage, Protocol):
    """
    Interface for graph refinement algorithms.
    """

    def refine(
        self,
        graph: Graph,
    ) -> Graph:
        """
        Refine graph.
        """
        ...

    def execute(
        self,
        input_data: Graph,
    ) -> StageResult[Graph]:
        """
        Execute graph refinement.
        """
        return StageResult(
            output=self.refine(input_data),
        )