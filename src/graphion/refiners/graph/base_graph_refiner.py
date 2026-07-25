"""
Base graph refiner implementation.
"""

from __future__ import annotations

from typing import Generic

from graphion.core.interfaces import GraphRefiner
from graphion.core.models import Graph

from graphion.core.types import (
    TId,
)


class BaseGraphRefiner(
    GraphRefiner[TId],
    Generic[TId],
):
    """
    Base implementation for graph refiners.

    Provides common refinement behavior and
    delegates actual refinement logic to subclasses.
    """

    def refine(
        self,
        graph: Graph[TId],
    ) -> Graph[TId]:
        """
        Refine graph.

        Must be implemented by subclasses.
        """

        raise NotImplementedError(
            "Graph refiner must implement refine()."
        )