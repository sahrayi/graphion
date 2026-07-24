"""
Identity graph refinement algorithm.
"""

from __future__ import annotations


from typing import Generic

from graphora.core.models import Graph

from .base_graph_refiner import BaseGraphRefiner

from graphora.core.types import (
    TId,
)


class IdentityGraphRefiner(
    BaseGraphRefiner[TId],
    Generic[TId],
):
    """
    Identity graph refiner.

    Returns the input graph unchanged.

    This implementation exists as a default no-op
    refinement stage in the pipeline.

    Responsibilities:

    - preserve graph structure
    - preserve nodes
    - preserve edges
    - provide a valid refinement stage
    - allow disabling refinement without special cases

    Properties
    ----------

    - deterministic
    - parameter free
    - zero modification
    """

    def refine(
        self,
        graph: Graph[TId],
    ) -> Graph[TId]:
        """
        Return graph without modification.

        Parameters
        ----------
        graph:
            Input graph.

        Returns
        -------
        Graph[TId]
            Same graph instance.
        """

        return graph