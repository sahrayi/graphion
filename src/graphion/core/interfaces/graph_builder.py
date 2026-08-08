"""
Graph builder interface.
"""

from __future__ import annotations

from collections.abc import Iterable

from typing import (
    Generic,
    Protocol,
)

from graphion.core.types import TId

from graphion.core.models import (
    Graph,
    RelationSet,
)

from graphion.core.results import (
    StageResult,
)

from .stage import Stage


class GraphBuilder(
    Stage,
    Protocol,
    Generic[TId],
):
    """
    Interface for graph construction algorithms.

    A GraphBuilder transforms a RelationSet
    into a Graph.

    Implementations define:

    - graph topology construction
    - edge selection strategy
    - sparsification rules
    - neighborhood extraction
    - graph directionality
    """

    @property
    def directed(
        self,
    ) -> bool:
        """
        Whether produced graph is directed.

        Implementations must declare
        the semantics of generated edges.
        """
        ...

    def build(
        self,
        relations: RelationSet[TId],
        nodes: Iterable[TId] | None = None,
    ) -> Graph[TId]:
        """
        Build graph from relations.

        Parameters
        ----------
        relations:
            Pairwise relations between entities.

        nodes:
            Optional explicit node collection.
            Used to preserve isolated nodes.
        """
        ...

    def execute(
        self,
        input_data: RelationSet[TId],
    ) -> StageResult[Graph[TId]]:
        """
        Execute graph building stage.
        """
        ...