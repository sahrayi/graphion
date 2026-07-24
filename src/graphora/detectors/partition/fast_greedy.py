"""
Fast Greedy community detection algorithm.
"""

from __future__ import annotations

from collections.abc import Hashable

from typing import (
    Generic,
    TypeVar,
)

from graphora.core.errors import InvalidGraphError
from graphora.core.models import Graph

from .base_partition_detector import (
    BasePartitionDetector,
)

from graphora.core.types import (
    TId,
)


class FastGreedy(
    BasePartitionDetector[TId],
    Generic[TId],
):
    """
    Weighted Fast Greedy community detector.

    Uses igraph backend internally.

    Fast Greedy is a hierarchical
    modularity optimization algorithm.

    Supported topology:

    - undirected graphs only


    Properties:

    - weighted
    - hierarchical
    - modularity based
    - deterministic
    """


    def __init__(
        self,
        *,
        target_partitions: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        target_partitions:
            Optional number of desired communities.

            If None, uses the modularity maximizing cut.
        """

        if (
            target_partitions is not None
            and target_partitions <= 0
        ):
            raise ValueError(
                "target_partitions must be greater than zero."
            )


        self.target_partitions = target_partitions


    # --------------------------------------------------
    # Capability
    # --------------------------------------------------

    @property
    def supports_directed(
        self,
    ) -> bool:
        return False


    @property
    def supports_undirected(
        self,
    ) -> bool:
        return True


    # --------------------------------------------------
    # Detection
    # --------------------------------------------------

    def _detect(
        self,
        graph: Graph[TId],
    ) -> list[set[TId]]:

        if graph.node_count == 0:
            return []


        if graph.node_count == 1:

            return [
                {
                    graph.nodes[0],
                }
            ]


        ig_graph = graph.to_igraph()


        weights = [
            edge.weight
            for edge in graph.edges
        ]


        dendrogram = (
            ig_graph.community_fastgreedy(
                weights=weights,
            )
        )


        if self.target_partitions is not None:

            communities = (
                dendrogram.as_clustering(
                    n=self.target_partitions,
                )
            )

        else:

            communities = (
                dendrogram.as_clustering()
            )


        node_ids = tuple(
            graph.nodes,
        )


        return [
            {
                node_ids[index]
                for index in community
            }
            for community in communities
        ]