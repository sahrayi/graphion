"""
Girvan-Newman community detection algorithm.
"""

from __future__ import annotations

from collections.abc import Hashable

from typing import (
    Generic,
    TypeVar,
)

from graphora.core.models import Graph

from .base_partition_detector import (
    BasePartitionDetector,
)

from graphora.core.types import (
    TId,
)


class GirvanNewman(
    BasePartitionDetector[TId],
    Generic[TId],
):
    """
    Weighted Girvan-Newman community detector.

    Uses edge betweenness removal
    to reveal hierarchical communities.

    Edge weights are interpreted as affinity
    and converted internally into distances.


    Supported topology:

    - undirected graphs only


    Properties:

    - hierarchical
    - weighted
    - deterministic
    - edge-betweenness based
    """

    def __init__(
        self,
        *,
        target_partitions: int | None = None,
    ) -> None:

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


        igraph = graph.to_igraph()


        # Convert affinity to distance
        distances = [
            1.0 / max(
                edge.weight,
                1e-12,
            )
            for edge in graph.edges
        ]


        igraph.es["weight"] = distances


        dendrogram = (
            igraph.community_edge_betweenness(
                weights="weight",
            )
        )


        if self.target_partitions is not None:

            clusters = (
                dendrogram.as_clustering(
                    self.target_partitions,
                )
            )

        else:

            clusters = (
                dendrogram.as_clustering()
            )


        return [
            {
                graph.nodes[index]
                for index in cluster
            }
            for cluster in clusters
        ]