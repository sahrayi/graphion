"""
Louvain community detection algorithm.
"""

from __future__ import annotations

from typing import (
    Generic,
)

import networkx as nx

from graphora.core.models import Graph

from .base_partition_detector import (
    BasePartitionDetector,
)

from graphora.core.types import (
    TId,
)


class Louvain(
    BasePartitionDetector[TId],
    Generic[TId],
):
    """
    Louvain community detection.

    Uses modularity optimization to detect
    communities in weighted undirected graphs.

    Edge weights are interpreted as affinity:

        higher weight = stronger connection


    Supported graphs:

    - undirected only


    Properties:

    - weighted
    - hierarchical
    - scalable
    - deterministic
    """

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
    # Initialization
    # --------------------------------------------------

    def __init__(
        self,
        *,
        resolution: float = 1.0,
        seed: int = 42,
    ) -> None:

        if resolution <= 0:
            raise ValueError(
                "resolution must be greater than zero."
            )

        self.resolution = resolution
        self.seed = seed


    # --------------------------------------------------
    # Detection
    # --------------------------------------------------

    def _detect(
        self,
        graph: Graph[TId],
    ) -> list[set[TId]]:
        """
        Detect communities using Louvain.
        """

        if graph.node_count == 0:
            return []


        if graph.node_count == 1:
            return [
                {
                    graph.nodes[0],
                }
            ]


        if graph.edge_count == 0:
            return [
                {
                    node,
                }
                for node in graph.nodes
            ]


        nx_graph = graph.to_networkx()


        communities = (
            nx.algorithms.community
            .louvain_communities(
                nx_graph,
                weight="weight",
                resolution=self.resolution,
                seed=self.seed,
            )
        )


        return [
            set(
                community,
            )
            for community in communities
        ]