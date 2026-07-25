"""
Leiden community detection algorithm.
"""

from __future__ import annotations

from typing import (
    Generic,
)

import leidenalg

from graphion.core.models import Graph

from .base_partition_detector import (
    BasePartitionDetector,
)

from graphion.core.types import (
    TId,
)


class Leiden(
    BasePartitionDetector[TId],
    Generic[TId],
):
    """
    Leiden community detection detector.

    Uses Leiden algorithm through
    igraph backend.

    Graph direction is inherited from
    Graphion Graph model.

    Properties
    ----------
    - weighted
    - scalable
    - deterministic
    - supports directed graphs
    - supports undirected graphs
    """


    # --------------------------------------------------
    # Capability
    # --------------------------------------------------

    @property
    def supports_directed(
        self,
    ) -> bool:

        return True


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
        iterations: int = -1,
    ) -> None:
        """
        Initialize Leiden detector.

        Parameters
        ----------
        resolution:
            Leiden resolution parameter.

        seed:
            Random seed for deterministic execution.

        iterations:
            Number of Leiden iterations.
            -1 means until convergence.
        """

        if resolution <= 0:
            raise ValueError(
                "resolution must be greater than zero."
            )

        self.resolution = resolution
        self.seed = seed
        self.iterations = iterations


    # --------------------------------------------------
    # Detection
    # --------------------------------------------------

    def _detect(
        self,
        graph: Graph[TId],
    ) -> list[set[TId]]:
        """
        Detect communities using Leiden.
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


        ig_graph = graph.to_igraph()


        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=self.resolution,
            seed=self.seed,
            n_iterations=self.iterations,
        )


        node_ids = tuple(
            graph.nodes,
        )


        return [
            {
                node_ids[index]
                for index in community
            }
            for community in partition
        ]