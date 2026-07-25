"""
Walktrap community detection algorithm using igraph backend.
"""

from __future__ import annotations

from typing import (
    Generic,
)

import igraph as ig

from graphion.core.models import Graph

from .base_partition_detector import (
    BasePartitionDetector,
)

from graphion.core.types import (
    TId,
)


class Walktrap(
    BasePartitionDetector[TId],
    Generic[TId],
):
    """
    Weighted Walktrap community detector.

    Uses igraph random-walk based hierarchical
    community detection.

    Edge weights are interpreted as affinity:

        higher weight = stronger connection


    Properties
    ----------

    - weighted
    - hierarchical
    - deterministic
    - sparse graph friendly
    """

    def __init__(
        self,
        *,
        walk_steps: int = 4,
    ) -> None:
        """
        Initialize Walktrap detector.

        Parameters
        ----------
        walk_steps:
            Number of random walk steps used
            to compute community similarity.
        """

        if walk_steps <= 0:
            raise ValueError(
                "walk_steps must be greater than zero."
            )

        self.walk_steps = walk_steps


    # --------------------------------------------------
    # Detection
    # --------------------------------------------------

    def _detect(
        self,
        graph: Graph[TId],
    ) -> list[set[TId]]:
        """
        Detect communities using igraph Walktrap.
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


        ig_graph, node_ids = self._to_igraph(
            graph,
        )


        dendrogram = ig_graph.community_walktrap(
            weights="weight",
            steps=self.walk_steps,
        )


        communities = dendrogram.as_clustering()


        return [
            {
                node_ids[index]
                for index in community
            }
            for community in communities
        ]


    # --------------------------------------------------
    # Backend conversion
    # --------------------------------------------------

    def _to_igraph(
        self,
        graph: Graph[TId],
    ) -> tuple[
        ig.Graph,
        tuple[TId, ...],
    ]:
        """
        Convert Graphora Graph into igraph.

        Graphora Graph remains the source of truth.
        """

        node_ids = tuple(
            sorted(
                graph.nodes,
                key=str,
            )
        )


        index_map = {
            node_id: index
            for index, node_id
            in enumerate(node_ids)
        }


        edges: list[
            tuple[int, int]
        ] = []

        weights: list[float] = []


        seen: dict[
            tuple[int, int],
            float,
        ] = {}


        for edge in graph.edges:

            if edge.source == edge.target:
                continue


            source = index_map[
                edge.source
            ]

            target = index_map[
                edge.target
            ]


            key = (
                min(source, target),
                max(source, target),
            )


            current = seen.get(
                key,
            )


            if (
                current is None
                or edge.weight > current
            ):
                seen[key] = edge.weight


        for (
            source,
            target,
        ), weight in sorted(
            seen.items(),
        ):

            edges.append(
                (
                    source,
                    target,
                )
            )

            weights.append(
                weight,
            )


        ig_graph = ig.Graph(
            n=len(node_ids),
            edges=edges,
            directed=False,
        )


        ig_graph.es["weight"] = weights


        return (
            ig_graph,
            node_ids,
        )