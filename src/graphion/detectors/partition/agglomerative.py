"""
Agglomerative clustering partition detector.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import (
    Generic,
    TypeVar,
)

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from graphion.core.models import Graph

from .base_partition_detector import (
    BasePartitionDetector,
)

from graphion.core.types import (
    TId,
)


class Agglomerative(
    BasePartitionDetector[TId],
    Generic[TId],
):
    """
    Undirected agglomerative clustering detector.

    Converts graph affinity into a precomputed
    distance matrix and applies hierarchical
    clustering.

    Edge weights are interpreted as affinity:

        distance = 1 - affinity


    Supported graphs:

    - undirected only

    Properties:

    - hierarchical
    - weighted
    - deterministic
    """


    def __init__(
        self,
        *,
        n_clusters: int = 2,
        linkage: str = "average",
    ) -> None:

        if n_clusters <= 0:
            raise ValueError(
                "n_clusters must be greater than zero."
            )

        allowed_linkages = {
            "single",
            "complete",
            "average",
        }

        if linkage not in allowed_linkages:
            raise ValueError(
                "linkage must be one of: "
                "single, complete, average."
            )

        self.n_clusters = n_clusters
        self.linkage = linkage


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


        if self.n_clusters > graph.node_count:

            raise ValueError(
                "n_clusters cannot exceed number of nodes."
            )


        distance = self._build_distance_matrix(
            graph,
        )


        model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric="precomputed",
            linkage=self.linkage,
        )


        labels = model.fit_predict(
            distance,
        )


        partitions: dict[
            int,
            set[TId],
        ] = {}


        for node, label in zip(
            graph.nodes,
            labels,
        ):

            partitions.setdefault(
                int(label),
                set(),
            ).add(
                node,
            )


        return [
            partitions[label]
            for label in sorted(
                partitions,
            )
        ]


    # --------------------------------------------------
    # Distance conversion
    # --------------------------------------------------

    def _build_distance_matrix(
        self,
        graph: Graph[TId],
    ) -> np.ndarray:
        """
        Convert undirected affinity graph
        into distance matrix.
        """

        size = graph.node_count


        index = {
            node: i
            for i, node in enumerate(
                graph.nodes,
            )
        }


        matrix = np.ones(
            (
                size,
                size,
            ),
            dtype=float,
        )


        np.fill_diagonal(
            matrix,
            0.0,
        )


        for edge in graph.edges:

            source = index[
                edge.source
            ]

            target = index[
                edge.target
            ]


            distance = (
                1.0
                -
                edge.weight
            )


            matrix[
                source,
                target,
            ] = distance


            matrix[
                target,
                source,
            ] = distance


        return matrix