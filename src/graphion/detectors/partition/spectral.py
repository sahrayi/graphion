"""
Spectral clustering partition detector.
"""

from __future__ import annotations

import warnings

from typing import Generic

import numpy as np

from sklearn.cluster import SpectralClustering

from graphion.core.models import Graph
from graphion.core.types import TId

from .base_partition_detector import BasePartitionDetector


class Spectral(
    BasePartitionDetector[TId],
    Generic[TId],
):
    """
    Spectral clustering partition detector.

    Uses graph affinity matrix and spectral
    embedding to detect partitions.

    Edge weights are interpreted as affinity:

        higher weight = stronger connection


    Properties:

    - weighted
    - graph based
    - embedding based
    - deterministic
    """

    def __init__(
        self,
        *,
        n_clusters: int = 2,
        random_state: int = 42,
    ) -> None:

        if n_clusters <= 0:
            raise ValueError(
                "n_clusters must be greater than zero."
            )

        self.n_clusters = n_clusters
        self.random_state = random_state


    def _detect(
        self,
        graph: Graph[TId],
    ) -> list[set[TId]]:
        """
        Detect partitions using spectral clustering.
        """

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


        # If every node should be its own cluster,
        # avoid unnecessary sklearn execution.
        if graph.node_count == self.n_clusters:
            return [
                {
                    node,
                }
                for node in graph.nodes
            ]


        affinity = self._build_affinity_matrix(
            graph,
        )


        model = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=self.random_state,
        )


        with warnings.catch_warnings():

            warnings.filterwarnings(
                "ignore",
                message=(
                    "Graph is not fully connected, "
                    "spectral embedding may not work as expected."
                ),
                category=UserWarning,
            )


            labels = model.fit_predict(
                affinity,
            )


        partitions: dict[int, set[TId]] = {}


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


        return sorted(
            partitions.values(),
            key=lambda partition: min(
                str(node)
                for node in partition
            ),
        )


    def _build_affinity_matrix(
        self,
        graph: Graph[TId],
    ) -> np.ndarray:
        """
        Build dense affinity matrix.

        Matrix order follows graph.nodes order.
        """

        size = graph.node_count


        index = {
            node: i
            for i, node in enumerate(
                graph.nodes,
            )
        }


        matrix = np.zeros(
            (
                size,
                size,
            ),
            dtype=float,
        )


        for edge in graph.edges:

            if edge.source == edge.target:
                continue


            if edge.weight < 0:
                raise ValueError(
                    "Spectral clustering requires "
                    "non-negative affinity weights."
                )


            source = index[edge.source]

            target = index[edge.target]


            matrix[
                source,
                target,
            ] = edge.weight


            if not graph.directed:

                matrix[
                    target,
                    source,
                ] = edge.weight


        return matrix