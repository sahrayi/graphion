"""
Weighted label propagation community detector.
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

class LabelPropagation(
    BasePartitionDetector[TId],
    Generic[TId],
):
    """
    Weighted Label Propagation detector.

    Undirected weighted community detection based on
    iterative label updates.

    Edge weights are interpreted as affinity:

        higher weight = stronger connection

    Properties
    ----------
    - weighted
    - parameter light
    - fast
    - deterministic implementation
    - suitable for sparse graphs

    Supported graphs
    ----------------
    - undirected graphs only
    """

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


    def __init__(
        self,
        *,
        max_iterations: int = 100,
    ) -> None:

        if max_iterations <= 0:
            raise ValueError(
                "max_iterations must be greater than zero."
            )

        self.max_iterations = max_iterations


    def _detect(
        self,
        graph: Graph[TId],
    ) -> list[set[TId]]:
        """
        Detect communities using weighted propagation.
        """

        if graph.node_count == 0:
            return []


        adjacency = self._build_adjacency(
            graph,
        )


        labels: dict[TId, TId] = {
            node: node
            for node in graph.nodes
        }


        for _ in range(
            self.max_iterations,
        ):

            changed = False


            for node in sorted(
                graph.nodes,
                key=str,
            ):

                neighbors = adjacency[node]

                if not neighbors:
                    continue


                scores: dict[TId, float] = {}


                for neighbor, weight in neighbors:

                    label = labels[neighbor]

                    scores[label] = (
                        scores.get(
                            label,
                            0.0,
                        )
                        + weight
                    )


                best_label = min(
                    scores,
                    key=lambda label: (
                        -scores[label],
                        str(label),
                    ),
                )


                if labels[node] != best_label:

                    labels[node] = best_label
                    changed = True


            if not changed:
                break


        partitions: dict[TId, set[TId]] = {}


        for node, label in labels.items():

            partitions.setdefault(
                label,
                set(),
            ).add(
                node,
            )


        return list(
            partitions.values(),
        )


    def _build_adjacency(
        self,
        graph: Graph[TId],
    ) -> dict[
        TId,
        list[
            tuple[TId, float]
        ],
    ]:
        """
        Build weighted undirected adjacency list.
        """

        adjacency: dict[
            TId,
            list[
                tuple[TId, float]
            ],
        ] = {
            node: []
            for node in graph.nodes
        }


        for edge in graph.edges:

            if edge.source == edge.target:
                continue


            adjacency[edge.source].append(
                (
                    edge.target,
                    edge.weight,
                )
            )


            adjacency[edge.target].append(
                (
                    edge.source,
                    edge.weight,
                )
            )


        for node in adjacency:

            adjacency[node].sort(
                key=lambda item: str(item[0]),
            )


        return adjacency