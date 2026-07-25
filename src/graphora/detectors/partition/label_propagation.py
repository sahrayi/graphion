"""
Weighted label propagation community detector.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic

from graphora.core.models import Graph
from graphora.core.types import TId

from .base_partition_detector import BasePartitionDetector


class LabelPropagation(BasePartitionDetector[TId], Generic[TId]):
    """
    Weighted Label Propagation detector.

    Undirected weighted community detection based on
    iterative label updates.

    Edge weights are interpreted as affinity:
        higher weight = stronger connection

    Update strategy:
        asynchronous propagation

    Labels are updated immediately during each
    iteration. Therefore, nodes processed later
    in an iteration can observe updates made
    earlier in the same iteration.

    Properties:
    - weighted
    - parameter light
    - fast
    - deterministic
    - suitable for sparse graphs

    Supported graphs:
    - undirected graphs only
    """

    @property
    def supports_directed(self) -> bool:
        return False

    @property
    def supports_undirected(self) -> bool:
        return True

    def __init__(
        self,
        *,
        max_iterations: int = 100,
        sort_key: Callable[[TId], str] | None = None,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than zero.")

        self.max_iterations = max_iterations
        self.sort_key: Callable[[TId], str] = (
            sort_key if sort_key is not None else lambda value: str(value)
        )

    def _detect(self, graph: Graph[TId]) -> list[set[TId]]:
        """
        Detect communities using weighted
        asynchronous label propagation.
        """
        if graph.node_count == 0:
            return []

        adjacency = self._build_adjacency(graph)

        # Initial label: each node starts with itself.
        labels: dict[TId, TId] = {node: node for node in graph.nodes}
        ordered_nodes = sorted(graph.nodes, key=self.sort_key)

        for _ in range(self.max_iterations):
            changed = False

            for node in ordered_nodes:
                neighbors = adjacency[node]

                # Isolated nodes keep their own label.
                if not neighbors:
                    continue

                scores: dict[TId, float] = {}

                for neighbor, weight in neighbors:
                    label = labels[neighbor]
                    scores[label] = scores.get(label, 0.0) + weight

                # Select strongest label.
                # Tie breaking is deterministic: lexicographically smaller label wins.
                best_label = min(
                    scores,
                    key=lambda label: (-scores[label], self.sort_key(label)),
                )

                if labels[node] != best_label:
                    labels[node] = best_label
                    changed = True

            if not changed:
                break

        partitions: dict[TId, set[TId]] = {}

        for node, label in labels.items():
            partitions.setdefault(label, set()).add(node)

        return list(partitions.values())

    def _build_adjacency(
        self, graph: Graph[TId]
    ) -> dict[TId, list[tuple[TId, float]]]:
        """
        Build weighted undirected adjacency list.
        """
        adjacency: dict[TId, list[tuple[TId, float]]] = {
            node: [] for node in graph.nodes
        }

        for edge in graph.edges:
            if edge.source == edge.target:
                continue

            adjacency[edge.source].append((edge.target, edge.weight))
            adjacency[edge.target].append((edge.source, edge.weight))

        for node in adjacency:
            adjacency[node].sort(key=lambda item: self.sort_key(item[0]))

        return adjacency