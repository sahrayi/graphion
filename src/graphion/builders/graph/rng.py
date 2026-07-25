"""
Relative Neighborhood Graph construction algorithm.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Generic

from graphion.core.models import Edge
from graphion.core.types import TId

from .base_graph_builder import BaseGraphBuilder


class RNG(BaseGraphBuilder[TId], Generic[TId]):
    """
    Relative Neighborhood Graph.

    Connects two nodes A and B if there is no
    third node C closer to both.

    Graphion affinity convention:

        higher affinity = closer

    Edge (A,B) survives iff there is no C such that:

        affinity(A,C) > affinity(A,B)

    and:

        affinity(B,C) > affinity(A,B)

    Properties:

    - parameter free
    - undirected
    - sparse
    - deterministic
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(directed=False, **kwargs)
        self._node_order: dict[TId, int] = {}

    def filter_edges(self, edges: list[Edge[TId]]) -> list[Edge[TId]]:
        """
        Apply RNG filtering.

        Pipeline:

        1. Remove self loops.
        2. Merge undirected duplicate edges.
        3. Build node ordering.
        4. Build adjacency index.
        5. Apply RNG criterion.
        6. Return symmetric edges.
        """
        edges = self.remove_self_loops(edges)
        edges = self.merge_duplicates(edges)

        self._build_node_order(edges)

        adjacency: dict[tuple[TId, TId], float] = {}
        neighbors_map: dict[TId, set[TId]] = defaultdict(set)

        for edge in edges:
            source = edge.source
            target = edge.target

            adjacency[self._edge_key(source, target)] = edge.weight
            neighbors_map[source].add(target)
            neighbors_map[target].add(source)

        kept: list[Edge[TId]] = []

        for edge in self._sort_edges(edges):
            common_neighbors = neighbors_map[edge.source].intersection(
                neighbors_map[edge.target]
            )

            if self._is_relative_neighbor(
                source=edge.source,
                target=edge.target,
                weight=edge.weight,
                adjacency=adjacency,
                candidates=common_neighbors,
            ):
                kept.append(edge)

        # Base contract:
        # undirected graph must have
        # symmetric edges.
        kept = self.make_symmetric(kept)

        return self._sort_edges(kept)

    def _is_relative_neighbor(
        self,
        source: TId,
        target: TId,
        weight: float,
        adjacency: dict[tuple[TId, TId], float],
        candidates: set[TId],
    ) -> bool:
        """
        Check RNG lune criterion.

        A candidate removes the edge if it has
        stronger affinity with both endpoints.
        """
        for candidate in candidates:
            source_weight = adjacency.get(self._edge_key(source, candidate))
            target_weight = adjacency.get(self._edge_key(target, candidate))

            if (
                source_weight is not None
                and target_weight is not None
                and source_weight > weight
                and target_weight > weight
            ):
                return False

        return True

    def _build_node_order(self, edges: list[Edge[TId]]) -> None:
        """
        Build deterministic node ordering.

        This avoids repeated string conversion
        inside the RNG inner loop.
        """
        nodes: set[TId] = set()

        for edge in edges:
            nodes.add(edge.source)
            nodes.add(edge.target)

        ordered_nodes = sorted(nodes, key=str)

        self._node_order = {
            node: index for index, node in enumerate(ordered_nodes)
        }

    def _edge_key(self, source: TId, target: TId) -> tuple[TId, TId]:
        """
        Normalize undirected edge ordering.

        Uses precomputed deterministic node
        ordering for fast lookup.
        """
        if self._node_order[source] <= self._node_order[target]:
            return (source, target)

        return (target, source)

    @staticmethod
    def _sort_edges(edges: list[Edge[TId]]) -> list[Edge[TId]]:
        """
        Deterministic edge sorting.

        Prefer native ordering when available,
        otherwise fall back to string ordering.
        """
        try:
            return sorted(
                edges,
                key=lambda edge: (edge.source, edge.target),
            )
        except TypeError:
            return sorted(
                edges,
                key=lambda edge: (str(edge.source), str(edge.target)),
            )