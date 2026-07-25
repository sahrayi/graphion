"""
Minimum Spanning Tree graph construction algorithm.
"""

from __future__ import annotations

from typing import Generic

from graphora.core.models import Edge
from graphora.core.types import TId

from .base_graph_builder import BaseGraphBuilder


class MST(
    BaseGraphBuilder[TId],
    Generic[TId],
):
    """
    Minimum Spanning Tree graph construction.

    Builds an undirected minimum spanning tree
    from affinity edges.

    Edge weights represent affinity:

        higher weight = stronger relation

    Kruskal internally minimizes:

        cost = 1 - affinity

    Properties:

    - parameter free
    - undirected
    - deterministic
    - sparse

    If the input graph is disconnected,
    the result is a minimum spanning forest.
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            directed=False,
            **kwargs,
        )

    def filter_edges(
        self,
        edges: list[Edge[TId]],
    ) -> list[Edge[TId]]:
        """
        Build a minimum spanning tree using
        Kruskal's algorithm.

        Pipeline:

        1. Remove self loops.
        2. Collapse directed duplicates.
        3. Sort by Kruskal cost.
        4. Apply union-find.
        5. Return deterministic forest.
        """

        edges = self.remove_self_loops(edges)

        # Collapse A->B and B->A into a single
        # undirected candidate while keeping the
        # strongest affinity.
        edges = self.make_symmetric(edges)

        edges.sort(
            key=lambda edge: (
                self._cost(edge.weight),
                self.sort_key(edge.source),
                self.sort_key(edge.target),
            ),
        )

        nodes = self._extract_nodes(edges)

        parent: dict[TId, TId] = {
            node: node
            for node in nodes
        }

        rank: dict[TId, int] = {
            node: 0
            for node in nodes
        }

        forest: list[Edge[TId]] = []

        for edge in edges:

            root_source = self._find(
                parent,
                edge.source,
            )

            root_target = self._find(
                parent,
                edge.target,
            )

            if root_source == root_target:
                continue

            self._union(
                parent,
                rank,
                root_source,
                root_target,
            )

            forest.append(edge)

        return sorted(
            forest,
            key=lambda edge: (
                self.sort_key(edge.source),
                self.sort_key(edge.target),
            ),
        )

    def _cost(
        self,
        affinity: float,
    ) -> float:
        """
        Convert affinity score to Kruskal cost.

        Since Graphora uses affinity where:

            higher = stronger relation

        Kruskal requires minimizing cost,
        therefore:

            cost = 1 - affinity
        """

        return 1.0 - affinity


    def _find(
        self,
        parent: dict[TId, TId],
        node: TId,
    ) -> TId:
        """
        Find the representative of a set.

        Uses path compression to reduce the
        amortized complexity of union-find.
        """

        if parent[node] != node:

            parent[node] = self._find(
                parent,
                parent[node],
            )

        return parent[node]


    def _union(
        self,
        parent: dict[TId, TId],
        rank: dict[TId, int],
        root_a: TId,
        root_b: TId,
    ) -> None:
        """
        Merge two disjoint sets.

        Uses union by rank to keep the
        internal tree shallow.
        """

        if rank[root_a] < rank[root_b]:

            parent[root_a] = root_b


        elif rank[root_a] > rank[root_b]:

            parent[root_b] = root_a


        else:

            parent[root_b] = root_a

            rank[root_a] += 1