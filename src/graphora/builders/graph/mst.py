"""
Minimum Spanning Tree graph construction algorithm.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Generic, TypeVar

from graphora.core.models import Edge

from .base_graph_builder import BaseGraphBuilder

from graphora.core.types import (
    TId
)


class MST(
    BaseGraphBuilder[TId],
    Generic[TId],
):
    """
    Minimum Spanning Tree graph construction.

    Builds an undirected MST from affinity edges.

    Edge weights represent affinity:

        higher weight = stronger relation


    Kruskal internally minimizes:

        cost = 1 - affinity


    Properties:

    - always undirected
    - deterministic
    - parameter free
    - sparse

    If input graph is disconnected,
    returns a minimum spanning forest.
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
        Build MST using Kruskal algorithm.

        Pipeline:

        1. Remove self loops.
        2. Collapse directed edges into undirected candidates.
        3. Sort by affinity cost.
        4. Apply union-find.
        """

        edges = self.remove_self_loops(
            edges,
        )

        edges = self._build_undirected_candidates(
            edges,
        )

        edges.sort(
            key=lambda edge: (
                self._cost(edge.weight),
                str(edge.source),
                str(edge.target),
            )
        )


        nodes = self._extract_nodes(
            edges,
        )


        parent = {
            node: node
            for node in nodes
        }

        rank = {
            node: 0
            for node in nodes
        }


        mst: list[Edge[TId]] = []


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


            mst.append(
                edge,
            )


        return sorted(
            mst,
            key=lambda edge: (
                str(edge.source),
                str(edge.target),
            ),
        )


    def _build_undirected_candidates(
        self,
        edges: list[Edge[TId]],
    ) -> list[Edge[TId]]:
        """
        Collapse directed affinity edges.

        Example:

            A -> B (0.8)
            B -> A (0.9)


        becomes:

            A -- B (0.9)
        """

        weights: dict[
            tuple[TId, TId],
            float,
        ] = {}


        for edge in edges:

            pair = self._normalize_pair(
                edge.source,
                edge.target,
            )

            current = weights.get(
                pair,
            )


            if (
                current is None
                or edge.weight > current
            ):
                weights[pair] = edge.weight


        return [
            Edge(
                source=source,
                target=target,
                weight=weight,
            )
            for (
                source,
                target,
            ), weight in sorted(
                weights.items(),
                key=lambda item: (
                    str(item[0][0]),
                    str(item[0][1]),
                ),
            )
        ]


    def _normalize_pair(
        self,
        source: TId,
        target: TId,
    ) -> tuple[TId, TId]:
        """
        Deterministic undirected ordering.
        """

        if str(source) <= str(target):

            return (
                source,
                target,
            )

        return (
            target,
            source,
        )


    def _cost(
        self,
        affinity: float,
    ) -> float:
        """
        Convert affinity to Kruskal cost.
        """

        return 1.0 - affinity


    def _find(
        self,
        parent: dict[TId, TId],
        node: TId,
    ) -> TId:
        """
        Find set representative.
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
        Merge two sets using rank.
        """

        if rank[root_a] < rank[root_b]:

            parent[root_a] = root_b


        elif rank[root_a] > rank[root_b]:

            parent[root_b] = root_a


        else:

            parent[root_b] = root_a
            rank[root_a] += 1