"""
Relative Neighborhood Graph construction algorithm.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Generic, TypeVar

from graphora.core.models import Edge

from .base_graph_builder import BaseGraphBuilder

from graphora.core.types import (
    TId
)


class RNG(
    BaseGraphBuilder[TId],
    Generic[TId],
):
    """
    Relative Neighborhood Graph.

    Connects two nodes A and B if there is no
    third node C closer to both.

    Graphora affinity convention:

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
        Apply RNG filtering.

        Pipeline:

        1. Remove self loops.
        2. Merge undirected duplicate edges.
        3. Apply RNG criterion.
        4. Return symmetric edges.
        """

        edges = self.remove_self_loops(
            edges,
        )

        edges = self.merge_duplicates(
            edges,
        )


        adjacency = {
            self._edge_key(
                edge.source,
                edge.target,
            ): edge.weight
            for edge in edges
        }


        nodes = self._extract_nodes(
            edges,
        )


        kept: list[Edge[TId]] = []


        for edge in sorted(
            edges,
            key=lambda edge: (
                str(edge.source),
                str(edge.target),
            ),
        ):

            if self._is_relative_neighbor(
                edge.source,
                edge.target,
                edge.weight,
                adjacency,
                nodes,
            ):

                kept.append(
                    edge,
                )


        # Base contract:
        # undirected graph must have
        # symmetric edges.

        kept = self.make_symmetric(
            kept,
        )


        return sorted(
            kept,
            key=lambda edge: (
                str(edge.source),
                str(edge.target),
            ),
        )


    def _is_relative_neighbor(
        self,
        source: TId,
        target: TId,
        weight: float,
        adjacency: dict[
            tuple[TId, TId],
            float,
        ],
        nodes: list[TId],
    ) -> bool:
        """
        Check RNG lune criterion.
        """

        for candidate in nodes:

            if (
                candidate == source
                or candidate == target
            ):
                continue


            source_candidate = adjacency.get(
                self._edge_key(
                    source,
                    candidate,
                )
            )

            target_candidate = adjacency.get(
                self._edge_key(
                    target,
                    candidate,
                )
            )


            if (
                source_candidate is not None
                and target_candidate is not None
                and source_candidate > weight
                and target_candidate > weight
            ):
                return False


        return True


    def _edge_key(
        self,
        source: TId,
        target: TId,
    ) -> tuple[TId, TId]:
        """
        Normalize undirected edge ordering.
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