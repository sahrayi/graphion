"""
Radius based graph construction algorithm.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Generic, TypeVar

from graphora.core.models import Edge

from .base_graph_builder import BaseGraphBuilder

from graphora.core.types import (
    TId,
)


class Radius(
    BaseGraphBuilder[TId],
    Generic[TId],
):
    """
    Radius graph construction.

    Keeps edges whose affinity satisfies
    a minimum affinity threshold.

    An edge is kept if:

        affinity >= radius


    Output directionality follows
    BaseGraphBuilder configuration.


    Supports:

    - directed radius graph

    - undirected radius graph


    Properties:

    - parameter controlled
    - deterministic output
    - sparse topology
    """

    def __init__(
        self,
        *,
        radius: float = 0.5,
        inclusive: bool = True,
        **kwargs,
    ) -> None:

        super().__init__(
            **kwargs,
        )

        if not 0.0 <= radius <= 1.0:
            raise ValueError(
                "Radius must be between 0 and 1."
            )

        self.radius = radius
        self.inclusive = inclusive


    def filter_edges(
        self,
        edges: list[Edge[TId]],
    ) -> list[Edge[TId]]:
        """
        Apply radius filtering.

        Pipeline:

        1. Remove self loops.
        2. Merge duplicate edges.
        3. Apply affinity threshold.
        4. Symmetrize if graph is undirected.
        """

        edges = self.remove_self_loops(
            edges,
        )

        edges = self.merge_duplicates(
            edges,
        )


        if self.inclusive:

            filtered = [
                edge
                for edge in edges
                if edge.weight >= self.radius
            ]

        else:

            filtered = [
                edge
                for edge in edges
                if edge.weight > self.radius
            ]


        if not self.directed:

            filtered = self.make_symmetric(
                filtered,
            )


        return sorted(
            filtered,
            key=lambda edge: (
                str(edge.source),
                str(edge.target),
            ),
        )