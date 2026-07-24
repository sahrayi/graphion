"""
Symmetric K-nearest neighbors graph construction algorithm.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
from typing import Generic, TypeVar

from graphora.core.models import Edge

from .base_graph_builder import BaseGraphBuilder

from graphora.core.types import (
    TId,
)


class SymmetricKNN(
    BaseGraphBuilder[TId],
    Generic[TId],
):
    """
    Symmetric K-nearest neighbors graph construction.

    Builds a symmetric KNN graph.

    The algorithm:

    1. Selects top-k neighbors for every node.
    2. Converts selected directed edges into
       an undirected symmetric graph.

    Example:

        A -> B (0.8)

    becomes:

        A -- B (0.8)


    When both directions exist:

        A -> B (0.8)
        B -> A (0.6)

    the maximum affinity is preserved:

        A -- B (0.8)


    Properties:

    - undirected output
    - sparse topology
    - deterministic
    - preserves strongest affinity


    Notes:

    This differs from MutualKNN:

    SymmetricKNN:
        keeps union of neighbors

    MutualKNN:
        keeps intersection of neighbors
    """

    def __init__(
        self,
        *,
        k: int = 10,
        **kwargs,
    ) -> None:

        super().__init__(
            directed=False,
            **kwargs,
        )

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        self.k = k


    def filter_edges(
        self,
        edges: list[Edge[TId]],
    ) -> list[Edge[TId]]:
        """
        Apply symmetric KNN sparsification.

        Pipeline:

        1. Remove self loops.
        2. Merge duplicate directed edges.
        3. Select top-k neighbors per node.
        4. Convert to symmetric graph.
        """

        edges = self.remove_self_loops(
            edges,
        )

        edges = self.merge_duplicates(
            edges,
        )

        selected = self._select_top_k(
            edges,
        )

        selected = self.make_symmetric(
            selected,
        )

        return sorted(
            selected,
            key=lambda edge: (
                str(edge.source),
                str(edge.target),
            ),
        )


    def _select_top_k(
        self,
        edges: list[Edge[TId]],
    ) -> list[Edge[TId]]:
        """
        Select strongest k outgoing neighbors per node.
        """

        outgoing: dict[
            TId,
            list[Edge[TId]],
        ] = defaultdict(list)


        for edge in edges:

            outgoing[
                edge.source
            ].append(
                edge,
            )


        selected: list[Edge[TId]] = []


        for node in sorted(
            outgoing,
            key=str,
        ):

            neighbors = sorted(
                outgoing[node],
                key=lambda edge: (
                    -edge.weight,
                    str(edge.target),
                ),
            )

            selected.extend(
                neighbors[: self.k]
            )


        return selected