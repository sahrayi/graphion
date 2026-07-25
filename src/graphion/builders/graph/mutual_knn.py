"""
Mutual K-nearest neighbors graph construction algorithm.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
from typing import Generic, TypeVar

from graphion.core.models import Edge

from .base_graph_builder import BaseGraphBuilder

from graphion.core.types import (
    TId
)


class MutualKNN(
    BaseGraphBuilder[TId],
    Generic[TId],
):
    """
    Mutual K-nearest neighbors graph construction.

    Keeps only reciprocal k-nearest neighbors.

    An edge:

        A -> B

    survives only if:

        A selects B
        and
        B selects A


    Output can be:

    Directed:

        A -> B
        B -> A


    Undirected:

        A -- B
    """

    def __init__(
        self,
        *,
        k: int = 10,
        directed: bool = True,
        **kwargs,
    ) -> None:

        super().__init__(
            directed=directed,
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
        Apply mutual KNN sparsification.

        Pipeline:

        1. Remove self loops.
        2. Merge duplicate edges.
        3. Select top-k neighbors.
        4. Keep reciprocal neighbors.
        5. Symmetrize if graph is undirected.
        """

        edges = self.remove_self_loops(
            edges,
        )

        edges = self.merge_duplicates(
            edges,
        )

        selected = self._select_knn(
            edges,
        )

        selected = self.apply_mutual(
            selected,
        )


        if not self.directed:

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


    def _select_knn(
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
                neighbors[: self.k],
            )


        return selected