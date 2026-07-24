"""
K-nearest neighbors graph construction algorithm.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
from typing import Generic, TypeVar

from graphora.core.models import Edge

from .base_graph_builder import BaseGraphBuilder

from graphora.core.types import (
    TId
)


class KNN(
    BaseGraphBuilder[TId],
    Generic[TId],
):
    """
    K-nearest neighbors graph construction.

    Keeps strongest k neighbors per node
    based on affinity.

    Topology modes:

    Default:

        directed KNN


    mutual=True:

        reciprocal KNN


    symmetric=True:

        symmetric undirected KNN
    """

    def __init__(
        self,
        *,
        k: int = 10,
        mutual: bool = False,
        symmetric: bool = False,
        **kwargs,
    ) -> None:

        if mutual and symmetric:
            raise ValueError(
                "mutual and symmetric cannot "
                "be enabled together."
            )

        super().__init__(
            directed=not symmetric,
            **kwargs,
        )

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        self.k = k
        self.mutual = mutual
        self.symmetric = symmetric


    def filter_edges(
        self,
        edges: list[Edge[TId]],
    ) -> list[Edge[TId]]:
        """
        Apply KNN sparsification.

        Pipeline:

        1. Remove self loops.
        2. Merge duplicate edges.
        3. Select top-k neighbors per node.
        4. Apply mutual filtering if requested.
        5. Apply symmetry conversion if requested.
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

        if self.mutual:

            selected = self.apply_mutual(
                selected,
            )


        if self.symmetric:

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
        Select strongest k outgoing neighbors
        for every node.
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