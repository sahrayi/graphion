"""
KNN graph construction with affinity threshold filtering.
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


class KNNThreshold(
    BaseGraphBuilder[TId],
    Generic[TId],
):
    """
    Threshold constrained K-nearest neighbors graph.

    Keeps strongest k neighbors per node
    among edges satisfying affinity threshold.

    Topology modes:

    Default:

        directed KNNThreshold


    mutual=True:

        reciprocal KNNThreshold


    symmetric=True:

        symmetric undirected KNNThreshold
    """

    def __init__(
        self,
        *,
        k: int = 10,
        threshold: float = 0.5,
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

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        self.k = k
        self.threshold = threshold
        self.mutual = mutual
        self.symmetric = symmetric


    def filter_edges(
        self,
        edges: list[Edge[TId]],
    ) -> list[Edge[TId]]:
        """
        Apply threshold constrained KNN.

        Pipeline:

        1. Remove self loops.
        2. Merge duplicate edges.
        3. Filter weak affinities.
        4. Select top-k neighbors.
        5. Apply mutual filtering.
        6. Apply symmetry conversion.
        """

        edges = self.remove_self_loops(
            edges,
        )

        edges = self.merge_duplicates(
            edges,
        )

        edges = [
            edge
            for edge in edges
            if edge.weight >= self.threshold
        ]

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