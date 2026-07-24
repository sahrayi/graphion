"""
Weighted K-nearest neighbors graph construction algorithm.
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


class WeightedKNN(
    BaseGraphBuilder[TId],
    Generic[TId],
):
    """
    Weighted K-nearest neighbors graph construction.

    Selects strongest k neighbors per node
    while preserving affinity weights.

    Optional local normalization can be applied
    on outgoing edges.

    Supports:

    - directed weighted KNN
    - mutual weighted KNN
    - symmetric weighted KNN

    Pipeline:

    1. Remove self loops.
    2. Merge duplicate edges.
    3. Select top-k neighbors.
    4. Optional mutual filtering.
    5. Optional local normalization.
    6. Optional symmetry conversion.
    """

    def __init__(
        self,
        *,
        k: int = 10,
        mutual: bool = False,
        symmetric: bool = False,
        normalize_outgoing: bool = False,
        directed: bool = True,
        **kwargs,
    ) -> None:

        if symmetric:
            directed = False

        super().__init__(
            directed=directed,
            **kwargs,
        )

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        if mutual and symmetric:
            raise ValueError(
                "mutual and symmetric cannot "
                "be enabled together."
            )

        self.k = k
        self.mutual = mutual
        self.symmetric = symmetric
        self.normalize_outgoing = (
            normalize_outgoing
        )


    def filter_edges(
        self,
        edges: list[Edge[TId]],
    ) -> list[Edge[TId]]:
        """
        Apply weighted KNN sparsification.

        Keeps original affinity values unless
        local normalization is enabled.
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


        if self.mutual:

            selected = self.apply_mutual(
                selected,
            )


        if self.normalize_outgoing:

            selected = self._normalize_weights(
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


    def _select_top_k(
        self,
        edges: list[Edge[TId]],
    ) -> list[Edge[TId]]:
        """
        Select strongest k outgoing neighbors.
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


    def _normalize_weights(
        self,
        edges: list[Edge[TId]],
    ) -> list[Edge[TId]]:
        """
        Normalize outgoing affinities.

        Each outgoing edge weight is divided
        by the strongest outgoing affinity
        of its source node.
        """

        maximums: dict[
            TId,
            float,
        ] = {}


        for edge in edges:

            maximums[edge.source] = max(
                maximums.get(
                    edge.source,
                    0.0,
                ),
                edge.weight,
            )


        normalized: list[Edge[TId]] = []


        for edge in edges:

            maximum = maximums[
                edge.source
            ]

            weight = (
                edge.weight / maximum
                if maximum > 0
                else 0.0
            )

            normalized.append(
                Edge(
                    source=edge.source,
                    target=edge.target,
                    weight=weight,
                )
            )


        return normalized