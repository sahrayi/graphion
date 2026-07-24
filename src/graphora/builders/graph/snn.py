"""
Shared Nearest Neighbor graph construction algorithm.
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


class SNN(
    BaseGraphBuilder[TId],
    Generic[TId],
):
    """
    Shared Nearest Neighbor graph construction.

    Builds an undirected graph where edge strength
    is based on neighborhood similarity.

    For nodes A and B:

        SNN(A,B) =
            |N(A) ∩ N(B)|
            ----------------
            |N(A) ∪ N(B)|


    where:

        N(X) = k nearest neighbors of X


    Properties:

    - sparse
    - noise resistant
    - undirected
    - deterministic
    """

    def __init__(
        self,
        *,
        k: int = 10,
        weighted: bool = False,
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
        self.weighted = weighted


    def filter_edges(
        self,
        edges: list[Edge[TId]],
    ) -> list[Edge[TId]]:
        """
        Build SNN graph.

        Pipeline:

        1. Remove self loops.
        2. Merge duplicate affinities.
        3. Build local kNN neighborhoods.
        4. Generate candidate pairs.
        5. Compute shared-neighbor similarity.
        6. Return undirected edges.
        """

        edges = self.remove_self_loops(
            edges,
        )

        edges = self.merge_duplicates(
            edges,
        )

        neighbors = self._build_neighbors(
            edges,
        )

        candidates = self._build_candidate_pairs(
            neighbors,
        )

        result: list[Edge[TId]] = []


        for source, target in sorted(
            candidates,
            key=lambda pair: (
                str(pair[0]),
                str(pair[1]),
            ),
        ):

            weight = self._shared_weight(
                source,
                target,
                neighbors,
            )

            if weight <= 0:
                continue


            result.append(
                Edge(
                    source=source,
                    target=target,
                    weight=weight,
                )
            )


        return sorted(
            result,
            key=lambda edge: (
                str(edge.source),
                str(edge.target),
            ),
        )


    def _build_neighbors(
        self,
        edges: list[Edge[TId]],
    ) -> dict[TId, list[TId]]:
        """
        Build k-nearest neighbor lists.

        Input graph is treated as undirected.
        """

        adjacency: dict[
            TId,
            list[Edge[TId]],
        ] = defaultdict(list)


        for edge in edges:

            adjacency[
                edge.source
            ].append(
                edge,
            )

            adjacency[
                edge.target
            ].append(
                Edge(
                    source=edge.target,
                    target=edge.source,
                    weight=edge.weight,
                )
            )


        neighbors: dict[
            TId,
            list[TId],
        ] = {}


        for node in sorted(
            adjacency,
            key=str,
        ):

            ranked = sorted(
                adjacency[node],
                key=lambda edge: (
                    -edge.weight,
                    str(edge.target),
                ),
            )


            neighbors[node] = [
                edge.target
                for edge in ranked[: self.k]
            ]


        return neighbors


    def _build_candidate_pairs(
        self,
        neighbors: dict[TId, list[TId]],
    ) -> set[tuple[TId, TId]]:
        """
        Build undirected candidate pairs from kNN neighborhoods.

        A pair is considered if at least one node
        selects the other as a neighbor.
        """

        pairs: set[
            tuple[TId, TId]
        ] = set()


        for source in sorted(
            neighbors,
            key=str,
        ):

            for target in neighbors[source]:

                if source == target:
                    continue

                pairs.add(
                    self._normalize_pair(
                        source,
                        target,
                    )
                )


        return pairs


    def _shared_weight(
        self,
        source: TId,
        target: TId,
        neighbors: dict[TId, list[TId]],
    ) -> float:
        """
        Compute shared-neighbor similarity.
        """

        if self.weighted:

            return self._weighted_shared_score(
                source,
                target,
                neighbors,
            )


        source_neighbors = set(
            neighbors.get(
                source,
                [],
            )
        )

        target_neighbors = set(
            neighbors.get(
                target,
                [],
            )
        )


        union = (
            source_neighbors
            |
            target_neighbors
        )


        if not union:
            return 0.0


        intersection = (
            source_neighbors
            &
            target_neighbors
        )


        return (
            len(intersection)
            /
            len(union)
        )


    def _weighted_shared_score(
        self,
        source: TId,
        target: TId,
        neighbors: dict[TId, list[TId]],
    ) -> float:
        """
        Compute rank weighted SNN similarity.
        """

        source_rank = {
            node: rank
            for rank, node in enumerate(
                neighbors.get(
                    source,
                    [],
                )
            )
        }


        target_rank = {
            node: rank
            for rank, node in enumerate(
                neighbors.get(
                    target,
                    [],
                )
            )
        }


        shared = (
            set(source_rank)
            &
            set(target_rank)
        )


        if not shared:
            return 0.0


        score = 0.0


        for node in shared:

            source_score = (
                len(source_rank)
                -
                source_rank[node]
            )

            target_score = (
                len(target_rank)
                -
                target_rank[node]
            )


            score += min(
                source_score,
                target_score,
            )


        size = min(
            len(source_rank),
            len(target_rank),
        )


        normalization = (
            size
            *
            (size + 1)
            /
            2
        )


        if normalization == 0:
            return 0.0


        return score / normalization


    def _normalize_pair(
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