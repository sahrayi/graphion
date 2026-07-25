"""
Shared Nearest Neighbor graph construction algorithm.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Generic

from graphora.core.models import Edge
from graphora.core.types import TId

from .base_graph_builder import BaseGraphBuilder


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

        # Deterministic ordering cache.
        self._node_order: dict[TId, int] = {}

    def filter_edges(
        self,
        edges: list[Edge[TId]],
    ) -> list[Edge[TId]]:
        """
        Build SNN graph.

        Pipeline:

        1. Remove self loops.
        2. Merge duplicate affinities.
        3. Build deterministic node ordering.
        4. Build local kNN neighborhoods.
        5. Generate candidate pairs.
        6. Compute shared-neighbor similarity.
        7. Return symmetric graph.
        """

        edges = self.remove_self_loops(edges)
        edges = self.merge_duplicates(edges)

        self._build_node_order(edges)

        neighbors = self._build_neighbors(edges)

        candidates = self._build_candidate_pairs(
            neighbors,
        )

        result: list[Edge[TId]] = []

        sorted_candidates = sorted(
            candidates,
            key=lambda pair: (
                self._node_order[pair[0]],
                self._node_order[pair[1]],
            ),
        )

        for source, target in sorted_candidates:

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

        result = self.make_symmetric(result)

        return self._sort_edges(result)

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

            adjacency[edge.source].append(edge)

            adjacency[edge.target].append(
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

        for node in self._sort_nodes(
            adjacency.keys(),
        ):

            ranked = sorted(
                adjacency[node],
                key=lambda edge: (
                    -edge.weight,
                    self._node_order[edge.target],
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
        Build symmetric candidate pairs.

        A pair is considered if at least one node
        selects the other as a nearest neighbor.
        """

        pairs: set[
            tuple[TId, TId]
        ] = set()

        for source in self._sort_nodes(
            neighbors.keys(),
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
        Compute rank-weighted shared-neighbor similarity.
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

    def _build_node_order(
        self,
        edges: list[Edge[TId]],
    ) -> None:
        """
        Build deterministic node ordering.

        Ordering is computed once and reused
        throughout the algorithm to avoid
        repeated string comparisons.
        """

        nodes: set[TId] = set()

        for edge in edges:
            nodes.add(edge.source)
            nodes.add(edge.target)

        ordered = sorted(
            nodes,
            key=self.sort_key,
        )

        self._node_order = {
            node: index
            for index, node in enumerate(
                ordered,
            )
        }

    def _normalize_pair(
        self,
        source: TId,
        target: TId,
    ) -> tuple[TId, TId]:
        """
        Normalize undirected edge ordering.
        """

        if (
            self._node_order[source]
            <= self._node_order[target]
        ):
            return (
                source,
                target,
            )

        return (
            target,
            source,
        )

    def _sort_nodes(
        self,
        nodes: Iterable[TId],
    ) -> list[TId]:
        """
        Return nodes in deterministic order.
        """

        return sorted(
            nodes,
            key=lambda node: self._node_order[node],
        )

    def _sort_edges(
        self,
        edges: list[Edge[TId]],
    ) -> list[Edge[TId]]:
        """
        Return edges in deterministic order.
        """

        return sorted(
            edges,
            key=lambda edge: (
                self._node_order[edge.source],
                self._node_order[edge.target],
            ),
        )