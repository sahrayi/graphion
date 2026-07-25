"""
Adaptive K-nearest neighbors graph construction algorithm.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Generic

from graphora.core.models import Edge
from graphora.core.types import TId

from .base_graph_builder import BaseGraphBuilder


class AdaptiveKNN(BaseGraphBuilder[TId], Generic[TId]):
    """
    Adaptive K-nearest neighbors graph construction.

    Selects a variable number of neighbors
    per node based on local affinity density.

    Dense regions:
        fewer neighbors

    Sparse regions:
        more neighbors

    Topology modes:
    - Default: directed adaptive KNN
    - mutual=True: reciprocal adaptive KNN
    - symmetric=True: undirected symmetric adaptive KNN
    """

    def __init__(
        self,
        *,
        min_k: int = 5,
        max_k: int = 20,
        density_threshold: float = 0.7,
        mutual: bool = False,
        symmetric: bool = False,
        **kwargs,
    ) -> None:
        if mutual and symmetric:
            raise ValueError("mutual and symmetric cannot be enabled together.")

        super().__init__(directed=not symmetric, **kwargs)

        if min_k <= 0:
            raise ValueError("min_k must be greater than zero.")
        if max_k < min_k:
            raise ValueError("max_k must be greater than or equal to min_k.")
        if not 0.0 <= density_threshold <= 1.0:
            raise ValueError("density_threshold must be between 0 and 1.")

        self.min_k = min_k
        self.max_k = max_k
        self.density_threshold = density_threshold
        self.mutual = mutual
        self.symmetric = symmetric

    def filter_edges(self, edges: list[Edge[TId]]) -> list[Edge[TId]]:
        """
        Apply adaptive KNN sparsification.

        Pipeline:
        1. Remove self loops.
        2. Merge duplicate directed edges.
        3. Select adaptive neighbors.
        4. Apply mutual filtering.
        5. Apply symmetry conversion.
        """
        edges = self.remove_self_loops(edges)
        edges = self.merge_duplicates(edges)

        selected = self._select_adaptive_neighbors(edges)

        if self.mutual:
            selected = self.apply_mutual(selected)

        if self.symmetric:
            selected = self.make_symmetric(selected)

        return sorted(
            selected,
            key=lambda edge: (
                self.sort_key(edge.source),
                self.sort_key(edge.target),
            ),
        )

    def _select_adaptive_neighbors(self, edges: list[Edge[TId]]) -> list[Edge[TId]]:
        """
        Select adaptive number of neighbors per node.
        """
        outgoing: dict[TId, list[Edge[TId]]] = defaultdict(list)

        for edge in edges:
            outgoing[edge.source].append(edge)

        selected: list[Edge[TId]] = []

        for node in sorted(outgoing, key=self.sort_key):
            neighbors = sorted(
                outgoing[node],
                key=lambda edge: (-edge.weight, self.sort_key(edge.target)),
            )
            k = self._adaptive_k(neighbors)
            selected.extend(neighbors[:k])

        return selected

    def _adaptive_k(self, neighbors: list[Edge[TId]]) -> int:
        """
        Estimate local neighborhood size.

        Dense neighborhood: high strong-edge ratio -> smaller k
        Sparse neighborhood: low strong-edge ratio -> larger k
        """
        if not neighbors:
            return self.min_k

        strong_count = sum(
            1 for edge in neighbors if edge.weight >= self.density_threshold
        )
        density = strong_count / len(neighbors)
        k_range = self.max_k - self.min_k

        adaptive_k = self.max_k - int(density * k_range)

        return max(
            self.min_k,
            min(adaptive_k, self.max_k),
        )