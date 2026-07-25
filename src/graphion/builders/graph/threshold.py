"""
Threshold based graph construction algorithm.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Generic, TypeVar

from graphion.core.models import Edge

from .base_graph_builder import BaseGraphBuilder

from graphion.core.types import (
    TId,
)


class Threshold(
    BaseGraphBuilder[TId],
    Generic[TId],
):
    """
    Threshold based graph construction.

    Keeps edges whose affinity is above
    a given threshold.

    Conditions:

        inclusive:

            weight >= threshold


        exclusive:

            weight > threshold


    Directionality is controlled by
    BaseGraphBuilder.

    Properties:

    - simple sparsification
    - deterministic
    - parameter controlled


    Pipeline:

    1. Remove self loops.
    2. Merge duplicate edges.
    3. Apply affinity threshold.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        inclusive: bool = True,
        **kwargs,
    ) -> None:

        super().__init__(
            **kwargs,
        )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        self.threshold = threshold
        self.inclusive = inclusive


    def filter_edges(
        self,
        edges: list[Edge[TId]],
    ) -> list[Edge[TId]]:
        """
        Apply threshold filtering.
        """

        edges = self.remove_self_loops(
            edges,
        )

        edges = self.merge_duplicates(
            edges,
        )


        if self.inclusive:

            edges = [
                edge
                for edge in edges
                if edge.weight >= self.threshold
            ]

        else:

            edges = [
                edge
                for edge in edges
                if edge.weight > self.threshold
            ]


        return sorted(
            edges,
            key=lambda edge: (
                str(edge.source),
                str(edge.target),
            ),
        )