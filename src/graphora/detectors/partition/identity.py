"""
Identity partition detector.

Creates one partition per graph node.
"""

from __future__ import annotations

from collections.abc import Hashable

from typing import (
    Generic,
    TypeVar,
)

from graphora.core.models import Graph

from .base_partition_detector import (
    BasePartitionDetector,
)

from graphora.core.types import (
    TId,
)


class IdentityPartitionDetector(
    BasePartitionDetector[TId],
    Generic[TId],
):
    """
    Identity partition detector.

    Each node becomes its own partition.

    This detector performs no clustering.


    Supported topology:

    - directed
    - undirected


    Useful for:

    - pipeline defaults
    - testing
    - debugging
    - disabling community detection
    """


    # --------------------------------------------------
    # Capability
    # --------------------------------------------------

    @property
    def supports_directed(
        self,
    ) -> bool:

        return True


    @property
    def supports_undirected(
        self,
    ) -> bool:

        return True


    # --------------------------------------------------
    # Detection
    # --------------------------------------------------

    def _detect(
        self,
        graph: Graph[TId],
    ) -> list[set[TId]]:
        """
        Create singleton partitions.
        """

        return [
            {
                node,
            }
            for node in graph.nodes
        ]