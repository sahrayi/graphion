"""
Edge data model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Generic

from graphora.core.errors import InvalidEdgeError
from graphora.core.types import TId


@dataclass(frozen=True, slots=True)
class Edge(Generic[TId]):
    """
    Immutable graph edge.

    An edge belongs to a Graph and represents the graph topology
    consumed by graph partitioning algorithms.

    The meaning of ``weight`` depends on the GraphBuilder that
    produced the graph.
    """

    source: TId
    target: TId
    weight: float

    def __post_init__(self) -> None:
        """
        Validate edge invariants.
        """
        if not isfinite(self.weight):
            raise InvalidEdgeError(
                "Edge weight must be a finite number."
            )