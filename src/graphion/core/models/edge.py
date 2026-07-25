"""
Edge data model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Generic

from graphion.core.errors import InvalidEdgeError
from graphion.core.types import TId


@dataclass(frozen=True, slots=True)
class Edge(Generic[TId]):
    """
    Immutable graph edge.

    An edge represents a connection between two nodes in a Graphion graph.

    Parameters
    ----------
    source:
        Identifier of the source node.

    target:
        Identifier of the target node.

    weight:
        Numeric edge weight.

    Notes
    -----
    The interpretation of ``weight`` depends on the GraphBuilder
    that created the graph. It may represent similarity, distance,
    capacity, or another domain-specific value.
    """

    source: TId
    target: TId
    weight: float

    def __post_init__(self) -> None:
        """
        Validate edge invariants.

        Raises
        ------
        InvalidEdgeError
            If endpoints are missing or weight is not finite.
        """
        if self.source is None:
            raise InvalidEdgeError("Edge source cannot be None.")

        if self.target is None:
            raise InvalidEdgeError("Edge target cannot be None.")

        if not isfinite(self.weight):
            raise InvalidEdgeError("Edge weight must be a finite number.")