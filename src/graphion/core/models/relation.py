"""
Relation data model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Generic

from graphion.core.errors import InvalidRelationError
from graphion.core.types import TId


@dataclass(frozen=True, slots=True)
class Relation(Generic[TId]):
    """
    Immutable relation between two entities.

    A relation is produced by a RelationBuilder and represents a
    measurable relationship between two entities.

    The meaning of ``weight`` depends on the algorithm that produced
    the relation (e.g. similarity, distance, affinity, correlation).
    """

    source: TId
    target: TId
    weight: float

    def __post_init__(self) -> None:
        """
        Validate relation invariants.

        Raises
        ------
        InvalidRelationError
            If endpoints are missing or weight is not finite.
        """
        if self.source is None:
            raise InvalidRelationError("Relation source cannot be None.")

        if self.target is None:
            raise InvalidRelationError("Relation target cannot be None.")

        if not isfinite(self.weight):
            raise InvalidRelationError("Relation weight must be a finite number.")