"""
Relation data model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Generic

from graphora.core.errors import InvalidRelationError
from graphora.core.types import TId


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
        """
        if not isfinite(self.weight):
            raise InvalidRelationError(
                "Relation weight must be a finite number."
            )