"""
RelationSet data model.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Generic

from graphion.core.errors import InvalidRelationSetError
from graphion.core.types import TId

from .relation import Relation


@dataclass(frozen=True, slots=True)
class RelationSet(Generic[TId]):
    """
    Immutable collection of relations.
    """

    relations: tuple[Relation[TId], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "relations", tuple(self.relations))

        seen: set[tuple[TId, TId]] = set()

        for relation in self.relations:
            key = (relation.source, relation.target)

            if key in seen:
                raise InvalidRelationSetError(
                    "Duplicate relations are not allowed."
                )

            seen.add(key)

    def __len__(self) -> int:
        return len(self.relations)

    def __iter__(self) -> Iterator[Relation[TId]]:
        return iter(self.relations)

    @property
    def is_empty(self) -> bool:
        return len(self) == 0