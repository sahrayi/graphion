"""
Tests for RelationSet model.
"""

from dataclasses import FrozenInstanceError

import pytest

from graphion.core.errors import InvalidRelationSetError
from graphion.core.models import Relation, RelationSet


def test_relation_set_creation() -> None:
    """
    RelationSet should be created successfully with valid relations.
    """
    relation_set = RelationSet(
        relations=[
            Relation(
                source=1,
                target=2,
                weight=0.8,
            ),
            Relation(
                source=2,
                target=3,
                weight=0.6,
            ),
        ]
    )

    assert len(relation_set) == 2
    assert isinstance(relation_set.relations, tuple)


def test_relation_set_is_immutable() -> None:
    """
    RelationSet should be immutable.
    """
    relation_set = RelationSet(
        relations=(
            Relation(
                source=1,
                target=2,
                weight=0.5,
            ),
        )
    )

    with pytest.raises(FrozenInstanceError):
        relation_set.relations = ()


def test_relation_set_converts_input_to_tuple() -> None:
    """
    Mutable input collections should not be stored directly.
    """
    relations = [
        Relation(
            source=1,
            target=2,
            weight=0.7,
        )
    ]

    relation_set = RelationSet(
        relations=relations,
    )

    relations.append(
        Relation(
            source=2,
            target=3,
            weight=0.4,
        )
    )

    assert len(relation_set) == 1


def test_relation_set_rejects_duplicate_relations() -> None:
    """
    RelationSet should reject duplicate source-target pairs.
    """
    with pytest.raises(InvalidRelationSetError):
        RelationSet(
            relations=(
                Relation(
                    source=1,
                    target=2,
                    weight=0.8,
                ),
                Relation(
                    source=1,
                    target=2,
                    weight=0.5,
                ),
            )
        )


def test_relation_set_iteration() -> None:
    """
    Iteration should return Relation objects.
    """
    relation_1 = Relation(
        source=1,
        target=2,
        weight=0.8,
    )

    relation_2 = Relation(
        source=2,
        target=3,
        weight=0.6,
    )

    relation_set = RelationSet(
        relations=(
            relation_1,
            relation_2,
        )
    )

    assert list(relation_set) == [
        relation_1,
        relation_2,
    ]


def test_empty_relation_set() -> None:
    """
    Empty RelationSet should be supported.
    """
    relation_set = RelationSet(
        relations=(),
    )

    assert len(relation_set) == 0
    assert relation_set.is_empty is True