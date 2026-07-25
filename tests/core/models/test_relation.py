"""
Tests for Relation model.
"""

from dataclasses import FrozenInstanceError
from math import inf, nan

import pytest

from graphion.core.errors import InvalidRelationError
from graphion.core.models import Relation


def test_relation_creation() -> None:
    """
    Relation should be created successfully with valid values.
    """
    relation = Relation(
        source=1,
        target=2,
        weight=0.85,
    )

    assert relation.source == 1
    assert relation.target == 2
    assert relation.weight == 0.85


def test_relation_is_immutable() -> None:
    """
    Relation should be immutable.
    """
    relation = Relation(
        source=1,
        target=2,
        weight=0.5,
    )

    with pytest.raises(FrozenInstanceError):
        relation.weight = 0.9


@pytest.mark.parametrize(
    "weight",
    [
        nan,
        inf,
        -inf,
    ],
)
def test_relation_rejects_non_finite_weight(
    weight: float,
) -> None:
    """
    Relation should reject NaN and infinite weights.
    """
    with pytest.raises(InvalidRelationError):
        Relation(
            source=1,
            target=2,
            weight=weight,
        )


@pytest.mark.parametrize(
    "weight",
    [
        0.0,
        1.0,
        -1.0,
        100.5,
    ],
)
def test_relation_accepts_any_finite_weight(
    weight: float,
) -> None:
    """
    Relation should accept any finite numeric weight.

    The meaning and range of weight is defined by the algorithm
    that produces the relation.
    """
    relation = Relation(
        source=1,
        target=2,
        weight=weight,
    )

    assert relation.weight == weight