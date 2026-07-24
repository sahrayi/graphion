"""
Tests for Edge model.
"""

from dataclasses import FrozenInstanceError
from math import inf, nan

import pytest

from graphora.core.errors import InvalidEdgeError
from graphora.core.models import Edge


def test_edge_creation() -> None:
    """
    Edge should be created successfully with valid values.
    """
    edge = Edge(
        source=1,
        target=2,
        weight=0.75,
    )

    assert edge.source == 1
    assert edge.target == 2
    assert edge.weight == 0.75


def test_edge_is_immutable() -> None:
    """
    Edge should be immutable.
    """
    edge = Edge(
        source=1,
        target=2,
        weight=0.5,
    )

    with pytest.raises(FrozenInstanceError):
        edge.weight = 1.0


@pytest.mark.parametrize(
    "weight",
    [
        nan,
        inf,
        -inf,
    ],
)
def test_edge_rejects_non_finite_weight(
    weight: float,
) -> None:
    """
    Edge should reject NaN and infinite weights.
    """
    with pytest.raises(InvalidEdgeError):
        Edge(
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
        42.5,
    ],
)
def test_edge_accepts_any_finite_weight(
    weight: float,
) -> None:
    """
    Edge should accept any finite weight.

    Weight semantics are defined by the graph builder.
    """
    edge = Edge(
        source=1,
        target=2,
        weight=weight,
    )

    assert edge.weight == weight