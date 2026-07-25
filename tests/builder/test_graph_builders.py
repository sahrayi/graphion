"""
Tests for graph builders.
"""

from __future__ import annotations

import pytest

from graphion.builders.graph import (
    AdaptiveKNN,
    KNN,
    KNNThreshold,
    MST,
    MutualKNN,
    Radius,
    SNN,
    SymmetricKNN,
    Threshold,
    WeightedKNN,
)

from graphion.core.models import Relation


# ==================================================
# Dummy relation builder
# ==================================================


class DummyRelationBuilder:
    """
    Dummy affinity converter for tests.
    """

    def affinity(
        self,
        raw_score: float,
    ) -> float:
        return raw_score


# ==================================================
# Test relations
# ==================================================


def build_test_relations():
    """
    Create deterministic relation set.
    """

    return [
        Relation(
            source="A",
            target="B",
            weight=0.9,
        ),
        Relation(
            source="A",
            target="C",
            weight=0.8,
        ),
        Relation(
            source="A",
            target="D",
            weight=0.2,
        ),

        Relation(
            source="B",
            target="A",
            weight=0.9,
        ),
        Relation(
            source="B",
            target="C",
            weight=0.7,
        ),
        Relation(
            source="B",
            target="D",
            weight=0.1,
        ),

        Relation(
            source="C",
            target="A",
            weight=0.8,
        ),
        Relation(
            source="C",
            target="B",
            weight=0.7,
        ),
        Relation(
            source="C",
            target="D",
            weight=0.4,
        ),

        Relation(
            source="D",
            target="A",
            weight=0.2,
        ),
        Relation(
            source="D",
            target="C",
            weight=0.4,
        ),
    ]


# ==================================================
# Builder configurations
# ==================================================


BUILDER_CONFIGS = [
    (
        KNN,
        {
            "k": 2,
        },
    ),

    (
        WeightedKNN,
        {
            "k": 2,
        },
    ),

    (
        MutualKNN,
        {
            "k": 2,
        },
    ),

    (
        SymmetricKNN,
        {
            "k": 2,
        },
    ),

    (
        AdaptiveKNN,
        {
            "min_k": 1,
            "max_k": 3,
        },
    ),

    (
        KNNThreshold,
        {
            "k": 2,
            "threshold": 0.5,
        },
    ),

    (
        Threshold,
        {
            "threshold": 0.5,
        },
    ),

    (
        Radius,
        {
            "radius": 0.5,
        },
    ),

    (
        SNN,
        {
            "k": 2,
        },
    ),

    (
        MST,
        {},
    ),
]


# ==================================================
# Helpers
# ==================================================


def create_builder(
    builder_cls,
    kwargs,
):
    """
    Create builder instance.
    """

    return builder_cls(
        relation_builder=DummyRelationBuilder(),
        **kwargs,
    )


# ==================================================
# Tests
# ==================================================


@pytest.mark.parametrize(
    "builder_cls,kwargs",
    BUILDER_CONFIGS,
)
def test_graph_builder_builds_graph(
    builder_cls,
    kwargs,
):
    """
    Every graph builder should create a graph.
    """

    relations = build_test_relations()

    builder = create_builder(
        builder_cls,
        kwargs,
    )

    graph = builder.build(
        relations,
    )

    assert graph is not None
    assert len(graph.nodes) > 0


@pytest.mark.parametrize(
    "builder_cls,kwargs",
    BUILDER_CONFIGS,
)
def test_graph_builder_has_no_self_loops(
    builder_cls,
    kwargs,
):
    """
    Graph builders should not create self loops.
    """

    relations = build_test_relations()

    builder = create_builder(
        builder_cls,
        kwargs,
    )

    graph = builder.build(
        relations,
    )

    for edge in graph.edges:
        assert edge.source != edge.target


@pytest.mark.parametrize(
    "builder_cls,kwargs",
    BUILDER_CONFIGS,
)
def test_graph_builder_deterministic(
    builder_cls,
    kwargs,
):
    """
    Same input should generate same graph.
    """

    relations = build_test_relations()

    builder = create_builder(
        builder_cls,
        kwargs,
    )

    graph1 = builder.build(
        relations,
    )

    graph2 = builder.build(
        relations,
    )

    assert graph1.nodes == graph2.nodes
    assert graph1.edges == graph2.edges


@pytest.mark.parametrize(
    "builder_cls,kwargs",
    BUILDER_CONFIGS,
)
def test_graph_edges_have_valid_weights(
    builder_cls,
    kwargs,
):
    """
    Edge weights must be valid affinities.
    """

    relations = build_test_relations()

    builder = create_builder(
        builder_cls,
        kwargs,
    )

    graph = builder.build(
        relations,
    )

    for edge in graph.edges:
        assert 0.0 <= edge.weight <= 1.0