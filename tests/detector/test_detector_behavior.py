"""
Detector behavior tests.

Tests actual community detection behavior
on controlled graph structures.
"""

from __future__ import annotations


import pytest

from graphora.core.models import (
    Edge,
    Graph,
)

from graphora.detectors.partition import (
    Agglomerative,
    ConnectedComponents,
    FastGreedy,
    LabelPropagation,
    Leiden,
    Louvain,
    Spectral,
    Walktrap,
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------


def two_community_graph() -> Graph[str]:
    """
    Two disconnected communities.

    Community 1:
        a - b - c

    Community 2:
        d - e - f
    """

    return Graph(
        nodes=(
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
        ),
        edges=(
            Edge("a", "b", 1.0),
            Edge("b", "c", 1.0),

            Edge("d", "e", 1.0),
            Edge("e", "f", 1.0),
        ),
    )


def single_community_graph() -> Graph[str]:
    """
    Dense single community.
    """

    return Graph(
        nodes=(
            "a",
            "b",
            "c",
            "d",
        ),
        edges=(
            Edge("a", "b", 1.0),
            Edge("a", "c", 1.0),
            Edge("a", "d", 1.0),
            Edge("b", "c", 1.0),
            Edge("b", "d", 1.0),
            Edge("c", "d", 1.0),
        ),
    )


def weighted_graph() -> Graph[str]:

    return Graph(
        nodes=(
            "a",
            "b",
            "c",
        ),
        edges=(
            Edge(
                "a",
                "b",
                1.0,
            ),
            Edge(
                "a",
                "c",
                0.01,
            ),
        ),
    )


# --------------------------------------------------
# Detector groups
# --------------------------------------------------


@pytest.mark.parametrize(
    "detector",
    [
        ConnectedComponents(),
        LabelPropagation(),
        Leiden(),
        Louvain(),
        FastGreedy(),
        Walktrap(),
    ],
)
def test_detectors_find_disconnected_communities(
    detector,
):

    result = detector.detect(
        two_community_graph(),
    )

    assert len(
        result.partitions
    ) == 2


@pytest.mark.parametrize(
    "detector",
    [
        Spectral(
            n_clusters=2,
        ),
        Agglomerative(
            n_clusters=2,
        ),
    ],
)
def test_fixed_cluster_detectors_respect_cluster_count(
    detector,
):

    result = detector.detect(
        two_community_graph(),
    )

    assert len(
        result.partitions
    ) == 2


# --------------------------------------------------
# Single community behavior
# --------------------------------------------------


@pytest.mark.parametrize(
    "detector",
    [
        Leiden(),
        Louvain(),
        FastGreedy(),
        Walktrap(),
    ],
)
def test_dense_graph_is_single_community(
    detector,
):

    result = detector.detect(
        single_community_graph(),
    )

    assert len(
        result.partitions
    ) == 1


# --------------------------------------------------
# Determinism
# --------------------------------------------------


@pytest.mark.parametrize(
    "detector",
    [
        Leiden(),
        Louvain(),
        FastGreedy(),
        Walktrap(),
        LabelPropagation(),
    ],
)
def test_weighted_detectors_are_deterministic(
    detector,
):

    graph = two_community_graph()

    first = detector.detect(
        graph,
    )

    for _ in range(3):

        assert detector.detect(
            graph,
        ) == first


# --------------------------------------------------
# Weight sanity
# --------------------------------------------------


def test_weighted_graph_preserves_strong_connection():

    detector = ConnectedComponents()

    result = detector.detect(
        weighted_graph(),
    )

    assert result.partitions == (
        frozenset(
            {
                "a",
                "b",
                "c",
            }
        ),
    )