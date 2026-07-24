"""
Detector order invariance tests.

Tests:

- node ordering does not affect partitions
- edge ordering does not affect partitions
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


def base_graph() -> Graph[str]:

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
            Edge(
                "a",
                "b",
                0.9,
            ),
            Edge(
                "b",
                "c",
                0.8,
            ),
            Edge(
                "d",
                "e",
                0.95,
            ),
            Edge(
                "e",
                "f",
                0.7,
            ),
            Edge(
                "c",
                "d",
                0.2,
            ),
        ),
    )


def reordered_node_graph() -> Graph[str]:

    return Graph(
        nodes=(
            "f",
            "e",
            "d",
            "c",
            "b",
            "a",
        ),
        edges=(
            Edge(
                "a",
                "b",
                0.9,
            ),
            Edge(
                "b",
                "c",
                0.8,
            ),
            Edge(
                "d",
                "e",
                0.95,
            ),
            Edge(
                "e",
                "f",
                0.7,
            ),
            Edge(
                "c",
                "d",
                0.2,
            ),
        ),
    )


def reordered_edge_graph() -> Graph[str]:

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
            Edge(
                "c",
                "d",
                0.2,
            ),
            Edge(
                "e",
                "f",
                0.7,
            ),
            Edge(
                "d",
                "e",
                0.95,
            ),
            Edge(
                "b",
                "c",
                0.8,
            ),
            Edge(
                "a",
                "b",
                0.9,
            ),
        ),
    )


def normalize_partitions(
    partitions,
):
    """
    Ignore partition ordering.
    """

    return {
        frozenset(
            partition,
        )
        for partition in partitions
    }


# --------------------------------------------------
# Tests
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

        Spectral(
            n_clusters=2,
        ),

        Agglomerative(
            n_clusters=2,
        ),
    ],
)
def test_detector_is_invariant_to_node_order(
    detector,
):

    expected = normalize_partitions(
        detector.detect(
            base_graph(),
        ).partitions
    )

    actual = normalize_partitions(
        detector.detect(
            reordered_node_graph(),
        ).partitions
    )

    assert actual == expected



@pytest.mark.parametrize(
    "detector",
    [
        ConnectedComponents(),

        LabelPropagation(),

        Leiden(),

        Louvain(),

        FastGreedy(),

        Walktrap(),

        Spectral(
            n_clusters=2,
        ),

        Agglomerative(
            n_clusters=2,
        ),
    ],
)
def test_detector_is_invariant_to_edge_order(
    detector,
):

    expected = normalize_partitions(
        detector.detect(
            base_graph(),
        ).partitions
    )

    actual = normalize_partitions(
        detector.detect(
            reordered_edge_graph(),
        ).partitions
    )

    assert actual == expected