"""
Degenerate graph behavior tests.

Tests:

- empty graph handling
- single node handling
- disconnected nodes handling
- self-loop tolerance
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


def empty_graph() -> Graph[str]:

    return Graph(
        nodes=(),
        edges=(),
    )


def single_node_graph() -> Graph[str]:

    return Graph(
        nodes=(
            "a",
        ),
        edges=(),
    )


def disconnected_graph() -> Graph[str]:

    return Graph(
        nodes=(
            "a",
            "b",
        ),
        edges=(),
    )


def self_loop_graph() -> Graph[str]:

    return Graph(
        nodes=(
            "a",
            "b",
        ),
        edges=(
            Edge(
                "a",
                "a",
                1.0,
            ),
            Edge(
                "a",
                "b",
                0.8,
            ),
        ),
    )


def normalize(
    partitions,
):
    return {
        frozenset(
            partition,
        )
        for partition in partitions
    }


# --------------------------------------------------
# Detector set
# --------------------------------------------------


@pytest.fixture
def detectors():

    return [
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
    ]


# --------------------------------------------------
# Empty graph
# --------------------------------------------------


def test_detectors_handle_empty_graph(
    detectors,
):

    for detector in detectors:

        result = detector.detect(
            empty_graph(),
        )

        assert result.partitions == ()


# --------------------------------------------------
# Single node
# --------------------------------------------------


def test_detectors_handle_single_node_graph(
    detectors,
):

    graph = single_node_graph()


    for detector in detectors:

        result = detector.detect(
            graph,
        )


        assert result.partitions == (
            frozenset(
                {
                    "a",
                }
            ),
        )


# --------------------------------------------------
# Disconnected nodes
# --------------------------------------------------


def test_detectors_preserve_disconnected_nodes(
    detectors,
):

    graph = disconnected_graph()


    expected = {
        frozenset(
            {
                "a",
            }
        ),
        frozenset(
            {
                "b",
            }
        ),
    }


    for detector in detectors:

        result = detector.detect(
            graph,
        )


        assert normalize(
            result.partitions,
        ) == expected


# --------------------------------------------------
# Self loop
# --------------------------------------------------


def test_detectors_tolerate_self_loops(
    detectors,
):

    graph = self_loop_graph()


    for detector in detectors:

        result = detector.detect(
            graph,
        )


        flattened = [
            node
            for partition in result.partitions
            for node in partition
        ]


        assert set(flattened) == {
            "a",
            "b",
        }

        assert len(flattened) == 2