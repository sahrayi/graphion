"""
Partition integrity contract tests.

Tests:

- all nodes are covered
- no node duplication
- no empty partitions
- detector output respects graph node set
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


def build_test_graph() -> Graph[str]:

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


# --------------------------------------------------
# Integrity contract
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
def test_detector_partition_integrity(
    detector,
):

    graph = build_test_graph()

    result = detector.detect(
        graph,
    )

    partitions = result.partitions


    # no empty partition

    assert all(
        len(partition) > 0
        for partition in partitions
    )


    # collect all nodes

    flattened = [
        node
        for partition in partitions
        for node in partition
    ]


    # every node appears exactly once

    assert set(flattened) == set(
        graph.nodes,
    )

    assert len(flattened) == len(
        set(flattened),
    )