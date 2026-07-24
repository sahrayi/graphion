"""
Partition count contract tests.

Tests:

- detectors respect requested partition count
- no empty partitions are produced
- all graph nodes remain covered
"""

from __future__ import annotations


import pytest

from graphora.core.models import (
    Edge,
    Graph,
)

from graphora.detectors.partition import (
    Agglomerative,
    FastGreedy,
    Spectral,
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------


def cluster_graph() -> Graph[str]:

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
                1.0,
            ),

            Edge(
                "b",
                "c",
                1.0,
            ),

            Edge(
                "d",
                "e",
                1.0,
            ),

            Edge(
                "e",
                "f",
                1.0,
            ),

            Edge(
                "c",
                "d",
                0.05,
            ),
        ),
    )


# --------------------------------------------------
# Helpers
# --------------------------------------------------


def assert_partition_integrity(
    graph,
    partitions,
):

    flattened = [
        node
        for partition in partitions
        for node in partition
    ]


    assert all(
        len(partition) > 0
        for partition in partitions
    )


    assert set(flattened) == set(
        graph.nodes,
    )


    assert len(flattened) == len(
        set(flattened),
    )


# --------------------------------------------------
# Exact partition count
# --------------------------------------------------


@pytest.mark.parametrize(
    "detector",
    [
        Spectral(
            n_clusters=2,
        ),

        Agglomerative(
            n_clusters=2,
        ),

        FastGreedy(
            target_partitions=2,
        ),
    ],
)
def test_detectors_respect_requested_partition_count(
    detector,
):

    graph = cluster_graph()


    result = detector.detect(
        graph,
    )


    assert len(
        result.partitions,
    ) == 2


    assert_partition_integrity(
        graph,
        result.partitions,
    )