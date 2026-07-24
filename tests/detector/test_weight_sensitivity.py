"""
Detector edge weight sensitivity tests.

Tests:

- weighted detectors react to affinity changes
- weak and strong bridges produce different structures
"""

from __future__ import annotations


import pytest

from graphora.core.models import (
    Edge,
    Graph,
)

from graphora.detectors.partition import (
    FastGreedy,
    Leiden,
    Louvain,
    Walktrap,
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------


def weighted_bridge_graph(
    bridge_weight: float,
) -> Graph[str]:

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
            # first community
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

            # second community
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

            # bridge
            Edge(
                "c",
                "d",
                bridge_weight,
            ),
        ),
    )


# --------------------------------------------------
# Weight sensitivity
# --------------------------------------------------


@pytest.mark.parametrize(
    "detector_factory",
    [
        Leiden,
        Louvain,
        FastGreedy,
        Walktrap,
    ],
)
def test_detectors_are_sensitive_to_edge_weights(
    detector_factory,
):

    weak_graph = weighted_bridge_graph(
        0.01,
    )

    strong_graph = weighted_bridge_graph(
        1.0,
    )


    weak_result = detector_factory().detect(
        weak_graph,
    )

    strong_result = detector_factory().detect(
        strong_graph,
    )


    assert (
        weak_result.partitions
        != strong_result.partitions
    )


    assert (
        weak_result.partitions
        != strong_result.partitions
    )