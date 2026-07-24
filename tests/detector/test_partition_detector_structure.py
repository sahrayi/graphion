"""
Structural behavior tests for partition detectors.
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
    GirvanNewman,
    Infomap,
    LabelPropagation,
    Leiden,
    Louvain,
    Spectral,
    Walktrap,
)


COMMUNITY_DETECTORS = [
    FastGreedy,
    GirvanNewman,
    Infomap,
    LabelPropagation,
    Leiden,
    Louvain,
    Spectral,
    Walktrap,
    Agglomerative,
]


def build_two_community_graph() -> Graph[str]:
    """
    Build graph with two dense regions
    connected by a weak bridge.
    """

    return Graph(
        nodes=(
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
        ),
        edges=(
            # dense region 1
            Edge(
                "A",
                "B",
                0.9,
            ),
            Edge(
                "B",
                "C",
                0.9,
            ),
            Edge(
                "A",
                "C",
                0.85,
            ),

            # dense region 2
            Edge(
                "D",
                "E",
                0.9,
            ),
            Edge(
                "E",
                "F",
                0.9,
            ),
            Edge(
                "D",
                "F",
                0.85,
            ),

            # weak connection
            Edge(
                "C",
                "D",
                0.01,
            ),
        ),
    )


@pytest.mark.parametrize(
    "detector_cls",
    COMMUNITY_DETECTORS,
)
def test_weak_bridge_does_not_collapse_graph(
    detector_cls,
):
    """
    Weak bridges should allow more than
    one partition to be detected.
    """

    graph = build_two_community_graph()

    detector = detector_cls()

    partitions = detector.detect(
        graph,
    )

    assert len(partitions) >= 2


@pytest.mark.parametrize(
    "detector_cls",
    COMMUNITY_DETECTORS,
)
def test_partition_structure_covers_all_nodes(
    detector_cls,
):
    """
    Detector output should form a valid partition.

    Requirements:
    - every node appears
    - no node appears twice
    """

    graph = build_two_community_graph()

    detector = detector_cls()

    partitions = detector.detect(
        graph,
    )

    flattened = [
        node
        for partition in partitions
        for node in partition
    ]

    assert set(flattened) == set(
        graph.nodes,
    )

    assert len(flattened) == len(
        set(flattened),
    )


@pytest.mark.parametrize(
    "detector_cls",
    COMMUNITY_DETECTORS,
)
def test_dense_groups_have_internal_cohesion(
    detector_cls,
):
    """
    Dense groups should not be completely
    destroyed by partitioning.

    The exact partition boundaries are
    algorithm dependent.
    """

    graph = build_two_community_graph()

    detector = detector_cls()

    partitions = detector.detect(
        graph,
    )

    left_group = {
        "A",
        "B",
        "C",
    }

    right_group = {
        "D",
        "E",
        "F",
    }

    left_overlap = max(
        len(
            partition.intersection(
                left_group,
            )
        )
        for partition in partitions
    )

    right_overlap = max(
        len(
            partition.intersection(
                right_group,
            )
        )
        for partition in partitions
    )

    assert left_overlap >= 2

    assert right_overlap >= 2