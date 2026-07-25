from __future__ import annotations

import pytest

from graphion.core.models import (
    Edge,
    Graph,
)

from graphion.detectors.partition import (
    Leiden,
    Louvain,
    Walktrap,
    Infomap,
    FastGreedy,
    GirvanNewman,
    LabelPropagation,
    Spectral,
    Agglomerative,
)


DETECTORS = [
    Leiden,
    Louvain,
    Walktrap,
    Infomap,
    FastGreedy,
    GirvanNewman,
    LabelPropagation,
    Spectral,
    Agglomerative,
]


def build_two_community_graph() -> Graph[str]:
    """
    Graph with two dense communities
    connected by a weak bridge.
    """

    nodes = (
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
    )

    edges = []

    community_a = [
        ("A", "B"),
        ("B", "C"),
        ("A", "D"),
        ("D", "E"),
        ("E", "F"),
        ("B", "E"),
        ("C", "F"),
    ]

    community_b = [
        ("G", "H"),
        ("H", "I"),
        ("G", "J"),
        ("J", "K"),
        ("K", "L"),
        ("H", "K"),
        ("I", "L"),
    ]

    for source, target in (
        community_a + community_b
    ):
        edges.append(
            Edge(
                source=source,
                target=target,
                weight=0.9,
            )
        )

    edges.append(
        Edge(
            source="F",
            target="G",
            weight=0.1,
        )
    )

    return Graph(
        nodes=nodes,
        edges=tuple(edges),
    )


@pytest.mark.parametrize(
    "detector_cls",
    DETECTORS,
)
def test_detector_finds_multiple_communities(
    detector_cls,
):
    """
    Community detectors should not collapse
    clearly separated communities into one.
    """

    graph = build_two_community_graph()

    detector = detector_cls()

    partitions = detector.detect(
        graph,
    )

    assert partitions.partition_count >= 2


@pytest.mark.parametrize(
    "detector_cls",
    DETECTORS,
)
def test_detector_keeps_dense_communities_together(
    detector_cls,
):
    """
    Nodes inside dense communities should
    mostly stay together.
    """

    graph = build_two_community_graph()

    detector = detector_cls()

    partitions = detector.detect(
        graph,
    )

    community_a = {
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    }

    community_b = {
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
    }

    detected = [
        set(partition)
        for partition in partitions
    ]

    assert any(
        len(partition & community_a) >= 4
        for partition in detected
    )

    assert any(
        len(partition & community_b) >= 4
        for partition in detected
    )