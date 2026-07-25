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
    ConnectedComponents,
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
    ConnectedComponents,
]


def assert_valid_partitions(
    graph: Graph,
    partitions,
):
    """
    Common partition validation.
    """

    assert partitions.partition_count > 0

    all_nodes = set()

    for partition in partitions:
        all_nodes.update(
            partition,
        )

    assert all_nodes == set(
        graph.nodes,
    )


def test_empty_graph():
    """
    Empty graph should return empty partitions.
    """

    graph = Graph(
        nodes=(),
        edges=(),
    )

    for detector_cls in DETECTORS:

        detector = detector_cls()

        partitions = detector.detect(
            graph,
        )

        assert partitions.is_empty


@pytest.mark.parametrize(
    "detector_cls",
    DETECTORS,
)
def test_single_node_graph(
    detector_cls,
):
    """
    Single isolated node should remain
    as a partition.
    """

    graph = Graph(
        nodes=(
            "A",
        ),
        edges=(),
    )

    detector = detector_cls()

    partitions = detector.detect(
        graph,
    )

    assert_valid_partitions(
        graph,
        partitions,
    )

    assert partitions.partition_count == 1

    assert partitions.partitions[0] == frozenset(
        {"A"},
    )


@pytest.mark.parametrize(
    "detector_cls",
    DETECTORS,
)
def test_disconnected_components(
    detector_cls,
):
    """
    Disconnected groups should not be
    merged into one community.
    """

    graph = Graph(
        nodes=(
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
        ),
        edges=(
            Edge(
                source="A",
                target="B",
                weight=1.0,
            ),
            Edge(
                source="B",
                target="C",
                weight=1.0,
            ),
            Edge(
                source="D",
                target="E",
                weight=1.0,
            ),
            Edge(
                source="E",
                target="F",
                weight=1.0,
            ),
        ),
    )

    detector = detector_cls()

    partitions = detector.detect(
        graph,
    )

    assert_valid_partitions(
        graph,
        partitions,
    )

    assert partitions.partition_count >= 2


@pytest.mark.parametrize(
    "detector_cls",
    DETECTORS,
)
def test_complete_graph(
    detector_cls,
):
    """
    Complete graph should not explode
    into many communities.
    """

    nodes = (
        "A",
        "B",
        "C",
        "D",
    )

    edges = []

    for i, source in enumerate(nodes):
        for target in nodes[i + 1:]:
            edges.append(
                Edge(
                    source=source,
                    target=target,
                    weight=1.0,
                )
            )

    graph = Graph(
        nodes=nodes,
        edges=tuple(edges),
    )

    detector = detector_cls()

    partitions = detector.detect(
        graph,
    )

    assert_valid_partitions(
        graph,
        partitions,
    )

    assert partitions.partition_count <= 2


@pytest.mark.parametrize(
    "detector_cls",
    DETECTORS,
)
def test_weighted_graph_preserves_nodes(
    detector_cls,
):
    """
    Weighted graphs should still produce
    valid complete partitions.
    """

    graph = Graph(
        nodes=(
            "A",
            "B",
            "C",
            "D",
        ),
        edges=(
            Edge(
                source="A",
                target="B",
                weight=0.9,
            ),
            Edge(
                source="B",
                target="C",
                weight=0.1,
            ),
            Edge(
                source="C",
                target="D",
                weight=0.8,
            ),
        ),
    )

    detector = detector_cls()

    partitions = detector.detect(
        graph,
    )

    assert_valid_partitions(
        graph,
        partitions,
    )