"""
Tests for partition detectors.
"""

from __future__ import annotations

import pytest

from graphion.core.models import (
    Edge,
    Graph,
    PartitionSet,
)

from graphion.detectors.partition import (
    Agglomerative,
    ConnectedComponents,
    FastGreedy,
    GirvanNewman,
    IdentityPartitionDetector,
    Infomap,
    LabelPropagation,
    Leiden,
    Louvain,
    Spectral,
    Walktrap,
)


DETECTORS = [
    IdentityPartitionDetector,
    ConnectedComponents,
    FastGreedy,
    GirvanNewman,
    Infomap,
    LabelPropagation,
    Leiden,
    Louvain,
    Walktrap,
    Agglomerative,
    Spectral,
]


def build_test_graph() -> Graph[str]:
    """
    Build deterministic weighted graph.

    Structure:

        A -- B -- C

        D -- E -- F

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
            Edge(
                source="A",
                target="B",
                weight=0.9,
            ),
            Edge(
                source="B",
                target="C",
                weight=0.85,
            ),
            Edge(
                source="D",
                target="E",
                weight=0.9,
            ),
            Edge(
                source="E",
                target="F",
                weight=0.85,
            ),
            Edge(
                source="C",
                target="D",
                weight=0.1,
            ),
        ),
    )


@pytest.mark.parametrize(
    "detector_cls",
    DETECTORS,
)
def test_detector_returns_partition_set(
    detector_cls,
):
    """
    Every detector should return PartitionSet.
    """

    graph = build_test_graph()

    detector = detector_cls()

    result = detector.detect(
        graph,
    )

    assert isinstance(
        result,
        PartitionSet,
    )


@pytest.mark.parametrize(
    "detector_cls",
    DETECTORS,
)
def test_detector_covers_all_nodes(
    detector_cls,
):
    """
    Every node should appear in exactly one partition.
    """

    graph = build_test_graph()

    detector = detector_cls()

    partitions = detector.detect(
        graph,
    )

    detected_nodes = set()

    for partition in partitions:

        detected_nodes.update(
            partition,
        )

    assert detected_nodes == set(
        graph.nodes,
    )


@pytest.mark.parametrize(
    "detector_cls",
    DETECTORS,
)
def test_detector_has_no_overlap(
    detector_cls,
):
    """
    Partitions should not overlap.
    """

    graph = build_test_graph()

    detector = detector_cls()

    partitions = detector.detect(
        graph,
    )

    seen = set()

    for partition in partitions:

        assert not (
            seen &
            set(partition)
        )

        seen.update(
            partition,
        )


@pytest.mark.parametrize(
    "detector_cls",
    DETECTORS,
)
def test_detector_handles_empty_graph(
    detector_cls,
):
    """
    Empty graph should return empty partitions.
    """

    graph = Graph(
        nodes=(),
        edges=(),
    )

    detector = detector_cls()

    partitions = detector.detect(
        graph,
    )

    assert isinstance(
        partitions,
        PartitionSet,
    )

    assert partitions.is_empty


@pytest.mark.parametrize(
    "detector_cls",
    DETECTORS,
)
def test_detector_is_deterministic(
    detector_cls,
):
    """
    Same graph should produce stable result.

    Detectors using randomness should configure
    deterministic behavior internally.
    """

    graph = build_test_graph()

    detector = detector_cls()

    first = detector.detect(
        graph,
    )

    second = detector.detect(
        graph,
    )

    assert first == second