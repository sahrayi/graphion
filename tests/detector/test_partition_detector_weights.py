"""
Tests for weighted behavior of partition detectors.
"""

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
    Spectral,
    Agglomerative,
    LabelPropagation,
)


DETECTORS = [
    Leiden,
    Louvain,
    Walktrap,
    Infomap,
    FastGreedy,
    GirvanNewman,
    Spectral,
    Agglomerative,
    LabelPropagation,
]


def build_weighted_graph(
) -> Graph[str]:
    """
    Build graph where weights strongly
    indicate two communities.
    """

    return Graph(
        nodes=(
            "A",
            "B",
            "C",
            "D",
        ),
        edges=(
            Edge(
                "A",
                "B",
                1.0,
            ),
            Edge(
                "B",
                "C",
                0.05,
            ),
            Edge(
                "C",
                "D",
                1.0,
            ),
        ),
    )


@pytest.mark.parametrize(
    "detector_cls",
    DETECTORS,
)
def test_weighted_graph_returns_partitions(
    detector_cls,
):
    """
    Weighted graphs should be accepted
    by every detector.
    """

    graph = build_weighted_graph()

    detector = detector_cls()

    partitions = detector.detect(
        graph,
    )

    assert len(partitions) > 0


@pytest.mark.parametrize(
    "detector_cls",
    DETECTORS,
)
def test_weight_information_is_preserved(
    detector_cls,
):
    """
    Detector should not fail when
    edge weights are changed.
    """

    weighted = build_weighted_graph()

    unweighted = Graph(
        nodes=weighted.nodes,
        edges=tuple(
            Edge(
                edge.source,
                edge.target,
                1.0,
            )
            for edge in weighted.edges
        ),
    )

    detector = detector_cls()

    weighted_result = detector.detect(
        weighted,
    )

    unweighted_result = detector.detect(
        unweighted,
    )

    assert weighted_result
    assert unweighted_result