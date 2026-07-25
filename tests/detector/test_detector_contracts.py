"""
Detector contract tests.

Ensures every partition detector
follows Graphora detector contracts.
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
    LabelPropagation,
    Leiden,
    Louvain,
    Spectral,
    Walktrap,
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------


def sample_graph() -> Graph[str]:

    return Graph(
        nodes=(
            "a",
            "b",
            "c",
            "d",
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
                0.8,
            ),
            Edge(
                "c",
                "d",
                0.7,
            ),
            Edge(
                "a",
                "d",
                0.2,
            ),
        ),
    )


def single_node_graph() -> Graph[str]:

    return Graph(
        nodes=(
            "a",
        ),
        edges=(),
    )


def empty_graph() -> Graph[str]:

    return Graph.empty()


# --------------------------------------------------
# Detector registry
# --------------------------------------------------


def detector_cases():

    return [
        pytest.param(
            ConnectedComponents(),
            id="connected_components",
        ),

        pytest.param(
            IdentityPartitionDetector(),
            id="identity",
        ),

        pytest.param(
            LabelPropagation(),
            id="label_propagation",
        ),

        pytest.param(
            Leiden(),
            id="leiden",
        ),

        pytest.param(
            Louvain(),
            id="louvain",
        ),

        pytest.param(
            FastGreedy(),
            id="fast_greedy",
        ),

        pytest.param(
            Walktrap(),
            id="walktrap",
        ),

        pytest.param(
            GirvanNewman(),
            id="girvan_newman",
        ),

        pytest.param(
            Spectral(
                n_clusters=2,
            ),
            id="spectral",
        ),

        pytest.param(
            Agglomerative(
                n_clusters=2,
            ),
            id="agglomerative",
        ),
    ]


# --------------------------------------------------
# Output contract
# --------------------------------------------------


@pytest.mark.parametrize(
    "detector",
    detector_cases(),
)
def test_detector_returns_partition_set(
    detector,
):

    result = detector.detect(
        sample_graph(),
    )

    assert isinstance(
        result,
        PartitionSet,
    )


@pytest.mark.parametrize(
    "detector",
    detector_cases(),
)
def test_detector_partitions_are_immutable(
    detector,
):

    result = detector.detect(
        sample_graph(),
    )

    for partition in result.partitions:

        assert isinstance(
            partition,
            frozenset,
        )


@pytest.mark.parametrize(
    "detector",
    detector_cases(),
)
def test_detector_partitions_cover_all_nodes_once(
    detector,
):

    graph = sample_graph()

    result = detector.detect(
        graph,
    )

    covered = []

    for partition in result.partitions:
        covered.extend(
            partition
        )

    assert set(covered) == set(
        graph.nodes
    )

    assert len(covered) == len(
        set(covered)
    )


# --------------------------------------------------
# Edge cases
# --------------------------------------------------


@pytest.mark.parametrize(
    "detector",
    detector_cases(),
)
def test_detector_handles_single_node_graph(
    detector,
):

    result = detector.detect(
        single_node_graph(),
    )

    assert result.partitions == (
        frozenset(
            {
                "a",
            }
        ),
    )


@pytest.mark.parametrize(
    "detector",
    detector_cases(),
)
def test_detector_handles_empty_graph(
    detector,
):

    result = detector.detect(
        empty_graph(),
    )

    assert result.partitions == ()