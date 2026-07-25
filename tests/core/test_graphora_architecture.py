"""
Graphora architecture and contract tests.

Tests:

- BasePartitionDetector normalization contract
- detector compatibility validation
- partition output contract
- isolated node preservation
- graph backend round trips
- deterministic behavior
- graph validation rules
"""

from __future__ import annotations


import math

import pytest

from graphion.core.errors import (
    InvalidEdgeError,
    InvalidGraphError,
)

from graphion.core.models import (
    Edge,
    Graph,
    PartitionSet,
)

from graphion.detectors.partition.base_partition_detector import (
    BasePartitionDetector,
)

from graphion.detectors.partition.connected_components import (
    ConnectedComponents,
)


# --------------------------------------------------
# Dummy detector for base contract
# --------------------------------------------------


class DummyDetector(
    BasePartitionDetector[str],
):
    """
    Detector used only for testing base behavior.
    """

    def _detect(
        self,
        graph,
    ):

        if graph.is_empty:
            return []

        return [
            {"b", "a"},
            {"c"},
        ]

# --------------------------------------------------
# Helpers
# --------------------------------------------------


def simple_graph() -> Graph[str]:

    return Graph(
        nodes=(
            "a",
            "b",
            "c",
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
        ),
    )


def isolated_graph() -> Graph[str]:

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
        ),
    )


# --------------------------------------------------
# Base detector contract
# --------------------------------------------------


def test_detector_returns_partition_set():

    result = DummyDetector().detect(
        simple_graph(),
    )

    assert isinstance(
        result,
        PartitionSet,
    )


def test_detector_normalizes_output():

    detector = DummyDetector()

    result = detector.detect(
        simple_graph(),
    )

    assert result.partitions == (
        frozenset(
            {
                "a",
                "b",
            }
        ),
        frozenset(
            {
                "c",
            }
        ),
    )


def test_empty_graph_detection():

    result = DummyDetector().detect(
        Graph.empty(),
    )

    assert result.partitions == ()


# --------------------------------------------------
# Directed compatibility
# --------------------------------------------------


def test_base_detector_rejects_unsupported_directed_graph():

    graph = Graph(
        nodes=(
            "a",
            "b",
        ),
        edges=(
            Edge(
                "a",
                "b",
                1.0,
            ),
        ),
        directed=True,
    )

    with pytest.raises(
        InvalidGraphError,
    ):
        DummyDetector().detect(
            graph,
        )


# --------------------------------------------------
# Graph validation
# --------------------------------------------------


def test_graph_rejects_duplicate_nodes():

    with pytest.raises(
        InvalidGraphError,
    ):
        Graph(
            nodes=(
                "a",
                "a",
            ),
            edges=(),
        )


def test_graph_rejects_unknown_edge_nodes():

    with pytest.raises(
        InvalidGraphError,
    ):
        Graph(
            nodes=(
                "a",
            ),
            edges=(
                Edge(
                    "a",
                    "b",
                    1.0,
                ),
            ),
        )


def test_edge_rejects_invalid_weight():

    with pytest.raises(
        InvalidEdgeError,
    ):
        Edge(
            "a",
            "b",
            math.inf,
        )


# --------------------------------------------------
# Isolated nodes
# --------------------------------------------------


def test_connected_components_preserves_isolated_nodes():

    detector = ConnectedComponents()

    result = detector.detect(
        isolated_graph(),
    )

    assert result.partitions == (
        frozenset(
            {
                "a",
                "b",
            }
        ),
        frozenset(
            {
                "c",
            }
        ),
        frozenset(
            {
                "d",
            }
        ),
    )


# --------------------------------------------------
# Backend round trips
# --------------------------------------------------


def test_networkx_round_trip():

    pytest.importorskip(
        "networkx",
    )

    graph = simple_graph()

    restored = Graph.from_networkx(
        graph.to_networkx(),
    )

    assert set(
        restored.nodes,
    ) == set(
        graph.nodes,
    )

    assert restored.edge_count == graph.edge_count

    assert restored.directed == graph.directed



def test_igraph_round_trip():

    pytest.importorskip(
        "igraph",
    )

    graph = simple_graph()

    restored = Graph.from_igraph(
        graph.to_igraph(),
    )

    assert set(
        restored.nodes,
    ) == set(
        graph.nodes,
    )

    assert restored.edge_count == graph.edge_count

    assert restored.directed == graph.directed


# --------------------------------------------------
# Determinism
# --------------------------------------------------


@pytest.mark.parametrize(
    "detector",
    [
        pytest.param(
            ConnectedComponents(),
            id="connected_components",
        ),
    ],
)
def test_detector_is_deterministic(
    detector,
):

    graph = simple_graph()

    first = detector.detect(
        graph,
    )

    for _ in range(5):

        assert detector.detect(
            graph,
        ) == first