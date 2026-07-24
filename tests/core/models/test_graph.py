"""
Tests for Graph model.
"""

from dataclasses import FrozenInstanceError

import pytest

from graphora.core.errors import InvalidGraphError
from graphora.core.models import Edge, Graph


def test_graph_creation() -> None:
    """
    Graph should be created successfully with valid data.
    """
    graph = Graph(
        nodes=[1, 2, 3],
        edges=[
            Edge(
                source=1,
                target=2,
                weight=0.8,
            ),
            Edge(
                source=2,
                target=3,
                weight=0.6,
            ),
        ],
    )

    assert graph.nodes == (1, 2, 3)
    assert len(graph.edges) == 2
    assert graph.node_count == 3
    assert graph.edge_count == 2


def test_graph_is_immutable() -> None:
    """
    Graph should be immutable.
    """
    graph = Graph(
        nodes=(1, 2),
        edges=(
            Edge(
                source=1,
                target=2,
                weight=0.5,
            ),
        ),
    )

    with pytest.raises(FrozenInstanceError):
        graph.nodes = (3, 4)


def test_graph_converts_inputs_to_tuple() -> None:
    """
    Mutable input collections should not be stored directly.
    """
    nodes = [1, 2]
    edges = [
        Edge(
            source=1,
            target=2,
            weight=0.5,
        )
    ]

    graph = Graph(
        nodes=nodes,
        edges=edges,
    )

    nodes.append(3)
    edges.append(
        Edge(
            source=2,
            target=3,
            weight=0.4,
        )
    )

    assert graph.nodes == (1, 2)
    assert len(graph.edges) == 1


def test_graph_rejects_duplicate_nodes() -> None:
    """
    Graph should reject duplicate node identifiers.
    """
    with pytest.raises(InvalidGraphError):
        Graph(
            nodes=(1, 1),
            edges=(),
        )


def test_graph_rejects_unknown_source_node() -> None:
    """
    Graph should reject edges with unknown source nodes.
    """
    with pytest.raises(InvalidGraphError):
        Graph(
            nodes=(1, 2),
            edges=(
                Edge(
                    source=3,
                    target=2,
                    weight=0.5,
                ),
            ),
        )


def test_graph_rejects_unknown_target_node() -> None:
    """
    Graph should reject edges with unknown target nodes.
    """
    with pytest.raises(InvalidGraphError):
        Graph(
            nodes=(1, 2),
            edges=(
                Edge(
                    source=1,
                    target=3,
                    weight=0.5,
                ),
            ),
        )


def test_graph_rejects_duplicate_edges() -> None:
    """
    Graph should reject duplicate edges.
    """
    with pytest.raises(InvalidGraphError):
        Graph(
            nodes=(1, 2),
            edges=(
                Edge(
                    source=1,
                    target=2,
                    weight=0.8,
                ),
                Edge(
                    source=1,
                    target=2,
                    weight=0.6,
                ),
            ),
        )


def test_graph_iteration() -> None:
    """
    Graph iteration should return node identifiers.
    """
    graph = Graph(
        nodes=(1, 2, 3),
        edges=(),
    )

    assert list(graph) == [1, 2, 3]


def test_empty_graph() -> None:
    """
    Empty graph should be supported.
    """
    graph = Graph(
        nodes=(),
        edges=(),
    )

    assert len(graph) == 0
    assert graph.node_count == 0
    assert graph.edge_count == 0
    assert graph.is_empty is True