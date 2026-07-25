"""
Detector parameter and failure behavior tests.

Tests:

- constructor validation
- runtime parameter validation
- directed graph compatibility
- empty topology handling
- detector capability contracts
"""

from __future__ import annotations

import pytest

from graphion.core.errors import (
    InvalidGraphError,
)

from graphion.core.models import (
    Edge,
    Graph,
)

from graphion.detectors.partition import (
    Agglomerative,
    ConnectedComponents,
    FastGreedy,
    LabelPropagation,
    Leiden,
    Louvain,
    Spectral,
    Walktrap,
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------


def small_graph() -> Graph[str]:
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
                1.0,
            ),
        ),
    )


def directed_graph() -> Graph[str]:
    return Graph(
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


def edgeless_graph() -> Graph[str]:
    return Graph(
        nodes=(
            "a",
            "b",
            "c",
        ),
        edges=(),
    )


# --------------------------------------------------
# Constructor validation
# --------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LabelPropagation(
            max_iterations=0,
        ),
        lambda: Walktrap(
            walk_steps=0,
        ),
        lambda: Leiden(
            resolution=0,
        ),
        lambda: Louvain(
            resolution=0,
        ),
        lambda: Spectral(
            n_clusters=0,
        ),
        lambda: Agglomerative(
            n_clusters=0,
        ),
        lambda: FastGreedy(
            target_partitions=0,
        ),
    ],
)
def test_detectors_reject_invalid_constructor_parameters(
    factory,
):
    with pytest.raises(
        ValueError,
    ):
        factory()


# --------------------------------------------------
# Runtime parameter validation
# --------------------------------------------------


@pytest.mark.parametrize(
    "detector",
    [
        Spectral(
            n_clusters=5,
        ),
        Agglomerative(
            n_clusters=5,
        ),
    ],
)
def test_cluster_count_cannot_exceed_nodes(
    detector,
):
    with pytest.raises(
        ValueError,
    ):
        detector.detect(
            small_graph(),
        )


# --------------------------------------------------
# Directed graph capability
# --------------------------------------------------


@pytest.mark.parametrize(
    "detector",
    [
        ConnectedComponents(),
        LabelPropagation(),
        FastGreedy(),
        Walktrap(),
    ],
)
def test_undirected_detectors_reject_directed_graph(
    detector,
):
    with pytest.raises(
        InvalidGraphError,
    ):
        detector.detect(
            directed_graph(),
        )


@pytest.mark.parametrize(
    "detector",
    [
        Leiden(),
        Louvain(),
    ],
)
def test_directed_capability_contract(
    detector,
):
    graph = directed_graph()

    if detector.supports_directed:

        result = detector.detect(
            graph,
        )

        assert len(
            result.partitions,
        ) > 0

    else:

        with pytest.raises(
            InvalidGraphError,
        ):
            detector.detect(
                graph,
            )


# --------------------------------------------------
# Empty topology
# --------------------------------------------------


@pytest.mark.parametrize(
    "detector",
    [
        ConnectedComponents(),
        LabelPropagation(),
        Leiden(),
        Louvain(),
        FastGreedy(),
        Walktrap(),
    ],
)
def test_detectors_handle_edgeless_graph(
    detector,
):
    result = detector.detect(
        edgeless_graph(),
    )

    assert len(
        result.partitions,
    ) == 3