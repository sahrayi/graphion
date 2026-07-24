"""
Detector determinism tests.

Tests:

- repeated execution produces identical partitions
- partition ordering is stable
"""

from __future__ import annotations


import pytest

from graphora.core.models import (
    Edge,
    Graph,
)

from graphora.detectors.partition import (
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


def deterministic_graph() -> Graph[str]:

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

            Edge(
                "d",
                "e",
                0.95,
            ),

            Edge(
                "e",
                "f",
                0.85,
            ),

            Edge(
                "c",
                "d",
                0.1,
            ),
        ),
    )


# --------------------------------------------------
# Determinism
# --------------------------------------------------


@pytest.mark.parametrize(
    "detector",
    [
        ConnectedComponents(),

        LabelPropagation(),

        Leiden(
            seed=42,
        ),

        Louvain(
            seed=42,
        ),

        FastGreedy(),

        Walktrap(),

        Spectral(
            n_clusters=2,
            random_state=42,
        ),

        Agglomerative(
            n_clusters=2,
        ),
    ],
)
def test_detector_output_is_deterministic(
    detector,
):

    graph = deterministic_graph()


    first = detector.detect(
        graph,
    )


    for _ in range(10):

        current = detector.detect(
            graph,
        )

        assert current == first