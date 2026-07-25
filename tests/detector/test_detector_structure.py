"""
Detector structural behavior tests.

Tests:

- detectors can discover obvious community separation
- weak bridges do not collapse communities
"""

from __future__ import annotations


import pytest

from graphion.core.models import (
    Edge,
    Graph,
)

from graphion.detectors.partition import (
    Agglomerative,
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


def community_graph() -> Graph[str]:

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
            # community 1
            Edge(
                "a",
                "b",
                1.0,
            ),
            Edge(
                "b",
                "c",
                1.0,
            ),

            # community 2
            Edge(
                "d",
                "e",
                1.0,
            ),
            Edge(
                "e",
                "f",
                1.0,
            ),

            # weak bridge
            Edge(
                "c",
                "d",
                0.05,
            ),
        ),
    )


# --------------------------------------------------
# Community separation
# --------------------------------------------------


@pytest.mark.parametrize(
    "detector",
    [
        LabelPropagation(),

        Leiden(),

        Louvain(),

        FastGreedy(),

        Walktrap(),

        Spectral(
            n_clusters=2,
        ),

        Agglomerative(
            n_clusters=2,
        ),
    ],
)
def test_detectors_preserve_obvious_community_structure(
    detector,
):

    graph = community_graph()

    result = detector.detect(
        graph,
    )

    partitions = result.partitions


    # must discover more than one group

    assert len(
        partitions,
    ) >= 2


    # every partition must contain nodes

    assert all(
        partition
        for partition in partitions
    )