"""
Detector noise robustness tests.

Tests:

- weak noisy edges do not collapse strong communities
- partitions remain complete
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


def noisy_community_graph() -> Graph[str]:

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
            # strong community A

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


            # strong community B

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


            # weak noisy edges

            Edge(
                "c",
                "d",
                0.01,
            ),

            Edge(
                "a",
                "f",
                0.02,
            ),

            Edge(
                "b",
                "e",
                0.01,
            ),
        ),
    )


def all_nodes(
    partitions,
):

    return {
        node
        for partition in partitions
        for node in partition
    }


# --------------------------------------------------
# Noise robustness
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
def test_detectors_are_robust_against_weak_noise(
    detector,
):

    graph = noisy_community_graph()


    result = detector.detect(
        graph,
    )


    partitions = result.partitions


    # should not collapse everything

    assert len(
        partitions,
    ) >= 2


    # no node loss

    assert all_nodes(
        partitions,
    ) == set(
        graph.nodes,
    )


    # no duplicate assignment

    flattened = [
        node
        for partition in partitions
        for node in partition
    ]

    assert len(
        flattened,
    ) == len(
        set(flattened),
    )