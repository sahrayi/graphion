"""
Weighted edge semantics tests.

Tests:

- detectors respect edge weights
- strong internal affinity dominates weak bridge edges
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


def weighted_community_graph() -> Graph[str]:
    """
    Two dense communities connected by a weak bridge.

    Expected:

        {a,b,c}

        {d,e,f}

    """

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

            Edge(
                "a",
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

            Edge(
                "d",
                "f",
                1.0,
            ),


            # weak bridge

            Edge(
                "c",
                "d",
                0.01,
            ),
        ),
    )


def normalize(
    partitions,
):
    return {
        frozenset(
            partition,
        )
        for partition in partitions
    }


# --------------------------------------------------
# Community detectors
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
def test_weighted_edges_preserve_strong_communities(
    detector,
):

    graph = weighted_community_graph()


    result = detector.detect(
        graph,
    )


    partitions = normalize(
        result.partitions,
    )


    assert {
        frozenset(
            {
                "a",
                "b",
                "c",
            }
        ),
        frozenset(
            {
                "d",
                "e",
                "f",
            }
        ),
    } == partitions