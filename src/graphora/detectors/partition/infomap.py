"""
Infomap community detection algorithm.
"""

from __future__ import annotations

import importlib

from collections.abc import Hashable

from typing import (
    Generic,
    TypeVar,
)

from graphora.core.models import Graph

from .base_partition_detector import (
    BasePartitionDetector,
)

from graphora.core.types import (
    TId,
)

class Infomap(
    BasePartitionDetector[TId],
    Generic[TId],
):
    """
    Infomap community detector.

    Uses the map equation and random walks
    to detect flow-based communities.

    Edge weights are interpreted as affinity.


    Supported topology:

    - directed
    - undirected


    Properties:

    - weighted
    - flow based
    - stochastic
    - reproducible with seed
    - suitable for sparse graphs
    """

    def __init__(
        self,
        *,
        two_level: bool = True,
        seed: int = 42,
    ) -> None:

        self.two_level = two_level
        self.seed = seed


    # --------------------------------------------------
    # Capability
    # --------------------------------------------------

    @property
    def supports_directed(
        self,
    ) -> bool:

        return True


    @property
    def supports_undirected(
        self,
    ) -> bool:

        return True


    # --------------------------------------------------
    # Detection
    # --------------------------------------------------

    def _detect(
        self,
        graph: Graph[TId],
    ) -> list[set[TId]]:

        if graph.node_count == 0:
            return []


        if graph.node_count == 1:

            return [
                {
                    graph.nodes[0],
                }
            ]


        args: list[str] = []


        if graph.directed:
            args.append(
                "--directed",
            )


        if self.two_level:
            args.append(
                "--two-level",
            )


        args.extend(
            [
                "--seed",
                str(self.seed),
            ]
        )


        infomap_module = importlib.import_module(
            "infomap",
        )


        InfomapEngine = (
            infomap_module.Infomap
        )


        engine = InfomapEngine(
            " ".join(args),
        )


        node_mapping = {
            node: index + 1
            for index, node
            in enumerate(
                graph.nodes,
            )
        }


        reverse_mapping = {
            index: node
            for node, index
            in node_mapping.items()
        }


        for edge in graph.edges:

            engine.add_link(
                node_mapping[edge.source],
                node_mapping[edge.target],
                edge.weight,
            )


        result = engine.run()


        partitions: dict[
            int,
            set[TId],
        ] = {}


        for node_id, module_id in result.modules().items():

            original_id = reverse_mapping.get(
                node_id,
            )


            if original_id is None:
                continue


            partitions.setdefault(
                module_id,
                set(),
            ).add(
                original_id,
            )


        return [
            partition
            for _, partition
            in sorted(
                partitions.items(),
                key=lambda item: item[0],
            )
        ]