"""
Infomap community detection algorithm.
"""

from __future__ import annotations

import importlib

from typing import Generic

from graphora.core.models import Graph
from graphora.core.types import TId

from .base_partition_detector import (
    BasePartitionDetector,
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
    """


    def __init__(
        self,
        *,
        two_level: bool = True,
        seed: int = 42,
    ) -> None:

        self.two_level = two_level
        self.seed = seed


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


        assigned_nodes: set[TId] = set()


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


            assigned_nodes.add(
                original_id,
            )


        # Preserve isolated or missing nodes.
        #
        # Infomap may omit nodes that do not
        # participate in flow.
        #
        # Graphora detectors must return a
        # complete partition of graph.nodes.
        for node in graph.nodes:

            if node not in assigned_nodes:

                partitions[
                    self._singleton_module_id(node)
                ] = {
                    node,
                }


        return [
            partition
            for _, partition
            in sorted(
                partitions.items(),
                key=lambda item: item[0],
            )
        ]


    def _singleton_module_id(
        self,
        node: TId,
    ) -> int:
        """
        Create deterministic singleton module id.

        Negative ids avoid collisions with
        Infomap generated module ids.
        """

        return -(
            abs(
                hash(node)
            )
        )