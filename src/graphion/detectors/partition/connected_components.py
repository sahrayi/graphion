"""
Connected components partition detector.
"""

from __future__ import annotations

from collections.abc import Hashable

from typing import (
    Generic,
    TypeVar,
)

from graphion.core.models import Graph

from .base_partition_detector import (
    BasePartitionDetector,
)

from graphion.core.types import (
    TId,
)


class ConnectedComponents(
    BasePartitionDetector[TId],
    Generic[TId],
):
    """
    Detect connected components in an undirected graph.

    Each connected component becomes one partition.


    Supported topology:

    - undirected graphs only


    Properties:

    - parameter free
    - deterministic
    - sparse graph friendly
    - preserves isolated nodes
    """


    # --------------------------------------------------
    # Capability
    # --------------------------------------------------

    @property
    def supports_directed(
        self,
    ) -> bool:
        """
        Directed graphs are not supported.

        Use a dedicated detector for:
        - weakly connected components
        - strongly connected components
        """

        return False


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
        """
        Execute depth-first traversal to find
        connected components.
        """


        adjacency: dict[
            TId,
            set[TId],
        ] = {
            node: set()
            for node in graph.nodes
        }


        for edge in graph.edges:

            if edge.source == edge.target:
                continue


            adjacency[
                edge.source
            ].add(
                edge.target,
            )

            adjacency[
                edge.target
            ].add(
                edge.source,
            )


        visited: set[TId] = set()

        partitions: list[set[TId]] = []


        for node in sorted(
            graph.nodes,
            key=str,
        ):

            if node in visited:
                continue


            component: set[TId] = set()

            stack: list[TId] = [
                node,
            ]


            while stack:

                current = stack.pop()


                if current in visited:
                    continue


                visited.add(
                    current,
                )

                component.add(
                    current,
                )


                for neighbor in sorted(
                    adjacency[current],
                    key=str,
                    reverse=True,
                ):

                    if neighbor not in visited:
                        stack.append(
                            neighbor,
                        )


            partitions.append(
                component,
            )


        return partitions