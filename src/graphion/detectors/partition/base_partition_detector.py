"""
Base partition detector implementation.
"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)

from collections.abc import (
    Hashable,
    Iterable,
)

from typing import (
    Generic,
    TypeVar,
)

from graphion.core.errors import InvalidGraphError

from graphion.core.interfaces import (
    PartitionDetector,
)

from graphion.core.models import (
    Graph,
    PartitionSet,
)


from graphion.core.types import (
    TId,
)


class BasePartitionDetector(
    PartitionDetector[TId],
    ABC,
    Generic[TId],
):
    """
    Base implementation for partition detectors.

    Responsibilities:

    - validate graph compatibility
    - execute detection pipeline
    - normalize detector output
    - guarantee deterministic representation


    Subclasses implement only:

    - detection algorithm
    - algorithm-specific parameters


    Graph conversion is handled by Graphion Graph model.
    Detectors should use:

        graph.to_networkx()

    or:

        graph.to_igraph()

    when required.
    """

    # --------------------------------------------------
    # Capability declaration
    # --------------------------------------------------

    @property
    def supports_directed(
        self,
    ) -> bool:
        """
        Whether detector supports directed graphs.

        Override in subclasses when required.
        """

        return False


    @property
    def supports_undirected(
        self,
    ) -> bool:
        """
        Whether detector supports undirected graphs.

        Override in subclasses when required.
        """

        return True


    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def detect(
        self,
        graph: Graph[TId],
    ) -> PartitionSet[TId]:
        """
        Detect graph partitions.
        """

        self._validate_graph(
            graph,
        )

        raw_partitions = self._detect(
            graph,
        )

        return self._normalize_partitions(
            raw_partitions,
        )


    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def _validate_graph(
        self,
        graph: Graph[TId],
    ) -> None:
        """
        Validate graph compatibility.
        """

        if graph.directed:

            if not self.supports_directed:

                raise InvalidGraphError(
                    f"{self.__class__.__name__} "
                    "does not support directed graphs."
                )

        else:

            if not self.supports_undirected:

                raise InvalidGraphError(
                    f"{self.__class__.__name__} "
                    "does not support undirected graphs."
                )


    # --------------------------------------------------
    # Normalization
    # --------------------------------------------------

    def _normalize_partitions(
        self,
        partitions: Iterable[
            Iterable[TId]
        ],
    ) -> PartitionSet[TId]:
        """
        Normalize detector output.

        Guarantees:

        - immutable partitions
        - deterministic ordering
        """

        normalized = [
            frozenset(
                partition,
            )
            for partition in partitions
        ]


        normalized.sort(
            key=lambda partition: tuple(
                sorted(
                    (
                        str(node)
                        for node in partition
                    )
                )
            )
        )


        return PartitionSet(
            partitions=tuple(
                normalized,
            ),
        )


    # --------------------------------------------------
    # Algorithm hook
    # --------------------------------------------------

    @abstractmethod
    def _detect(
        self,
        graph: Graph[TId],
    ) -> Iterable[
        Iterable[TId]
    ]:
        """
        Execute partition detection.

        Subclasses receive Graphion Graph.

        They should:

        - use graph.to_networkx()
        - or graph.to_igraph()

        when required.

        They should NOT:

        - rebuild graph
        - modify topology
        - convert relations
        - normalize output
        """

        ...