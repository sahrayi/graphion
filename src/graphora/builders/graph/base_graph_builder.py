"""
Base graph builder.
"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)

from collections.abc import (
    Callable,
    Iterable,
)

from typing import (
    Generic,
)

from graphora.core.interfaces import (
    GraphBuilder,
    RelationBuilder,
)

from graphora.core.models import (
    Edge,
    Graph,
    RelationSet,
)

from graphora.core.types import (
    TId
)


class BaseGraphBuilder(
    GraphBuilder[TId],
    Generic[TId],
    ABC,
):
    """
    Base implementation for graph builders.

    Responsibilities:

    - relation -> edge conversion
    - common edge cleanup
    - graph directionality metadata
    - deterministic output

    Subclasses define topology selection rules.
    """

    def __init__(
        self,
        *,
        relation_builder: RelationBuilder[TId],
        directed: bool = True,
        sort_key: Callable[[TId], object] | None = None,
    ) -> None:

        self.relation_builder = relation_builder

        self._directed = directed

        self.sort_key = (
            sort_key
            if sort_key is not None
            else str
        )


    # --------------------------------------------------
    # Properties
    # --------------------------------------------------

    @property
    def directed(
        self,
    ) -> bool:
        """
        Generated graph directionality.
        """

        return self._directed


    # --------------------------------------------------
    # Build pipeline
    # --------------------------------------------------

    def build(
        self,
        relations: RelationSet[TId],
        nodes: Iterable[TId] | None = None,
    ) -> Graph[TId]:

        edges = self._build_edges(
            relations,
        )

        edges = self.post_process_edges(
            edges,
        )

        edges = self.filter_edges(
            edges,
        )

        graph_nodes = self._extract_nodes(
            edges,
            nodes,
        )

        # Important:
        # Some topology algorithms (e.g. SNN)
        # can legitimately produce zero edges.
        #
        # In this case graph nodes must still
        # be preserved from input relations.

        if not graph_nodes:

            graph_nodes = self._extract_relation_nodes(
                relations,
            )


        return Graph(
            nodes=tuple(graph_nodes),
            edges=tuple(edges),
            directed=self.directed,
        )


    # --------------------------------------------------
    # Relation conversion
    # --------------------------------------------------

    def _build_edges(
        self,
        relations: RelationSet[TId],
    ) -> list[Edge[TId]]:

        return [
            Edge(
                source=relation.source,
                target=relation.target,
                weight=self.relation_builder.affinity(
                    relation.weight,
                ),
            )
            for relation in relations
        ]


    # --------------------------------------------------
    # Hooks
    # --------------------------------------------------

    def post_process_edges(
        self,
        edges: list[Edge[TId]],
    ) -> list[Edge[TId]]:
        """
        Optional preprocessing hook.
        """

        return edges


    # --------------------------------------------------
    # Shared edge helpers
    # --------------------------------------------------

    def remove_self_loops(
        self,
        edges: Iterable[Edge[TId]],
    ) -> list[Edge[TId]]:
        """
        Remove self edges.
        """

        return [
            edge
            for edge in edges
            if edge.source != edge.target
        ]


    def merge_duplicates(
        self,
        edges: Iterable[Edge[TId]],
    ) -> list[Edge[TId]]:
        """
        Merge duplicate directed edges.

        Keeps strongest affinity.

        Example:

            A -> B 0.7
            A -> B 0.9

        becomes:

            A -> B 0.9


        Reverse edges remain separate:

            A -> B

            B -> A

        """

        merged: dict[
            tuple[TId, TId],
            Edge[TId],
        ] = {}


        for edge in edges:

            key = (
                edge.source,
                edge.target,
            )

            existing = merged.get(
                key,
            )

            if (
                existing is None
                or edge.weight > existing.weight
            ):

                merged[key] = edge


        return list(
            merged.values()
        )


    def apply_mutual(
        self,
        edges: Iterable[Edge[TId]],
    ) -> list[Edge[TId]]:
        """
        Keep reciprocal directed edges only.

        Example:

            A -> B
            B -> A

        survive.

        Direction is preserved.
        """

        edges = list(
            edges
        )


        pairs = {
            (
                edge.source,
                edge.target,
            )
            for edge in edges
        }


        return [
            edge
            for edge in edges
            if (
                edge.target,
                edge.source,
            )
            in pairs
        ]


    def make_symmetric(
        self,
        edges: Iterable[Edge[TId]],
    ) -> list[Edge[TId]]:
        """
        Convert directed edges into undirected edges.

        For:

            A -> B (0.7)
            B -> A (0.9)


        returns:

            A -- B (0.9)


        Since Graph(directed=False)
        treats edges as undirected,
        only one normalized edge is returned.
        """

        weights: dict[
            tuple[TId, TId],
            float,
        ] = {}


        for edge in edges:

            pair = self._normalize_pair(
                edge.source,
                edge.target,
            )


            current = weights.get(
                pair,
            )


            if (
                current is None
                or edge.weight > current
            ):

                weights[pair] = edge.weight


        return [
            Edge(
                source=source,
                target=target,
                weight=weight,
            )
            for (
                source,
                target,
            ), weight in sorted(
                weights.items(),
                key=lambda item: (
                    self.sort_key(item[0][0]),
                    self.sort_key(item[0][1]),
                ),
            )
        ]


    # --------------------------------------------------
    # Node extraction
    # --------------------------------------------------

    def _extract_nodes(
        self,
        edges: Iterable[Edge[TId]],
        nodes: Iterable[TId] | None = None,
    ) -> list[TId]:
        """
        Extract graph nodes from edges and optional input.
        """

        node_set: set[TId] = set()


        if nodes is not None:

            node_set.update(
                nodes,
            )


        for edge in edges:

            node_set.add(
                edge.source,
            )

            node_set.add(
                edge.target,
            )


        return sorted(
            node_set,
            key=self.sort_key,
        )


    def _extract_relation_nodes(
        self,
        relations: RelationSet[TId],
    ) -> list[TId]:
        """
        Extract nodes directly from relations.

        Used when topology builder
        produces no edges.
        """

        node_set: set[TId] = set()


        for relation in relations:

            node_set.add(
                relation.source,
            )

            node_set.add(
                relation.target,
            )


        return sorted(
            node_set,
            key=self.sort_key,
        )


    # --------------------------------------------------
    # Algorithm hook
    # --------------------------------------------------

    @abstractmethod
    def filter_edges(
        self,
        edges: list[Edge[TId]],
    ) -> list[Edge[TId]]:
        """
        Select graph topology.

        Examples:

        - threshold
        - KNN
        - adaptive KNN
        - MST
        """
        ...


    # --------------------------------------------------
    # Utility
    # --------------------------------------------------

    def _normalize_pair(
        self,
        source: TId,
        target: TId,
    ) -> tuple[TId, TId]:
        """
        Normalize undirected edge ordering.
        """

        if self.sort_key(source) <= self.sort_key(target):

            return (
                source,
                target,
            )


        return (
            target,
            source,
        )