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

from graphion.core.interfaces import (
    GraphBuilder,
)

from graphion.core.models import (
    Edge,
    Graph,
    RelationSet,
)

from graphion.core.types import (
    TId,
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

    Relation weights are assumed to already represent
    the final affinity values. GraphBuilder does not
    calculate or transform relation weights.
    """

    def __init__(
        self,
        *,
        directed: bool = True,
        sort_key: Callable[[TId], object] | None = None,
    ) -> None:

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

        print(
            "[Graphion] Building graph..."
        )

        print(
            f"[Graphion] Input relations: {len(relations)}"
        )

        edges = self._build_edges(
            relations,
        )

        print(
            f"[Graphion] Converted edges: {len(edges)}"
        )

        edges = self.post_process_edges(
            edges,
        )

        print(
            "[Graphion] Applying topology filter..."
        )

        before_filter = len(edges)

        edges = self.filter_edges(
            edges,
        )

        print(
            f"[Graphion] Edges after filtering: "
            f"{len(edges)} "
            f"(removed {before_filter - len(edges)})"
        )

        graph_nodes = self._extract_nodes(
            edges,
            nodes,
        )

        # Important:
        # Some topology algorithms can legitimately
        # produce zero edges.
        #
        # In this case graph nodes must still be
        # preserved from the explicitly supplied nodes
        # or from the input relations.

        if not graph_nodes:

            graph_nodes = self._extract_relation_nodes(
                relations,
            )

        graph = Graph(
            nodes=tuple(graph_nodes),
            edges=tuple(edges),
            directed=self.directed,
        )

        print(
            "[Graphion] Graph created"
        )

        print(
            f"[Graphion] Nodes: {len(graph.nodes)}"
        )

        print(
            f"[Graphion] Edges: {len(graph.edges)}"
        )

        print(
            f"[Graphion] Directed: {graph.directed}"
        )

        return graph

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
                weight=relation.weight,
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

        When both directions exist, the strongest
        affinity is preserved.
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

        if self.sort_key(source) <= self.sort_key(target):

            return (
                source,
                target,
            )

        return (
            target,
            source,
        )