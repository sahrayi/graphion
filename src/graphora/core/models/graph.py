"""
Graph data model.
"""

from __future__ import annotations

import math

from collections.abc import (
    Iterable,
    Iterator,
)

from dataclasses import dataclass
from typing import Generic

from graphora.core.errors import InvalidGraphError
from graphora.core.types import TId

from .edge import Edge


@dataclass(
    frozen=True,
    slots=True,
)
class Graph(Generic[TId]):
    """
    Immutable Graphora graph model.

    Independent from external graph libraries.
    """

    nodes: tuple[TId, ...]
    edges: tuple[Edge[TId], ...]
    directed: bool = False

    def __post_init__(self) -> None:

        object.__setattr__(
            self,
            "nodes",
            tuple(self.nodes),
        )

        object.__setattr__(
            self,
            "edges",
            tuple(self.edges),
        )

        node_set = set(self.nodes)

        if len(node_set) != len(self.nodes):
            raise InvalidGraphError(
                "Duplicate node identifiers are not allowed."
            )

        seen: set[tuple[TId, TId]] = set()

        for edge in self.edges:

            if edge.source not in node_set:
                raise InvalidGraphError(
                    f"Unknown source node: {edge.source!r}"
                )

            if edge.target not in node_set:
                raise InvalidGraphError(
                    f"Unknown target node: {edge.target!r}"
                )

            if not math.isfinite(
                edge.weight,
            ):
                raise InvalidGraphError(
                    "Edge weight must be finite."
                )

            key = (
                edge.source,
                edge.target,
            )

            if self.directed:

                if key in seen:
                    raise InvalidGraphError(
                        "Duplicate directed edges are not allowed."
                    )

            else:

                reverse = (
                    edge.target,
                    edge.source,
                )

                if (
                    key in seen
                    or reverse in seen
                ):
                    raise InvalidGraphError(
                        "Duplicate undirected edges are not allowed."
                    )

            seen.add(key)

    # --------------------------------------------------
    # Constructors
    # --------------------------------------------------

    @classmethod
    def empty(
        cls,
        *,
        directed: bool = False,
    ) -> "Graph[TId]":

        return cls(
            nodes=(),
            edges=(),
            directed=directed,
        )


    @classmethod
    def from_edges(
        cls,
        edges: Iterable[Edge[TId]],
        *,
        directed: bool = False,
    ) -> "Graph[TId]":

        edges = tuple(edges)

        nodes: set[TId] = set()

        for edge in edges:

            nodes.add(
                edge.source,
            )

            nodes.add(
                edge.target,
            )

        return cls(
            nodes=tuple(
                sorted(
                    nodes,
                    key=str,
                )
            ),
            edges=edges,
            directed=directed,
        )


    # --------------------------------------------------
    # Basic protocols
    # --------------------------------------------------

    def __len__(self) -> int:
        return len(self.nodes)


    def __iter__(
        self,
    ) -> Iterator[TId]:

        return iter(
            self.nodes,
        )


    @property
    def node_count(
        self,
    ) -> int:

        return len(self.nodes)


    @property
    def edge_count(
        self,
    ) -> int:

        return len(self.edges)


    @property
    def is_empty(
        self,
    ) -> bool:

        return self.node_count == 0


    # --------------------------------------------------
    # Query operations
    # --------------------------------------------------

    def has_node(
        self,
        node_id: TId,
    ) -> bool:

        return node_id in self.nodes


    def has_edge(
        self,
        source: TId,
        target: TId,
    ) -> bool:

        for edge in self.edges:

            if (
                edge.source == source
                and edge.target == target
            ):
                return True

            if (
                not self.directed
                and edge.source == target
                and edge.target == source
            ):
                return True

        return False


    def neighbors(
        self,
        node_id: TId,
    ) -> tuple[TId, ...]:

        if not self.has_node(
            node_id,
        ):
            raise InvalidGraphError(
                f"Unknown node: {node_id!r}"
            )

        result: list[TId] = []

        for edge in self.edges:

            if edge.source == node_id:

                result.append(
                    edge.target,
                )

            elif (
                not self.directed
                and edge.target == node_id
            ):

                result.append(
                    edge.source,
                )


        return tuple(
            sorted(
                result,
                key=str,
            )
        )


    def degree(
        self,
        node_id: TId,
    ) -> int:
        """
        Return node degree.

        For directed graphs this returns
        outgoing degree.
        """

        return len(
            self.neighbors(
                node_id,
            )
        )


    # --------------------------------------------------
    # Mutation-like immutable operations
    # --------------------------------------------------

    def add_node(
        self,
        node_id: TId,
    ) -> "Graph[TId]":

        if self.has_node(
            node_id,
        ):
            return self

        return Graph(
            nodes=self.nodes + (
                node_id,
            ),
            edges=self.edges,
            directed=self.directed,
        )


    def add_edge(
        self,
        source: TId,
        target: TId,
        weight: float = 1.0,
    ) -> "Graph[TId]":

        if self.has_edge(
            source,
            target,
        ):
            raise InvalidGraphError(
                "Edge already exists."
            )

        graph = self

        if not graph.has_node(
            source,
        ):
            graph = graph.add_node(
                source,
            )

        if not graph.has_node(
            target,
        ):
            graph = graph.add_node(
                target,
            )

        return Graph(
            nodes=graph.nodes,
            edges=graph.edges + (
                Edge(
                    source=source,
                    target=target,
                    weight=float(weight),
                ),
            ),
            directed=self.directed,
        )


    def update_edge_weight(
        self,
        source: TId,
        target: TId,
        weight: float,
    ) -> "Graph[TId]":

        updated = False

        edges: list[Edge[TId]] = []

        for edge in self.edges:

            match_edge = (
                edge.source == source
                and edge.target == target
            )

            reverse_match = (
                not self.directed
                and edge.source == target
                and edge.target == source
            )

            if match_edge or reverse_match:

                edges.append(
                    Edge(
                        source=edge.source,
                        target=edge.target,
                        weight=float(weight),
                    )
                )

                updated = True

            else:

                edges.append(
                    edge,
                )


        if not updated:
            raise InvalidGraphError(
                "Edge does not exist."
            )


        return Graph(
            nodes=self.nodes,
            edges=tuple(edges),
            directed=self.directed,
        )


    def remove_node(
        self,
        node_id: TId,
    ) -> "Graph[TId]":

        if not self.has_node(
            node_id,
        ):
            return self

        return Graph(
            nodes=tuple(
                node
                for node in self.nodes
                if node != node_id
            ),
            edges=tuple(
                edge
                for edge in self.edges
                if (
                    edge.source != node_id
                    and edge.target != node_id
                )
            ),
            directed=self.directed,
        )


    def remove_edge(
        self,
        source: TId,
        target: TId,
    ) -> "Graph[TId]":

        return Graph(
            nodes=self.nodes,
            edges=tuple(
                edge
                for edge in self.edges
                if not (
                    (
                        edge.source == source
                        and edge.target == target
                    )
                    or (
                        not self.directed
                        and edge.source == target
                        and edge.target == source
                    )
                )
            ),
            directed=self.directed,
        )


    # --------------------------------------------------
    # External conversions
    # --------------------------------------------------

    def to_networkx(self):

        try:
            import networkx as nx

        except ImportError as exc:

            raise ImportError(
                "NetworkX is required."
            ) from exc


        graph = (
            nx.DiGraph()
            if self.directed
            else nx.Graph()
        )


        graph.add_nodes_from(
            self.nodes,
        )


        for edge in self.edges:

            graph.add_edge(
                edge.source,
                edge.target,
                weight=edge.weight,
            )


        return graph


    @classmethod
    def from_networkx(
        cls,
        graph,
    ) -> "Graph[TId]":

        return cls(
            nodes=tuple(
                graph.nodes()
            ),
            edges=tuple(
                Edge(
                    source=u,
                    target=v,
                    weight=float(
                        data.get(
                            "weight",
                            1.0,
                        )
                    ),
                )
                for u, v, data
                in graph.edges(data=True)
            ),
            directed=graph.is_directed(),
        )


    def to_igraph(self):

        try:
            import igraph

        except ImportError as exc:

            raise ImportError(
                "python-igraph is required."
            ) from exc


        graph = igraph.Graph(
            directed=self.directed,
        )


        graph.add_vertices(
            len(self.nodes),
        )


        index = {
            node: i
            for i, node in enumerate(
                self.nodes,
            )
        }


        graph.add_edges(
            [
                (
                    index[edge.source],
                    index[edge.target],
                )
                for edge in self.edges
            ]
        )


        graph.vs["id"] = list(
            self.nodes,
        )

        graph.es["weight"] = [
            edge.weight
            for edge in self.edges
        ]


        return graph


    @classmethod
    def from_igraph(
        cls,
        graph,
    ) -> "Graph[TId]":

        nodes = tuple(
            graph.vs["id"]
            if "id"
            in graph.vs.attributes()
            else tuple(
                range(
                    graph.vcount()
                )
            )
        )


        edges = tuple(
            Edge(
                source=nodes[source],
                target=nodes[target],
                weight=float(
                    graph.es[index]["weight"]
                    if "weight"
                    in graph.es.attributes()
                    else 1.0
                ),
            )
            for index, (
                source,
                target,
            )
            in enumerate(
                graph.get_edgelist()
            )
        )


        return cls(
            nodes=nodes,
            edges=edges,
            directed=graph.is_directed(),
        )