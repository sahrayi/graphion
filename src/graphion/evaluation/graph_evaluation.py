from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from statistics import median, mean, pstdev
from typing import Any, cast

import networkx as nx
import numpy as np

from graphion.core.models import Graph


class GraphEvaluation:
    """Evaluate structural properties of a Graphion Graph."""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self._nx_graph = graph.to_networkx()

    @property
    def nx_graph(self) -> Graph | nx.DiGraph:
        """Return the underlying networkx graph object."""
        return self._nx_graph

    @property
    def node_count(self) -> int:
        """Return the total number of nodes in the graph."""
        return self.graph.node_count

    @property
    def edge_count(self) -> int:
        """Return the total number of edges in the graph."""
        return self.graph.edge_count

    @staticmethod
    def _empty_stats() -> dict[str, float]:
        """Return a default dictionary of zero-initialized statistical values."""
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}

    @staticmethod
    def _nan_stats() -> dict[str, float]:
        """Return a default dictionary of NaN-initialized statistical values."""
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "median": float("nan"), "std": float("nan")}

    @staticmethod
    def _compute_stats(values: Sequence[float] | np.ndarray) -> dict[str, float]:
        """Compute statistical metrics (min, max, mean, median, std) for a given collection of values."""
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return GraphEvaluation._empty_stats()
        return {"min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean()), "median": float(np.median(arr)), "std": float(arr.std())}

    def _degrees(self) -> list[int]:
        """Return a list of degrees for all nodes in the graph."""
        deg_iter = cast(Iterable[tuple[Any, int]], self.nx_graph.out_degree() if self.graph.directed else self.nx_graph.degree())
        return [int(degree) for _, degree in deg_iter]

    def density(self) -> float:
        """Calculate and return the density of the graph."""
        return 0.0 if self.node_count <= 1 else float(nx.density(self.nx_graph))

    def average_degree(self) -> float:
        """Calculate and return the average degree of nodes."""
        d = self._degrees()
        return float(mean(d)) if d else 0.0

    def min_degree(self) -> int:
        """Return the minimum degree among all nodes."""
        d = self._degrees()
        return min(d) if d else 0

    def max_degree(self) -> int:
        """Return the maximum degree among all nodes."""
        d = self._degrees()
        return max(d) if d else 0

    def median_degree(self) -> float:
        """Return the median degree across all nodes."""
        d = self._degrees()
        return float(median(d)) if d else 0.0

    def degree_std(self) -> float:
        """Calculate and return the population standard deviation of node degrees."""
        d = self._degrees()
        return float(pstdev(d)) if len(d) > 1 else 0.0

    def degree_statistics(self) -> dict[str, float]:
        """Return comprehensive statistics for node degrees."""
        return {"min": float(self.min_degree()), "max": float(self.max_degree()), "mean": self.average_degree(), "median": self.median_degree(), "std": self.degree_std()}

    def degree_entropy(self) -> float:
        """Calculate and return the Shannon entropy of the degree distribution."""
        degrees = self._degrees()
        if not degrees:
            return 0.0
        counts = Counter(degrees)
        probabilities = np.asarray(list(counts.values()), dtype=float)
        probabilities /= probabilities.sum()
        probabilities = probabilities[probabilities > 0]
        return -float(np.sum(probabilities * np.log2(probabilities)))

    def is_connected(self) -> bool:
        """Check graph connectivity. Directed graphs use weak connectivity."""
        if self.node_count == 0:
            return True
        return nx.is_weakly_connected(cast(nx.DiGraph, self.nx_graph)) if self.graph.directed else nx.is_connected(self.nx_graph)

    def connected_component_count(self) -> int:
        """Number of connected components. For directed graphs weak connectivity is used."""
        if self.node_count == 0:
            return 0
        return nx.number_weakly_connected_components(self.nx_graph) if self.graph.directed else nx.number_connected_components(self.nx_graph)

    def edge_connectivity(self) -> float:
        """Return minimum number of edges whose removal disconnects the graph."""
        if self.node_count <= 1:
            return 0.0
        graph = self.nx_graph.to_undirected() if self.graph.directed else self.nx_graph
        if not nx.is_connected(graph):
            return 0.0
        try:
            return float(nx.edge_connectivity(graph))
        except Exception:
            return float("nan")

    def node_connectivity(self) -> float:
        """Return minimum number of nodes whose removal disconnects the graph."""
        if self.node_count <= 1:
            return 0.0
        graph = self.nx_graph.to_undirected() if self.graph.directed else self.nx_graph
        if not nx.is_connected(graph):
            return 0.0
        try:
            return float(nx.node_connectivity(graph))
        except Exception:
            return float("nan")

    def average_clustering(self) -> float:
        """Calculate and return the average clustering coefficient of the graph."""
        if self.graph.directed:
            return float("nan")
        try:
            return float(nx.average_clustering(self.nx_graph, weight="weight"))
        except Exception:
            return float("nan")

    def transitivity(self) -> float:
        """Calculate and return the graph transitivity."""
        if self.graph.directed:
            return float("nan")
        try:
            return float(nx.transitivity(self.nx_graph))
        except Exception:
            return float("nan")

    def _largest_connected_component_graph(self) -> nx.Graph:
        """Return the largest connected component as a subgraph."""
        graph = self.nx_graph.to_undirected() if self.graph.directed else self.nx_graph
        if graph.number_of_nodes() == 0:
            return graph
        largest = max(nx.connected_components(graph), key=len)
        return graph.subgraph(largest).copy()

    def diameter(self) -> float:
        """Calculate and return the diameter of the largest connected component."""
        g = self._largest_connected_component_graph()
        if g.number_of_nodes() <= 1:
            return 0.0
        try:
            return float(nx.diameter(g))
        except Exception:
            return float("nan")

    def radius(self) -> float:
        """Calculate and return the radius of the largest connected component."""
        g = self._largest_connected_component_graph()
        if g.number_of_nodes() <= 1:
            return 0.0
        try:
            return float(nx.radius(g))
        except Exception:
            return float("nan")

    def average_shortest_path_length(self) -> float:
        """Calculate and return the average shortest path length."""
        g = self._largest_connected_component_graph()
        if g.number_of_nodes() <= 1:
            return 0.0
        try:
            return float(nx.average_shortest_path_length(g))
        except Exception:
            return float("nan")

    def global_efficiency(self) -> float:
        """Calculate and return the global efficiency of the graph."""
        if self.graph.directed:
            return float("nan")
        try:
            return float(nx.global_efficiency(self.nx_graph))
        except Exception:
            return float("nan")

    def local_efficiency(self) -> float:
        """Calculate and return the average local efficiency of the graph."""
        if self.graph.directed:
            return float("nan")
        try:
            return float(nx.local_efficiency(self.nx_graph))
        except Exception:
            return float("nan")

    def eccentricity_statistics(self) -> dict[str, float]:
        """Compute and return statistics for node eccentricities."""
        g = self._largest_connected_component_graph()
        if g.number_of_nodes() <= 1:
            return self._empty_stats()
        try:
            vals = np.asarray(list(nx.eccentricity(g).values()), dtype=float)
            return self._compute_stats(vals)
        except Exception:
            return self._nan_stats()

    def wiener_index(self) -> float:
        """Calculate and return the Wiener index of the graph."""
        g = self._largest_connected_component_graph()
        if g.number_of_nodes() <= 1:
            return 0.0
        try:
            return float(nx.wiener_index(g))
        except Exception:
            return float("nan")

    def _centrality_statistics(self, values: dict[Any, float]) -> dict[str, float]:
        """Compute statistics for a given dictionary of centrality values."""
        return self._compute_stats(np.asarray(list(values.values()), dtype=float)) if values else self._empty_stats()

    def degree_centrality(self) -> dict[Any, float]:
        """Calculate and return degree centrality for all nodes."""
        try:
            vals = nx.out_degree_centrality(self.nx_graph) if self.graph.directed else nx.degree_centrality(self.nx_graph)
            return {k: float(v) for k, v in vals.items()}
        except Exception:
            return {}

    def degree_centrality_statistics(self) -> dict[str, float]:
        """Return statistics for degree centralities."""
        return self._centrality_statistics(self.degree_centrality())

    def betweenness_centrality(self) -> dict[Any, float]:
        """Calculate and return betweenness centrality for all nodes."""
        try:
            return {k: float(v) for k, v in nx.betweenness_centrality(self.nx_graph, weight="weight").items()}
        except Exception:
            return {}

    def betweenness_centrality_statistics(self) -> dict[str, float]:
        """Return statistics for betweenness centralities."""
        return self._centrality_statistics(self.betweenness_centrality())

    def closeness_centrality(self) -> dict[Any, float]:
        """Calculate and return closeness centrality for all nodes."""
        try:
            return {k: float(v) for k, v in nx.closeness_centrality(self.nx_graph).items()}
        except Exception:
            return {}

    def closeness_centrality_statistics(self) -> dict[str, float]:
        """Return statistics for closeness centralities."""
        return self._centrality_statistics(self.closeness_centrality())

    def eigenvector_centrality(self) -> dict[Any, float]:
        """Calculate and return eigenvector centrality for all nodes."""
        try:
            return {k: float(v) for k, v in nx.eigenvector_centrality(self.nx_graph, weight="weight", max_iter=1000).items()}
        except Exception:
            return {}

    def eigenvector_centrality_statistics(self) -> dict[str, float]:
        """Return statistics for eigenvector centralities."""
        return self._centrality_statistics(self.eigenvector_centrality())

    def pagerank(self) -> dict[Any, float]:
        """Calculate and return PageRank scores for all nodes."""
        try:
            return {k: float(v) for k, v in nx.pagerank(self.nx_graph, weight="weight").items()}
        except Exception:
            return {}

    def pagerank_statistics(self) -> dict[str, float]:
        """Return statistics for PageRank scores."""
        return self._centrality_statistics(self.pagerank())

    def katz_centrality(self) -> dict[Any, float]:
        """Calculate and return Katz centrality for all nodes."""
        try:
            return {k: float(v) for k, v in nx.katz_centrality(self.nx_graph, weight="weight").items()}
        except Exception:
            return {}

    def katz_centrality_statistics(self) -> dict[str, float]:
        """Return statistics for Katz centralities."""
        return self._centrality_statistics(self.katz_centrality())

    def harmonic_centrality(self) -> dict[Any, float]:
        """Calculate and return harmonic centrality for all nodes."""
        try:
            return {k: float(v) for k, v in nx.harmonic_centrality(self.nx_graph).items()}
        except Exception:
            return {}

    def harmonic_centrality_statistics(self) -> dict[str, float]:
        """Return statistics for harmonic centralities."""
        return self._centrality_statistics(self.harmonic_centrality())

    def load_centrality(self) -> dict[Any, float]:
        """Calculate and return load centrality for all nodes."""
        try:
            return {k: float(v) for k, v in nx.load_centrality(self.nx_graph, weight="weight").items()}
        except Exception:
            return {}

    def load_centrality_statistics(self) -> dict[str, float]:
        """Return statistics for load centralities."""
        return self._centrality_statistics(self.load_centrality())

    def core_number(self) -> dict[Any, int]:
        """Calculate and return the core number for each node."""
        if self.graph.directed:
            return {}
        try:
            return nx.core_number(self.nx_graph)
        except Exception:
            return {}

    def k_core_statistics(self) -> dict[str, float]:
        """Return statistics for k-core numbers."""
        values = self.core_number()
        if not values:
            return self._empty_stats()
        return self._compute_stats(np.asarray(list(values.values()), dtype=float))

    def clustering_statistics(self) -> dict[str, float]:
        """Return statistics for local clustering coefficients."""
        if self.graph.directed:
            return self._nan_stats()
        try:
            coeffs = nx.clustering(self.nx_graph, weight="weight")
            return self._compute_stats(np.asarray(list(coeffs.values()), dtype=float))
        except Exception:
            return self._nan_stats()

    def triangle_count(self) -> int:
        """Count and return the total number of triangles in the graph."""
        if self.graph.directed:
            return 0
        try:
            return int(sum(nx.triangles(self.nx_graph).values()) // 3)
        except Exception:
            return 0

    def square_clustering(self) -> float:
        """Calculate and return the average square clustering coefficient."""
        if self.graph.directed:
            return float("nan")
        try:
            vals = nx.square_clustering(self.nx_graph)
            return float(np.mean(list(vals.values()))) if vals else 0.0
        except Exception:
            return float("nan")

    def largest_connected_component_ratio(self) -> float:
        """Return the ratio of nodes in the largest connected component to total nodes."""
        if self.node_count == 0:
            return 0.0
        return float(self._largest_connected_component_graph().number_of_nodes() / self.node_count)

    def assortativity_coefficient(self) -> float:
        """Calculate and return the degree assortativity coefficient."""
        try:
            return float(nx.degree_assortativity_coefficient(self.nx_graph))
        except Exception:
            return float("nan")

    def edge_weight_statistics(self) -> dict[str, float]:
        """Compute and return statistics for edge weights."""
        weights = np.asarray([edge.weight for edge in self.graph.edges if getattr(edge, "weight", None) is not None], dtype=float)
        if weights.size == 0:
            return self._empty_stats()
        return self._compute_stats(weights)

    def adjacency_spectrum(self) -> list[float]:
        """Return adjacency eigenvalue spectrum. For directed graphs the absolute magnitude of complex eigenvalues is returned."""
        if self.node_count == 0:
            return []
        try:
            matrix = nx.to_numpy_array(self.nx_graph, weight="weight")
            vals = np.linalg.eigvals(matrix) if self.graph.directed else np.linalg.eigvalsh(matrix)
            spectrum = [float(abs(v)) if self.graph.directed else float(v) for v in vals]
            return sorted(spectrum, reverse=True)
        except Exception:
            return []

    def spectral_radius(self) -> float:
        """Largest absolute adjacency eigenvalue."""
        spectrum = self.adjacency_spectrum()
        if not spectrum:
            return 0.0
        return float(max(abs(v) for v in spectrum))

    def laplacian_spectrum(self) -> list[float]:
        """Return the Laplacian matrix eigenvalue spectrum."""
        if self.graph.directed or self.node_count == 0:
            return []
        try:
            matrix = nx.laplacian_matrix(self.nx_graph, weight="weight").toarray()
            return sorted(float(v) for v in np.linalg.eigvalsh(matrix))
        except Exception:
            return []

    def laplacian_zero_eigenvalue_count(self) -> int:
        """Count and return the number of zero eigenvalues in the Laplacian spectrum."""
        spectrum = self.laplacian_spectrum()
        return int(sum(abs(v) < 1e-10 for v in spectrum)) if spectrum else 0

    def algebraic_connectivity(self) -> float:
        """Return the algebraic connectivity (Fiedler value) of the graph."""
        spectrum = self.laplacian_spectrum()
        return float(spectrum[1]) if len(spectrum) >= 2 else 0.0

    def adjacency_spectral_gap(self) -> float:
        """Calculate and return the adjacency spectral gap."""
        if self.graph.directed:
            return float("nan")
        spectrum = self.adjacency_spectrum()
        if len(spectrum) < 2:
            return 0.0
        vals = sorted((abs(v) for v in spectrum), reverse=True)
        return float(vals[0] - vals[1])

    def normalized_laplacian_spectrum(self) -> list[float]:
        """Return the normalized Laplacian matrix eigenvalue spectrum."""
        if self.graph.directed or self.node_count == 0:
            return []
        try:
            matrix = nx.normalized_laplacian_matrix(self.nx_graph, weight="weight").toarray()
            return sorted(float(v) for v in np.linalg.eigvalsh(matrix))
        except Exception:
            return []

    def normalized_algebraic_connectivity(self) -> float:
        """Return the normalized algebraic connectivity."""
        if self.graph.directed:
            return float("nan")
        spectrum = self.normalized_laplacian_spectrum()
        if len(spectrum) < 2:
            return 0.0
        return float(spectrum[1])

    def spectral_statistics(self, spectrum: list[float]) -> dict[str, float]:
        """Compute and return statistical metrics for a given spectrum."""
        return self._compute_stats(np.asarray(spectrum, dtype=float)) if spectrum else self._empty_stats()