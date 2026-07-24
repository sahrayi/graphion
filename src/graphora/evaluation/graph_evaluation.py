"""
Graph evaluation utilities.

This module provides structural evaluation metrics
for Graphora Graph objects.

The goal is to compare graph builders from a
research perspective.

Examples
--------
>>> evaluator = GraphEvaluation(graph)
"""

from __future__ import annotations

from collections import Counter
from statistics import (
    mean,
    median,
    pstdev,
)

import numpy as np
import networkx as nx
from typing import Any

from graphora.core.models import Graph


class GraphEvaluation:
    """
    Evaluate structural properties of a graph.

    Parameters
    ----------
    graph:
        Graphora Graph instance.

    Notes
    -----
    Evaluation is intentionally lazy.

    Every metric is computed only when requested.
    """

    def __init__(
        self,
        graph: Graph,
    ) -> None:

        self.graph = graph

        self._nx_graph = graph.to_networkx()

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    @property
    def nx_graph(self):

        return self._nx_graph

    @property
    def node_count(self) -> int:

        return self.graph.node_count

    @property
    def edge_count(self) -> int:

        return self.graph.edge_count

    def _degrees(self) -> list[int]:

        if self.graph.directed:
            return [
                degree
                for _, degree
                in self.nx_graph.out_degree()
            ]

        return [
            degree
            for _, degree
            in self.nx_graph.degree()
        ]

    # ==================================================
    # Basic graph statistics
    # ==================================================

    def density(self) -> float:

        if self.node_count <= 1:
            return 0.0

        return float(
            nx.density(
                self.nx_graph,
            )
        )

    def average_degree(self) -> float:

        degrees = self._degrees()

        if not degrees:
            return 0.0

        return mean(
            degrees,
        )

    def min_degree(self) -> int:

        degrees = self._degrees()

        if not degrees:
            return 0

        return min(
            degrees,
        )

    def max_degree(self) -> int:

        degrees = self._degrees()

        if not degrees:
            return 0

        return max(
            degrees,
        )

    def median_degree(self) -> float:

        degrees = self._degrees()

        if not degrees:
            return 0.0

        return median(
            degrees,
        )

    def degree_std(self) -> float:

        degrees = self._degrees()

        if len(degrees) <= 1:
            return 0.0

        return pstdev(
            degrees,
        )

    def degree_statistics(
            self,
    ) -> dict[str, float]:

        return {
            "min": float(
                self.min_degree()
            ),
            "max": float(
                self.max_degree()
            ),
            "mean": float(
                self.average_degree()
            ),
            "median": float(
                self.median_degree()
            ),
            "std": float(
                self.degree_std()
            ),
        }

    def degree_entropy(
            self,
    ) -> float:

        degrees = self._degrees()

        if not degrees:
            return 0.0

        counts = Counter(degrees)

        probabilities = np.asarray(
            list(counts.values()),
            dtype=float,
        )

        probabilities /= probabilities.sum()

        return float(
            -np.sum(
                probabilities * np.log2(probabilities)
            )
        )

    # ==================================================
    # Connectivity
    # ==================================================

    def is_weakly_connected(self) -> bool:

        if self.node_count == 0:
            return True

        if self.graph.directed:
            return nx.is_strongly_connected(
                self.nx_graph,
            )

        return nx.is_connected(
            self.nx_graph,
        )

    def edge_connectivity(self) -> float:

        if self.node_count <= 1:
            return 0.0

        try:
            return float(
                nx.edge_connectivity(
                    self.nx_graph,
                )
            )

        except nx.NetworkXError:
            return float("nan")

    def node_connectivity(self) -> float:

        if self.node_count <= 1:
            return 0.0

        try:
            return float(
                nx.node_connectivity(
                    self.nx_graph,
                )
            )

        except nx.NetworkXError:
            return float("nan")

    def average_clustering(
        self,
    ) -> float:

        if self.graph.directed:
            return 0.0

        return nx.average_clustering(
            self.nx_graph,
            weight="weight",
        )

    def transitivity(
        self,
    ) -> float:

        if self.graph.directed:
            return 0.0

        return nx.transitivity(
            self.nx_graph,
        )

    def connected_component_count(
            self,
    ) -> int:

        if self.graph.directed:
            return nx.number_strongly_connected_components(
                self.nx_graph,
            )

        return nx.number_connected_components(
            self.nx_graph,
        )

    # ==================================================
    # Path metrics
    # ==================================================

    def _largest_component_graph(
            self,
    ) -> nx.Graph:

        graph = self.nx_graph

        if graph.number_of_nodes() == 0:
            return graph

        if self.graph.directed:
            components = nx.strongly_connected_components(graph)
        else:
            components = nx.connected_components(graph)

        largest = max(
            components,
            key=len,
            default=None,
        )

        if largest is None:
            return graph

        return graph.subgraph(
            largest
        ).copy()

    def diameter(self) -> float:

        graph = self._largest_component_graph()

        if graph.number_of_nodes() <= 1:
            return 0.0

        try:
            return float(
                nx.diameter(
                    graph,
                )
            )

        except Exception:
            return float("nan")

    def radius(self) -> float:

        graph = self._largest_component_graph()

        if graph.number_of_nodes() <= 1:
            return 0.0

        try:
            return float(
                nx.radius(
                    graph,
                )
            )

        except Exception:
            return float("nan")

    def average_shortest_path_length(
        self,
    ) -> float:

        graph = self._largest_component_graph()

        if graph.number_of_nodes() <= 1:
            return 0.0

        try:

            return float(
                nx.average_shortest_path_length(
                    graph,
                )
            )

        except Exception:

            return float("nan")

    def global_efficiency(
        self,
    ) -> float:

        if self.graph.directed:
            return float("nan")

        try:

            return float(
                nx.global_efficiency(
                    self.nx_graph,
                )
            )

        except Exception:

            return float("nan")

    def local_efficiency(
        self,
    ) -> float:

        if self.graph.directed:
            return float("nan")

        try:

            return float(
                nx.local_efficiency(
                    self.nx_graph,
                )
            )

        except Exception:

            return float("nan")

    def eccentricity_statistics(
        self,
    ) -> dict[str, float]:

        graph = self._largest_component_graph()

        if graph.number_of_nodes() <= 1:

            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }

        try:

            values = list(
                nx.eccentricity(
                    graph,
                ).values()
            )

        except Exception:

            return {
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
            }

        return {
            "min": float(min(values)),
            "max": float(max(values)),
            "mean": float(mean(values)),
            "median": float(median(values)),
            "std": (
                float(pstdev(values))
                if len(values) > 1
                else 0.0
            ),
        }

    def wiener_index(
        self,
    ) -> float:

        graph = self._largest_component_graph()

        try:

            return float(
                nx.wiener_index(
                    graph,
                )
            )

        except Exception:

            return float("nan")

    # ==================================================
    # Centrality metrics
    # ==================================================

    def _centrality_statistics(
            self,
            values: dict,
    ) -> dict[str, float]:

        if not values:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }

        numbers = np.asarray(
            list(values.values()),
            dtype=float,
        )

        return {
            "min": float(numbers.min()),
            "max": float(numbers.max()),
            "mean": float(numbers.mean()),
            "median": float(np.median(numbers)),
            "std": float(numbers.std()),
        }

    def degree_centrality(
            self,
    ) -> dict:

        try:

            if self.graph.directed:
                return nx.out_degree_centrality(
                    self.nx_graph,
                )

            return nx.degree_centrality(
                self.nx_graph,
            )

        except Exception:

            return {}

    def degree_centrality_statistics(
        self,
    ) -> dict[str, float]:

        return self._centrality_statistics(
            self.degree_centrality(),
        )

    def betweenness_centrality(self) -> dict:

        try:
            return nx.betweenness_centrality(
                self.nx_graph,
                weight="weight",
            )

        except Exception:
            return {}

    def betweenness_centrality_statistics(
        self,
    ) -> dict[str, float]:

        return self._centrality_statistics(
            self.betweenness_centrality(),
        )

    def closeness_centrality(self) -> dict:

        try:
            return nx.closeness_centrality(
                self.nx_graph,
            )

        except Exception:
            return {}

    def closeness_centrality_statistics(
        self,
    ) -> dict[str, float]:

        return self._centrality_statistics(
            self.closeness_centrality(),
        )

    def eigenvector_centrality(
        self,
    ) -> dict:

        try:

            return nx.eigenvector_centrality(
                self.nx_graph,
                weight="weight",
                max_iter=1000,
            )

        except Exception:

            return {}

    def eigenvector_centrality_statistics(
        self,
    ) -> dict[str, float]:

        return self._centrality_statistics(
            self.eigenvector_centrality(),
        )

    def pagerank(self) -> dict:

        try:
            return nx.pagerank(
                self.nx_graph,
                weight="weight",
            )

        except Exception:
            return {}

    def pagerank_statistics(
        self,
    ) -> dict[str, float]:

        return self._centrality_statistics(
            self.pagerank(),
        )

    def katz_centrality(
        self,
    ) -> dict:

        try:

            return nx.katz_centrality(
                self.nx_graph,
                weight="weight",
            )

        except Exception:

            return {}

    def katz_centrality_statistics(
        self,
    ) -> dict[str, float]:

        return self._centrality_statistics(
            self.katz_centrality(),
        )

    def harmonic_centrality(self) -> dict:

        try:
            return nx.harmonic_centrality(
                self.nx_graph,
            )

        except Exception:
            return {}

    def harmonic_centrality_statistics(
        self,
    ) -> dict[str, float]:

        return self._centrality_statistics(
            self.harmonic_centrality(),
        )

    def load_centrality(
            self,
    ) -> dict[Any, float]:

        try:

            values = nx.load_centrality(
                self.nx_graph,
                weight="weight",
            )

            if not isinstance(values, dict):
                return {}

            return {
                node: float(score)
                for node, score in values.items()
            }

        except Exception:

            return {}

    def load_centrality_statistics(
        self,
    ) -> dict[str, float]:

        return self._centrality_statistics(
            self.load_centrality(),
        )

    # ==================================================
    # Core metrics
    # ==================================================

    def core_number(
            self,
    ) -> dict:

        if self.graph.directed:
            return {}

        try:
            return nx.core_number(
                self.nx_graph
            )

        except Exception:
            return {}

    def k_core_statistics(
            self,
    ) -> dict[str, float]:

        values = list(
            self.core_number().values()
        )

        if not values:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }

        return {
            "min": float(min(values)),
            "max": float(max(values)),
            "mean": float(mean(values)),
            "median": float(median(values)),
            "std": float(
                pstdev(values)
                if len(values) > 1
                else 0.0
            ),
        }

    # ==================================================
    # Community and structural metrics
    # ==================================================

    def clustering_statistics(
            self,
    ) -> dict[str, float]:
        """
        Statistics of local clustering coefficients.

        Returns
        -------
        dict
            min / max / mean / median / std of node clustering coefficients.
        """

        if self.graph.directed:
            return {
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
            }

        try:

            coefficients = nx.clustering(
                self.nx_graph,
                weight="weight",
            )

            if not isinstance(coefficients, dict):
                return {
                    "min": float("nan"),
                    "max": float("nan"),
                    "mean": float("nan"),
                    "median": float("nan"),
                    "std": float("nan"),
                }

            values: dict[Any, float] = {
                node: float(value)
                for node, value in coefficients.items()
            }

        except Exception:

            return {
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
            }

        return self._centrality_statistics(values)


    def triangle_count(
        self,
    ) -> int:

        if self.graph.directed:
            return 0

        try:

            triangles = nx.triangles(
                self.nx_graph,
            )

            return int(
                sum(triangles.values())
                // 3
            )

        except Exception:

            return 0

    def square_clustering(
            self,
    ) -> float:

        if self.graph.directed:
            return float("nan")

        try:

            values = nx.square_clustering(
                self.nx_graph,
            )

            if not values:
                return 0.0

            return float(
                np.mean(
                    list(values.values())
                )
            )

        except Exception:

            return float("nan")

    def largest_connected_component_ratio(
            self,
    ) -> float:

        if self.node_count == 0:
            return 0.0

        graph = self._largest_component_graph()

        return (
                graph.number_of_nodes()
                / self.node_count
        )


    def assortativity_coefficient(
        self,
    ) -> float:

        try:

            return float(
                nx.degree_assortativity_coefficient(
                    self.nx_graph,
                )
            )

        except Exception:

            return float("nan")

    def edge_weight_statistics(
            self,
    ) -> dict[str, float]:

        weights = np.fromiter(
            (
                edge.weight
                for edge in self.graph.edges
            ),
            dtype=float,
        )

        if weights.size == 0:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }

        return {
            "min": float(weights.min()),
            "max": float(weights.max()),
            "mean": float(weights.mean()),
            "median": float(np.median(weights)),
            "std": float(weights.std()),
        }

    # ==================================================
    # Spectral metrics
    # ==================================================

    def adjacency_spectrum(
            self,
    ) -> list[float]:
        """For directed graphs the absolute value of eigenvalues is returned."""

        if self.node_count == 0:
            return []

        try:

            matrix = nx.to_numpy_array(
                self.nx_graph,
                weight="weight",
            )

            if self.graph.directed:

                values = np.linalg.eigvals(matrix)

                spectrum = [
                    float(abs(v))
                    for v in values
                ]

            else:

                values = np.linalg.eigvalsh(matrix)

                spectrum = [
                    float(v)
                    for v in values
                ]

            return sorted(
                spectrum,
                reverse=True,
            )

        except Exception:

            return []

    def spectral_radius(
            self,
    ) -> float:
        """
        Largest absolute adjacency eigenvalue.
        """

        spectrum = self.adjacency_spectrum()

        if not spectrum:
            return 0.0

        return float(
            max(spectrum)
        )

    def laplacian_spectrum(
            self,
    ) -> list[float]:

        if self.graph.directed:
            return []

        if self.node_count == 0:
            return []

        try:

            laplacian = nx.laplacian_matrix(
                self.nx_graph,
                weight="weight",
            ).toarray()

            values = np.linalg.eigvalsh(
                laplacian,
            )

            return sorted(
                (
                    float(v)
                    for v in values
                )
            )

        except Exception:

            return []

    def laplacian_zero_eigenvalue_count(
            self,
    ) -> int:
        """
        Number of zero Laplacian eigenvalues.

        Equals the number of connected components.
        """

        spectrum = self.laplacian_spectrum()

        tolerance = 1e-10

        return sum(
            abs(v) < tolerance
            for v in spectrum
        )


    def algebraic_connectivity(
        self,
    ) -> float:

        """
        Second smallest Laplacian eigenvalue.

        Measures graph connectivity strength.
        """

        spectrum = self.laplacian_spectrum()


        if len(spectrum) < 2:
            return 0.0


        return float(
            spectrum[1]
        )

    def adjacency_spectral_gap(
            self,
    ) -> float:
        """
        Difference between largest and second largest
        absolute adjacency eigenvalues.
        """

        if self.graph.directed:
            return float("nan")

        spectrum = self.adjacency_spectrum()

        if len(spectrum) < 2:
            return 0.0

        values = sorted(
            (
                abs(value)
                for value in spectrum
            ),
            reverse=True,
        )

        return float(
            values[0] - values[1]
        )

    def normalized_laplacian_spectrum(
            self,
    ) -> list[float]:

        if self.graph.directed:
            return []

        if self.node_count == 0:
            return []

        try:

            laplacian = (
                nx.normalized_laplacian_matrix(
                    self.nx_graph,
                    weight="weight",
                )
                .toarray()
            )

            values = np.linalg.eigvalsh(
                laplacian,
            )

            return sorted(
                (
                    float(v)
                    for v in values
                )
            )

        except Exception:

            return []

    def normalized_laplacian_gap(
            self,
    ) -> float:

        if self.graph.directed:
            return float("nan")

        spectrum = (
            self.normalized_laplacian_spectrum()
        )

        if len(spectrum) < 2:
            return 0.0

        return float(
            spectrum[1] - spectrum[0]
        )

    def spectral_statistics(
            self,
            spectrum: list[float],
    ) -> dict[str, float]:

        if not spectrum:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }

        values = np.asarray(
            spectrum,
            dtype=float,
        )

        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std()),
        }

