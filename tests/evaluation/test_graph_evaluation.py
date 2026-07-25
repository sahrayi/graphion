from __future__ import annotations

import numpy as np
import pytest
import networkx as nx

from graphion.core.models import Graph
from graphion.core.models.edge import Edge
from graphion.evaluation.graph_evaluation import GraphEvaluation


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def undirected_graph():
    """Create a simple undirected graph for testing using immutable chain methods."""
    g = (
        Graph.empty(directed=False)
        .add_edge(1, 2, weight=1.0)
        .add_edge(2, 3, weight=2.0)
        .add_edge(3, 4, weight=1.0)
        .add_edge(4, 1, weight=1.0)
        .add_edge(1, 3, weight=1.5)
    )
    return g


@pytest.fixture
def directed_graph():
    """Create a simple directed graph for testing using Edge dataclass."""
    g = Graph.from_edges(
        [
            Edge(source=1, target=2, weight=1.0),
            Edge(source=2, target=3, weight=2.0),
            Edge(source=3, target=1, weight=1.5),
        ],
        directed=True,
    )
    return g


@pytest.fixture
def empty_graph():
    """Create an empty graph."""
    return Graph.empty()


# ============================================================
# Basic Properties Tests
# ============================================================

class TestGraphEvaluationBasic:

    def test_basic_properties(self, undirected_graph):
        evaluator = GraphEvaluation(undirected_graph)
        assert evaluator.node_count == 4
        assert evaluator.edge_count == 5
        assert isinstance(evaluator.nx_graph, nx.Graph)

    def test_empty_graph_properties(self, empty_graph):
        evaluator = GraphEvaluation(empty_graph)
        assert evaluator.node_count == 0
        assert evaluator.edge_count == 0
        assert evaluator.density() == 0.0
        assert evaluator.is_connected() is True
        assert evaluator.connected_component_count() == 0


# ============================================================
# Degree Statistics Tests
# ============================================================

class TestGraphEvaluationDegrees:

    def test_degree_metrics(self, undirected_graph):
        evaluator = GraphEvaluation(undirected_graph)
        assert evaluator.average_degree() == pytest.approx(2.5)
        assert evaluator.min_degree() == 2
        assert evaluator.max_degree() == 3
        assert evaluator.median_degree() == 2.5
        assert evaluator.degree_std() >= 0.0

    def test_degree_statistics_dict(self, undirected_graph):
        evaluator = GraphEvaluation(undirected_graph)
        stats = evaluator.degree_statistics()
        assert set(stats.keys()) == {"min", "max", "mean", "median", "std"}
        assert stats["min"] == 2.0
        assert stats["max"] == 3.0

    def test_degree_entropy(self, undirected_graph):
        evaluator = GraphEvaluation(undirected_graph)
        entropy = evaluator.degree_entropy()
        assert entropy >= 0.0


# ============================================================
# Connectivity & Path Tests
# ============================================================

class TestGraphEvaluationConnectivity:

    def test_connectivity_undirected(self, undirected_graph):
        evaluator = GraphEvaluation(undirected_graph)
        assert evaluator.is_connected() is True
        assert evaluator.connected_component_count() == 1
        assert evaluator.edge_connectivity() >= 1.0
        assert evaluator.node_connectivity() >= 1.0

    def test_connectivity_directed(self, directed_graph):
        evaluator = GraphEvaluation(directed_graph)
        assert evaluator.is_connected() is True
        assert evaluator.connected_component_count() == 1

    def test_paths_and_distances(self, undirected_graph):
        evaluator = GraphEvaluation(undirected_graph)
        assert evaluator.diameter() >= 1.0
        assert evaluator.radius() >= 1.0
        assert evaluator.average_shortest_path_length() >= 1.0
        assert evaluator.wiener_index() > 0.0


# ============================================================
# Clustering & Efficiency Tests
# ============================================================

class TestGraphEvaluationClustering:

    def test_clustering_coefficients(self, undirected_graph):
        evaluator = GraphEvaluation(undirected_graph)
        assert 0.0 <= evaluator.average_clustering() <= 1.0
        assert 0.0 <= evaluator.transitivity() <= 1.0
        assert evaluator.triangle_count() > 0
        assert 0.0 <= evaluator.square_clustering() <= 1.0

    def test_directed_clustering_edge_case(self, directed_graph):
        evaluator = GraphEvaluation(directed_graph)
        assert np.isnan(evaluator.average_clustering())
        assert np.isnan(evaluator.transitivity())
        assert evaluator.triangle_count() == 0

    def test_efficiency(self, undirected_graph):
        evaluator = GraphEvaluation(undirected_graph)
        assert 0.0 <= evaluator.global_efficiency() <= 1.0
        assert 0.0 <= evaluator.local_efficiency() <= 1.0

    def test_assortativity(self, undirected_graph):
        evaluator = GraphEvaluation(undirected_graph)
        assert -1.0 <= evaluator.assortativity_coefficient() <= 1.0


# ============================================================
# Centrality Tests
# ============================================================

class TestGraphEvaluationCentralities:

    def test_centrality_dictionaries(self, undirected_graph):
        evaluator = GraphEvaluation(undirected_graph)

        for method in [
            evaluator.degree_centrality,
            evaluator.betweenness_centrality,
            evaluator.closeness_centrality,
            evaluator.eigenvector_centrality,
            evaluator.pagerank,
            evaluator.katz_centrality,
            evaluator.harmonic_centrality,
            evaluator.load_centrality,
        ]:
            res = method()
            assert isinstance(res, dict)
            assert len(res) == 4

    def test_centrality_statistics(self, undirected_graph):
        evaluator = GraphEvaluation(undirected_graph)
        stats = evaluator.degree_centrality_statistics()
        assert "mean" in stats
        assert "std" in stats

    def test_core_number(self, undirected_graph):
        evaluator = GraphEvaluation(undirected_graph)
        cores = evaluator.core_number()
        assert isinstance(cores, dict)
        k_stats = evaluator.k_core_statistics()
        assert "mean" in k_stats


# ============================================================
# Spectral Analysis Tests
# ============================================================

class TestGraphEvaluationSpectrals:

    def test_edge_weight_statistics(self, undirected_graph):
        evaluator = GraphEvaluation(undirected_graph)
        stats = evaluator.edge_weight_statistics()
        assert stats["min"] == 1.0
        assert stats["max"] == 2.0

    def test_adjacency_spectrum(self, undirected_graph):
        evaluator = GraphEvaluation(undirected_graph)
        spec = evaluator.adjacency_spectrum()
        assert len(spec) == 4
        assert evaluator.spectral_radius() > 0.0
        assert evaluator.adjacency_spectral_gap() >= 0.0

    def test_laplacian_spectrum(self, undirected_graph):
        evaluator = GraphEvaluation(undirected_graph)
        lap_spec = evaluator.laplacian_spectrum()
        assert len(lap_spec) == 4
        assert evaluator.laplacian_zero_eigenvalue_count() >= 1
        assert evaluator.algebraic_connectivity() >= 0.0

    def test_normalized_laplacian(self, undirected_graph):
        evaluator = GraphEvaluation(undirected_graph)
        norm_spec = evaluator.normalized_laplacian_spectrum()
        assert len(norm_spec) == 4
        assert evaluator.normalized_algebraic_connectivity() >= 0.0

    def test_directed_spectral_edge_case(self, directed_graph):
        evaluator = GraphEvaluation(directed_graph)
        assert evaluator.laplacian_spectrum() == []
        assert np.isnan(evaluator.adjacency_spectral_gap())
        assert np.isnan(evaluator.normalized_algebraic_connectivity())