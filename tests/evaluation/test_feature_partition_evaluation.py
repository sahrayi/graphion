from __future__ import annotations

import numpy as np
import pytest

from graphion.core.models import FeatureSet, PartitionSet
from graphion.core.errors import InvalidPartitionSetError
from graphion.evaluation.feature_partition_evaluation import FeaturePartitionEvaluation


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def feature_set():
    """
    Two clearly separated clusters:

    Cluster 0: (0,0), (1,0), (0,1)
    Cluster 1: (10,10), (11,10), (10,11)
    """
    return FeatureSet.from_lists(
        ids=[1, 2, 3, 4, 5, 6],
        features=[
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (10.0, 10.0),
            (11.0, 10.0),
            (10.0, 11.0),
        ],
    )


@pytest.fixture
def partition_set():
    return PartitionSet.from_labels(
        ids=[1, 2, 3, 4, 5, 6],
        labels=[0, 0, 0, 1, 1, 1],
    )


@pytest.fixture
def evaluation(feature_set, partition_set):
    return FeaturePartitionEvaluation(feature_set, partition_set)


@pytest.fixture
def single_cluster_partition(feature_set):
    return PartitionSet.from_labels(
        ids=[1, 2, 3, 4, 5, 6],
        labels=[0, 0, 0, 0, 0, 0],
    )


@pytest.fixture
def valid_data():
    fs = FeatureSet.from_lists(
        ids=[1, 2, 3, 4],
        features=[
            (1.0, 1.0),
            (1.5, 1.5),
            (5.0, 5.0),
            (5.5, 5.5),
        ],
    )
    ps = PartitionSet.from_labels(
        ids=[1, 2, 3, 4],
        labels=[0, 0, 1, 1],
    )
    return fs, ps


@pytest.fixture
def evaluator(valid_data):
    fs, ps = valid_data
    return FeaturePartitionEvaluation(fs, ps)


# ============================================================
# Initialization & Validation Tests
# ============================================================

class TestInitialization:

    def test_initialization(self, evaluation):
        assert evaluation.sample_count == 6
        assert evaluation.dimension == 2

    def test_feature_matrix_is_numpy_array(self, evaluation):
        assert isinstance(evaluation.feature_matrix, np.ndarray)

    def test_labels_are_numpy_array(self, evaluation):
        assert isinstance(evaluation.labels, np.ndarray)

    def test_invalid_partition_size(self, feature_set):
        partition = PartitionSet.from_labels(
            ids=[1, 2],
            labels=[0, 1],
        )
        with pytest.raises(InvalidPartitionSetError):
            FeaturePartitionEvaluation(feature_set, partition)

    def test_invalid_numerical_values_error(self):
        fs = FeatureSet.from_lists(ids=[1, 2], features=[(1.0, np.nan), (3.0, 4.0)])
        ps = PartitionSet.from_labels(ids=[1, 2], labels=[0, 1])
        with pytest.raises(ValueError):
            FeaturePartitionEvaluation(fs, ps)


# ============================================================
# Properties & Basic Metrics Tests
# ============================================================

class TestProperties:

    def test_sample_count(self, evaluation):
        assert evaluation.sample_count == 6

    def test_dimension(self, evaluation):
        assert evaluation.dimension == 2

    def test_cluster_count(self, evaluation):
        assert evaluation.cluster_count == 2

    def test_cluster_sizes(self, evaluation):
        assert evaluation.cluster_sizes == (3, 3)

    def test_cluster_count_and_sizes(self, evaluator):
        assert evaluator.cluster_count == 2
        assert evaluator.cluster_sizes == (2, 2)

    def test_labels_are_read_only(self, evaluation):
        labels = evaluation.labels
        assert labels.flags.writeable is False
        with pytest.raises(ValueError):
            labels[0] = 99

    def test_feature_matrix_is_read_only(self, evaluation):
        matrix = evaluation.feature_matrix
        assert matrix.flags.writeable is False
        with pytest.raises(ValueError):
            matrix[0][0] = 100


# ============================================================
# Cluster Statistics & Ratios
# ============================================================

class TestClusterStatistics:

    def test_cluster_size_statistics(self, evaluation):
        stats = evaluation.cluster_size_statistics()
        assert stats == {
            "min": 3.0,
            "max": 3.0,
            "mean": 3.0,
            "median": 3.0,
            "std": 0.0,
        }

    def test_largest_cluster_ratio(self, evaluation):
        assert evaluation.largest_cluster_ratio() == pytest.approx(0.5)

    def test_smallest_cluster_ratio(self, evaluation):
        assert evaluation.smallest_cluster_ratio() == pytest.approx(0.5)

    def test_singleton_cluster_count(self, evaluation):
        assert evaluation.singleton_cluster_count() == 0

    def test_singleton_cluster_ratio(self, evaluation):
        assert evaluation.singleton_cluster_ratio() == 0.0


# ============================================================
# Cluster Balance
# ============================================================

class TestClusterBalance:

    def test_balanced_clusters(self, evaluation):
        assert evaluation.cluster_balance() == pytest.approx(1.0)

    def test_unbalanced_clusters(self, feature_set):
        partition = PartitionSet.from_labels(
            ids=[1, 2, 3, 4, 5, 6],
            labels=[0, 0, 0, 0, 0, 1],
        )
        evaluation_unbalanced = FeaturePartitionEvaluation(
            feature_set,
            partition,
        )
        assert evaluation_unbalanced.cluster_balance() == pytest.approx(1 / 5)


# ============================================================
# Internal Cluster Helpers
# ============================================================

class TestClusterHelpers:

    def test_cluster_indices(self, evaluation):
        indices = evaluation._cluster_indices()
        assert len(indices) == 2
        assert np.array_equal(indices[0], np.array([0, 1, 2]))
        assert np.array_equal(indices[1], np.array([3, 4, 5]))

    def test_cluster_centroids(self, evaluation):
        centroids = evaluation._cluster_centroids()
        expected = np.array(
            [
                [1 / 3, 1 / 3],
                [31 / 3, 31 / 3],
            ]
        )
        assert centroids.shape == (2, 2)
        assert np.allclose(centroids, expected)


# ============================================================
# Distance & Cohesion / Separation Tests
# ============================================================

class TestDistanceMetrics:

    def test_within_and_centroid_distances(self, evaluator):
        within_stats = evaluator.within_cluster_distance_statistics()
        centroid_stats = evaluator.centroid_distance_statistics()

        assert within_stats["mean"] >= 0.0
        assert centroid_stats["mean"] > 0.0

    def test_separation_ratio(self, evaluator):
        ratio = evaluator.separation_ratio()
        assert ratio > 0.0 and np.isfinite(ratio)


# ============================================================
# Clustering Quality Scores Tests
# ============================================================

class TestClusteringQualityScores:

    def test_valid_scores(self, evaluator):
        assert -1.0 <= evaluator.silhouette_score() <= 1.0
        assert evaluator.davies_bouldin_score() >= 0.0
        assert evaluator.calinski_harabasz_score() > 0.0

    def test_single_cluster_edge_case(self):
        fs = FeatureSet.from_lists(ids=[1, 2], features=[(1.0, 2.0), (3.0, 4.0)])
        ps = PartitionSet.from_labels(ids=[1, 2], labels=[0, 0])
        eval_single = FeaturePartitionEvaluation(fs, ps)

        assert np.isnan(eval_single.silhouette_score())
        assert np.isnan(eval_single.davies_bouldin_score())
        assert np.isnan(eval_single.calinski_harabasz_score())
        assert np.isnan(eval_single.separation_ratio())
        assert np.isnan(eval_single.quality_summary()["silhouette"])

    def test_quality_summary(self, evaluator):
        summary = evaluator.quality_summary()
        expected_keys = {"silhouette", "davies_bouldin", "calinski_harabasz", "separation_ratio"}
        assert expected_keys.issubset(summary.keys())
        assert all(np.isfinite(v) for v in summary.values())