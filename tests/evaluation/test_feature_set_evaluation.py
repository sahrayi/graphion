from __future__ import annotations

import numpy as np
import pytest

from graphion.core.models import FeatureSet
from graphion.evaluation.feature_set_evaluation import FeatureSetEvaluation

# ============================================================
# Constants & Fixtures
# ============================================================

EMPTY_STATS = {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}

@pytest.fixture
def numeric_feature_set():
    return FeatureSet.from_lists(ids=[1, 2, 3], features=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])

@pytest.fixture
def empty_feature_set():
    return FeatureSet.from_lists([], [])

@pytest.fixture
def non_numeric_feature_set():
    return FeatureSet.from_lists(ids=[1, 2, 3], features=[{"a", "b"}, {"a"}, {"c"}])

@pytest.fixture
def evaluation(numeric_feature_set):
    return FeatureSetEvaluation(numeric_feature_set)

# ============================================================
# Tests
# ============================================================

class TestProperties:
    def test_feature_count(self, evaluation):
        assert evaluation.feature_count == 3

    def test_dimension(self, evaluation):
        assert evaluation.dimension == 2

    def test_is_empty_false(self, evaluation):
        assert evaluation.is_empty is False

    def test_empty_feature_set_properties(self, empty_feature_set):
        ev = FeatureSetEvaluation(empty_feature_set)
        assert ev.feature_count == 0
        assert ev.dimension is None
        assert ev.is_empty is True


class TestNumericDetection:
    def test_numeric_feature_set(self, evaluation):
        assert evaluation.is_numeric() is True

    def test_non_numeric_feature_set(self, non_numeric_feature_set):
        assert FeatureSetEvaluation(non_numeric_feature_set).is_numeric() is False

    def test_empty_feature_set_is_numeric(self, empty_feature_set):
        assert FeatureSetEvaluation(empty_feature_set).is_numeric() is True


class TestFeatureMatrixShape:
    def test_numeric_matrix_shape(self, evaluation):
        assert evaluation.feature_matrix_shape() == (3, 2)

    def test_empty_matrix_shape(self, empty_feature_set):
        assert FeatureSetEvaluation(empty_feature_set).feature_matrix_shape() == (0, 0)

    def test_non_numeric_matrix_shape(self, non_numeric_feature_set):
        assert FeatureSetEvaluation(non_numeric_feature_set).feature_matrix_shape() is None


class TestFeatureDimensions:
    def test_feature_dimensions(self, evaluation):
        assert evaluation.feature_dimensions() == [2, 2, 2]

    def test_empty_feature_dimensions(self, empty_feature_set):
        assert FeatureSetEvaluation(empty_feature_set).feature_dimensions() == []

    def test_non_numeric_dimensions(self, non_numeric_feature_set):
        assert FeatureSetEvaluation(non_numeric_feature_set).feature_dimensions() == []


class TestFeatureValueStatistics:
    def test_numeric_statistics(self, evaluation):
        stats = evaluation.feature_value_statistics()
        assert stats["min"] == 1.0
        assert stats["max"] == 6.0
        assert stats["mean"] == pytest.approx(3.5)
        assert stats["median"] == pytest.approx(3.5)
        assert stats["std"] == pytest.approx(np.std([1, 2, 3, 4, 5, 6]))

    def test_empty_statistics(self, empty_feature_set):
        assert FeatureSetEvaluation(empty_feature_set).feature_value_statistics() == EMPTY_STATS

    def test_non_numeric_statistics(self, non_numeric_feature_set):
        stats = FeatureSetEvaluation(non_numeric_feature_set).feature_value_statistics()
        for value in stats.values():
            assert np.isnan(value)


class TestDimensionStatistics:
    def test_dimension_statistics(self, evaluation):
        assert evaluation.dimension_statistics() == {"min": 2.0, "max": 2.0, "mean": 2.0, "median": 2.0, "std": 0.0}

    def test_empty_dimension_statistics(self, empty_feature_set):
        assert FeatureSetEvaluation(empty_feature_set).dimension_statistics() == EMPTY_STATS

    def test_non_numeric_dimension_statistics(self, non_numeric_feature_set):
        assert FeatureSetEvaluation(non_numeric_feature_set).dimension_statistics() == EMPTY_STATS


class TestFeatureMeanStatistics:
    def test_feature_mean_statistics(self, evaluation):
        stats, expected = evaluation.feature_mean_statistics(), np.mean([[1., 2.], [3., 4.], [5., 6.]], axis=0)
        assert stats["min"] == pytest.approx(expected.min())
        assert stats["mean"] == pytest.approx(expected.mean())

    def test_empty_feature_mean_statistics(self, empty_feature_set):
        assert FeatureSetEvaluation(empty_feature_set).feature_mean_statistics() == EMPTY_STATS


class TestFeatureStdStatistics:
    def test_feature_std_statistics(self, evaluation):
        stats, expected = evaluation.feature_std_statistics(), np.std([[1., 2.], [3., 4.], [5., 6.]], axis=0)
        assert stats["min"] == pytest.approx(expected.min())
        assert stats["std"] == pytest.approx(expected.std())

    def test_empty_feature_std_statistics(self, empty_feature_set):
        assert FeatureSetEvaluation(empty_feature_set).feature_std_statistics() == EMPTY_STATS


class TestFeatureVarianceStatistics:
    def test_feature_variance_statistics(self, evaluation):
        stats, expected = evaluation.feature_variance_statistics(), np.var([[1., 2.], [3., 4.], [5., 6.]], axis=0)
        assert stats["min"] == pytest.approx(expected.min())
        assert stats["mean"] == pytest.approx(expected.mean())

    def test_empty_feature_variance_statistics(self, empty_feature_set):
        assert FeatureSetEvaluation(empty_feature_set).feature_variance_statistics() == EMPTY_STATS


class TestFeatureRangeStatistics:
    def test_feature_range_statistics(self, evaluation):
        matrix = np.array([[1., 2.], [3., 4.], [5., 6.]])
        expected = np.max(matrix, axis=0) - np.min(matrix, axis=0)
        stats = evaluation.feature_range_statistics()
        assert stats["min"] == pytest.approx(expected.min())

    def test_empty_feature_range_statistics(self, empty_feature_set):
        assert FeatureSetEvaluation(empty_feature_set).feature_range_statistics() == EMPTY_STATS


class TestFeatureMagnitudeStatistics:
    def test_feature_magnitude_statistics(self, evaluation):
        stats, expected = evaluation.feature_magnitude_statistics(), np.nanmax(np.abs([[1., 2.], [3., 4.], [5., 6.]]), axis=0)
        assert stats["min"] == pytest.approx(expected.min())


class TestFeatureNormStatistics:
    def test_feature_norm_statistics(self, evaluation):
        stats, expected = evaluation.feature_norm_statistics(), np.linalg.norm([[1., 2.], [3., 4.], [5., 6.]], axis=1)
        assert stats["min"] == pytest.approx(expected.min())

    def test_empty_feature_norm_statistics(self, empty_feature_set):
        assert FeatureSetEvaluation(empty_feature_set).feature_norm_statistics() == EMPTY_STATS


class TestPairwiseDistanceStatistics:
    def test_pairwise_distance_statistics(self, evaluation):
        stats = evaluation.pairwise_distance_statistics()
        assert stats["mean"] > 0.0 and stats["std"] >= 0.0

    def test_single_feature_pairwise_distance(self):
        fs = FeatureSet.from_lists(ids=[1], features=[(1.0, 2.0)])
        assert FeatureSetEvaluation(fs).pairwise_distance_statistics() == EMPTY_STATS


class TestCosineSimilarityStatistics:
    def test_cosine_similarity_statistics(self, evaluation):
        stats = evaluation.cosine_similarity_statistics()
        assert -1.0 <= stats["mean"] <= 1.0 and stats["std"] >= 0.0

    def test_single_feature_cosine_similarity(self):
        fs = FeatureSet.from_lists(ids=[1], features=[(1.0, 2.0)])
        assert FeatureSetEvaluation(fs).cosine_similarity_statistics() == EMPTY_STATS


class TestNearestNeighborDistanceStatistics:
    def test_nearest_neighbor_distance_statistics(self, evaluation):
        stats = evaluation.nearest_neighbor_distance_statistics()
        assert stats["mean"] > 0.0 and stats["std"] >= 0.0

    def test_single_feature_nearest_neighbor_statistics(self):
        fs = FeatureSet.from_lists(ids=[1], features=[(1.0, 2.0)])
        assert FeatureSetEvaluation(fs).nearest_neighbor_distance_statistics() == EMPTY_STATS


class TestSparsityStatistics:
    def test_sparsity_statistics(self):
        fs = FeatureSet.from_lists(ids=[1, 2], features=[(0.0, 1.0, 0.0), (1.0, 0.0, 1.0)])
        stats = FeatureSetEvaluation(fs).sparsity_statistics()
        assert stats["min"] == pytest.approx(1 / 3)
        assert stats["mean"] == pytest.approx(0.5)

    def test_empty_sparsity_statistics(self, empty_feature_set):
        assert FeatureSetEvaluation(empty_feature_set).sparsity_statistics() == EMPTY_STATS


class TestZeroVector:
    def test_zero_vector_count(self):
        fs = FeatureSet.from_lists(ids=[1, 2, 3], features=[(0., 0.), (1., 2.), (0., 0.)])
        evaluation = FeatureSetEvaluation(fs)
        assert evaluation.zero_vector_count() == 2
        assert evaluation.zero_vector_ratio() == pytest.approx(2 / 3)

    def test_empty_zero_vector(self, empty_feature_set):
        ev = FeatureSetEvaluation(empty_feature_set)
        assert ev.zero_vector_count() == 0
        assert ev.zero_vector_ratio() == pytest.approx(0.0)


class TestDuplicateFeatures:
    def test_duplicate_feature_count(self):
        fs = FeatureSet.from_lists(ids=[1, 2, 3, 4], features=[(1., 2.), (1., 2.), (3., 4.), (3., 4.)])
        evaluation = FeatureSetEvaluation(fs)
        assert evaluation.duplicate_feature_count() == 2
        assert evaluation.duplicate_feature_ratio() == pytest.approx(0.5)

    def test_no_duplicate_features(self, evaluation):
        assert evaluation.duplicate_feature_count() == 0
        assert evaluation.duplicate_feature_ratio() == pytest.approx(0.0)


class TestConstantDimensions:
    def test_constant_dimension_count(self):
        fs = FeatureSet.from_lists(ids=[1, 2, 3], features=[(1., 5.), (2., 5.), (3., 5.)])
        evaluation = FeatureSetEvaluation(fs)
        assert evaluation.constant_dimension_count() == 1
        assert evaluation.constant_dimension_ratio() == pytest.approx(0.5)

    def test_no_constant_dimensions(self, evaluation):
        assert evaluation.constant_dimension_count() == 0
        assert evaluation.constant_dimension_ratio() == pytest.approx(0.0)


class TestInvalidValues:
    def test_invalid_value_count(self):
        fs = FeatureSet.from_lists(ids=[1, 2, 3], features=[(1., np.nan), (np.inf, 2.), (5., 6.)])
        assert FeatureSetEvaluation(fs).invalid_value_count() == 2

    def test_no_invalid_values(self, evaluation):
        assert evaluation.invalid_value_count() == 0


class TestDistanceConcentration:
    def test_distance_concentration(self, evaluation):
        value = evaluation.distance_concentration()
        assert value >= 0.0 and np.isfinite(value)

    def test_single_vector_distance_concentration(self):
        fs = FeatureSet.from_lists(ids=[1], features=[(1.0, 2.0)])
        assert FeatureSetEvaluation(fs).distance_concentration() == pytest.approx(0.0)


class TestNearestNeighborHubness:
    def test_cosine_metric(self, evaluation):
        stats = evaluation.nearest_neighbor_hub_statistics()
        assert stats["min"] >= 0 and stats["mean"] >= 0

    def test_euclidean_metric(self, evaluation):
        stats = evaluation.nearest_neighbor_hub_statistics(metric="euclidean")
        assert stats["min"] >= 0 and stats["max"] >= 0

    def test_invalid_metric(self, evaluation):
        with pytest.raises(ValueError):
            evaluation.nearest_neighbor_hub_statistics(metric="invalid")


class TestCorrelation:
    def test_feature_correlation_statistics(self, evaluation):
        stats = evaluation.feature_correlation_statistics()
        assert -1.0 <= stats["mean"] <= 1.0 and stats["std"] >= 0.0

    def test_single_dimension_returns_empty(self):
        fs = FeatureSet.from_lists(ids=[1, 2, 3], features=[(1.0,), (2.0,), (3.0,)])
        assert FeatureSetEvaluation(fs).feature_correlation_statistics() == EMPTY_STATS


class TestCovariance:
    def test_covariance_statistics(self, evaluation):
        stats = evaluation.covariance_statistics()
        assert np.isfinite(stats["mean"])

    def test_single_dimension_covariance(self):
        fs = FeatureSet.from_lists(ids=[1, 2], features=[(1.0,), (2.0,)])
        assert FeatureSetEvaluation(fs).covariance_statistics() == EMPTY_STATS


class TestEffectiveDimension:
    def test_effective_dimension(self, evaluation):
        value = evaluation.effective_dimension()
        assert 0.0 < value <= evaluation.dimension

    def test_empty_effective_dimension(self, empty_feature_set):
        assert FeatureSetEvaluation(empty_feature_set).effective_dimension() == pytest.approx(0.0)

    def test_single_sample_effective_dimension(self):
        fs = FeatureSet.from_lists(ids=[1], features=[(1.0, 2.0)])
        assert FeatureSetEvaluation(fs).effective_dimension() == pytest.approx(0.0)

class TestNonNumericMatrix:

    def test_feature_matrix_raises_type_error(
        self,
        non_numeric_feature_set,
    ):
        evaluation = FeatureSetEvaluation(
            non_numeric_feature_set
        )

        with pytest.raises(TypeError):
            evaluation._feature_matrix()