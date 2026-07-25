from __future__ import annotations

import numpy as np
import pytest

from graphora.core.models import PartitionSet
from graphora.evaluation.partition_set_evaluation import PartitionSetEvaluation


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def balanced_partition_set():
    """Create a balanced partition set: partitions of equal size."""
    # Partition 0: nodes 1, 2
    # Partition 1: nodes 3, 4
    return PartitionSet.from_labels(
        ids=[1, 2, 3, 4],
        labels=[0, 0, 1, 1],
    )


@pytest.fixture
def imbalanced_partition_set():
    """Create an imbalanced partition set with singletons and large partitions."""
    # Partition 0: nodes 1, 2, 3, 4, 5 (size 5)
    # Partition 1: node 6 (size 1 - singleton)
    # Partition 2: node 7 (size 1 - singleton)
    return PartitionSet.from_labels(
        ids=[1, 2, 3, 4, 5, 6, 7],
        labels=[0, 0, 0, 0, 0, 1, 2],
    )


@pytest.fixture
def empty_partition_set():
    """Create an empty partition set."""
    return PartitionSet.from_labels(
        ids=[],
        labels=[],
    )


# ============================================================
# Basic Properties Tests
# ============================================================

class TestPartitionSetEvaluationBasic:

    def test_basic_properties(self, balanced_partition_set):
        evaluator = PartitionSetEvaluation(balanced_partition_set)
        assert evaluator.partition_count == 2
        assert evaluator.node_count == 4
        assert sorted(evaluator.partition_sizes) == [2, 2]

    def test_empty_partition_properties(self, empty_partition_set):
        evaluator = PartitionSetEvaluation(empty_partition_set)
        assert evaluator.partition_count == 0
        assert evaluator.node_count == 0
        assert evaluator.partition_sizes == []
        assert evaluator.average_partition_size() == 0.0
        assert evaluator.min_partition_size() == 0
        assert evaluator.max_partition_size() == 0
        assert evaluator.median_partition_size() == 0.0
        assert evaluator.partition_size_std() == 0.0


# ============================================================
# Size Statistics Tests
# ============================================================

class TestPartitionSetEvaluationSizes:

    def test_size_statistics_balanced(self, balanced_partition_set):
        evaluator = PartitionSetEvaluation(balanced_partition_set)
        stats = evaluator.partition_size_statistics()
        assert stats == {
            "min": 2.0,
            "max": 2.0,
            "mean": 2.0,
            "median": 2.0,
            "std": 0.0,
        }

    def test_size_statistics_imbalanced(self, imbalanced_partition_set):
        evaluator = PartitionSetEvaluation(imbalanced_partition_set)
        stats = evaluator.partition_size_statistics()
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == pytest.approx(7 / 3)


# ============================================================
# Balance & Dispersion Metrics Tests
# ============================================================

class TestPartitionSetEvaluationDispersion:

    def test_balance_ratio(self, balanced_partition_set, imbalanced_partition_set):
        eval_bal = PartitionSetEvaluation(balanced_partition_set)
        eval_imb = PartitionSetEvaluation(imbalanced_partition_set)

        assert eval_bal.balance_ratio() == pytest.approx(1.0)
        assert eval_imb.balance_ratio() == pytest.approx(1 / 5)

    def test_coefficient_of_variation(self, balanced_partition_set, imbalanced_partition_set):
        eval_bal = PartitionSetEvaluation(balanced_partition_set)
        eval_imb = PartitionSetEvaluation(imbalanced_partition_set)

        assert eval_bal.coefficient_of_variation() == pytest.approx(0.0)
        assert eval_imb.coefficient_of_variation() > 0.0

    def test_gini_coefficient(self, balanced_partition_set, imbalanced_partition_set):
        eval_bal = PartitionSetEvaluation(balanced_partition_set)
        eval_imb = PartitionSetEvaluation(imbalanced_partition_set)

        assert eval_bal.gini_coefficient() == pytest.approx(0.0)
        assert 0.0 < eval_imb.gini_coefficient() <= 1.0


# ============================================================
# Entropy & Diversity Tests
# ============================================================

class TestPartitionSetEvaluationEntropy:

    def test_partition_size_entropy(self, balanced_partition_set):
        evaluator = PartitionSetEvaluation(balanced_partition_set)
        # For sizes [2, 2], total = 4, probabilities = [0.5, 0.5]
        # Entropy = - (0.5*log2(0.5) + 0.5*log2(0.5)) = 1.0
        assert evaluator.partition_size_entropy() == pytest.approx(1.0)

    def test_effective_partition_count(self, balanced_partition_set):
        evaluator = PartitionSetEvaluation(balanced_partition_set)
        # 2 ** 1.0 = 2.0
        assert evaluator.effective_partition_count() == pytest.approx(2.0)

    def test_empty_entropy_edge_case(self, empty_partition_set):
        evaluator = PartitionSetEvaluation(empty_partition_set)
        assert evaluator.partition_size_entropy() == 0.0
        assert evaluator.effective_partition_count() == 1.0


# ============================================================
# Singleton & Composition Tests
# ============================================================

class TestPartitionSetEvaluationSingletons:

    def test_singleton_metrics_balanced(self, balanced_partition_set):
        evaluator = PartitionSetEvaluation(balanced_partition_set)
        assert evaluator.singleton_partition_count() == 0
        assert evaluator.singleton_ratio() == 0.0
        assert evaluator.non_singleton_partition_count() == 2
        assert evaluator.non_singleton_ratio() == 1.0

    def test_singleton_metrics_imbalanced(self, imbalanced_partition_set):
        evaluator = PartitionSetEvaluation(imbalanced_partition_set)
        # Sizes: [5, 1, 1] -> 2 singletons, 1 non-singleton, total partitions = 3
        assert evaluator.singleton_partition_count() == 2
        assert evaluator.singleton_ratio() == pytest.approx(2 / 3)
        assert evaluator.non_singleton_partition_count() == 1
        assert evaluator.non_singleton_ratio() == pytest.approx(1 / 3)

    def test_largest_partition_ratio(self, imbalanced_partition_set):
        evaluator = PartitionSetEvaluation(imbalanced_partition_set)
        # Max size = 5, total nodes = 7
        assert evaluator.largest_partition_ratio() == pytest.approx(5 / 7)

    def test_size_histogram(self, imbalanced_partition_set):
        evaluator = PartitionSetEvaluation(imbalanced_partition_set)
        histogram = evaluator.size_histogram()
        # Sizes: 5 appears once, 1 appears twice
        assert histogram == {1: 2, 5: 1}