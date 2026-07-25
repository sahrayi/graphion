"""
Partition evaluation utilities.

This module provides structural evaluation metrics
for Graphora PartitionSet objects.
"""

from __future__ import annotations

from statistics import mean, median, pstdev

import numpy as np

from graphion.core.models import PartitionSet


class PartitionSetEvaluation:
    """
    Evaluate structural properties of a PartitionSet.

    Parameters
    ----------
    partition_set:
        Graphora PartitionSet instance.

    Notes
    -----
    All metrics are computed lazily.
    """

    def __init__(self, partition_set: PartitionSet) -> None:
        self.partition_set = partition_set

    @property
    def partition_count(self) -> int:
        """Return the total number of partitions in the set."""
        return self.partition_set.partition_count

    @property
    def node_count(self) -> int:
        """Return the total number of nodes across all partitions."""
        return self.partition_set.node_count

    @property
    def partition_sizes(self) -> list[int]:
        """Return a list containing the size of each partition."""
        return list(self.partition_set.partition_sizes)

    def _size_statistics(self, values: list[int]) -> dict[str, float]:
        """Compute comprehensive statistical metrics (min, max, mean, median, std) for partition sizes."""
        if not values:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
        values_arr = np.asarray(values, dtype=float)
        return {
            "min": float(values_arr.min()),
            "max": float(values_arr.max()),
            "mean": float(values_arr.mean()),
            "median": float(np.median(values_arr)),
            "std": float(values_arr.std()),
        }

    def average_partition_size(self) -> float:
        """Calculate and return the arithmetic mean of partition sizes."""
        sizes = self.partition_sizes
        return float(mean(sizes)) if sizes else 0.0

    def min_partition_size(self) -> int:
        """Return the size of the smallest partition."""
        sizes = self.partition_sizes
        return min(sizes) if sizes else 0

    def max_partition_size(self) -> int:
        """Return the size of the largest partition."""
        sizes = self.partition_sizes
        return max(sizes) if sizes else 0

    def median_partition_size(self) -> float:
        """Return the median value among all partition sizes."""
        sizes = self.partition_sizes
        return float(median(sizes)) if sizes else 0.0

    def partition_size_std(self) -> float:
        """Calculate and return the population standard deviation of partition sizes."""
        sizes = self.partition_sizes
        return float(pstdev(sizes)) if len(sizes) > 1 else 0.0

    def partition_size_statistics(self) -> dict[str, float]:
        """Return comprehensive statistical summary for partition sizes."""
        return self._size_statistics(self.partition_sizes)

    def balance_ratio(self) -> float:
        """
        Smallest partition divided by largest partition.

        Returns
        -------
        float
            1.0 means perfectly balanced.
            0.0 means highly imbalanced.
        """
        sizes = self.partition_sizes
        if not sizes:
            return 0.0
        largest = max(sizes)
        return float(min(sizes) / largest) if largest != 0 else 0.0

    def coefficient_of_variation(self) -> float:
        """Calculate and return the relative dispersion (coefficient of variation) of partition sizes."""
        sizes = self.partition_sizes
        if not sizes:
            return 0.0
        avg = mean(sizes)
        return float(self.partition_size_std() / avg) if avg != 0 else 0.0

    def partition_size_entropy(self) -> float:
        """Calculate and return the Shannon entropy of the partition size distribution."""
        sizes = self.partition_sizes
        if not sizes:
            return 0.0
        probabilities = np.asarray(sizes, dtype=float)
        total = probabilities.sum()
        if total == 0:
            return 0.0
        probabilities /= total
        probabilities = probabilities[probabilities > 0]
        return float(-np.sum(probabilities * np.log2(probabilities)))

    def gini_coefficient(self) -> float:
        """
        Gini coefficient of partition sizes.

        0.0 -> perfectly balanced
        1.0 -> maximally imbalanced
        """
        sizes = np.asarray(self.partition_sizes, dtype=float)
        if sizes.size == 0 or np.all(sizes == 0):
            return 0.0
        sizes = np.sort(sizes)
        n = sizes.size
        index = np.arange(1, n + 1)
        numerator = np.sum((2 * index - n - 1) * sizes)
        denominator = n * sizes.sum()
        return float(numerator / denominator)

    def largest_partition_ratio(self) -> float:
        """Calculate and return the fraction of total nodes contained within the largest partition."""
        if self.node_count == 0:
            return 0.0
        sizes = self.partition_sizes
        return float(max(sizes) / self.node_count) if sizes else 0.0

    def effective_partition_count(self) -> float:
        """
        Hill number of order 1 (Perplexity).

        Equal to:
            2 ** Shannon entropy
        """
        return float(2 ** self.partition_size_entropy())

    def singleton_partition_count(self) -> int:
        """Count and return the number of partitions that contain exactly one node."""
        return sum(size == 1 for size in self.partition_sizes)

    def singleton_ratio(self) -> float:
        """Calculate and return the fraction of singleton partitions relative to total partitions."""
        if self.partition_count == 0:
            return 0.0
        return float(self.singleton_partition_count() / self.partition_count)

    def non_singleton_partition_count(self) -> int:
        """Count and return the number of partitions containing two or more nodes."""
        return sum(size > 1 for size in self.partition_sizes)

    def non_singleton_ratio(self) -> float:
        """Calculate and return the fraction of non-singleton partitions relative to total partitions."""
        if self.partition_count == 0:
            return 0.0
        return float(self.non_singleton_partition_count() / self.partition_count)

    def size_histogram(self) -> dict[int, int]:
        """
        Histogram of partition sizes.
        Returns
        -------
        dict
            key = partition size
            value = number of partitions
        """
        histogram: dict[int, int] = {}
        for size in self.partition_sizes:
            histogram[size] = histogram.get(size, 0) + 1
        return dict(sorted(histogram.items()))