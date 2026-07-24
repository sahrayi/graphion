"""
Partition evaluation utilities.

This module provides structural evaluation metrics
for Graphora PartitionSet objects.
"""

from __future__ import annotations

from statistics import (
    mean,
    median,
    pstdev,
)

import numpy as np

from graphora.core.models import PartitionSet


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

    def __init__(
        self,
        partition_set: PartitionSet,
    ) -> None:

        self.partition_set = partition_set

    # ==================================================
    # Helpers
    # ==================================================

    @property
    def partition_count(self) -> int:

        return self.partition_set.partition_count

    @property
    def node_count(self) -> int:

        return self.partition_set.node_count

    @property
    def partition_sizes(self) -> list[int]:

        return list(
            self.partition_set.partition_sizes
        )

    def _size_statistics(
            self,
            values: list[int],
    ) -> dict[str, float]:

        if not values:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }

        values = np.asarray(
            values,
            dtype=float,
        )

        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std()),
        }

    # ==================================================
    # Basic statistics
    # ==================================================

    def average_partition_size(
        self,
    ) -> float:

        sizes = self.partition_sizes

        if not sizes:
            return 0.0

        return float(
            mean(sizes)
        )

    def min_partition_size(
        self,
    ) -> int:

        sizes = self.partition_sizes

        if not sizes:
            return 0

        return min(sizes)

    def max_partition_size(
        self,
    ) -> int:

        sizes = self.partition_sizes

        if not sizes:
            return 0

        return max(sizes)

    def median_partition_size(
        self,
    ) -> float:

        sizes = self.partition_sizes

        if not sizes:
            return 0.0

        return float(
            median(sizes)
        )

    def partition_size_std(
        self,
    ) -> float:

        sizes = self.partition_sizes

        if len(sizes) <= 1:
            return 0.0

        return float(
            pstdev(sizes)
        )

    def partition_size_statistics(
        self,
    ) -> dict[str, float]:

        return self._size_statistics(
            self.partition_sizes
        )

    # ==================================================
    # Balance metrics
    # ==================================================

    def balance_ratio(
        self,
    ) -> float:
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

        if largest == 0:
            return 0.0

        return (
            min(sizes)
            / largest
        )

    def coefficient_of_variation(
        self,
    ) -> float:
        """
        Relative dispersion of partition sizes.
        """

        sizes = self.partition_sizes

        if not sizes:
            return 0.0

        avg = mean(sizes)

        if avg == 0:
            return 0.0

        return (
            self.partition_size_std()
            / avg
        )

    # ==================================================
    # Distribution metrics
    # ==================================================

    def partition_size_entropy(
        self,
    ) -> float:
        """
        Shannon entropy of partition size distribution.
        """

        sizes = self.partition_sizes

        if not sizes:
            return 0.0

        probabilities = np.asarray(
            sizes,
            dtype=float,
        )

        probabilities /= probabilities.sum()

        return float(
            -np.sum(
                probabilities
                * np.log2(probabilities)
            )
        )

    def gini_coefficient(
        self,
    ) -> float:
        """
        Gini coefficient of partition sizes.

        0.0 -> perfectly balanced
        1.0 -> maximally imbalanced
        """

        sizes = np.asarray(
            self.partition_sizes,
            dtype=float,
        )

        if sizes.size == 0:
            return 0.0

        if np.all(sizes == 0):
            return 0.0

        sizes = np.sort(sizes)

        n = sizes.size

        index = np.arange(
            1,
            n + 1,
        )

        numerator = np.sum(
            (2 * index - n - 1)
            * sizes
        )

        denominator = (
            n * sizes.sum()
        )

        return float(
            numerator / denominator
        )

    def largest_partition_ratio(
        self,
    ) -> float:
        """
        Fraction of nodes contained
        in the largest partition.
        """

        if self.node_count == 0:
            return 0.0

        sizes = self.partition_sizes

        if not sizes:
            return 0.0

        return (
            max(sizes)
            / self.node_count
        )

    def effective_partition_count(
        self,
    ) -> float:
        """
        Hill number of order 1 (Perplexity).

        Equal to:

            2 ** Shannon entropy
        """

        return float(
            2
            ** self.partition_size_entropy()
        )

    # ==================================================
    # Structural metrics
    # ==================================================

    def singleton_partition_count(
        self,
    ) -> int:
        """
        Number of singleton partitions.
        """

        return sum(
            size == 1
            for size in self.partition_sizes
        )

    def singleton_ratio(
        self,
    ) -> float:
        """
        Fraction of singleton partitions.
        """

        if self.partition_count == 0:
            return 0.0

        return (
            self.singleton_partition_count()
            / self.partition_count
        )

    def non_singleton_partition_count(
        self,
    ) -> int:
        """
        Number of partitions containing
        at least two nodes.
        """

        return sum(
            size > 1
            for size in self.partition_sizes
        )

    def non_singleton_ratio(
        self,
    ) -> float:
        """
        Fraction of non-singleton partitions.
        """

        if self.partition_count == 0:
            return 0.0

        return (
            self.non_singleton_partition_count()
            / self.partition_count
        )

    def size_histogram(
        self,
    ) -> dict[int, int]:
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

