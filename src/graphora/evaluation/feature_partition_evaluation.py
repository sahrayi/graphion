"""
FeaturePartition evaluation utilities.

Evaluate clustering quality of a PartitionSet over
a Graphora FeatureSet.
"""

from __future__ import annotations

import numpy as np

from graphora.core.models import (
    FeatureSet,
    PartitionSet,
)
from graphora.core.errors import (
    InvalidPartitionSetError,
)


class FeaturePartitionEvaluation:
    """
    Evaluate quality of a partition over a FeatureSet.

    Parameters
    ----------
    feature_set:
        Feature representations.

    partition_set:
        Partition assignment of the same entities.

    Notes
    -----
    Expensive representations are created only once
    and cached.
    """

    def __init__(
        self,
        feature_set: FeatureSet,
        partition_set: PartitionSet,
    ) -> None:

        self.feature_set = feature_set
        self.partition_set = partition_set

        (
            self._ids,
            self._feature_matrix,
        ) = self.feature_set.to_numpy()

        (
            _,
            self._labels,
        ) = self.partition_set.to_labels(
            self._ids,
        )

        self._feature_matrix = np.asarray(
            self._feature_matrix,
            dtype=float,
        )

        self._labels = np.asarray(
            self._labels,
            dtype=int,
        )

        if (
            self._feature_matrix.ndim != 2
        ):
            raise TypeError(
                "FeatureSet must contain numerical vectors."
            )

        if (
            len(self._labels)
            !=
            self._feature_matrix.shape[0]
        ):
            raise InvalidPartitionSetError(
                "FeatureSet and PartitionSet size mismatch."
            )

    # ==================================================
    # Basic properties
    # ==================================================

    @property
    def sample_count(
        self,
    ) -> int:
        """
        Number of feature vectors.
        """

        return self._feature_matrix.shape[0]

    @property
    def dimension(
        self,
    ) -> int:
        """
        Feature dimension.
        """

        return self._feature_matrix.shape[1]

    @property
    def cluster_count(
        self,
    ) -> int:
        """
        Number of partitions.
        """

        return self.partition_set.partition_count

    @property
    def cluster_sizes(
        self,
    ) -> tuple[int, ...]:
        """
        Size of every cluster.
        """

        return self.partition_set.partition_sizes

    @property
    def labels(
            self,
    ) -> np.ndarray:
        """
        Cluster labels.

        Returned as read-only array.
        """

        labels = self._labels.copy()
        labels.flags.writeable = False

        return labels

    @property
    def feature_matrix(
            self,
    ) -> np.ndarray:
        """
        Feature matrix.

        Returned as read-only copy.
        """

        matrix = self._feature_matrix.copy()
        matrix.flags.writeable = False

        return matrix

    # ==================================================
    # Internal helpers
    # ==================================================

    def _require_multiple_clusters(
        self,
    ) -> None:
        """
        Ensure at least two clusters exist.
        """

        if self.cluster_count < 2:
            raise ValueError(
                "Metric requires at least two clusters."
            )

    # ==================================================
    # Cluster size metrics
    # ==================================================

    def cluster_size_statistics(
        self,
    ) -> dict[str, float]:
        """
        Statistics of cluster sizes.
        """

        sizes = np.asarray(
            self.cluster_sizes,
            dtype=float,
        )

        if sizes.size == 0:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }

        return {
            "min": float(sizes.min()),
            "max": float(sizes.max()),
            "mean": float(sizes.mean()),
            "median": float(np.median(sizes)),
            "std": float(sizes.std()),
        }


    def largest_cluster_ratio(
        self,
    ) -> float:
        """
        Fraction of samples belonging
        to the largest cluster.
        """

        if self.sample_count == 0:
            return 0.0

        return (
            max(self.cluster_sizes)
            /
            self.sample_count
        )


    def smallest_cluster_ratio(
        self,
    ) -> float:
        """
        Fraction of samples belonging
        to the smallest cluster.
        """

        if self.sample_count == 0:
            return 0.0

        return (
            min(self.cluster_sizes)
            /
            self.sample_count
        )


    def singleton_cluster_count(
        self,
    ) -> int:
        """
        Number of singleton clusters.
        """

        return int(
            np.sum(
                np.asarray(
                    self.cluster_sizes,
                ) == 1
            )
        )


    def singleton_cluster_ratio(
        self,
    ) -> float:
        """
        Fraction of singleton clusters.
        """

        if self.cluster_count == 0:
            return 0.0

        return (
            self.singleton_cluster_count()
            /
            self.cluster_count
        )


    def cluster_balance(
        self,
    ) -> float:
        """
        Measure cluster balance.

        Defined as

            min_cluster_size / max_cluster_size

        Returns
        -------
        float

            1.0
                perfectly balanced

            -> 0
                highly imbalanced
        """

        if self.cluster_count == 0:
            return 0.0

        sizes = np.asarray(
            self.cluster_sizes,
            dtype=float,
        )

        maximum = sizes.max()

        if maximum == 0:
            return 0.0

        return float(
            sizes.min()
            /
            maximum
        )

    # ==================================================
    # Internal geometry helpers
    # ==================================================

    def _cluster_indices(
        self,
    ) -> list[np.ndarray]:
        """
        Indices of samples belonging to each cluster.
        """

        return [
            np.where(
                self._labels == label
            )[0]
            for label in np.unique(
                self._labels
            )
        ]


    def _cluster_centroids(
        self,
    ) -> np.ndarray:
        """
        Centroid of every cluster.
        """

        indices = self._cluster_indices()

        return np.asarray(
            [
                self._feature_matrix[idx].mean(
                    axis=0
                )
                for idx in indices
            ]
        )

    def within_cluster_distance_statistics(
        self,
    ) -> dict[str, float]:
        """
        Statistics of distances between
        samples and their own cluster centroid.
        """

        indices = self._cluster_indices()
        centroids = self._cluster_centroids()

        values = []

        for idx, centroid in zip(
            indices,
            centroids,
        ):

            vectors = self._feature_matrix[idx]

            distances = np.linalg.norm(
                vectors - centroid,
                axis=1,
            )

            values.extend(
                distances.tolist()
            )

        values = np.asarray(
            values,
            dtype=float,
        )

        if values.size == 0:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }

        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std()),
        }

    def centroid_distance_statistics(
        self,
    ) -> dict[str, float]:
        """
        Statistics of distances between
        cluster centroids.
        """

        self._require_multiple_clusters()

        centroids = self._cluster_centroids()

        from sklearn.metrics.pairwise import (
            euclidean_distances,
        )

        distances = euclidean_distances(
            centroids,
        )

        values = distances[
            np.triu_indices(
                distances.shape[0],
                k=1,
            )
        ]

        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std()),
        }

    def separation_ratio(
        self,
    ) -> float:
        """
        Ratio of cluster separation
        to cluster cohesion.

        Higher is better.
        """

        cohesion = (
            self.within_cluster_distance_statistics()[
                "mean"
            ]
        )

        separation = (
            self.centroid_distance_statistics()[
                "mean"
            ]
        )

        if cohesion == 0:
            return float("inf")

        return float(
            separation / cohesion
        )

    def silhouette_score(
        self,
        metric: str = "euclidean",
    ) -> float:
        """
        Compute silhouette score.

        Higher is better.

        Parameters
        ----------
        metric:
            Distance metric passed to sklearn.
        """

        self._require_multiple_clusters()

        from sklearn.metrics import (
            silhouette_score,
        )

        return float(
            silhouette_score(
                self._feature_matrix,
                self._labels,
                metric=metric,
            )
        )

    def davies_bouldin_score(
        self,
    ) -> float:
        """
        Compute Davies-Bouldin index.

        Lower is better.
        """

        self._require_multiple_clusters()

        from sklearn.metrics import (
            davies_bouldin_score,
        )

        return float(
            davies_bouldin_score(
                self._feature_matrix,
                self._labels,
            )
        )

    def calinski_harabasz_score(
        self,
    ) -> float:
        """
        Compute Calinski-Harabasz index.

        Higher is better.
        """

        self._require_multiple_clusters()

        from sklearn.metrics import (
            calinski_harabasz_score,
        )

        return float(
            calinski_harabasz_score(
                self._feature_matrix,
                self._labels,
            )
        )

    def quality_summary(
            self,
    ) -> dict[str, float]:
        """
        Summary of clustering quality metrics.

        Metrics that require multiple clusters are returned
        as NaN when unavailable.
        """

        if self.cluster_count < 2:
            return {
                "silhouette": float("nan"),
                "davies_bouldin": float("nan"),
                "calinski_harabasz": float("nan"),
                "separation_ratio": float("nan"),
            }

        return {
            "silhouette": self.silhouette_score(),
            "davies_bouldin": self.davies_bouldin_score(),
            "calinski_harabasz": self.calinski_harabasz_score(),
            "separation_ratio": self.separation_ratio(),
        }

