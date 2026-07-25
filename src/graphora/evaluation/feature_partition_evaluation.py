from __future__ import annotations

import numpy as np

from graphora.core.models import FeatureSet, PartitionSet
from graphora.core.errors import InvalidPartitionSetError


class FeaturePartitionEvaluation:
    """Evaluate clustering quality of a partition over a FeatureSet."""

    def __init__(self, feature_set: FeatureSet, partition_set: PartitionSet) -> None:
        """Initialize the evaluator with a FeatureSet and a PartitionSet, validating dimensions and values."""
        self.feature_set = feature_set
        self.partition_set = partition_set

        ids, matrix = self.feature_set.to_numpy()
        _, labels = self.partition_set.to_labels(ids)

        matrix = np.asarray(matrix, dtype=float)
        labels = np.asarray(labels, dtype=int)

        if matrix.ndim != 2:
            raise TypeError("FeatureSet must contain numerical vectors.")
        if len(labels) != matrix.shape[0]:
            raise InvalidPartitionSetError("FeatureSet and PartitionSet size mismatch.")
        if not np.isfinite(matrix).all():
            raise ValueError("FeatureSet contains invalid numerical values.")

        self._feature_matrix = matrix
        self._labels = labels

    @staticmethod
    def _empty_stats() -> dict[str, float]:
        """Return a default dictionary of statistical values set to zero."""
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}

    @staticmethod
    def _compute_stats(values: np.ndarray) -> dict[str, float]:
        """Compute basic statistical metrics (min, max, mean, median, std) for a given array."""
        if values.size == 0:
            return FeaturePartitionEvaluation._empty_stats()
        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std()),
        }

    @property
    def sample_count(self) -> int:
        """Get the total number of samples in the feature matrix."""
        return self._feature_matrix.shape[0]

    @property
    def dimension(self) -> int:
        """Get the dimensionality (number of features) of each sample."""
        return self._feature_matrix.shape[1]

    @property
    def cluster_count(self) -> int:
        """Get the total number of unique clusters present in the labels."""
        return int(len(np.unique(self._labels)))

    @property
    def cluster_sizes(self) -> tuple[int, ...]:
        """Get a tuple containing the size of each cluster."""
        _, counts = np.unique(self._labels, return_counts=True)
        return tuple(counts.tolist())

    @property
    def labels(self) -> np.ndarray:
        """Get a read-only copy of the cluster labels array."""
        labels = self._labels.copy()
        labels.flags.writeable = False
        return labels

    @property
    def feature_matrix(self) -> np.ndarray:
        """Get a read-only copy of the underlying feature matrix."""
        matrix = self._feature_matrix.copy()
        matrix.flags.writeable = False
        return matrix

    def cluster_size_statistics(self) -> dict[str, float]:
        """Compute statistical metrics for the sizes of the clusters."""
        return self._compute_stats(np.asarray(self.cluster_sizes, dtype=float))

    def largest_cluster_ratio(self) -> float:
        """Calculate the ratio of the largest cluster size to the total number of samples."""
        return float(max(self.cluster_sizes) / self.sample_count) if self.sample_count else 0.0

    def smallest_cluster_ratio(self) -> float:
        """Calculate the ratio of the smallest cluster size to the total number of samples."""
        return float(min(self.cluster_sizes) / self.sample_count) if self.sample_count else 0.0

    def singleton_cluster_count(self) -> int:
        """Count the number of clusters that contain exactly one sample."""
        return int(np.sum(np.asarray(self.cluster_sizes) == 1))

    def singleton_cluster_ratio(self) -> float:
        """Calculate the ratio of singleton clusters to the total number of clusters."""
        return float(self.singleton_cluster_count() / self.cluster_count) if self.cluster_count else 0.0

    def cluster_balance(self) -> float:
        """Calculate the balance ratio between the smallest and largest cluster sizes."""
        if not self.cluster_count:
            return 0.0
        sizes = np.asarray(self.cluster_sizes, dtype=float)
        maximum = sizes.max()
        return float(sizes.min() / maximum) if maximum else 0.0

    def _cluster_indices(self) -> list[np.ndarray]:
        """Extract and return a list of sample indices for each unique cluster."""
        return [np.where(self._labels == label)[0] for label in np.unique(self._labels)]

    def _cluster_centroids(self) -> np.ndarray:
        """Compute and return the geometric centroids for all clusters."""
        indices = self._cluster_indices()
        if not indices:
            return np.empty((0, self.dimension))
        return np.asarray([self._feature_matrix[idx].mean(axis=0) for idx in indices])

    def within_cluster_distance_statistics(self) -> dict[str, float]:
        """Compute distance statistics between samples and their respective cluster centroids (cohesion)."""
        indices = self._cluster_indices()
        if not indices:
            return self._empty_stats()
        centroids = self._cluster_centroids()
        values = []
        for idx, centroid in zip(indices, centroids):
            distances = np.linalg.norm(self._feature_matrix[idx] - centroid, axis=1)
            values.extend(distances.tolist())
        return self._compute_stats(np.asarray(values, dtype=float))

    def centroid_distance_statistics(self) -> dict[str, float]:
        """Compute distance statistics between different cluster centroids (separation)."""
        centroids = self._cluster_centroids()
        if centroids.shape[0] < 2:
            return self._empty_stats()
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(centroids)
        values = distances[np.triu_indices(distances.shape[0], k=1)]
        return self._compute_stats(values)

    def separation_ratio(self) -> float:
        """Calculate the ratio of inter-cluster separation to intra-cluster cohesion."""
        if self.cluster_count < 2:
            return float("nan")
        cohesion = self.within_cluster_distance_statistics()["mean"]
        separation = self.centroid_distance_statistics()["mean"]
        return float(separation / cohesion) if cohesion != 0 else float("inf")

    def silhouette_score(self, metric: str = "euclidean") -> float:
        """Compute the silhouette score for the clustering partition."""
        if self.cluster_count < 2 or self.sample_count <= self.cluster_count:
            return float("nan")
        from sklearn.metrics import silhouette_score
        try:
            return float(silhouette_score(self._feature_matrix, self._labels, metric=metric))
        except ValueError:
            return float("nan")

    def davies_bouldin_score(self) -> float:
        """Compute the Davies-Bouldin index for the clustering partition."""
        if self.cluster_count < 2 or self.sample_count <= self.cluster_count:
            return float("nan")
        from sklearn.metrics import davies_bouldin_score
        try:
            return float(davies_bouldin_score(self._feature_matrix, self._labels))
        except ValueError:
            return float("nan")

    def calinski_harabasz_score(self) -> float:
        """Compute the Calinski-Harabasz index for the clustering partition."""
        if self.cluster_count < 2 or self.sample_count <= self.cluster_count:
            return float("nan")
        from sklearn.metrics import calinski_harabasz_score
        try:
            return float(calinski_harabasz_score(self._feature_matrix, self._labels))
        except ValueError:
            return float("nan")

    def quality_summary(self) -> dict[str, float]:
        """Generate a comprehensive summary dictionary of primary clustering quality metrics."""
        if self.cluster_count < 2:
            return {k: float("nan") for k in ["silhouette", "davies_bouldin", "calinski_harabasz", "separation_ratio"]}
        return {
            "silhouette": self.silhouette_score(),
            "davies_bouldin": self.davies_bouldin_score(),
            "calinski_harabasz": self.calinski_harabasz_score(),
            "separation_ratio": self.separation_ratio(),
        }