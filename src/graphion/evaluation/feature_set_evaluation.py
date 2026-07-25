from __future__ import annotations

from statistics import mean, median, pstdev
import numpy as np
from graphion.core.models import FeatureSet


class FeatureSetEvaluation:
    """Evaluate structural properties of a FeatureSet (Lazy evaluation)."""

    def __init__(self, feature_set: FeatureSet) -> None:
        self.feature_set = feature_set
        self._numpy_cache: np.ndarray | None = None

    @property
    def feature_count(self) -> int:
        return len(self.feature_set)

    @property
    def dimension(self) -> int | None:
        return self.feature_set.dimension

    @property
    def is_empty(self) -> bool:
        return self.feature_set.is_empty

    def _to_numpy(self) -> np.ndarray | None:
        if self._numpy_cache is not None:
            return self._numpy_cache
        try:
            _, matrix = self.feature_set.to_numpy()
            matrix = np.asarray(matrix, dtype=float)
        except (TypeError, ValueError):
            return None
        if matrix.ndim != 2:
            return None
        self._numpy_cache = matrix
        return matrix

    def is_numeric(self) -> bool:
        return self._to_numpy() is not None

    def _feature_matrix(self) -> np.ndarray:
        matrix = self._to_numpy()
        if matrix is None:
            raise TypeError("FeatureSet contains non numerical features.")
        return matrix

    @staticmethod
    def _empty_stats() -> dict[str, float]:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}

    @staticmethod
    def _nan_stats() -> dict[str, float]:
        return {k: float("nan") for k in ["min", "max", "mean", "median", "std"]}

    @staticmethod
    def _compute_stats(values: np.ndarray,) -> dict[str, float]:
        if values.size == 0:
            return FeatureSetEvaluation._empty_stats()
        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std()),
        }

    @staticmethod
    def _get_triu_values(matrix: np.ndarray) -> np.ndarray:
        return matrix[np.triu_indices(matrix.shape[0], k=1)]

    def feature_dimensions(self) -> list[int]:
        dimensions = []
        for feature in self.feature_set.features:
            try:
                array = np.asarray(feature)
            except Exception:
                continue
            if array.ndim == 1:
                dimensions.append(array.shape[0])
        return dimensions

    def dimension_statistics(self) -> dict[str, float]:
        dims = self.feature_dimensions()
        if not dims:
            return self._empty_stats()
        return {
            "min": float(min(dims)),
            "max": float(max(dims)),
            "mean": float(mean(dims)),
            "median": float(median(dims)),
            "std": float(pstdev(dims) if len(dims) > 1 else 0.0),
        }

    def feature_matrix_shape(self) -> tuple[int, int] | None:
        try:
            matrix = self._feature_matrix()
        except TypeError:
            return None
        return (matrix.shape[0], matrix.shape[1]) if matrix.ndim == 2 else None

    def feature_value_statistics(self) -> dict[str, float]:
        matrix = self._to_numpy()
        return self._nan_stats() if matrix is None else self._compute_stats(matrix.flatten())

    def feature_mean_statistics(self) -> dict[str, float]:
        matrix = self._feature_matrix()
        if matrix.size == 0:
            return self._empty_stats()
        return self._compute_stats(np.mean(matrix, axis=0))

    def feature_std_statistics(self) -> dict[str, float]:
        matrix = self._feature_matrix()
        if matrix.size == 0:
            return self._empty_stats()
        return self._compute_stats(np.std(matrix, axis=0))

    def feature_variance_statistics(self) -> dict[str, float]:
        matrix = self._feature_matrix()
        if matrix.size == 0:
            return self._empty_stats()
        return self._compute_stats(np.var(matrix, axis=0))

    def feature_range_statistics(self) -> dict[str, float]:
        matrix = self._feature_matrix()
        if matrix.size == 0:
            return self._empty_stats()
        return self._compute_stats(np.max(matrix, axis=0) - np.min(matrix, axis=0))

    def feature_magnitude_statistics(self) -> dict[str, float]:
        matrix = self._feature_matrix()
        if matrix.size == 0:
            return self._empty_stats()
        scales = np.nanmax(np.abs(matrix), axis=0)
        return self._empty_stats() if scales.size == 0 else self._compute_stats(scales)

    def feature_norm_statistics(self) -> dict[str, float]:
        matrix = self._feature_matrix()
        if matrix.size == 0:
            return self._empty_stats()
        return self._compute_stats(np.linalg.norm(matrix, axis=1))

    def pairwise_distance_statistics(self) -> dict[str, float]:
        matrix = self._feature_matrix()
        if matrix.shape[0] <= 1:
            return self._empty_stats()
        from sklearn.metrics.pairwise import euclidean_distances
        values = self._get_triu_values(euclidean_distances(matrix))
        return self._empty_stats() if values.size == 0 else self._compute_stats(values)

    def cosine_similarity_statistics(self) -> dict[str, float]:
        matrix = self._feature_matrix()
        if matrix.shape[0] <= 1:
            return self._empty_stats()
        from sklearn.metrics.pairwise import cosine_similarity
        values = self._get_triu_values(cosine_similarity(matrix))
        return self._empty_stats() if values.size == 0 else self._compute_stats(values)

    def nearest_neighbor_distance_statistics(self) -> dict[str, float]:
        matrix = self._feature_matrix()
        if matrix.shape[0] <= 1:
            return self._empty_stats()
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(matrix)
        np.fill_diagonal(distances, np.inf)
        return self._compute_stats(np.min(distances, axis=1))

    def sparsity_statistics(self) -> dict[str, float]:
        matrix = self._feature_matrix()
        if matrix.size == 0:
            return self._empty_stats()
        return self._compute_stats(np.sum(matrix == 0, axis=1) / matrix.shape[1])

    def zero_vector_count(self) -> int:
        matrix = self._feature_matrix()
        return int(np.sum(np.all(matrix == 0, axis=1))) if matrix.size else 0

    def zero_vector_ratio(self) -> float:
        return self.zero_vector_count() / self.feature_count if self.feature_count else 0.0

    def duplicate_feature_count(self) -> int:
        matrix = self._feature_matrix()
        if matrix.size == 0:
            return 0
        _, counts = np.unique(matrix, axis=0, return_counts=True)
        return int(np.sum(counts - 1))

    def duplicate_feature_ratio(self) -> float:
        return self.duplicate_feature_count() / self.feature_count if self.feature_count else 0.0

    def constant_dimension_count(self) -> int:
        matrix = self._feature_matrix()
        if matrix.size == 0:
            return 0
        return int(np.sum(np.isclose(np.var(matrix, axis=0), 0.0)))

    def constant_dimension_ratio(self) -> float:
        dimension = self.dimension
        return self.constant_dimension_count() / dimension if dimension else 0.0

    def invalid_value_count(self) -> int:
        matrix = self._feature_matrix()
        return int(np.sum(~np.isfinite(matrix))) if matrix.size else 0

    def distance_concentration(self) -> float:
        stats = self.pairwise_distance_statistics()
        return float(stats["std"] / stats["mean"]) if stats["mean"] != 0 else 0.0

    def nearest_neighbor_hub_statistics(self, metric: str = "cosine") -> dict[str, float]:
        matrix = self._feature_matrix()
        if matrix.shape[0] <= 1:
            return self._empty_stats()
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
        if metric == "cosine":
            distances = 1.0 - cosine_similarity(matrix)
        elif metric == "euclidean":
            distances = euclidean_distances(matrix)
        else:
            raise ValueError(f"Unsupported metric: {metric}. Use 'cosine' or 'euclidean'.")
        np.fill_diagonal(distances, np.inf)
        counts = np.bincount(np.argmin(distances, axis=1), minlength=matrix.shape[0])
        return self._compute_stats(counts)

    def feature_correlation_statistics(self) -> dict[str, float]:
        matrix = self._feature_matrix()
        if matrix.size == 0 or matrix.shape[0] <= 1 or matrix.shape[1] <= 1:
            return self._empty_stats()
        correlation = np.corrcoef(matrix, rowvar=False)
        if correlation.ndim != 2:
            return self._empty_stats()
        values = self._get_triu_values(correlation)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return self._empty_stats()
        return self._compute_stats(values)

    def covariance_statistics(self) -> dict[str, float]:
        matrix = self._feature_matrix()
        if matrix.size == 0 or matrix.shape[0] <= 1 or matrix.shape[1] <= 1:
            return self._empty_stats()
        covariance = np.cov(matrix, rowvar=False)
        if covariance.ndim != 2:
            return self._empty_stats()
        values = self._get_triu_values(covariance)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return self._empty_stats()
        return self._compute_stats(values)

    def effective_dimension(self) -> float:
        matrix = self._feature_matrix()
        if matrix.size == 0:
            return 0.0
        if matrix.shape[1] <= 1:
            return float(matrix.shape[1])
        if matrix.shape[0] <= 1:
            return 0.0
        covariance = np.cov(matrix, rowvar=False)
        if covariance.ndim != 2:
            return 0.0
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        if eigenvalues.size == 0:
            return 0.0
        denominator = np.sum(eigenvalues ** 2)
        if denominator == 0:
            return 0.0
        return float((np.sum(eigenvalues) ** 2) / denominator)