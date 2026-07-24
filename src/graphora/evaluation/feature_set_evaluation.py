"""
FeatureSet evaluation utilities.

This module provides structural evaluation metrics
for Graphora FeatureSet objects.
"""

from __future__ import annotations

from statistics import (
    mean,
    median,
    pstdev,
)

import numpy as np

from graphora.core.models import FeatureSet


class FeatureSetEvaluation:
    """
    Evaluate structural properties of a FeatureSet.

    Parameters
    ----------
    feature_set:
        Graphora FeatureSet instance.

    Notes
    -----
    Evaluation is intentionally lazy.
    """

    def __init__(
        self,
        feature_set: FeatureSet,
    ) -> None:

        self.feature_set = feature_set

        self._numpy_cache = None


    # ==================================================
    # Basic properties
    # ==================================================

    @property
    def feature_count(
        self,
    ) -> int:

        return len(
            self.feature_set,
        )


    @property
    def dimension(
        self,
    ) -> int | None:

        return self.feature_set.dimension


    @property
    def is_empty(
        self,
    ) -> bool:

        return self.feature_set.is_empty


    # ==================================================
    # Helpers
    # ==================================================

    def _to_numpy(
            self,
    ) -> np.ndarray | None:
        """
        Convert numerical FeatureSet into numpy matrix.

        Returns
        -------
        np.ndarray | None
            Numeric 2D feature matrix.
        """

        if self._numpy_cache is not None:
            return self._numpy_cache

        try:

            _ids, matrix = (
                self.feature_set.to_numpy()
            )

            matrix = np.asarray(
                matrix,
                dtype=float,
            )


        except (
                TypeError,
                ValueError,
        ):

            return None

        if matrix.ndim != 2:
            return None

        self._numpy_cache = matrix

        return matrix


    def is_numeric(
        self,
    ) -> bool:
        """
        Check whether FeatureSet contains
        numerical vectors.
        """

        return (
            self._to_numpy()
            is not None
        )


    # ==================================================
    # Feature shape statistics
    # ==================================================

    def feature_dimensions(
            self,
    ) -> list[int]:
        """
        Return dimension of every vector feature.
        """

        dimensions = []

        for feature in self.feature_set.features:

            try:

                array = np.asarray(feature)

            except Exception:

                continue

            if array.ndim == 1:
                dimensions.append(
                    array.shape[0]
                )

        return dimensions


    def dimension_statistics(
        self,
    ) -> dict[str, float]:
        """
        Statistics of feature dimensions.
        """

        dimensions = (
            self.feature_dimensions()
        )

        if not dimensions:

            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }


        return {
            "min": float(
                min(dimensions)
            ),
            "max": float(
                max(dimensions)
            ),
            "mean": float(
                mean(dimensions)
            ),
            "median": float(
                median(dimensions)
            ),
            "std": float(
                pstdev(dimensions)
                if len(dimensions) > 1
                else 0.0
            ),
        }


    # ==================================================
    # Numeric feature statistics
    # ==================================================

    def feature_matrix_shape(
            self,
    ) -> tuple[int, int] | None:

        try:
            matrix = self._feature_matrix()

        except TypeError:
            return None

        if matrix.ndim != 2:
            return None

        return (
            matrix.shape[0],
            matrix.shape[1],
        )


    def feature_value_statistics(
        self,
    ) -> dict[str, float]:
        """
        Statistics over all numerical feature values.
        """

        matrix = self._to_numpy()

        if matrix is None:

            return {
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
            }


        values = matrix.flatten()


        if values.size == 0:

            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }


        return {
            "min": float(
                values.min()
            ),
            "max": float(
                values.max()
            ),
            "mean": float(
                values.mean()
            ),
            "median": float(
                np.median(values)
            ),
            "std": float(
                values.std()
            ),
        }

    # ==================================================
    # Feature distribution metrics
    # ==================================================

    def _feature_matrix(
            self,
    ) -> np.ndarray:

        matrix = self._to_numpy()

        if matrix is None:
            raise TypeError(
                "FeatureSet contains non numerical features."
            )

        return matrix


    def feature_mean_statistics(
        self,
    ) -> dict[str, float]:
        """
        Statistics of feature dimension means.
        """

        matrix = self._feature_matrix()

        if matrix.size == 0:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }


        values = np.mean(
            matrix,
            axis=0,
        )


        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std()),
        }


    def feature_std_statistics(
        self,
    ) -> dict[str, float]:
        """
        Statistics of feature dimension standard deviations.
        """

        matrix = self._feature_matrix()

        if matrix.size == 0:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }


        values = np.std(
            matrix,
            axis=0,
        )


        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std()),
        }


    def feature_variance_statistics(
        self,
    ) -> dict[str, float]:
        """
        Statistics of feature dimension variances.
        """

        matrix = self._feature_matrix()

        if matrix.size == 0:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }


        values = np.var(
            matrix,
            axis=0,
        )


        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std()),
        }


    def feature_range_statistics(
        self,
    ) -> dict[str, float]:
        """
        Statistics of feature dimension ranges.
        """

        matrix = self._feature_matrix()

        if matrix.size == 0:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }


        ranges = (
            np.max(
                matrix,
                axis=0,
            )
            -
            np.min(
                matrix,
                axis=0,
            )
        )


        return {
            "min": float(ranges.min()),
            "max": float(ranges.max()),
            "mean": float(ranges.mean()),
            "median": float(np.median(ranges)),
            "std": float(ranges.std()),
        }

    def feature_magnitude_statistics(
        self,
    ) -> dict[str, float]:
        """
        Statistics of feature dimension scales.

        Measures relative scale differences between
        feature dimensions.

        For every dimension:

            scale = max(abs(values))

        Returns statistics over dimension scales.

        Useful for:
            - detecting unbalanced feature dimensions
            - checking preprocessing quality
            - identifying dominant dimensions

        Returns
        -------
        dict[str, float]
            min / max / mean / median / std
        """

        matrix = self._feature_matrix()

        if matrix.size == 0:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }

        scales = np.nanmax(
            np.abs(matrix),
            axis=0,
        )


        if scales.size == 0:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }

        return {
            "min": float(
                scales.min()
            ),
            "max": float(
                scales.max()
            ),
            "mean": float(
                scales.mean()
            ),
            "median": float(
                np.median(scales)
            ),
            "std": float(
                scales.std()
            ),
        }

    def feature_norm_statistics(
        self,
    ) -> dict[str, float]:
        """
        Statistics of feature vector L2 norms.
        """

        matrix = self._feature_matrix()

        if matrix.size == 0:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }


        norms = np.linalg.norm(
            matrix,
            axis=1,
        )


        return {
            "min": float(norms.min()),
            "max": float(norms.max()),
            "mean": float(norms.mean()),
            "median": float(np.median(norms)),
            "std": float(norms.std()),
        }

    # ==================================================
    # Feature geometry metrics
    # ==================================================

    def pairwise_distance_statistics(
        self,
    ) -> dict[str, float]:
        """
        Statistics of pairwise euclidean distances.

        Computes distances between all feature vectors.
        """

        matrix = self._feature_matrix()

        if matrix.shape[0] <= 1:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }

        from sklearn.metrics.pairwise import (
            euclidean_distances,
        )

        distances = euclidean_distances(
            matrix,
        )

        values = distances[
            np.triu_indices(
                distances.shape[0],
                k=1,
            )
        ]

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


    def cosine_similarity_statistics(
        self,
    ) -> dict[str, float]:
        """
        Statistics of pairwise cosine similarities.

        Suitable for embedding spaces.
        """

        matrix = self._feature_matrix()

        if matrix.shape[0] <= 1:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }


        from sklearn.metrics.pairwise import (
            cosine_similarity,
        )

        similarities = cosine_similarity(
            matrix,
        )

        values = similarities[
            np.triu_indices(
                similarities.shape[0],
                k=1,
            )
        ]

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


    def nearest_neighbor_distance_statistics(
        self,
    ) -> dict[str, float]:
        """
        Statistics of nearest neighbor euclidean distances.

        Measures local feature density.
        """

        matrix = self._feature_matrix()

        if matrix.shape[0] <= 1:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }


        from sklearn.metrics.pairwise import (
            euclidean_distances,
        )


        distances = euclidean_distances(
            matrix,
        )


        np.fill_diagonal(
            distances,
            np.inf,
        )


        nearest = np.min(
            distances,
            axis=1,
        )


        return {
            "min": float(nearest.min()),
            "max": float(nearest.max()),
            "mean": float(nearest.mean()),
            "median": float(np.median(nearest)),
            "std": float(nearest.std()),
        }


    def sparsity_statistics(
        self,
    ) -> dict[str, float]:
        """
        Statistics of zero-value ratio.

        Useful for sparse vectors.
        """

        matrix = self._feature_matrix()

        if matrix.size == 0:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }


        zero_ratio = (
            np.sum(
                matrix == 0,
                axis=1,
            )
            /
            matrix.shape[1]
        )


        return {
            "min": float(zero_ratio.min()),
            "max": float(zero_ratio.max()),
            "mean": float(zero_ratio.mean()),
            "median": float(np.median(zero_ratio)),
            "std": float(zero_ratio.std()),
        }

    # ==================================================
    # Feature quality metrics
    # ==================================================

    def zero_vector_count(
        self,
    ) -> int:
        """
        Number of completely zero feature vectors.
        """

        matrix = self._feature_matrix()

        if matrix.size == 0:
            return 0

        return int(
            np.sum(
                np.all(
                    matrix == 0,
                    axis=1,
                )
            )
        )


    def zero_vector_ratio(
        self,
    ) -> float:
        """
        Ratio of zero feature vectors.
        """

        if self.feature_count == 0:
            return 0.0

        return (
            self.zero_vector_count()
            /
            self.feature_count
        )


    def duplicate_feature_count(
        self,
    ) -> int:
        """
        Number of duplicated feature vectors.

        Counts extra occurrences.

        Example:

            [a, a, b]

        returns:

            1
        """

        matrix = self._feature_matrix()

        if matrix.size == 0:
            return 0


        _, counts = np.unique(
            matrix,
            axis=0,
            return_counts=True,
        )


        return int(
            np.sum(
                counts - 1
            )
        )


    def duplicate_feature_ratio(
        self,
    ) -> float:
        """
        Ratio of duplicated feature vectors.
        """

        if self.feature_count == 0:
            return 0.0

        return (
            self.duplicate_feature_count()
            /
            self.feature_count
        )


    def constant_dimension_count(
        self,
    ) -> int:
        """
        Number of dimensions with zero variance.
        """

        matrix = self._feature_matrix()

        if matrix.size == 0:
            return 0


        variances = np.var(
            matrix,
            axis=0,
        )

        return int(
            np.sum(
                np.isclose(
                    variances,
                    0.0,
                )
            )
        )


    def constant_dimension_ratio(
        self,
    ) -> float:
        """
        Ratio of dimensions without variation.
        """

        dimension = self.dimension

        if not dimension:
            return 0.0


        return (
            self.constant_dimension_count()
            /
            dimension
        )

    def invalid_value_count(
            self,
    ) -> int:
        """
        Count NaN and infinite values.
        """

        matrix = self._feature_matrix()

        if matrix.size == 0:
            return 0

        return int(
            np.sum(
                ~np.isfinite(matrix)
            )
        )

    # ==================================================
    # Advanced feature space metrics
    # ==================================================

    def distance_concentration(
        self,
    ) -> float:
        """
        Measure distance concentration.

        Defined as:

            std(distance) / mean(distance)

        Lower values indicate stronger concentration,
        which is common in very high dimensional spaces.
        """

        statistics = (
            self.pairwise_distance_statistics()
        )

        mean_distance = statistics["mean"]
        std_distance = statistics["std"]

        if mean_distance == 0:
            return 0.0

        return float(
            std_distance / mean_distance
        )

    def nearest_neighbor_hub_statistics(
            self,
            metric: str = "cosine",
    ) -> dict[str, float]:
        """
        Analyze hubness in nearest neighbor graph.

        Counts how often each sample appears
        as nearest neighbor of other samples.

        Parameters
        ----------
        metric:
            Distance metric used for nearest neighbor search.

            Supported:
                - "cosine"
                - "euclidean"

            Default:
                "cosine"

            Cosine distance is recommended for
            semantic embeddings.

        Returns
        -------
        dict[str, float]
            Statistics of nearest neighbor occurrence counts.
        """

        matrix = self._feature_matrix()

        if matrix.shape[0] <= 1:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }

        from sklearn.metrics.pairwise import (
            cosine_similarity,
            euclidean_distances,
        )

        if metric == "cosine":

            similarities = cosine_similarity(
                matrix,
            )

            distances = 1.0 - similarities


        elif metric == "euclidean":

            distances = euclidean_distances(
                matrix,
            )


        else:

            raise ValueError(
                f"Unsupported metric: {metric}. "
                "Use 'cosine' or 'euclidean'."
            )

        np.fill_diagonal(
            distances,
            np.inf,
        )

        nearest_indices = np.argmin(
            distances,
            axis=1,
        )

        counts = np.bincount(
            nearest_indices,
            minlength=matrix.shape[0],
        )

        return {
            "min": float(
                counts.min()
            ),
            "max": float(
                counts.max()
            ),
            "mean": float(
                counts.mean()
            ),
            "median": float(
                np.median(counts)
            ),
            "std": float(
                counts.std()
            ),
        }


    def feature_correlation_statistics(
        self,
    ) -> dict[str, float]:
        """
        Statistics of feature dimension correlations.
        """

        matrix = self._feature_matrix()

        if matrix.shape[1] <= 1:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }


        correlation = np.corrcoef(
            matrix,
            rowvar=False,
        )


        values = correlation[
            np.triu_indices(
                correlation.shape[0],
                k=1,
            )
        ]


        values = values[
            np.isfinite(values)
        ]


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


    def covariance_statistics(
        self,
    ) -> dict[str, float]:
        """
        Statistics of covariance values between dimensions.
        """

        matrix = self._feature_matrix()

        if matrix.shape[1] <= 1:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
            }


        covariance = np.cov(
            matrix,
            rowvar=False,
        )


        values = covariance[
            np.triu_indices(
                covariance.shape[0],
                k=1,
            )
        ]


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


    def effective_dimension(
        self,
    ) -> float:
        """
        Estimate effective dimensionality.

        Based on participation ratio:

            (sum(lambda))^2 / sum(lambda^2)

        where lambda are covariance eigenvalues.
        """

        matrix = self._feature_matrix()

        if matrix.shape[1] <= 1:
            return float(
                matrix.shape[1]
            )


        covariance = np.cov(
            matrix,
            rowvar=False,
        )


        eigenvalues = np.linalg.eigvalsh(
            covariance,
        )

        eigenvalues = eigenvalues[
            eigenvalues > 1e-12
        ]


        if eigenvalues.size == 0:
            return 0.0


        return float(
            (
                np.sum(eigenvalues)
                ** 2
            )
            /
            np.sum(
                eigenvalues ** 2
            )
        )