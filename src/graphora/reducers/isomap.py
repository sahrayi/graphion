"""
Isomap dimensionality reduction.

Uses scikit-learn Isomap implementation internally.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Generic

import numpy as np
from sklearn.manifold import Isomap as SklearnIsomap

from .base_reducer import BaseReducer

from graphora.core.types import TId


class Isomap(
    BaseReducer[
        TId,
        Sequence[float],
        tuple[float, ...],
    ],
    Generic[TId],
):
    """
    Isomap based feature reducer.

    Nonlinear manifold learning algorithm
    that preserves geodesic distances.

    Suitable for:

    - small and medium datasets
    - nonlinear manifold structures
    - exploratory embeddings


    Notes:

    - n_neighbors must be smaller than the number
      of samples.
    - Isomap is not recommended for very large
      datasets.
    """

    def __init__(
        self,
        *,
        output_dimension: int = 2,
        n_neighbors: int = 5,
        metric: str = "minkowski",
    ) -> None:

        super().__init__(
            output_dimension=output_dimension,
        )

        if output_dimension <= 0:
            raise ValueError(
                "output_dimension must be greater than zero."
            )

        if n_neighbors <= 0:
            raise ValueError(
                "n_neighbors must be greater than zero."
            )

        self.n_neighbors = n_neighbors
        self.metric = metric


    def reduce_features(
        self,
        features: tuple[Sequence[float], ...],
    ) -> tuple[tuple[float, ...], ...]:
        """
        Apply Isomap transformation.

        Steps:

        1. Validate input.
        2. Validate neighborhood size.
        3. Fit Isomap model.
        4. Convert output into immutable tuples.
        """

        if not features:
            return ()


        matrix = np.asarray(
            features,
            dtype=float,
        )


        if matrix.ndim != 2:
            raise ValueError(
                "Isomap requires "
                "a 2-dimensional feature matrix."
            )


        n_samples = matrix.shape[0]


        if self.n_neighbors >= n_samples:
            raise ValueError(
                "n_neighbors must be smaller "
                "than number of samples. "
                f"Got n_neighbors={self.n_neighbors}, "
                f"n_samples={n_samples}."
            )


        if self.output_dimension >= min(
            matrix.shape,
        ):
            raise ValueError(
                "output_dimension must be smaller "
                "than min(samples, features)."
            )


        model = SklearnIsomap(
            n_components=self.output_dimension,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
        )


        reduced = model.fit_transform(
            matrix,
        )


        return tuple(
            tuple(
                float(value)
                for value in row
            )
            for row in reduced
        )