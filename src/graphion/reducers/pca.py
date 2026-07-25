"""
Principal Component Analysis reducer.
"""

from __future__ import annotations

import numpy as np

from typing import Generic

from sklearn.decomposition import PCA as SKLearnPCA

from graphion.core.types import TId

from .base_reducer import BaseReducer


class PCA(
    BaseReducer[
        TId,
        tuple[float, ...],
        tuple[float, ...],
    ],
    Generic[TId],
):
    """
    Principal Component Analysis reducer.

    Reduces dense numerical feature vectors
    into a lower-dimensional representation.

    Example:

        (768,)
            |
            v
        PCA
            |
            v
        (128,)

    Properties:

    - linear dimensionality reduction
    - deterministic
    - preserves maximum variance
    - suitable for dense numerical features
    """

    def __init__(
        self,
        *,
        output_dimension: int,
        whiten: bool = False,
        random_state: int | None = None,
        **kwargs,
    ) -> None:

        super().__init__(
            output_dimension=output_dimension,
            **kwargs,
        )

        if output_dimension <= 0:
            raise ValueError(
                "output_dimension must be greater than zero."
            )

        self.model = SKLearnPCA(
            n_components=output_dimension,
            whiten=whiten,
            random_state=random_state,
        )


    def reduce_features(
        self,
        features: tuple[
            tuple[float, ...],
            ...,
        ],
    ) -> tuple[
        tuple[float, ...],
        ...,
    ]:
        """
        Apply PCA transformation.

        Steps:

        1. Convert feature vectors into numpy matrix.
        2. Validate input dimensions.
        3. Fit PCA model.
        4. Transform features.
        5. Convert result back to immutable tuples.
        """

        if not features:
            return ()


        matrix = np.asarray(
            features,
            dtype=float,
        )


        if matrix.ndim != 2:
            raise ValueError(
                "PCA requires a 2-dimensional feature matrix."
            )


        if matrix.shape[0] == 0:
            return ()


        if matrix.shape[1] == 0:
            raise ValueError(
                "PCA cannot reduce empty feature vectors."
            )


        if not np.isfinite(matrix).all():
            raise ValueError(
                "features contain NaN or infinite values."
            )


        max_components = min(
            matrix.shape[0],
            matrix.shape[1],
        )


        if self.output_dimension > max_components:

            raise ValueError(
                "output_dimension cannot be greater "
                "than min(number of samples, "
                "input feature dimension)."
            )


        reduced = self.model.fit_transform(
            matrix,
        )


        return tuple(
            tuple(
                float(value)
                for value in row
            )
            for row in reduced
        )