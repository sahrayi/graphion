"""
Truncated Singular Value Decomposition dimensionality reduction.
"""

from __future__ import annotations

from typing import Generic

import numpy as np
from sklearn.decomposition import TruncatedSVD as SklearnTruncatedSVD

from graphora.core.types import TId

from .base_reducer import BaseReducer


class TruncatedSVD(
    BaseReducer[
        TId,
        tuple[float, ...],
        tuple[float, ...],
    ],
    Generic[TId],
):
    """
    Truncated SVD dimensionality reduction.

    Reduces high-dimensional feature vectors
    using truncated singular value decomposition.

    Supports:

    - dense numerical vectors
    - sparse-friendly reduction
    - large dimensional feature spaces

    Example:

        (10000,)
            |
            v
        TruncatedSVD
            |
            v
        (256,)

    """

    def __init__(
        self,
        *,
        output_dimension: int,
        n_iter: int = 5,
        random_state: int | None = None,
        **kwargs,
    ) -> None:

        super().__init__(
            output_dimension=output_dimension,
            **kwargs,
        )

        if n_iter <= 0:
            raise ValueError(
                "n_iter must be greater than zero."
            )

        self.model = SklearnTruncatedSVD(
            n_components=output_dimension,
            n_iter=n_iter,
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
        Apply truncated SVD reduction.

        Steps:

        1. Convert features into matrix.
        2. Fit SVD model.
        3. Transform features.
        4. Convert result into immutable tuples.
        """

        if not features:
            return ()

        matrix = np.asarray(
            features,
            dtype=float,
        )

        if matrix.ndim != 2:
            raise ValueError(
                "Truncated SVD requires "
                "a 2-dimensional feature matrix."
            )

        if self.output_dimension >= min(
            matrix.shape,
        ):
            raise ValueError(
                "output_dimension must be smaller "
                "than min(samples, features)."
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