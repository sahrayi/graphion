"""
Random Projection reducer.
"""

from __future__ import annotations

from typing import Generic

import numpy as np

from sklearn.random_projection import (
    GaussianRandomProjection,
)

from graphion.core.types import TId

from .base_reducer import BaseReducer


class RandomProjection(
    BaseReducer[
        TId,
        tuple[float, ...],
        tuple[float, ...],
    ],
    Generic[TId],
):
    """
    Gaussian Random Projection reducer.

    Projects high-dimensional numerical feature
    vectors into a lower-dimensional space.

    Example:

        (768,)
            |
            v
        RandomProjection
            |
            v
        (128,)


    Advantages:

    - fast
    - memory efficient
    - suitable for large datasets
    - approximately preserves pairwise distances
    """

    def __init__(
        self,
        *,
        output_dimension: int,
        eps: float = 0.1,
        random_state: int | None = None,
        **kwargs,
    ) -> None:

        if output_dimension <= 0:
            raise ValueError(
                "output_dimension must be "
                "greater than zero."
            )

        if eps <= 0:
            raise ValueError(
                "eps must be greater than zero."
            )

        super().__init__(
            output_dimension=output_dimension,
            **kwargs,
        )

        self.eps = eps
        self.random_state = random_state

        self.model = GaussianRandomProjection(
            n_components=output_dimension,
            eps=eps,
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
        Apply random projection.

        Steps:

        1. Handle empty input.
        2. Convert features to numpy matrix.
        3. Validate dimensions.
        4. Fit projection matrix.
        5. Transform features.
        6. Convert output back to immutable tuples.
        """

        if not features:
            return ()


        matrix = np.asarray(
            features,
            dtype=float,
        )


        if matrix.ndim != 2:
            raise ValueError(
                "Random projection requires "
                "a 2-dimensional feature matrix."
            )


        if matrix.shape[1] == 0:
            raise ValueError(
                "Feature dimension cannot be zero."
            )


        if not np.isfinite(
            matrix,
        ).all():

            raise ValueError(
                "Features contain NaN or infinite values."
            )


        if self.output_dimension > matrix.shape[1]:

            raise ValueError(
                "output_dimension cannot be greater "
                "than input feature dimension."
            )


        projected = self.model.fit_transform(
            matrix,
        )


        return tuple(
            tuple(
                float(value)
                for value in row
            )
            for row in projected
        )