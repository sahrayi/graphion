"""
Random Projection reducer.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Generic, TypeVar

import numpy as np
from sklearn.random_projection import (
    GaussianRandomProjection,
)

from graphora.core.models import FeatureSet

from .base_reducer import BaseReducer


from graphora.core.types import (
    TId,
)


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
    - preserves pairwise distances approximately

    """

    def __init__(
        self,
        *,
        output_dimension: int,
        eps: float = 0.1,
        random_state: int | None = None,
        **kwargs,
    ) -> None:

        super().__init__(
            output_dimension=output_dimension,
            **kwargs,
        )

        if eps <= 0:
            raise ValueError(
                "eps must be greater than zero."
            )

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

        1. Convert features to numpy matrix.
        2. Fit projection matrix.
        3. Transform features.
        4. Convert output back to immutable tuples.
        """

        matrix = np.asarray(
            features,
            dtype=float,
        )

        if matrix.ndim != 2:
            raise ValueError(
                "Random projection requires "
                "a 2-dimensional feature matrix."
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