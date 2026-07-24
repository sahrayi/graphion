"""
Principal Component Analysis reducer.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Generic, TypeVar

import numpy as np
from sklearn.decomposition import PCA as SKLearnPCA

from graphora.core.models import FeatureSet

from .base_reducer import BaseReducer

from graphora.core.types import (
    TId,
)


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

        1. Convert FeatureSet features to numpy matrix.
        2. Fit PCA model.
        3. Transform features.
        4. Convert result back to immutable tuples.
        """

        matrix = np.asarray(
            features,
            dtype=float,
        )

        if matrix.ndim != 2:
            raise ValueError(
                "PCA requires a 2-dimensional feature matrix."
            )

        if self.output_dimension > matrix.shape[1]:
            raise ValueError(
                "output_dimension cannot be greater "
                "than input feature dimension."
            )

        if self.output_dimension > matrix.shape[0]:
            raise ValueError(
                "output_dimension cannot be greater "
                "than number of samples."
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