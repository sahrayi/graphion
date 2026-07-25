"""
t-SNE dimensionality reduction.

Uses scikit-learn TSNE implementation internally.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Generic

import numpy as np
from sklearn.manifold import TSNE as SklearnTSNE

from .base_reducer import BaseReducer

from graphion.core.types import TId


class TSNE(
    BaseReducer[
        TId,
        Sequence[float],
        tuple[float, ...],
    ],
    Generic[TId],
):
    """
    t-SNE based feature reducer.

    t-SNE is a nonlinear dimensionality reduction
    algorithm that preserves local neighborhood
    structures.

    Best suited for:

    - visualization
    - exploratory analysis
    - small and medium datasets

    Notes:

    - t-SNE does not support transforming
      unseen samples after fitting.
    - Reduced dimensions should usually be 2 or 3.
    """

    def __init__(
        self,
        *,
        output_dimension: int = 2,
        perplexity: float = 30.0,
        learning_rate: float | str = "auto",
        max_iter: int = 1000,
        metric: str = "euclidean",
        random_state: int | None = 42,
    ) -> None:

        super().__init__(
            output_dimension=output_dimension,
        )

        if output_dimension <= 0:
            raise ValueError(
                "output_dimension must be greater than zero."
            )

        if perplexity <= 0:
            raise ValueError(
                "perplexity must be greater than zero."
            )

        if max_iter <= 0:
            raise ValueError(
                "max_iter must be greater than zero."
            )

        self.output_dimension = output_dimension
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.metric = metric
        self.random_state = random_state


    def reduce_features(
        self,
        features: tuple[Sequence[float], ...],
    ) -> tuple[tuple[float, ...], ...]:
        """
        Apply t-SNE transformation.
        """

        if not features:
            return ()


        matrix = np.asarray(
            features,
            dtype=float,
        )


        if matrix.ndim != 2:
            raise ValueError(
                "t-SNE requires a 2-dimensional feature matrix."
            )


        n_samples = matrix.shape[0]


        if self.perplexity >= n_samples:
            raise ValueError(
                "perplexity must be smaller than "
                "number of samples. "
                f"Got perplexity={self.perplexity}, "
                f"samples={n_samples}."
            )


        model = SklearnTSNE(
            n_components=self.output_dimension,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            metric=self.metric,
            random_state=self.random_state,
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