"""
UMAP dimensionality reduction.

Uses umap-learn implementation internally.
"""

from __future__ import annotations

import importlib
import warnings

from collections.abc import Sequence
from typing import Generic

import numpy as np

from .base_reducer import BaseReducer

from graphora.core.types import TId


class UMAP(
    BaseReducer[
        TId,
        Sequence[float],
        tuple[float, ...],
    ],
    Generic[TId],
):
    """
    UMAP based feature dimensionality reduction.

    Reduces high dimensional feature vectors
    into a lower dimensional representation.

    Uses external ``umap-learn`` package.


    Requires:

        pip install umap-learn


    Properties:

    - nonlinear
    - manifold based
    - preserves local structure
    - suitable for visualization
      and embedding exploration
    """

    def __init__(
        self,
        *,
        output_dimension: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        random_state: int | None = 42,
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


        if not 0.0 <= min_dist <= 1.0:
            raise ValueError(
                "min_dist must be between 0 and 1."
            )


        self.output_dimension = output_dimension
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state


        self._backend = self._load_backend()


    def _load_backend(
        self,
    ):
        """
        Load external umap-learn implementation.
        """

        try:

            module = importlib.import_module(
                "umap.umap_",
            )

        except ImportError as exc:

            raise ImportError(
                "UMAP requires 'umap-learn'. "
                "Install with: pip install umap-learn"
            ) from exc


        return module.UMAP


    def reduce_features(
        self,
        features: tuple[Sequence[float], ...],
    ) -> tuple[tuple[float, ...], ...]:
        """
        Apply UMAP transformation.

        Steps:

        1. Validate input.
        2. Validate neighborhood size.
        3. Fit UMAP model.
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
                "UMAP requires "
                "a 2-dimensional feature matrix."
            )


        n_samples = matrix.shape[0]


        if n_samples < 3:
            raise ValueError(
                "UMAP requires at least "
                "three samples."
            )


        if self.n_neighbors >= n_samples:
            raise ValueError(
                "n_neighbors must be smaller "
                "than number of samples. "
                f"Got n_neighbors={self.n_neighbors}, "
                f"n_samples={n_samples}."
            )


        model = self._backend(
            n_components=self.output_dimension,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
        )


        with warnings.catch_warnings():

            warnings.filterwarnings(
                "ignore",
                message=(
                    "n_jobs value .* overridden to 1 "
                    "by setting random_state"
                ),
                category=UserWarning,
                module="umap",
            )


            reduced = model.fit_transform(
                matrix,
            )


        if np.isnan(
            reduced,
        ).any():

            raise RuntimeError(
                "UMAP produced NaN values."
            )


        return tuple(
            tuple(
                float(value)
                for value in row
            )
            for row in reduced
        )