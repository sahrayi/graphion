"""
Identity feature reducer.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Generic, TypeVar

from graphion.core.models import FeatureSet

from .base_reducer import BaseReducer


from graphion.core.types import (
    TId, TFeature,
)


class IdentityReducer(
    BaseReducer[
        TId,
        TFeature,
        TFeature,
    ],
    Generic[TId, TFeature],
):
    """
    Identity reducer.

    Returns features unchanged.

    This reducer is useful when:

    - dimensionality reduction is disabled
    - testing pipeline stages
    - comparing reduced vs original features

    The feature dimension remains unchanged.
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:

        # Identity does not reduce dimensions.
        # Keep value for BaseReducer compatibility.
        super().__init__(
            output_dimension=1,
            **kwargs,
        )

    def reduce(
        self,
        features: FeatureSet[TId, TFeature],
    ) -> FeatureSet[TId, TFeature]:
        """
        Return original FeatureSet unchanged.
        """

        return features

    def reduce_features(
        self,
        features: tuple[TFeature, ...],
    ) -> tuple[TFeature, ...]:
        """
        Identity transformation.

        Not used because reduce() is overridden.
        """

        return features