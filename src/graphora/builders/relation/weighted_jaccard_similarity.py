"""
Weighted Jaccard similarity relation metric.
"""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import Any

from .base_relation_builder import (
    BaseRelationBuilder,
)


class WeightedJaccardSimilarity(
    BaseRelationBuilder,
):
    """
    Weighted Jaccard similarity metric.

    Designed for weighted set features:

    - TF-IDF feature maps
    - keyword importance scores
    - entity confidence scores
    - weighted categorical features

    Formula:

        similarity =
            Σ min(wA_i, wB_i)
            -----------------
            Σ max(wA_i, wB_i)

    Raw score range:

        0 <= score <= 1
    """

    @property
    def name(
        self,
    ) -> str:
        return "weighted_jaccard"


    def prepare_vector(
        self,
        vector: Any,
    ) -> dict[Hashable, float]:
        """
        Convert weighted feature collection
        into a normalized dictionary.

        Accepted input:

            [
                ("feature_a", 0.8),
                ("feature_b", 0.4),
            ]

        becomes:

            {
                "feature_a": 0.8,
                "feature_b": 0.4,
            }

        Duplicate features are aggregated.
        """

        if isinstance(
            vector,
            dict,
        ):
            items = vector.items()

        else:
            try:
                items = iter(vector)

            except TypeError as exc:
                raise TypeError(
                    "Weighted Jaccard features must be "
                    "a mapping or iterable of (feature, weight) pairs."
                ) from exc


        weighted_features: dict[
            Hashable,
            float,
        ] = {}


        for item in items:

            if (
                not isinstance(
                    item,
                    tuple,
                )
                or len(item) != 2
            ):
                raise TypeError(
                    "Weighted Jaccard features must be "
                    "provided as (feature, weight) pairs."
                )


            feature, weight = item


            if not isinstance(
                feature,
                Hashable,
            ):
                raise TypeError(
                    "Weighted Jaccard feature keys "
                    "must be hashable."
                )


            if not isinstance(
                weight,
                Real,
            ):
                raise TypeError(
                    "Weighted Jaccard weights "
                    "must be numeric."
                )


            weight = float(weight)


            if weight < 0:
                raise ValueError(
                    "Weighted Jaccard weights "
                    "cannot be negative."
                )


            weighted_features[feature] = (
                weighted_features.get(
                    feature,
                    0.0,
                )
                +
                weight
            )


        return weighted_features



    def score(
        self,
        source: dict[Hashable, float],
        target: dict[Hashable, float],
    ) -> float:
        """
        Compute Weighted Jaccard similarity.

        Formula:

            Σ min(source_i, target_i)
            -------------------------
            Σ max(source_i, target_i)
        """

        keys = (
            source.keys()
            |
            target.keys()
        )


        numerator = 0.0
        denominator = 0.0


        for key in keys:

            source_weight = source.get(
                key,
                0.0,
            )

            target_weight = target.get(
                key,
                0.0,
            )


            numerator += min(
                source_weight,
                target_weight,
            )

            denominator += max(
                source_weight,
                target_weight,
            )


        if denominator == 0:
            return 0.0


        return numerator / denominator



    def affinity(
        self,
        raw_score: float,
    ) -> float:
        """
        Convert Weighted Jaccard similarity
        into graph affinity.

        Weighted Jaccard is already normalized.

        Returns:

            0 <= affinity <= 1
        """

        return max(
            0.0,
            min(
                1.0,
                raw_score,
            ),
        )