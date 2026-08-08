"""
Base reducer implementation.
"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)

from typing import (
    Generic,
)

from graphion.core.models import (
    FeatureSet,
)

from graphion.core.results import (
    StageResult,
)

from graphion.core.interfaces import (
    Reducer,
)

from graphion.core.types import (
    TId,
    TFeature,
    TOutput,
)


class BaseReducer(
    Reducer,
    ABC,
    Generic[
        TId,
        TFeature,
        TOutput,
    ],
):
    """
    Base class for feature reduction algorithms.

    ```
    Responsibilities:

    - define reducer contract
    - preserve entity identifiers
    - convert reduced representations
    - provide Stage execution compatibility


    Subclasses only implement:

        reduce_features()

    """

    def __init__(
        self,
        *,
        output_dimension: int,
    ) -> None:

        if output_dimension <= 0:

            raise ValueError(
                "output_dimension must be greater than zero."
            )


        self.output_dimension = (
            output_dimension
        )


    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def reduce(
        self,
        features: FeatureSet[
            TId,
            TFeature,
        ],
    ) -> FeatureSet[
        TId,
        TOutput,
    ]:
        """
        Reduce FeatureSet dimensionality.

        Pipeline:

        1. Extract feature matrix.
        2. Run reduction algorithm.
        3. Rebuild FeatureSet preserving ids.
        """


        print(
            f"[Graphion] Feature reduction started "
            f"({self.__class__.__name__})"
        )

        print(
            "[Graphion] Reduction:"
            f" samples={len(features.ids)},"
            f" target_dimension={self.output_dimension}"
        )


        ids = features.ids


        reduced_features = self.reduce_features(
            features.features,
        )


        if len(ids) != len(reduced_features):

            raise ValueError(
                "Reducer output size does not match input size."
            )


        output = FeatureSet.from_lists(
            ids=ids,
            features=reduced_features,
        )


        print(
            "[Graphion] Feature reduction finished"
        )


        return output



    def execute(
        self,
        input_data: FeatureSet[
            TId,
            TFeature,
        ],
    ) -> StageResult[
        FeatureSet[
            TId,
            TOutput,
        ]
    ]:
        """
        Execute reducer stage.
        """


        return StageResult(
            output=self.reduce(
                input_data,
            ),
        )



    # --------------------------------------------------
    # Algorithm hook
    # --------------------------------------------------

    @abstractmethod
    def reduce_features(
        self,
        features: tuple[
            TFeature,
            ...
        ],
    ) -> tuple[
        TOutput,
        ...
    ]:
        """
        Apply dimensionality reduction algorithm.

        Implemented by subclasses.
        """

        ...