"""
FeatureSet data model.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic

import numpy as np

from graphora.core.errors import InvalidFeatureSetError
from graphora.core.types import TId, TFeature


@dataclass(frozen=True, slots=True)
class FeatureSet(Generic[TId, TFeature]):
    """
    Immutable collection of feature representations.

    Each feature representation is associated
    with a unique entity identifier.

    FeatureSet is intentionally agnostic about
    feature semantics.

    Supported examples:

    - dense vectors
    - embeddings
    - sparse dictionaries
    - categorical features
    - custom feature objects
    """

    ids: tuple[TId, ...]
    features: tuple[TFeature, ...]

    def __post_init__(self) -> None:
        """
        Validate and normalize FeatureSet data.
        """
        ids = tuple(self.ids)
        features = tuple(self._freeze_feature(f) for f in self.features)

        self._validate_ids(ids)
        self._validate_feature_count(ids, features)
        self._validate_dimensions(features)

        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "features", features)

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    @staticmethod
    def _validate_ids(ids: tuple[TId, ...]) -> None:
        """
        Validate entity identifiers.
        """
        if any(identifier is None for identifier in ids):
            raise InvalidFeatureSetError("Feature identifiers cannot be None.")

        if len(set(ids)) != len(ids):
            raise InvalidFeatureSetError("Duplicate identifiers are not allowed.")

    @staticmethod
    def _validate_feature_count(
        ids: tuple[TId, ...],
        features: tuple[TFeature, ...],
    ) -> None:
        """
        Ensure ids and features have equal length.
        """
        if len(ids) != len(features):
            raise InvalidFeatureSetError(
                "The number of ids must match the number of features."
            )

    @staticmethod
    def _feature_dimension(feature) -> int | None:
        """
        Return dimension for sequence-based features.

        Non-vector objects are ignored.
        """
        if isinstance(feature, Sequence) and not isinstance(feature, (str, bytes)):
            return len(feature)
        return None

    @classmethod
    def _validate_dimensions(cls, features: tuple[TFeature, ...]) -> None:
        """
        Ensure vector-like features have consistent dimensions.

        Non-vector features are ignored because FeatureSet
        supports arbitrary feature objects.
        """
        dimensions: set[int] = set()
        for feature in features:
            dimension = cls._feature_dimension(feature)
            if dimension is not None:
                dimensions.add(dimension)

        if len(dimensions) > 1:
            raise InvalidFeatureSetError(
                "All feature vectors must have the same dimension."
            )

    # --------------------------------------------------
    # Constructors
    # --------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        data: Mapping[TId, TFeature],
    ) -> FeatureSet[TId, TFeature]:
        """
        Create FeatureSet from id -> feature mapping.
        """
        return cls(
            ids=tuple(data.keys()),
            features=tuple(data.values()),
        )

    @classmethod
    def from_lists(
        cls,
        ids: Sequence[TId],
        features: Sequence[TFeature],
    ) -> FeatureSet[TId, TFeature]:
        """
        Create FeatureSet from python sequences.
        """
        return cls(
            ids=tuple(ids),
            features=tuple(features),
        )

    @classmethod
    def from_numpy(
        cls,
        ids: Sequence[TId],
        matrix,
    ) -> FeatureSet[TId, tuple[float, ...]]:
        """
        Create FeatureSet from numpy matrix.

        Expected shape:

            (samples, features)
        """
        matrix = np.asarray(matrix)
        if matrix.ndim != 2:
            raise InvalidFeatureSetError("Feature matrix must be two-dimensional.")

        rows = matrix.tolist()
        return cls(
            ids=tuple(ids),
            features=tuple(tuple(float(value) for value in row) for row in rows),
        )

    @classmethod
    def from_dataframe(
        cls,
        dataframe,
        *,
        id_column: str,
    ) -> FeatureSet[TId, tuple[float, ...]]:
        """
        Create FeatureSet from pandas DataFrame.
        """
        ids = tuple(dataframe[id_column].tolist())
        matrix = dataframe.drop(columns=[id_column]).to_numpy()
        return cls.from_numpy(ids, matrix)

    # --------------------------------------------------
    # Converters
    # --------------------------------------------------

    def to_dict(self) -> dict[TId, TFeature]:
        """
        Convert FeatureSet into id -> feature mapping.
        """
        return dict(self)

    def to_numpy(self):
        """
        Convert numerical features to numpy matrix.

        Raises
        ------
        TypeError
            If features are not numerical vectors.
        """
        try:
            if self.is_empty:
                matrix = np.empty((0, 0), dtype=float)
            else:
                matrix = np.asarray(self.features, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("FeatureSet contains non numerical features.") from exc

        return self.ids, matrix

    def to_dataframe(
        self,
        *,
        id_column: str = "id",
    ):
        """
        Convert numerical FeatureSet into pandas DataFrame.
        """
        import pandas as pd

        _, matrix = self.to_numpy()
        dataframe = pd.DataFrame(
            matrix,
            columns=[f"feature_{index}" for index in range(matrix.shape[1])],
        )
        dataframe.insert(0, id_column, self.ids)
        return dataframe

    # --------------------------------------------------
    # Protocols
    # --------------------------------------------------

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self) -> Iterator[tuple[TId, TFeature]]:
        return iter(zip(self.ids, self.features))

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    @property
    def dimension(self) -> int | None:
        """
        Return feature dimension.

        Returns None for empty FeatureSets,
        inconsistent dimensions, or non-vector features.
        """
        if self.is_empty:
            return None

        dimensions: set[int] = set()
        for feature in self.features:
            dimension = self._feature_dimension(feature)
            if dimension is not None:
                dimensions.add(dimension)

        if len(dimensions) == 1:
            return dimensions.pop()

        return None

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    @staticmethod
    def _freeze_feature(feature):
        """
        Recursively convert mutable sequences
        into immutable tuples.

        Dictionaries and custom objects are
        intentionally left unchanged.
        """
        if isinstance(feature, list):
            return tuple(FeatureSet._freeze_feature(item) for item in feature)
        if isinstance(feature, tuple):
            return tuple(FeatureSet._freeze_feature(item) for item in feature)
        return feature