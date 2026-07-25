"""
Tests for FeatureSet model.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from graphion.core.errors import (
    InvalidFeatureSetError,
)

from graphion.core.models import (
    FeatureSet,
)


def test_feature_set_creation() -> None:
    """
    FeatureSet should be created successfully
    with valid feature vectors.
    """

    feature_set = FeatureSet(
        ids=[1, 2, 3],
        features=[
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ],
    )

    assert len(feature_set) == 3

    assert feature_set.ids == (
        1,
        2,
        3,
    )

    assert feature_set.features == (
        (
            0.1,
            0.2,
        ),
        (
            0.3,
            0.4,
        ),
        (
            0.5,
            0.6,
        ),
    )


def test_feature_set_is_immutable() -> None:
    """
    FeatureSet should be immutable.
    """

    feature_set = FeatureSet(
        ids=(1, 2),
        features=(
            (0.1, 0.2),
            (0.3, 0.4),
        ),
    )

    with pytest.raises(
        (FrozenInstanceError, AttributeError),
    ):
        feature_set.ids = (
            3,
            4,
        )


def test_feature_set_copies_mutable_inputs() -> None:
    """
    Mutable input collections should not
    affect stored values.
    """

    ids = [
        1,
        2,
    ]

    features = [
        [
            0.1,
            0.2,
        ],
        [
            0.3,
            0.4,
        ],
    ]

    feature_set = FeatureSet(
        ids=ids,
        features=features,
    )

    ids.append(3)
    features[0].append(9.9)

    assert feature_set.ids == (
        1,
        2,
    )

    assert feature_set.features == (
        (
            0.1,
            0.2,
        ),
        (
            0.3,
            0.4,
        ),
    )


def test_feature_set_from_dict() -> None:
    """
    FeatureSet should be creatable from mapping.
    """

    feature_set = FeatureSet.from_dict(
        {
            "A": [
                1,
                2,
            ],
            "B": [
                3,
                4,
            ],
        }
    )

    assert feature_set.ids == (
        "A",
        "B",
    )

    assert feature_set.dimension == 2


def test_feature_set_to_dict() -> None:
    """
    FeatureSet should convert back to mapping.
    """

    feature_set = FeatureSet.from_dict(
        {
            "A": [
                1,
                2,
            ],
            "B": [
                3,
                4,
            ],
        }
    )

    assert feature_set.to_dict() == {
        "A": (
            1,
            2,
        ),
        "B": (
            3,
            4,
        ),
    }


def test_feature_set_numpy_roundtrip() -> None:
    """
    FeatureSet should support numpy conversion.
    """

    matrix = np.array(
        [
            [
                1,
                2,
            ],
            [
                3,
                4,
            ],
        ]
    )

    feature_set = FeatureSet.from_numpy(
        ids=[
            "A",
            "B",
        ],
        matrix=matrix,
    )

    ids, result = feature_set.to_numpy()

    assert ids == (
        "A",
        "B",
    )

    assert np.array_equal(
        result,
        matrix.astype(float),
    )


def test_feature_set_dataframe_roundtrip() -> None:
    """
    FeatureSet should support pandas conversion.
    """

    dataframe = pd.DataFrame(
        {
            "id": [
                "A",
                "B",
            ],
            "feature_0": [
                1,
                3,
            ],
            "feature_1": [
                2,
                4,
            ],
        }
    )

    feature_set = FeatureSet.from_dataframe(
        dataframe,
        id_column="id",
    )

    result = feature_set.to_dataframe()

    assert list(
        result["id"]
    ) == [
        "A",
        "B",
    ]

    assert list(
        result.columns
    ) == [
        "id",
        "feature_0",
        "feature_1",
    ]


def test_feature_set_from_lists() -> None:
    """
    FeatureSet should support plain python lists.
    """

    feature_set = FeatureSet.from_lists(
        ids=[
            1,
            2,
        ],
        features=[
            [
                0.1,
                0.2,
            ],
            [
                0.3,
                0.4,
            ],
        ],
    )

    assert len(feature_set) == 2


def test_feature_set_rejects_length_mismatch() -> None:
    """
    FeatureSet should reject different ids
    and feature counts.
    """

    with pytest.raises(
        InvalidFeatureSetError,
    ):
        FeatureSet(
            ids=(
                1,
                2,
                3,
            ),
            features=(
                (
                    0.1,
                ),
                (
                    0.2,
                ),
            ),
        )


def test_feature_set_rejects_duplicate_ids() -> None:
    """
    FeatureSet should reject duplicate identifiers.
    """

    with pytest.raises(
        InvalidFeatureSetError,
    ):
        FeatureSet(
            ids=(
                1,
                1,
            ),
            features=(
                (
                    0.1,
                ),
                (
                    0.2,
                ),
            ),
        )


def test_feature_set_iteration() -> None:
    """
    Iteration should return id-feature pairs.
    """

    feature_set = FeatureSet(
        ids=(
            1,
            2,
        ),
        features=(
            (
                0.1,
            ),
            (
                0.2,
            ),
        ),
    )

    assert list(feature_set) == [
        (
            1,
            (
                0.1,
            ),
        ),
        (
            2,
            (
                0.2,
            ),
        ),
    ]


def test_empty_feature_set() -> None:
    """
    Empty FeatureSet should be supported.
    """

    feature_set = FeatureSet(
        ids=(),
        features=(),
    )

    assert feature_set.is_empty is True
    assert feature_set.dimension is None