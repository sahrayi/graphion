from __future__ import annotations

import pytest

from graphion.builders.relation import (
    AngularSimilarity,
    BrayCurtisDistance,
    CanberraDistance,
    ChebyshevDistance,
    CosineSimilarity,
    DiceSimilarity,
    DotProduct,
    EuclideanDistance,
    HammingDistance,
    JaccardSimilarity,
    ManhattanDistance,
    MinkowskiDistance,
    OverlapCoefficient,
    PearsonCorrelation,
    RBFSimilarity,
    TanimotoSimilarity,
    WeightedJaccardSimilarity,
)

from graphion.core.models import FeatureSet


BUILDER_CONFIGS = [
    (AngularSimilarity, {}),
    (BrayCurtisDistance, {}),
    (CanberraDistance, {}),
    (ChebyshevDistance, {}),
    (CosineSimilarity, {}),
    (DiceSimilarity, {}),
    (DotProduct, {}),
    (EuclideanDistance, {}),
    (HammingDistance, {}),
    (JaccardSimilarity, {}),
    (ManhattanDistance, {}),
    (MinkowskiDistance, {}),
    (OverlapCoefficient, {}),
    (PearsonCorrelation, {}),
    (RBFSimilarity, {}),
    (TanimotoSimilarity, {}),
    (WeightedJaccardSimilarity, {}),
]


def build_test_features(
    builder_cls,
):
    """
    Create FeatureSet compatible
    with each relation builder.
    """

    if builder_cls is WeightedJaccardSimilarity:

        return FeatureSet(
            ids=(
                "a",
                "b",
                "c",
            ),
            features=(
                (
                    ("x", 1.0),
                    ("y", 0.5),
                ),
                (
                    ("x", 0.8),
                    ("z", 0.3),
                ),
                (
                    ("y", 0.7),
                    ("z", 0.9),
                ),
            ),
        )


    return FeatureSet(
        ids=(
            "a",
            "b",
            "c",
        ),
        features=(
            [
                1.0,
                0.0,
                1.0,
            ],
            [
                0.8,
                1.0,
                0.0,
            ],
            [
                0.0,
                1.0,
                1.0,
            ],
        ),
    )


def create_builder(
    builder_cls,
    kwargs,
):
    return builder_cls(
        **kwargs,
    )


def relation_signature(
    relations,
):
    return [
        (
            relation.source,
            relation.target,
            relation.weight,
        )
        for relation in relations
    ]


@pytest.mark.parametrize(
    "builder_cls,kwargs",
    BUILDER_CONFIGS,
)
def test_relation_builder_creates_relations(
    builder_cls,
    kwargs,
):

    features = build_test_features(
        builder_cls,
    )

    builder = create_builder(
        builder_cls,
        kwargs,
    )

    relations = builder.build(
        features,
    )

    assert relations

    assert len(relations) > 0



@pytest.mark.parametrize(
    "builder_cls,kwargs",
    BUILDER_CONFIGS,
)
def test_relation_builder_has_no_self_relations(
    builder_cls,
    kwargs,
):

    features = build_test_features(
        builder_cls,
    )

    builder = create_builder(
        builder_cls,
        kwargs,
    )

    relations = builder.build(
        features,
    )

    for relation in relations:

        assert relation.source != relation.target



@pytest.mark.parametrize(
    "builder_cls,kwargs",
    BUILDER_CONFIGS,
)
def test_relation_builder_is_deterministic(
    builder_cls,
    kwargs,
):

    features = build_test_features(
        builder_cls,
    )

    builder = create_builder(
        builder_cls,
        kwargs,
    )

    relations1 = builder.build(
        features,
    )

    relations2 = builder.build(
        features,
    )

    assert relation_signature(
        relations1,
    ) == relation_signature(
        relations2,
    )



@pytest.mark.parametrize(
    "builder_cls,kwargs",
    BUILDER_CONFIGS,
)
def test_relation_affinity_is_valid(
    builder_cls,
    kwargs,
):

    features = build_test_features(
        builder_cls,
    )

    builder = create_builder(
        builder_cls,
        kwargs,
    )

    relations = builder.build(
        features,
    )

    for relation in relations:

        affinity = builder.affinity(
            relation.weight,
        )

        assert -1e-10 <= affinity <= 1 + 1e-10