from __future__ import annotations

import numpy as np

from graphion.core.models import FeatureSet

from graphion.reducers import (
    IdentityReducer,
    PCA,
    RandomProjection,
    TruncatedSVD,
    UMAP,
    TSNE,
    Isomap,
    Autoencoder,
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------


def create_random_feature_set(
    samples: int = 100,
    dimension: int = 768,
) -> FeatureSet:

    rng = np.random.default_rng(
        seed=42,
    )

    matrix = rng.random(
        (
            samples,
            dimension,
        )
    )

    return FeatureSet.from_numpy(
        ids=[
            f"node_{index}"
            for index in range(samples)
        ],
        matrix=matrix,
    )


def assert_reduced_feature_set(
    original: FeatureSet,
    reduced: FeatureSet,
    expected_dimension: int,
) -> None:

    assert len(reduced) == len(original)

    assert reduced.ids == original.ids

    assert reduced.dimension == expected_dimension

    assert all(
        isinstance(feature, tuple)
        for feature in reduced.features
    )


# --------------------------------------------------
# Identity
# --------------------------------------------------


def test_identity_reducer():

    features = create_random_feature_set()

    reducer = IdentityReducer()

    result = reducer.reduce(
        features,
    )

    assert result.ids == features.ids

    assert result.features == features.features

    assert result.dimension == features.dimension


# --------------------------------------------------
# PCA
# --------------------------------------------------


def test_pca_reducer():

    features = create_random_feature_set()

    reducer = PCA(
        output_dimension=32,
    )

    result = reducer.reduce(
        features,
    )

    assert_reduced_feature_set(
        features,
        result,
        32,
    )


# --------------------------------------------------
# Random Projection
# --------------------------------------------------


def test_random_projection_reducer():

    features = create_random_feature_set()

    reducer = RandomProjection(
        output_dimension=32,
        random_state=42,
    )

    result = reducer.reduce(
        features,
    )

    assert_reduced_feature_set(
        features,
        result,
        32,
    )


# --------------------------------------------------
# Truncated SVD
# --------------------------------------------------


def test_truncated_svd_reducer():

    features = create_random_feature_set(
        samples=200,
        dimension=768,
    )

    reducer = TruncatedSVD(
        output_dimension=32,
        random_state=42,
    )

    result = reducer.reduce(
        features,
    )

    assert_reduced_feature_set(
        features,
        result,
        32,
    )


# --------------------------------------------------
# UMAP
# --------------------------------------------------


def test_umap_reducer():

    features = create_random_feature_set()

    reducer = UMAP(
        output_dimension=2,
        n_neighbors=10,
        random_state=42,
    )

    result = reducer.reduce(
        features,
    )

    assert_reduced_feature_set(
        features,
        result,
        2,
    )


# --------------------------------------------------
# t-SNE
# --------------------------------------------------


def test_tsne_reducer():

    features = create_random_feature_set(
        samples=100,
        dimension=64,
    )

    reducer = TSNE(
        output_dimension=2,
        perplexity=10,
        max_iter=250,
        random_state=42,
    )

    result = reducer.reduce(
        features,
    )

    assert_reduced_feature_set(
        features,
        result,
        2,
    )


# --------------------------------------------------
# Isomap
# --------------------------------------------------


def test_isomap_reducer():

    features = create_random_feature_set(
        samples=100,
        dimension=64,
    )

    reducer = Isomap(
        output_dimension=2,
        n_neighbors=10,
    )

    result = reducer.reduce(
        features,
    )

    assert_reduced_feature_set(
        features,
        result,
        2,
    )


# --------------------------------------------------
# Autoencoder
# --------------------------------------------------


def test_autoencoder_reducer():

    features = create_random_feature_set(
        samples=100,
        dimension=64,
    )

    reducer = Autoencoder(
        output_dimension=8,
        hidden_dimension=32,
        epochs=5,
        random_state=42,
    )

    result = reducer.reduce(
        features,
    )

    assert_reduced_feature_set(
        features,
        result,
        8,
    )