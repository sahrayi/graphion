"""
Detailed correctness tests for CosineSimilarity.

The tests distinguish between:

- raw cosine similarity
- graph affinity

By default, BaseRelationBuilder.build() returns affinities.

Use:

    as_affinity=False

when testing the native cosine metric.
"""

from __future__ import annotations

import math

import pytest

from graphion.builders.relation.cosine_similarity import (
    CosineSimilarity,
)
from graphion.core.models import FeatureSet


# ==================================================
# Helpers
# ==================================================


def get_relation(
    relations,
    source,
    target,
):
    """
    Return a specific relation from a RelationSet.
    """

    matches = [
        relation
        for relation in relations
        if (
            relation.source == source
            and relation.target == target
        )
    ]

    assert matches, (
        f"Relation {source!r} -> {target!r} "
        "was not generated."
    )

    return matches[0]


# ==================================================
# Cosine similarity correctness
# ==================================================


class TestCosineSimilarityCorrectness:
    """
    Tests for the mathematical correctness of
    cosine similarity.

    These tests explicitly request raw scores because
    RelationBuilder.build() defaults to affinity output.
    """

    def test_known_value_orthogonal_vectors(self):
        """
        Orthogonal vectors must have cosine similarity 0.

        [1, 0] · [0, 1] = 0
        """

        feature_set = FeatureSet.from_lists(
            ids=["v1", "v2"],
            features=[
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )

        builder = CosineSimilarity()

        relations = builder.build(
            feature_set,
            as_affinity=False,
        )

        relation = get_relation(
            relations,
            "v1",
            "v2",
        )

        assert abs(
            relation.weight
        ) < 1e-10

    def test_known_value_identical_vectors(self):
        """
        Identical vectors must have cosine similarity 1.

        cosine(v, v) = 1
        """

        feature_set = FeatureSet.from_lists(
            ids=["v1", "v2"],
            features=[
                [3.0, 4.0],
                [3.0, 4.0],
            ],
        )

        builder = CosineSimilarity()

        relations = builder.build(
            feature_set,
            as_affinity=False,
        )

        relation = get_relation(
            relations,
            "v1",
            "v2",
        )

        assert abs(
            relation.weight - 1.0
        ) < 1e-10

    def test_known_value_opposite_vectors(self):
        """
        Opposite vectors must have cosine similarity -1.

        cosine(v, -v) = -1
        """

        feature_set = FeatureSet.from_lists(
            ids=["v1", "v2"],
            features=[
                [1.0, 0.0],
                [-1.0, 0.0],
            ],
        )

        builder = CosineSimilarity()

        relations = builder.build(
            feature_set,
            as_affinity=False,
        )

        relation = get_relation(
            relations,
            "v1",
            "v2",
        )

        assert abs(
            relation.weight + 1.0
        ) < 1e-10

    def test_known_value_manual_calculation(self):
        """
        Verify cosine similarity using manual calculation.

        a = [3, 4]
        b = [1, 0]

        a · b = 3

        ||a|| = 5
        ||b|| = 1

        cosine(a, b) = 3 / 5 = 0.6
        """

        feature_set = FeatureSet.from_lists(
            ids=["a", "b"],
            features=[
                [3.0, 4.0],
                [1.0, 0.0],
            ],
        )

        builder = CosineSimilarity()

        relations = builder.build(
            feature_set,
            as_affinity=False,
        )

        relation = get_relation(
            relations,
            "a",
            "b",
        )

        expected = 0.6

        assert abs(
            relation.weight - expected
        ) < 1e-10

    def test_symmetry_property(self):
        """
        Cosine similarity is symmetric.

        cosine(a, b) == cosine(b, a)
        """

        feature_set = FeatureSet.from_lists(
            ids=["x", "y"],
            features=[
                [2.0, 3.0],
                [4.0, 5.0],
            ],
        )

        builder = CosineSimilarity()

        relations = builder.build(
            feature_set,
            as_affinity=False,
        )

        xy = get_relation(
            relations,
            "x",
            "y",
        )

        yx = get_relation(
            relations,
            "y",
            "x",
        )

        assert abs(
            xy.weight - yx.weight
        ) < 1e-10

    def test_zero_vector_handling(self):
        """
        This implementation defines cosine similarity
        involving a zero vector as 0.
        """

        feature_set = FeatureSet.from_lists(
            ids=["zero", "nonzero"],
            features=[
                [0.0, 0.0],
                [3.0, 4.0],
            ],
        )

        builder = CosineSimilarity()

        relations = builder.build(
            feature_set,
            as_affinity=False,
        )

        zero_nonzero = get_relation(
            relations,
            "zero",
            "nonzero",
        )

        nonzero_zero = get_relation(
            relations,
            "nonzero",
            "zero",
        )

        assert zero_nonzero.weight == 0.0
        assert nonzero_zero.weight == 0.0

    def test_raw_score_preserves_negative_cosine(self):
        """
        Raw cosine similarity must remain negative.

        build(as_affinity=False) must not modify
        the native metric output.
        """

        feature_set = FeatureSet.from_lists(
            ids=["a", "b"],
            features=[
                [1.0],
                [-1.0],
            ],
        )

        builder = CosineSimilarity()

        relations = builder.build(
            feature_set,
            as_affinity=False,
        )

        relation = get_relation(
            relations,
            "a",
            "b",
        )

        assert relation.weight == -1.0

    def test_default_build_returns_affinity(self):
        """
        build() defaults to as_affinity=True.

        Therefore negative cosine similarity must be
        converted to zero.
        """

        feature_set = FeatureSet.from_lists(
            ids=["a", "b"],
            features=[
                [1.0],
                [-1.0],
            ],
        )

        builder = CosineSimilarity()

        relations = builder.build(
            feature_set,
        )

        relation = get_relation(
            relations,
            "a",
            "b",
        )

        assert relation.weight == 0.0


# ==================================================
# Affinity conversion
# ==================================================


class TestCosineAffinity:
    """
    Tests for cosine-to-affinity conversion.
    """

    def test_negative_score_maps_to_zero(self):
        """
        cosine = -1 -> affinity = 0
        """

        builder = CosineSimilarity()

        assert (
            builder.affinity(-1.0)
            == 0.0
        )

    def test_zero_score_maps_to_zero(self):
        """
        cosine = 0 -> affinity = 0
        """

        builder = CosineSimilarity()

        assert (
            builder.affinity(0.0)
            == 0.0
        )

    def test_positive_score_is_preserved(self):
        """
        Positive cosine values are preserved.
        """

        builder = CosineSimilarity()

        assert (
            builder.affinity(0.6)
            == 0.6
        )

    def test_maximum_score_maps_to_one(self):
        """
        cosine = 1 -> affinity = 1
        """

        builder = CosineSimilarity()

        assert (
            builder.affinity(1.0)
            == 1.0
        )

    def test_affinity_range(self):
        """
        Affinity must remain in [0, 1].
        """

        builder = CosineSimilarity()

        scores = [
            -1.0,
            -0.75,
            -0.5,
            -0.1,
            0.0,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
        ]

        for score in scores:

            affinity = builder.affinity(
                score
            )

            assert (
                0.0
                <= affinity
                <= 1.0
            )


# ==================================================
# Build affinity behavior
# ==================================================


class TestCosineBuildAffinity:
    """
    Tests verifying the as_affinity behavior of build().
    """

    def test_build_as_affinity_true(self):
        """
        Explicit as_affinity=True must store affinity.
        """

        feature_set = FeatureSet.from_lists(
            ids=["a", "b"],
            features=[
                [1.0],
                [-1.0],
            ],
        )

        builder = CosineSimilarity()

        relations = builder.build(
            feature_set,
            as_affinity=True,
        )

        relation = get_relation(
            relations,
            "a",
            "b",
        )

        assert relation.weight == 0.0

    def test_build_as_affinity_false(self):
        """
        Explicit as_affinity=False must store raw score.
        """

        feature_set = FeatureSet.from_lists(
            ids=["a", "b"],
            features=[
                [1.0],
                [-1.0],
            ],
        )

        builder = CosineSimilarity()

        relations = builder.build(
            feature_set,
            as_affinity=False,
        )

        relation = get_relation(
            relations,
            "a",
            "b",
        )

        assert relation.weight == -1.0

    def test_affinity_output_is_bounded(self):
        """
        All generated affinity weights must be in [0, 1].
        """

        feature_set = FeatureSet.from_lists(
            ids=["a", "b", "c"],
            features=[
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
        )

        builder = CosineSimilarity()

        relations = builder.build(
            feature_set,
            as_affinity=True,
        )

        for relation in relations:

            assert (
                0.0
                <= relation.weight
                <= 1.0
            )

    def test_raw_output_can_be_negative(self):
        """
        Raw cosine output may contain negative values.
        """

        feature_set = FeatureSet.from_lists(
            ids=["a", "b"],
            features=[
                [1.0, 0.0],
                [-1.0, 0.0],
            ],
        )

        builder = CosineSimilarity()

        relations = builder.build(
            feature_set,
            as_affinity=False,
        )

        for relation in relations:

            assert (
                -1.0
                <= relation.weight
                <= 1.0
            )


# ==================================================
# Normalization
# ==================================================


class TestCosineNormalization:
    """
    Tests for optional L2 feature normalization.
    """

    def test_normalization_does_not_change_cosine_result(self):
        """
        L2 normalization must not change cosine similarity.

        Cosine similarity is scale invariant.
        """

        feature_set = FeatureSet.from_lists(
            ids=["p", "q"],
            features=[
                [3.0, 4.0],
                [5.0, 12.0],
            ],
        )

        builder_unnormalized = CosineSimilarity(
            normalize=False,
        )

        builder_normalized = CosineSimilarity(
            normalize=True,
        )

        relations_unnormalized = (
            builder_unnormalized.build(
                feature_set,
                as_affinity=False,
            )
        )

        relations_normalized = (
            builder_normalized.build(
                feature_set,
                as_affinity=False,
            )
        )

        raw_unnormalized = get_relation(
            relations_unnormalized,
            "p",
            "q",
        )

        raw_normalized = get_relation(
            relations_normalized,
            "p",
            "q",
        )

        expected = 63.0 / 65.0

        assert abs(
            raw_unnormalized.weight
            - expected
        ) < 1e-10

        assert abs(
            raw_normalized.weight
            - expected
        ) < 1e-10

    def test_normalized_vector_has_unit_norm(self):
        """
        prepare_vector(normalize=True) must return
        a unit vector for non-zero input.
        """

        builder = CosineSimilarity(
            normalize=True,
        )

        vector = builder.prepare_vector(
            [3.0, 4.0],
        )

        norm = math.sqrt(
            sum(
                value * value
                for value in vector
            )
        )

        assert abs(
            norm - 1.0
        ) < 1e-10

    def test_zero_vector_remains_zero_when_normalized(self):
        """
        Zero vector must remain unchanged.
        """

        builder = CosineSimilarity(
            normalize=True,
        )

        vector = builder.prepare_vector(
            [0.0, 0.0],
        )

        assert vector == [
            0.0,
            0.0,
        ]

    def test_normalization_preserves_direction(self):
        """
        Normalization must preserve vector direction.
        """

        builder = CosineSimilarity(
            normalize=True,
        )

        vector = builder.prepare_vector(
            [3.0, 4.0],
        )

        assert abs(
            vector[0] - 0.6
        ) < 1e-10

        assert abs(
            vector[1] - 0.8
        ) < 1e-10


# ==================================================
# Edge cases
# ==================================================


class TestCosineEdgeCases:
    """
    Tests for numerical and dimensional edge cases.
    """

    def test_high_dimensional_vectors(self):
        """
        Cosine similarity must work for high-dimensional vectors.
        """

        feature_set = FeatureSet.from_lists(
            ids=["v1", "v2"],
            features=[
                [
                    float(i)
                    for i in range(1000)
                ],
                [
                    float(i + 1)
                    for i in range(1000)
                ],
            ],
        )

        builder = CosineSimilarity()

        relations = builder.build(
            feature_set,
            as_affinity=False,
        )

        for relation in relations:

            assert (
                -1.0
                <= relation.weight
                <= 1.0
            )

    def test_high_dimensional_affinity(self):
        """
        Affinity output for high-dimensional vectors
        must remain in [0, 1].
        """

        feature_set = FeatureSet.from_lists(
            ids=["v1", "v2"],
            features=[
                [
                    float(i)
                    for i in range(1000)
                ],
                [
                    float(i + 1)
                    for i in range(1000)
                ],
            ],
        )

        builder = CosineSimilarity()

        relations = builder.build(
            feature_set,
            as_affinity=True,
        )

        for relation in relations:

            assert (
                0.0
                <= relation.weight
                <= 1.0
            )

    def test_small_magnitude_values(self):
        """
        Very small values must be handled correctly.
        """

        feature_set = FeatureSet.from_lists(
            ids=["tiny1", "tiny2"],
            features=[
                [1e-10, 2e-10],
                [2e-10, 4e-10],
            ],
        )

        builder = CosineSimilarity()

        relations = builder.build(
            feature_set,
            as_affinity=False,
        )

        relation = get_relation(
            relations,
            "tiny1",
            "tiny2",
        )

        assert abs(
            relation.weight - 1.0
        ) < 1e-8

    def test_large_magnitude_values(self):
        """
        Large values must be handled correctly.
        """

        feature_set = FeatureSet.from_lists(
            ids=["large1", "large2"],
            features=[
                [1e10, 2e10],
                [2e10, 4e10],
            ],
        )

        builder = CosineSimilarity()

        relations = builder.build(
            feature_set,
            as_affinity=False,
        )

        relation = get_relation(
            relations,
            "large1",
            "large2",
        )

        assert abs(
            relation.weight - 1.0
        ) < 1e-10

    def test_single_dimension_vectors(self):
        """
        One-dimensional opposite vectors must have
        raw cosine similarity -1.
        """

        feature_set = FeatureSet.from_lists(
            ids=["positive", "negative"],
            features=[
                [5.0],
                [-5.0],
            ],
        )

        builder = CosineSimilarity()

        relations = builder.build(
            feature_set,
            as_affinity=False,
        )

        positive_negative = get_relation(
            relations,
            "positive",
            "negative",
        )

        negative_positive = get_relation(
            relations,
            "negative",
            "positive",
        )

        assert abs(
            positive_negative.weight + 1.0
        ) < 1e-10

        assert abs(
            negative_positive.weight + 1.0
        ) < 1e-10


# ==================================================
# Validation
# ==================================================


class TestCosineValidation:
    """
    Tests for input validation.
    """

    def test_mismatched_vector_dimensions(self):
        """
        Vectors with different dimensions must raise ValueError.
        """

        builder = CosineSimilarity()

        with pytest.raises(ValueError):
            builder.score(
                [1.0, 2.0],
                [1.0],
            )

    def test_empty_vector_is_rejected(self):
        """
        Empty feature vectors must be rejected.
        """

        builder = CosineSimilarity()

        with pytest.raises(ValueError):
            builder.prepare_vector([])

    def test_non_numeric_vector_is_rejected(self):
        """
        Non-numeric feature values must be rejected.
        """

        builder = CosineSimilarity()

        with pytest.raises(TypeError):
            builder.prepare_vector(
                [1.0, "invalid"],
            )


# ==================================================
# Determinism
# ==================================================


class TestCosineConsistency:
    """
    Tests for deterministic behavior.
    """

    def test_consistency_across_calls(self):
        """
        Repeated builds with identical input must
        produce identical relation sets.
        """

        feature_set = FeatureSet.from_lists(
            ids=["a", "b", "c"],
            features=[
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
        )

        builder = CosineSimilarity()

        result1 = builder.build(
            feature_set,
            as_affinity=False,
        )

        result2 = builder.build(
            feature_set,
            as_affinity=False,
        )

        assert result1.relations == result2.relations

    def test_relation_order_is_deterministic(self):
        """
        Relation generation order must be deterministic.
        """

        feature_set = FeatureSet.from_lists(
            ids=["a", "b", "c"],
            features=[
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
        )

        builder = CosineSimilarity()

        result = builder.build(
            feature_set,
            as_affinity=False,
        )

        pairs = [
            (
                relation.source,
                relation.target,
            )
            for relation in result.relations
        ]

        assert pairs == [
            ("a", "b"),
            ("a", "c"),
            ("b", "a"),
            ("b", "c"),
            ("c", "a"),
            ("c", "b"),
        ]


# ==================================================
# Main
# ==================================================


if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
        ]
    )