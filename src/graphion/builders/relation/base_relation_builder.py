"""
Base relation builder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from graphion.core.interfaces import RelationBuilder
from graphion.core.models import FeatureSet, Relation, RelationSet
from graphion.core.types import TFeature, TId, TPrepared


class BaseRelationBuilder(RelationBuilder, ABC):
    """
    Base implementation for relation builders.

    Generates pairwise relations between entities.

    Notes
    -----
    This implementation performs exhaustive pairwise comparison.

    Time complexity:
        O(n²)

    Suitable for small and medium datasets.
    """

    def __init__(
        self,
        *,
        feature_extractor: Callable[[TFeature], Any] | None = None,
        include_self: bool = False,
        symmetric: bool = False,
        sort_key: Callable[[TId], Any] | None = None,
    ) -> None:
        self.feature_extractor = (
            feature_extractor
            if feature_extractor is not None
            else self._default_extractor
        )
        self.include_self = include_self
        self.symmetric = symmetric
        self.sort_key = (
            sort_key
            if sort_key is not None
            else str
        )

    def build(self, features: FeatureSet[TId, TFeature]) -> RelationSet[TId]:
        """Build pairwise relations."""
        prepared_features = tuple(
            self.prepare_vector(self.feature_extractor(f))
            for f in features.features
        )

        relations = [
            Relation(
                source=src_id,
                target=tgt_id,
                weight=float(self.score(prepared_features[src_idx], prepared_features[tgt_idx])),
            )
            for src_idx, src_id in enumerate(features.ids)
            for tgt_idx, tgt_id in enumerate(features.ids)
            if self.include_self or src_idx != tgt_idx
        ]

        if self.symmetric:
            relations = self._symmetrize(relations)

        return RelationSet(relations=tuple(relations))

    @abstractmethod
    def prepare_vector(self, vector: Any) -> TPrepared:
        """Convert raw feature representation into metric-specific representation."""
        ...

    @abstractmethod
    def score(self, source: TPrepared, target: TPrepared) -> float:
        """Calculate raw metric score."""
        ...

    @abstractmethod
    def affinity(self, raw_score: float) -> float:
        """Convert raw metric score into graph affinity."""
        ...

    @staticmethod
    def _default_extractor(feature: TFeature) -> TFeature:
        """Return feature unchanged."""
        return feature

    def _symmetrize(self, relations: list[Relation[TId]]) -> list[Relation[TId]]:
        """Convert directed relations into symmetric form."""
        weights: dict[tuple[TId, TId], float] = {}

        for rel in relations:
            src, tgt = rel.source, rel.target
            key = (src, tgt) if src == tgt else tuple(sorted((src, tgt), key=self.sort_key))
            weights[key] = max(weights.get(key, float("-inf")), rel.weight)

        output: list[Relation[TId]] = []
        for key in sorted(weights.keys(), key=self._sort_relation_key):
            src, tgt = key
            weight = weights[key]
            output.append(Relation(source=src, target=tgt, weight=weight))
            if src != tgt:
                output.append(Relation(source=tgt, target=src, weight=weight))

        return output

    def _sort_relation_key(self, relation: tuple[TId, TId]) -> tuple:
        return (self.sort_key(relation[0]), self.sort_key(relation[1]))