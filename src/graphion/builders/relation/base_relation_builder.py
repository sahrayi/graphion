"""
Base relation builder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from tqdm import tqdm

from graphion.core.interfaces import RelationBuilder

from graphion.core.models import (
    FeatureSet,
    Relation,
    RelationSet,
)

from graphion.core.types import (
    TFeature,
    TId,
    TPrepared,
)


class BaseRelationBuilder(
    RelationBuilder,
    ABC,
):
    """
    Base implementation for relation builders.

    Generates pairwise relations between entities.

    Notes
    -----
    This implementation performs exhaustive pairwise comparison.

    Time complexity:
        O(n²)

    Suitable for small and medium datasets.

    Relation weights
    ----------------
    By default, ``build()`` converts raw metric scores into
    graph affinities before storing them in ``Relation.weight``.

    This keeps the graph-building layer independent from the
    underlying metric.
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

    # ==================================================
    # Build
    # ==================================================

    def build(
        self,
        features: FeatureSet[TId, TFeature],
        *,
        as_affinity: bool = True,
    ) -> RelationSet[TId]:
        """
        Build pairwise relations.

        Parameters
        ----------
        features:
            Feature vectors indexed by entity id.

        as_affinity:
            If True, convert raw metric scores into graph
            affinities before storing them in Relation.weight.

            If False, store the raw metric scores directly.

            Defaults to True.

        Returns
        -------
        RelationSet[TId]
            Generated pairwise relations.
        """

        print()
        print("[Relation Builder]")
        print("-" * 70)

        print("Preparing features...")

        prepared_features = tuple(
            self.prepare_vector(
                self.feature_extractor(feature)
            )
            for feature in features.features
        )

        print(
            f"✓ Prepared features: "
            f"{len(prepared_features):,}"
        )

        print()

        print("Computing pairwise relations...")

        n = len(features.ids)

        print(
            f"✓ Entities       : {n:,}"
        )

        print(
            f"✓ Comparisons    : "
            f"{n * n:,}"
        )

        relations: list[Relation[TId]] = []

        for src_idx, src_id in tqdm(
            enumerate(features.ids),
            total=n,
            desc="Relations",
        ):

            source_vector = prepared_features[src_idx]

            for tgt_idx, tgt_id in enumerate(
                features.ids
            ):

                if (
                    not self.include_self
                    and src_idx == tgt_idx
                ):
                    continue

                raw_score = float(
                    self.score(
                        source_vector,
                        prepared_features[tgt_idx],
                    )
                )

                weight = (
                    self.affinity(raw_score)
                    if as_affinity
                    else raw_score
                )

                relations.append(
                    Relation(
                        source=src_id,
                        target=tgt_id,
                        weight=float(weight),
                    )
                )

        print()

        print(
            f"✓ Generated relations: "
            f"{len(relations):,}"
        )

        if as_affinity:
            print(
                "✓ Relation weights: affinity"
            )
        else:
            print(
                "✓ Relation weights: raw scores"
            )

        if self.symmetric:

            print()

            print("Symmetrizing relations...")

            relations = self._symmetrize(
                relations
            )

            print(
                f"✓ Symmetric relations: "
                f"{len(relations):,}"
            )

        print()

        print("Relation building completed")

        print("-" * 70)

        return RelationSet(
            relations=tuple(relations)
        )

    # ==================================================
    # Metric API
    # ==================================================

    @abstractmethod
    def prepare_vector(
        self,
        vector: Any,
    ) -> TPrepared:
        """
        Convert raw feature representation
        into metric-specific representation.
        """
        ...

    @abstractmethod
    def score(
        self,
        source: TPrepared,
        target: TPrepared,
    ) -> float:
        """
        Calculate raw metric score.

        The returned value represents the native metric
        output and is not required to be in the range
        [0, 1].
        """
        ...

    @abstractmethod
    def affinity(
        self,
        raw_score: float,
    ) -> float:
        """
        Convert raw metric score into graph affinity.

        Returns
        -------
        float
            Graph affinity in the range:

                0 <= affinity <= 1
        """
        ...

    # ==================================================
    # Utilities
    # ==================================================

    @staticmethod
    def _default_extractor(
        feature: TFeature,
    ) -> TFeature:
        """
        Return feature unchanged.
        """

        return feature

    # ==================================================
    # Symmetrization
    # ==================================================

    def _symmetrize(
        self,
        relations: list[Relation[TId]],
    ) -> list[Relation[TId]]:
        """
        Convert directed relations into symmetric form.

        For every unordered pair, the strongest relation
        weight is retained and emitted in both directions.

        Important
        ---------
        The operation is metric-agnostic.

        It works equally for raw scores and affinities.
        """

        weights: dict[
            tuple[TId, TId],
            float,
        ] = {}

        for rel in relations:

            src = rel.source

            tgt = rel.target

            key = (
                (src, tgt)
                if src == tgt
                else tuple(
                    sorted(
                        (src, tgt),
                        key=self.sort_key,
                    )
                )
            )

            weights[key] = max(
                weights.get(
                    key,
                    float("-inf"),
                ),
                rel.weight,
            )

        output: list[Relation[TId]] = []

        for key in sorted(
            weights.keys(),
            key=self._sort_relation_key,
        ):

            src, tgt = key

            weight = weights[key]

            output.append(
                Relation(
                    source=src,
                    target=tgt,
                    weight=weight,
                )
            )

            if src != tgt:

                output.append(
                    Relation(
                        source=tgt,
                        target=src,
                        weight=weight,
                    )
                )

        return output

    # ==================================================
    # Sorting
    # ==================================================

    def _sort_relation_key(
        self,
        relation: tuple[TId, TId],
    ) -> tuple:

        return (
            self.sort_key(relation[0]),
            self.sort_key(relation[1]),
        )