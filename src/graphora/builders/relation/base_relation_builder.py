"""
Base relation builder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from collections.abc import (
    Callable,
    Hashable,
    Iterable,
)

from typing import (
    Any,
    TypeVar,
)

from graphora.core.interfaces import RelationBuilder

from graphora.core.models import (
    FeatureSet,
    Relation,
    RelationSet,
)


from graphora.core.types import (
    TId, TFeature, TPrepared
)


class BaseRelationBuilder(
    RelationBuilder,
    ABC,
):
    """
    Base implementation for relation builders.

    This class provides common pairwise relation
    generation logic.

    Subclasses define only:

    - prepare_vector()
    - score()
    - affinity()

    Feature representation is intentionally
    not restricted here.

    Examples:

    - dense numeric vectors
    - sparse mappings
    - weighted sets
    - custom representations
    """


    def __init__(
        self,
        *,
        feature_extractor: Callable[
            [TFeature],
            Any,
        ]
        | None = None,

        include_self: bool = False,

        symmetric: bool = False,

        sort_key: Callable[
            [TId],
            Any,
        ]
        | None = None,

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


    def build(
        self,
        features: FeatureSet[
            TId,
            TFeature,
        ],
    ) -> RelationSet[TId]:
        """
        Build relations between all entities.

        The generated relation weight stores
        raw metric score.

        Affinity conversion is intentionally
        separated and handled by affinity().
        """

        prepared_features = tuple(
            self.prepare_vector(
                self.feature_extractor(
                    feature,
                )
            )
            for feature in features.features
        )


        relations: list[
            Relation[TId]
        ] = []


        for source_index, source_id in enumerate(
            features.ids
        ):

            for target_index, target_id in enumerate(
                features.ids
            ):

                if (
                    not self.include_self
                    and source_index == target_index
                ):
                    continue


                raw_score = self.score(
                    prepared_features[source_index],
                    prepared_features[target_index],
                )


                relations.append(
                    Relation(
                        source=source_id,
                        target=target_id,
                        weight=float(raw_score),
                    )
                )


        if self.symmetric:
            relations = self._symmetrize(
                relations,
            )


        return RelationSet(
            relations=tuple(relations),
        )


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
        Calculate raw similarity/distance score.
        """
        ...


    @abstractmethod
    def affinity(
        self,
        raw_score: float,
    ) -> float:
        """
        Convert raw metric score into
        graph affinity.
        """
        ...


    @staticmethod
    def _default_extractor(
        feature: TFeature,
    ) -> TFeature:
        """
        Default feature extractor.

        Returns feature unchanged.

        Feature interpretation belongs
        to metric implementation.
        """

        return feature



    def _symmetrize(
        self,
        relations: list[
            Relation[TId]
        ],
    ) -> list[
        Relation[TId]
    ]:
        """
        Convert directed relations into symmetric ones.

        For duplicate directions:

            A -> B
            B -> A

        maximum weight is preserved.

        Output ordering is deterministic.
        """


        weights: dict[
            frozenset[TId]
            | tuple[TId, TId],
            float,
        ] = {}


        for relation in relations:

            if relation.source == relation.target:

                key = (
                    relation.source,
                    relation.target,
                )

            else:

                key = frozenset(
                    (
                        relation.source,
                        relation.target,
                    )
                )


            weights[key] = max(
                weights.get(
                    key,
                    float("-inf"),
                ),
                relation.weight,
            )


        output: list[
            Relation[TId]
        ] = []


        for key in sorted(
            weights.keys(),
            key=self._sort_relation_key,
        ):

            weight = weights[key]


            if isinstance(
                key,
                tuple,
            ):

                source, target = key

                output.append(
                    Relation(
                        source=source,
                        target=target,
                        weight=weight,
                    )
                )

                continue



            source, target = sorted(
                key,
                key=self.sort_key,
            )


            output.append(
                Relation(
                    source=source,
                    target=target,
                    weight=weight,
                )
            )

            output.append(
                Relation(
                    source=target,
                    target=source,
                    weight=weight,
                )
            )


        return output



    def _sort_relation_key(
        self,
        key,
    ) -> tuple:

        if isinstance(
            key,
            tuple,
        ):
            return (
                self.sort_key(key[0]),
                self.sort_key(key[1]),
            )


        return tuple(
            sorted(
                (
                    self.sort_key(item)
                    for item in key
                )
            )
        )