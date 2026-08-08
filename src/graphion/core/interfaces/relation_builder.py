"""
Relation builder interface.
"""

from __future__ import annotations

from typing import (
    Generic,
    Protocol,
)

from graphion.core.types import TId

from graphion.core.models import (
    FeatureSet,
    RelationSet,
)

from graphion.core.results import StageResult

from .stage import Stage


class RelationBuilder(
    Stage,
    Protocol,
    Generic[TId],
):
    """
    Interface for building relations between entities.

    Implementations calculate relations between entities
    represented by feature vectors.

    Responsibilities:

    - feature comparison
    - relation score generation
    - optional raw-score to affinity conversion

    Relation.weight contains either:

    - the raw metric score, when ``as_affinity=False``
    - the graph affinity, when ``as_affinity=True``

    Graph builders consume relation weights as affinities
    and must not perform metric-specific conversion.
    """

    @property
    def name(
        self,
    ) -> str:
        """
        Metric name.

        Examples
        --------
        cosine
        euclidean
        pearson
        """
        ...

    def build(
        self,
        features: FeatureSet[TId],
        *,
        as_affinity: bool = True,
    ) -> RelationSet[TId]:
        """
        Build relations from a feature set.

        Parameters
        ----------
        features:
            Feature vectors indexed by entity id.

        as_affinity:
            If True, convert the generated raw metric scores
            into graph affinities before storing them in
            Relation.weight.

            If False, keep the raw metric scores unchanged.

            Defaults to True.

        Returns
        -------
        RelationSet[TId]

            Relations containing either raw metric scores
            or graph affinities, depending on ``as_affinity``.
        """
        ...

    def affinity(
        self,
        raw_score: float,
    ) -> float:
        """
        Convert a raw metric score into graph affinity.

        Parameters
        ----------
        raw_score:
            Raw metric output.

        Returns
        -------
        float

            Normalized graph affinity in the range:

                0 <= affinity <= 1
        """
        ...

    def execute(
        self,
        input_data: FeatureSet[TId],
    ) -> StageResult[RelationSet[TId]]:
        """
        Execute relation building stage.

        The stage execution uses the default behavior of
        ``build()``, meaning relation weights are returned
        as graph affinities.
        """

        return StageResult(
            output=self.build(
                input_data,
                as_affinity=True,
            ),
        )
