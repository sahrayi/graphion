"""
Relation builder interface.
"""

from __future__ import annotations

from typing import (
    Generic,
    Protocol,
)

from graphora.core.types import TId

from graphora.core.models import (
    FeatureSet,
    RelationSet,
)
from graphora.core.results import StageResult

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
    - raw metric conversion into graph affinity

    Relation.weight stores the raw metric output.

    Graph construction must never interpret raw scores
    directly. It should always use affinity().
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
    ) -> RelationSet[TId]:
        """
        Build relations from feature set.

        Parameters
        ----------
        features:
            Feature vectors indexed by entity id.

        Returns
        -------
        RelationSet[TId]

            Relations containing raw metric scores.
        """
        ...

    def affinity(
        self,
        raw_score: float,
    ) -> float:
        """
        Convert raw metric score into graph affinity.

        Returns
        -------
        float

            Normalized affinity:

                0 <= affinity <= 1
        """
        ...

    def execute(
        self,
        input_data: FeatureSet[TId],
    ) -> StageResult[RelationSet[TId]]:
        """
        Execute relation building stage.
        """

        return StageResult(
            output=self.build(
                input_data,
            ),
        )