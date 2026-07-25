"""
Base partition refiner implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic

from graphion.core.models import PartitionSet
from graphion.core.results import StageResult

from graphion.core.interfaces import PartitionRefiner

from graphion.core.types import (
    TId,
)


class BasePartitionRefiner(
    PartitionRefiner,
    ABC,
    Generic[TId],
):
    """
    Base class for partition refinement algorithms.

    Responsibilities:

    - define refinement contract
    - preserve immutable PartitionSet model
    - provide Stage execution compatibility

    Subclasses only implement:

        refine_partitions()

    """


    def refine(
        self,
        partitions: PartitionSet[TId],
    ) -> PartitionSet[TId]:
        """
        Refine partitions.

        Pipeline:

        1. Receive immutable PartitionSet.
        2. Apply refinement algorithm.
        3. Validate returned partitions.
        4. Build new PartitionSet.
        """

        refined_partitions = self.refine_partitions(
            partitions.partitions,
        )

        return PartitionSet(
            partitions=tuple(
                refined_partitions,
            ),
        )


    def execute(
        self,
        input_data: PartitionSet[TId],
    ) -> StageResult[
        PartitionSet[TId]
    ]:
        """
        Execute partition refinement stage.
        """

        return StageResult(
            output=self.refine(
                input_data,
            ),
        )


    @abstractmethod
    def refine_partitions(
        self,
        partitions: tuple[frozenset[TId], ...],
    ) -> tuple[frozenset[TId], ...]:
        """
        Apply refinement algorithm.

        Implemented by subclasses.
        """
        ...