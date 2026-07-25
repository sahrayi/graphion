"""
Identity partition refiner.

Returns partitions unchanged.
"""

from __future__ import annotations

from typing import Generic

from .base_partition_refiner import (
    BasePartitionRefiner,
)

from graphion.core.types import (
    TId,
)


class IdentityPartitionRefiner(
    BasePartitionRefiner[TId],
    Generic[TId],
):
    """
    Identity partition refinement.

    Keeps partitions unchanged.

    Useful as:

    - default pipeline stage
    - testing refiner interface
    - disabling refinement
    """


    def refine_partitions(
        self,
        partitions: tuple[
            frozenset[TId],
            ...,
        ],
    ) -> tuple[
        frozenset[TId],
        ...,
    ]:
        """
        Return original partitions.
        """

        return partitions