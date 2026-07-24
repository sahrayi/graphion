"""
Tests for PartitionSet model.
"""

from dataclasses import FrozenInstanceError

import pytest

from graphora.core.errors import InvalidPartitionSetError
from graphora.core.models import PartitionSet


def test_partition_set_creation() -> None:
    """
    PartitionSet should be created successfully with valid partitions.
    """
    partition_set = PartitionSet(
        partitions=[
            {1, 2},
            {3, 4},
        ],
    )

    assert len(partition_set) == 2
    assert partition_set.partition_count == 2
    assert partition_set.partitions == (
        frozenset({1, 2}),
        frozenset({3, 4}),
    )


def test_partition_set_is_immutable() -> None:
    """
    PartitionSet should be immutable.
    """
    partition_set = PartitionSet(
        partitions=(
            {1, 2},
        ),
    )

    with pytest.raises(FrozenInstanceError):
        partition_set.partitions = ()


def test_partition_set_converts_partitions_to_frozenset() -> None:
    """
    Mutable partition inputs should not be stored directly.
    """
    first_partition = {1, 2}
    second_partition = {3, 4}

    partition_set = PartitionSet(
        partitions=[
            first_partition,
            second_partition,
        ],
    )

    first_partition.add(5)

    assert partition_set.partitions == (
        frozenset({1, 2}),
        frozenset({3, 4}),
    )


def test_partition_set_rejects_overlapping_partitions() -> None:
    """
    A node cannot belong to multiple partitions.
    """
    with pytest.raises(InvalidPartitionSetError):
        PartitionSet(
            partitions=(
                {1, 2},
                {2, 3},
            ),
        )


def test_partition_set_iteration() -> None:
    """
    Iteration should return partitions.
    """
    partition_set = PartitionSet(
        partitions=(
            {1, 2},
            {3, 4},
        ),
    )

    assert list(partition_set) == [
        frozenset({1, 2}),
        frozenset({3, 4}),
    ]


def test_empty_partition_set() -> None:
    """
    Empty PartitionSet should be supported.
    """
    partition_set = PartitionSet(
        partitions=(),
    )

    assert len(partition_set) == 0
    assert partition_set.partition_count == 0
    assert partition_set.is_empty is True