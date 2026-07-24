"""
PartitionSet data model.
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)

from dataclasses import dataclass
from typing import Generic

from graphora.core.errors import (
    InvalidPartitionSetError,
)
from graphora.core.types import (
    TId,
)


@dataclass(
    frozen=True,
    slots=True,
)
class PartitionSet(Generic[TId]):
    """
    Immutable collection of graph partitions.

    Each partition is represented as a frozenset
    of node identifiers.
    """

    partitions: tuple[frozenset[TId], ...]

    def __post_init__(self) -> None:

        object.__setattr__(
            self,
            "partitions",
            tuple(
                frozenset(partition)
                for partition in self.partitions
            ),
        )

        seen: set[TId] = set()

        for partition in self.partitions:

            overlap = (
                seen.intersection(
                    partition,
                )
            )

            if overlap:

                raise InvalidPartitionSetError(
                    "A node cannot belong to multiple partitions."
                )

            seen.update(
                partition,
            )

    # --------------------------------------------------
    # Constructors
    # --------------------------------------------------

    @classmethod
    def from_sets(
        cls,
        partitions: Iterable[
            Iterable[TId]
        ],
    ) -> "PartitionSet[TId]":

        return cls(
            partitions=tuple(
                frozenset(partition)
                for partition in partitions
            ),
        )

    @classmethod
    def from_dict(
        cls,
        labels: Mapping[
            TId,
            int,
        ],
    ) -> "PartitionSet[TId]":

        groups: dict[
            int,
            set[TId],
        ] = {}

        for node, label in labels.items():

            groups.setdefault(
                label,
                set(),
            ).add(node)

        ordered = tuple(
            frozenset(groups[label])
            for label in sorted(groups)
        )

        return cls(
            partitions=ordered,
        )

    @classmethod
    def from_labels(
        cls,
        ids: Sequence[TId],
        labels: Sequence[int],
    ) -> "PartitionSet[TId]":

        if len(ids) != len(labels):

            raise InvalidPartitionSetError(
                "ids and labels must have the same length."
            )

        return cls.from_dict(
            dict(
                zip(
                    ids,
                    labels,
                )
            )
        )

    # --------------------------------------------------
    # Converters
    # --------------------------------------------------

    def to_sets(
        self,
    ) -> list[set[TId]]:
        """
        Convert partitions into mutable sets.
        """

        return [
            set(partition)
            for partition
            in self.partitions
        ]

    def to_dict(
        self,
    ) -> dict[TId, int]:
        """
        Convert to node -> partition_id mapping.
        """

        result: dict[
            TId,
            int,
        ] = {}

        for index, partition in enumerate(
            self.partitions,
        ):

            for node in partition:

                result[node] = index

        return result

    def to_labels(
            self,
            ids: Sequence[TId],
    ) -> tuple[
        tuple[TId, ...],
        tuple[int, ...],
    ]:
        """
        Convert partition assignment into labels
        following provided id order.

        Raises
        ------
        InvalidPartitionSetError
            If provided ids and partition nodes
            do not represent the same set.
        """

        mapping = self.to_dict()

        provided_ids = set(ids)

        partition_ids = set(mapping)

        if provided_ids != partition_ids:
            missing = provided_ids - partition_ids
            extra = partition_ids - provided_ids

            raise InvalidPartitionSetError(
                "Partition nodes and provided ids "
                "must match exactly. "
                f"Missing: {missing}, Extra: {extra}"
            )

        labels = tuple(
            mapping[node]
            for node in ids
        )

        return (
            tuple(ids),
            labels,
        )

    # --------------------------------------------------
    # Protocols
    # --------------------------------------------------

    def __len__(
        self,
    ) -> int:

        return len(
            self.partitions,
        )

    def __iter__(
        self,
    ) -> Iterator[
        frozenset[TId]
    ]:

        return iter(
            self.partitions,
        )

    @property
    def partition_count(
        self,
    ) -> int:

        return len(
            self.partitions,
        )

    @property
    def is_empty(
        self,
    ) -> bool:

        return (
            self.partition_count == 0
        )

    @property
    def partition_sizes(self) -> tuple[int, ...]:
        return tuple(len(partition) for partition in self.partitions)

    @property
    def node_count(self) -> int:
        return sum(self.partition_sizes)