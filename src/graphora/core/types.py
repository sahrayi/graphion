"""
Common generic type variables used throughout Graphora.

This module defines the shared type variables that are used by the
core data models, interfaces, pipeline, and algorithms.
"""

from __future__ import annotations

from collections.abc import Hashable

from typing import TypeVar

#: Identifier type for entities, nodes, and partitions.
TId = TypeVar(
    "TId",
    bound=Hashable,
)

#: Feature representation type.
TFeature = TypeVar("TFeature")

#: Generic input type.
TInput = TypeVar("TInput")

#: Generic output type.
TOutput = TypeVar("TOutput")

TPrepared = TypeVar(
    "TPrepared",
)