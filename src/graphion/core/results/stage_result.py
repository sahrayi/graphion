"""
Stage execution result model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic

from graphion.core.types import TOutput


@dataclass(frozen=True, slots=True)
class StageResult(Generic[TOutput]):
    """
    Immutable result produced by a pipeline stage.

    A stage may produce:
    - an output object
    - metadata
    - evaluation information (future use)
    """

    output: TOutput

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )