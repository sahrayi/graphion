"""
Base interface for executable pipeline stages.
"""

from __future__ import annotations

from typing import Any, Protocol

from graphora.core.results import StageResult
from graphora.core.types import TInput, TOutput

class Stage(
    Protocol[TInput, TOutput],
):
    def execute(
        self,
        input_data: TInput,
    ) -> StageResult[TOutput]:
        """
        Execute stage transformation.
        """
        ...