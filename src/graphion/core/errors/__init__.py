"""
Graphion core error definitions.
"""

from .models import (
    InvalidEdgeError,
    InvalidFeatureSetError,
    InvalidGraphError,
    InvalidPartitionSetError,
    InvalidRelationError,
    InvalidRelationSetError,
    ModelError,
)

__all__ = [
    "ModelError",
    "InvalidFeatureSetError",
    "InvalidRelationError",
    "InvalidRelationSetError",
    "InvalidEdgeError",
    "InvalidGraphError",
    "InvalidPartitionSetError",
]