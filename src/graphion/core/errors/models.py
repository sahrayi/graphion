"""
Errors related to Graphion core data models.
"""

from __future__ import annotations


class ModelError(Exception):
    """
    Base class for all model-related errors.
    """


class InvalidFeatureSetError(ModelError):
    """
    Raised when a FeatureSet violates its invariants.
    """


class InvalidRelationSetError(ModelError):
    """
    Raised when a RelationSet violates its invariants.
    """


class InvalidGraphError(ModelError):
    """
    Raised when a Graph violates its invariants.
    """


class InvalidPartitionSetError(ModelError):
    """
    Raised when a PartitionSet violates its invariants.
    """

class InvalidRelationError(ModelError):
    """Raised when a Relation violates its invariants."""


class InvalidEdgeError(ModelError):
    """Raised when an Edge violates its invariants."""