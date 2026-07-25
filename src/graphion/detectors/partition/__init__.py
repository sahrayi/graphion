"""
Partition detectors.

Community detection algorithms
for graphion partition stage.
"""

from .base_partition_detector import BasePartitionDetector
from .agglomerative import Agglomerative
from .connected_components import ConnectedComponents
from .fast_greedy import FastGreedy
from .girvan_newman import GirvanNewman
from .identity import IdentityPartitionDetector
from .infomap import Infomap
from .label_propagation import LabelPropagation
from .leiden import Leiden
from .louvain import Louvain
from .spectral import Spectral
from .walktrap import Walktrap

__all__ = [
    "BasePartitionDetector",
    "IdentityPartitionDetector",
    "ConnectedComponents",
    "LabelPropagation",
    "Louvain",
    "Leiden",
    "FastGreedy",
    "Walktrap",
    "Infomap",
    "GirvanNewman",
    "Spectral",
    "Agglomerative",
]