"""
Partition detectors.

Community detection algorithms
for Graphora partition stage.
"""

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