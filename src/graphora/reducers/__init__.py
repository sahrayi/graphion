"""
Feature dimensionality reduction algorithms.

Public reducer API.
"""

from .base_reducer import BaseReducer
from .identity import IdentityReducer
from .pca import PCA
from .random_projection import RandomProjection
from .truncated_svd import TruncatedSVD
from .umap import UMAP
from .tsne import TSNE
from .isomap import Isomap
from .autoencoder import Autoencoder


__all__ = [
    "BaseReducer",
    "IdentityReducer",
    "PCA",
    "RandomProjection",
    "TruncatedSVD",
    "UMAP",
    "TSNE",
    "Isomap",
    "Autoencoder",
]