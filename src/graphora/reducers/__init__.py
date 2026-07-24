"""
Feature dimensionality reduction algorithms.

Public reducer API.
"""

from .identity import IdentityReducer
from .pca import PCA
from .random_projection import RandomProjection
from .truncated_svd import TruncatedSVD
from .umap import UMAP
from .tsne import TSNE
from .isomap import Isomap
from .autoencoder import Autoencoder


__all__ = [
    "IdentityReducer",
    "PCA",
    "RandomProjection",
    "TruncatedSVD",
    "UMAP",
    "TSNE",
    "Isomap",
    "Autoencoder",
]