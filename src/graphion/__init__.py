"""
Graphora.

A graph-based framework for building semantic graphs,
community detection, refinement, dimensionality reduction,
and evaluation.
"""

__version__ = "0.0.1"


# ============================================================
# Core Models
# ============================================================

from .core.models import (
    Edge,
    FeatureSet,
    Graph,
    PartitionSet,
    Relation,
    RelationSet,
)


# ============================================================
# Core Interfaces
# ============================================================

from .core.interfaces import (
    GraphBuilder,
    GraphRefiner,
    PartitionDetector,
    PartitionRefiner,
    Reducer,
    RelationBuilder,
    Stage,
)


# ============================================================
# Core Results
# ============================================================

from .core.results import (
    StageResult,
)


# ============================================================
# Graph Relation Builders
# ============================================================

from .builders.relation import (
    AngularSimilarity,
    BaseNumericRelationBuilder,
    BaseRelationBuilder,
    BrayCurtisDistance,
    CanberraDistance,
    ChebyshevDistance,
    CosineSimilarity,
    DiceSimilarity,
    DotProduct,
    EuclideanDistance,
    HammingDistance,
    JaccardSimilarity,
    ManhattanDistance,
    MinkowskiDistance,
    OverlapCoefficient,
    PearsonCorrelation,
    RBFSimilarity,
    TanimotoSimilarity,
    WeightedJaccardSimilarity,
)


# ============================================================
# Graph Builders
# ============================================================

from .builders.graph import (
    AdaptiveKNN,
    BaseGraphBuilder,
    KNN,
    KNNThreshold,
    MST,
    MutualKNN,
    Radius,
    RNG,
    SNN,
    SymmetricKNN,
    Threshold,
    WeightedKNN,
)


# ============================================================
# Partition Detectors
# ============================================================

from .detectors.partition import (
    Agglomerative,
    BasePartitionDetector,
    ConnectedComponents,
    FastGreedy,
    GirvanNewman,
    IdentityPartitionDetector,
    Infomap,
    LabelPropagation,
    Leiden,
    Louvain,
    Spectral,
    Walktrap,
)


# ============================================================
# Reducers
# ============================================================

from .reducers import (
    Autoencoder,
    BaseReducer,
    IdentityReducer,
    Isomap,
    PCA,
    RandomProjection,
    TruncatedSVD,
    TSNE,
    UMAP,
)


# ============================================================
# Graph Refiners
# ============================================================

from .refiners.graph import (
    BaseGraphRefiner,
    IdentityGraphRefiner,
)


# ============================================================
# Partition Refiners
# ============================================================

from .refiners.partition import (
    BasePartitionRefiner,
    IdentityPartitionRefiner,
)


# ============================================================
# Evaluation
# ============================================================

from .evaluation import (
    FeaturePartitionEvaluation,
    FeatureSetEvaluation,
    GraphEvaluation,
    PartitionSetEvaluation,
)


# ============================================================
# Public API
# ============================================================

__all__ = [

    # version
    "__version__",

    # models
    "Edge",
    "FeatureSet",
    "Graph",
    "PartitionSet",
    "Relation",
    "RelationSet",

    # interfaces
    "GraphBuilder",
    "GraphRefiner",
    "PartitionDetector",
    "PartitionRefiner",
    "Reducer",
    "RelationBuilder",
    "Stage",

    # results
    "StageResult",

    # relation builders
    "AngularSimilarity",
    "BaseNumericRelationBuilder",
    "BaseRelationBuilder",
    "BrayCurtisDistance",
    "CanberraDistance",
    "ChebyshevDistance",
    "CosineSimilarity",
    "DiceSimilarity",
    "DotProduct",
    "EuclideanDistance",
    "HammingDistance",
    "JaccardSimilarity",
    "ManhattanDistance",
    "MinkowskiDistance",
    "OverlapCoefficient",
    "PearsonCorrelation",
    "RBFSimilarity",
    "TanimotoSimilarity",
    "WeightedJaccardSimilarity",

    # graph builders
    "AdaptiveKNN",
    "BaseGraphBuilder",
    "KNN",
    "KNNThreshold",
    "MST",
    "MutualKNN",
    "Radius",
    "RNG",
    "SNN",
    "SymmetricKNN",
    "Threshold",
    "WeightedKNN",

    # detectors
    "Agglomerative",
    "BasePartitionDetector",
    "ConnectedComponents",
    "FastGreedy",
    "GirvanNewman",
    "IdentityPartitionDetector",
    "Infomap",
    "LabelPropagation",
    "Leiden",
    "Louvain",
    "Spectral",
    "Walktrap",

    # reducers
    "Autoencoder",
    "IdentityReducer",
    "BaseReducer",
    "Isomap",
    "PCA",
    "RandomProjection",
    "TruncatedSVD",
    "TSNE",
    "UMAP",

    # refiners
    "BaseGraphRefiner",
    "IdentityGraphRefiner",
    "BasePartitionRefiner",
    "IdentityPartitionRefiner",

    # evaluation
    "FeaturePartitionEvaluation",
    "FeatureSetEvaluation",
    "GraphEvaluation",
    "PartitionSetEvaluation",
]