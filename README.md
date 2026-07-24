# Graphora

**Graphora** is a modular and extensible Python framework for building graph-based data processing pipelines.

Instead of providing a single clustering or graph algorithm, Graphora offers reusable building blocks for constructing complete graph analytics workflows, from feature relations to graph construction, partition detection, refinement, reduction, and evaluation.

The framework is designed around interchangeable pipeline stages, allowing researchers and developers to combine different algorithms with a consistent interface.

> **Project status:** Early development (pre-release)

---

# Features

Graphora currently provides the following components:

### Relation Builders

Build pairwise relations between feature vectors using multiple similarity and distance metrics.

Implemented metrics include:

- Cosine Similarity
- Dot Product
- Pearson Correlation
- Angular Similarity
- Euclidean Distance
- Manhattan Distance
- Chebyshev Distance
- Minkowski Distance
- Canberra Distance
- Bray-Curtis Distance
- Hamming Distance
- Jaccard Similarity
- Weighted Jaccard
- Dice Similarity
- Overlap Coefficient
- Tanimoto Similarity
- RBF Similarity

---

### Graph Builders

Construct graphs from pairwise relations using multiple strategies.

Implemented builders include:

- Threshold Graph
- Radius Graph
- k-Nearest Neighbors (kNN)
- Mutual kNN
- Symmetric kNN
- Weighted kNN
- Adaptive kNN
- kNN + Threshold
- Shared Nearest Neighbor (SNN)
- Relative Neighborhood Graph (RNG)
- Minimum Spanning Tree (MST)

---

### Graph Refiners

Graph transformation stages.

Currently available:

- Identity Refiner

---

### Partition Detectors

Detect graph communities using interchangeable algorithms.

Implemented algorithms include:

- Connected Components
- Leiden
- Louvain
- Label Propagation
- Walktrap
- Fast Greedy
- Girvan-Newman
- Spectral Clustering
- Agglomerative Clustering
- Infomap
- Identity

---

### Partition Refiners

Post-process detected communities.

Currently available:

- Identity Refiner

---

### Reducers

Dimensionality reduction modules.

Implemented reducers:

- PCA
- Truncated SVD
- Random Projection
- UMAP
- t-SNE
- Isomap
- Autoencoder
- Identity

---

### Evaluation

Built-in evaluation utilities for:

- Feature sets
- Graphs
- Partition sets
- Feature partitions

---

# Design Principles

Graphora is designed around several core principles:

- Modular architecture
- Consistent interfaces
- Pluggable algorithms
- Strong typing
- Clear separation of responsibilities
- Pipeline-oriented execution
- Easy extensibility

Every processing stage follows a common execution interface, making it straightforward to replace one algorithm with another.

---

# Project Structure

```
graphora/
│
├── builders/
│   ├── relation/
│   └── graph/
│
├── reducers/
│
├── refiners/
│   ├── graph/
│   └── partition/
│
├── detectors/
│   └── partition/
│
├── evaluation/
│
└── core/
    ├── interfaces/
    ├── models/
    ├── results/
    ├── errors/
    └── types.py
```

---

# Installation

Graphora is currently under active development.

Clone the repository:

```bash
git clone https://github.com/sahrayi/graphora.git
```

Install in editable mode:

```bash
pip install -e .
```

---

# Roadmap

Planned features include:

- Additional graph construction algorithms
- More community detection methods
- Hierarchical partition refinement
- Graph embedding modules
- Visualization utilities
- Benchmark suite
- Documentation website
- Comprehensive test coverage

---

# Related Projects

Graphora serves as the foundation for higher-level graph analytics libraries.

For example:

- GraphTopic (graph-based topic discovery)

---

# License

MIT License