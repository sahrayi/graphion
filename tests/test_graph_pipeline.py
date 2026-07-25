import numpy as np

from graphion.core.models import FeatureSet

from graphion.builders.relation.cosine_similarity import (
    CosineSimilarity,
)

from graphion.builders.graph.weighted_knn import (
    WeightedKNN,
)

from graphion.detectors.partition.leiden import (
    Leiden,
)


def test_random_feature_graph_pipeline():
    """
    Full pipeline test:

    Random feature matrix
        ->
    FeatureSet
        ->
    RelationSet
        ->
    Weighted KNN Graph
        ->
    Leiden Partition
    """

    # --------------------------------------------------
    # 1. Create random feature matrix
    # --------------------------------------------------

    rng = np.random.default_rng(
        seed=42,
    )

    matrix = rng.random(
        (
            100,
            768,
        )
    )

    assert matrix.shape == (
        100,
        768,
    )


    # --------------------------------------------------
    # 2. Create FeatureSet
    # --------------------------------------------------

    feature_set = FeatureSet.from_numpy(
        ids=[
            f"node_{index}"
            for index in range(
                100,
            )
        ],
        matrix=matrix,
    )

    assert len(feature_set) == 100
    assert feature_set.dimension == 768


    # --------------------------------------------------
    # 3. Build relations
    # --------------------------------------------------

    relation_builder = CosineSimilarity()

    relations = relation_builder.build(
        feature_set,
    )

    assert len(relations) > 0


    # --------------------------------------------------
    # 4. Build graph
    # --------------------------------------------------

    graph_builder = WeightedKNN(
        relation_builder=relation_builder,
        k=10,
        symmetric=True,
    )

    graph = graph_builder.build(
        relations,
        nodes=feature_set.ids,
    )

    assert graph is not None

    assert len(graph.nodes) == 100

    assert len(graph.edges) > 0


    # --------------------------------------------------
    # 5. Detect partitions
    # --------------------------------------------------

    detector = Leiden()

    partitions = detector.detect(
        graph,
    )

    assert partitions is not None

    assert len(partitions) > 0