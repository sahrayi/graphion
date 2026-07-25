"""
Tests for Graphion core interfaces.
"""

from graphion.core.interfaces import (
    GraphBuilder,
    GraphRefiner,
    PartitionDetector,
    PartitionRefiner,
    Reducer,
    RelationBuilder,
)
from graphion.core.models import (
    FeatureSet,
    Graph,
    PartitionSet,
    RelationSet,
)
from graphion.core.results import StageResult

class DummyReducer:
    """
    Dummy Reducer implementation.
    """

    def reduce(
        self,
        features: FeatureSet,
    ) -> FeatureSet:
        return features

    def execute(
        self,
        input_data: FeatureSet,
    ) -> StageResult[FeatureSet]:
        return StageResult(
            output=self.reduce(input_data),
        )


class DummyRelationBuilder:
    """
    Dummy RelationBuilder implementation.
    """

    def build(
        self,
        features: FeatureSet,
    ) -> RelationSet:
        return RelationSet(
            relations=(),
        )

    def execute(
        self,
        input_data: FeatureSet,
    ) -> StageResult[RelationSet]:
        return StageResult(
            output=self.build(input_data),
        )


class DummyGraphBuilder:
    """
    Dummy GraphBuilder implementation.
    """

    def build(
        self,
        relations: RelationSet,
    ) -> Graph:
        return Graph(
            nodes=(),
            edges=(),
        )

    def execute(
        self,
        input_data: RelationSet,
    ) -> StageResult[Graph]:
        return StageResult(
            output=self.build(input_data),
        )


class DummyGraphRefiner:
    """
    Dummy GraphRefiner implementation.
    """

    def refine(
        self,
        graph: Graph,
    ) -> Graph:
        return graph

    def execute(
        self,
        input_data: Graph,
    ) -> StageResult[Graph]:
        return StageResult(
            output=self.refine(input_data),
        )


class DummyPartitionDetector:
    """
    Dummy PartitionDetector implementation.
    """

    def detect(
        self,
        graph: Graph,
    ) -> PartitionSet:
        return PartitionSet(
            partitions=(),
        )

    def execute(
        self,
        input_data: Graph,
    ) -> StageResult[PartitionSet]:
        return StageResult(
            output=self.detect(input_data),
        )


class DummyPartitionRefiner:
    """
    Dummy PartitionRefiner implementation.
    """

    def refine(
        self,
        partitions: PartitionSet,
    ) -> PartitionSet:
        return partitions

    def execute(
        self,
        input_data: PartitionSet,
    ) -> StageResult[PartitionSet]:
        return StageResult(
            output=self.refine(input_data),
        )


def test_reducer_contract() -> None:
    """
    Reducer should expose reduce and execute.
    """
    reducer = DummyReducer()

    features = FeatureSet(
        ids=(),
        features=(),
    )

    assert reducer.reduce(features) == features

    result = reducer.execute(features)

    assert isinstance(result, StageResult)
    assert result.output == features


def test_relation_builder_contract() -> None:
    """
    RelationBuilder should expose build and execute.
    """
    builder = DummyRelationBuilder()

    result = builder.execute(
        FeatureSet(
            ids=(),
            features=(),
        )
    )

    assert isinstance(result, StageResult)
    assert isinstance(
        result.output,
        RelationSet,
    )


def test_graph_builder_contract() -> None:
    """
    GraphBuilder should expose build and execute.
    """
    builder = DummyGraphBuilder()

    result = builder.execute(
        RelationSet(
            relations=(),
        )
    )

    assert isinstance(result, StageResult)
    assert isinstance(
        result.output,
        Graph,
    )


def test_graph_refiner_contract() -> None:
    """
    GraphRefiner should expose refine and execute.
    """
    refiner = DummyGraphRefiner()

    graph = Graph(
        nodes=(),
        edges=(),
    )

    result = refiner.execute(graph)

    assert isinstance(result, StageResult)
    assert result.output is graph


def test_partition_detector_contract() -> None:
    """
    PartitionDetector should expose detect and execute.
    """
    detector = DummyPartitionDetector()

    result = detector.execute(
        Graph(
            nodes=(),
            edges=(),
        )
    )

    assert isinstance(result, StageResult)
    assert isinstance(
        result.output,
        PartitionSet,
    )


def test_partition_refiner_contract() -> None:
    """
    PartitionRefiner should expose refine and execute.
    """
    refiner = DummyPartitionRefiner()

    partitions = PartitionSet(
        partitions=(),
    )

    result = refiner.execute(partitions)

    assert isinstance(result, StageResult)
    assert result.output is partitions