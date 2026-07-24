import pytest

from graphora.core.models import (
    Graph,
    Edge,
)

from graphora.detectors.partition import (
    Spectral,
    Walktrap,
    Infomap,
    Louvain,
    Leiden,
)


DETECTORS_WITH_PARAMETERS = [
    Spectral,
    Walktrap,
    Infomap,
    Louvain,
    Leiden,
]


def build_graph():
    return Graph(
        nodes=(
            "A",
            "B",
            "C",
            "D",
        ),
        edges=(
            Edge(
                source="A",
                target="B",
                weight=0.9,
            ),
            Edge(
                source="B",
                target="C",
                weight=0.1,
            ),
            Edge(
                source="C",
                target="D",
                weight=0.85,
            ),
        ),
    )


@pytest.mark.parametrize(
    "detector_cls",
    DETECTORS_WITH_PARAMETERS,
)
def test_detector_execute_returns_stage_result(
    detector_cls,
):
    """
    Every detector should work through Stage interface.
    """

    graph = build_graph()

    detector = detector_cls()

    result = detector.execute(
        graph,
    )

    assert result.output is not None


def test_invalid_spectral_clusters():
    """
    Spectral should reject invalid cluster count.
    """

    with pytest.raises(
        ValueError,
    ):
        Spectral(
            n_clusters=0,
        )


def test_walktrap_invalid_steps():
    """
    Walktrap should reject invalid walk length.
    """

    with pytest.raises(
        ValueError,
    ):
        Walktrap(
            walk_steps=0,
        )


def test_infomap_deterministic_seed():
    """
    Infomap should accept deterministic seed.
    """

    detector = Infomap(
        seed=42,
    )

    assert detector.seed == 42


def test_leiden_invalid_resolution():
    """
    Leiden should reject invalid resolution.
    """

    with pytest.raises(
        ValueError,
    ):
        Leiden(
            resolution=0,
        )


def test_louvain_invalid_resolution():
    """
    Louvain should reject invalid resolution.
    """

    with pytest.raises(
        ValueError,
    ):
        Louvain(
            resolution=0,
        )