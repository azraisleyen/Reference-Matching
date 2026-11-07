from dataclasses import replace
import numpy as np
import pytest
from uav_refmatch.features.normalization import normalize
from uav_refmatch.matching.retrieval import retrieve
from uav_refmatch.config import Retrieval


@pytest.mark.parametrize(
    "values", [np.zeros((1, 3)), np.array([[np.nan, 1]]), np.ones(3), np.empty((0, 3))]
)
def test_invalid_embeddings(values):
    with pytest.raises(ValueError):
        normalize(values)


def test_normalization():
    np.testing.assert_allclose(normalize([[3, 4]]), [[0.6, 0.8]])


def test_order_budget_and_static(anchor):
    candidates = np.array([[1.0, 0, 0], [1.0, 0, 0], [0.8, 0.6, 0]])
    refs = [anchor, replace(anchor, memory_id="mem_0001")]
    pairs = retrieve(candidates, refs, Retrieval()).pairs
    assert [(p.candidate_index, p.memory_id) for p in pairs] == [
        (0, "R0"),
        (0, "mem_0001"),
        (1, "R0"),
    ]
    static = retrieve(candidates, [anchor], Retrieval()).pairs
    assert len(static) == 3 and all(p.memory_id == "R0" for p in static)


def test_below_threshold_scores_remain_observable(anchor):
    result = retrieve(np.array([[0.6, 0.8, 0]], dtype=np.float32), [anchor], Retrieval())
    assert not result.pairs
    assert result.candidate_best_matches[0].cosine == pytest.approx(0.6)


def test_anchor_pair_can_be_reserved_within_budget():
    from uav_refmatch.core.types import ReferenceEntry
    from uav_refmatch.config import Retrieval
    from uav_refmatch.matching.retrieval import retrieve

    refs = [
        ReferenceEntry(
            name,
            "anchor" if name == "R0" else "dynamic",
            None,
            None,
            None,
            np.zeros((8, 8, 3), np.uint8),
            np.array(vector),
        )
        for name, vector in [("R0", [0.6, 0.8]), ("mem_1", [1.0, 0.0]), ("mem_2", [0.99, 0.1])]
    ]
    result = retrieve(
        np.array([[1.0, 0.0]]),
        refs,
        Retrieval(max_geometry_pairs=2, detect_min_cosine=0.3, reserve_anchor_pair=True),
    )
    assert len(result.pairs) == 2
    assert result.pairs[-1].memory_id == "R0"
    # Reservation must not bypass the geometry similarity threshold.
    result = retrieve(
        np.array([[1.0, 0.0]]),
        refs,
        Retrieval(max_geometry_pairs=2, detect_min_cosine=0.9, reserve_anchor_pair=True),
    )
    assert all(p.memory_id != "R0" for p in result.pairs)
