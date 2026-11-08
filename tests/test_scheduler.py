from dataclasses import replace
import pytest
from uav_refmatch.config import Geometry, Retrieval, Memory
from uav_refmatch.core.types import Pair, VerifiedPair, GeometryEvidence
from uav_refmatch.matching.scheduler import initial_pairs, retry_attempts
from uav_refmatch.memory.dynamic import DynamicReferenceMemory


def test_existing_anchor_evidence_admits_memory_winner(anchor, found, entry):
    memory = DynamicReferenceMemory(
        anchor, Memory(anchor_only_admission=True, confirmation_frames=1)
    )
    prediction = replace(found, matched_memory_id="mem_0001", anchor_geometry=found.geometry)
    memory.update(prediction, entry, 0.85)
    assert memory.events[-1].event_type == "ACCEPTED"
    assert memory.events[-1].evidence["geometry_reference"] == "R0"
    assert prediction.matched_memory_id == "mem_0001"
    assert len(memory.active(0)) == 1
    assert len(memory.active(1)) == 2


def test_weak_anchor_cannot_borrow_memory_geometry(anchor, found, entry):
    memory = DynamicReferenceMemory(
        anchor, Memory(anchor_only_admission=True, confirmation_frames=1)
    )
    prediction = replace(
        found, matched_memory_id="mem_0001", anchor_geometry=replace(found.geometry, num_inliers=8)
    )
    memory.update(prediction, entry, 0.85)
    assert memory.events[-1].evidence["failed_gate"] == "geometry_support"
    assert len(memory.entries) == 1


@pytest.mark.parametrize(
    "policy,first", [("legacy", "R0"), ("cosine", "mem_1"), ("evidence", "R0")]
)
def test_retry_priority_is_explicit(policy, first):
    verified = [
        VerifiedPair(Pair(0, "mem_1", 0.9), GeometryEvidence(num_inliers=6, transform_valid=True)),
        VerifiedPair(Pair(0, "R0", 0.6), GeometryEvidence(num_inliers=8, transform_valid=True)),
    ]
    config = Geometry(
        rotation_retry_policy=policy,
        rotation_retry_anchor_first=True,
        candidate_rotation_fallbacks=(-45, 45),
    )
    attempts = retry_attempts(verified, config)
    assert [p.memory_id for p, _ in attempts[:2]] == [first, first]
    assert [a for _, a in attempts[:2]] == [-45, 45]


@pytest.mark.parametrize("capacity", [1, 4, 8, 10])
def test_larger_bank_does_not_consume_reserved_retry_budget(capacity):
    pairs = [Pair(0, f"mem_{i}", 0.9 - i * 0.01) for i in range(capacity)] + [Pair(0, "R0", 0.5)]
    retrieval = Retrieval(max_geometry_pairs=4, reserve_anchor_pair=True)
    geometry = Geometry(reserved_rotation_calls=2, candidate_rotation_fallbacks=(-45, 45))
    selected = initial_pairs(
        pairs[:3] + [pairs[-1]] if len(pairs) > 4 else pairs, retrieval, geometry
    )
    assert len(selected) <= 2
    assert any(p.memory_id == "R0" for p in selected)


def test_anchor_support_is_not_borrowed_from_another_candidate(found):
    from uav_refmatch.memory.policies import select_anchor_geometry

    pairs = [
        VerifiedPair(Pair(0, "R0", 0.9), found.geometry),
        VerifiedPair(Pair(1, "mem_1", 0.95), found.geometry),
    ]
    assert select_anchor_geometry(pairs, 1, Memory()) is None
    assert select_anchor_geometry(pairs, 0, Memory()) == found.geometry


def test_stronger_inlier_count_cannot_hide_qualifying_anchor_ratio(found):
    from uav_refmatch.memory.policies import select_anchor_geometry

    weak = replace(found.geometry, num_inliers=100, inlier_ratio=0.1)
    pairs = [
        VerifiedPair(Pair(0, "R0", 0.9), weak),
        VerifiedPair(Pair(0, "R0", 0.9), found.geometry, 45),
    ]
    assert select_anchor_geometry(pairs, 0, Memory()) == found.geometry
