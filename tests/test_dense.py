"""Dense verification must preserve coordinates, budgets and memory trust boundaries."""

from types import SimpleNamespace
import numpy as np
import pytest
import torch
from uav_refmatch.features.spatial import SpatialFeatures
from uav_refmatch.matching.dense import mutual_matches, DenseVerificationBackend
from uav_refmatch.config import PatchVerification, Geometry


def test_mutual_matches_preserve_original_crop_coordinates():
    a = SpatialFeatures(np.eye(4, dtype=np.float32), (2, 2), (100, 200))
    b = SpatialFeatures(np.eye(4, dtype=np.float32)[[2, 0, 3, 1]], (2, 2), (40, 80))
    m = mutual_matches(a, b)
    assert np.allclose(m.reference_points, [[50, 25], [150, 25], [50, 75], [150, 75]])
    assert np.allclose(m.candidate_points, [[60, 10], [60, 30], [20, 10], [20, 30]])
    assert np.allclose(m.confidence, 1)


def test_spatial_features_reject_bad_grid_and_nonfinite_values():
    with pytest.raises(ValueError):
        SpatialFeatures(np.eye(4), (3, 2), (10, 10))
    with pytest.raises(ValueError):
        SpatialFeatures(np.full((4, 2), np.nan), (2, 2), (10, 10))


def test_dense_cache_is_bounded_and_content_addressed():
    class Encoder:
        calls = 0

        def encode_spatial(self, image, size):
            self.calls += 1
            return SpatialFeatures(np.eye(4, dtype=np.float32), (2, 2), image.shape[:2])

    e = Encoder()
    b = DenseVerificationBackend(e, PatchVerification(cache_entries=2), Geometry())
    a = np.zeros((4, 4, 3), np.uint8)
    b.features(a)
    b.features(a.copy())
    assert e.calls == 1
    b.features(a + 1)
    b.features(a + 2)
    assert len(b.cache) == 2
    b.features(a)
    assert e.calls == 4


def test_spatial_encoder_excludes_cls_and_register_tokens():
    from uav_refmatch.features.dinov3 import DinoV3Encoder

    e = DinoV3Encoder.__new__(DinoV3Encoder)
    e.device = "cpu"
    e.dtype = torch.float32
    e.processor = lambda **kwargs: {"pixel_values": torch.zeros((1, 3, 32, 32))}

    class Model:
        config = SimpleNamespace(patch_size=16, num_register_tokens=4)

        def __call__(self, **kwargs):
            hidden = torch.cat([torch.full((5, 4), 99.0), torch.eye(4)]).unsqueeze(0)
            return SimpleNamespace(last_hidden_state=hidden)

    e.model = Model()
    f = e.encode_spatial(np.zeros((100, 200, 3), np.uint8), 32)
    assert f.grid == (2, 2) and f.image_shape == (100, 200)
    assert np.allclose(f.descriptors, np.eye(4))


def test_dense_evidence_does_not_enter_long_term_memory():
    from uav_refmatch.memory.policies import MemoryAdmissionPolicy
    from uav_refmatch.config import Memory
    from uav_refmatch.core.types import GeometryEvidence

    g = GeometryEvidence(
        100, 80, 0.8, 1.0, 0.5, True, True, "geometry_passed", {"backend": "dino_patch"}
    )
    p = SimpleNamespace(
        status="FOUND", dino_cosine=0.95, anchor_geometry=None, matched_memory_id="R0", geometry=g
    )
    reason, e = MemoryAdmissionPolicy(Memory()).evaluate(p, SimpleNamespace(), 0.95)
    assert reason == "REJECTED_GEOMETRY"
    assert e["failed_gate"] == "dense_evidence_requires_fine_confirmation"


@pytest.mark.parametrize("budget, expected", [(1, "NOT_FOUND"), (4, "FOUND")])
def test_patch_fallback_respects_total_budget_and_never_admits_coarse_memory(
    tmp_path, budget, expected
):
    import json
    from uav_refmatch.config import Config
    from uav_refmatch.core.types import Proposal, FrameRecord, GeometryEvidence, LocalMatches
    from uav_refmatch.outputs.recorder import Recorder
    from uav_refmatch.pipeline import Pipeline

    class Proposals:
        def propose(self, image):
            return [Proposal(0, (10, 10, 90, 90), "test")]

    class Encoder:
        def encode(self, images):
            return np.tile([1.0, 0.0], (len(images), 1))

        def encode_spatial(self, image, size):
            return SpatialFeatures(np.eye(16, dtype=np.float32), (4, 4), image.shape[:2])

    class Matcher:
        def match(self, *args):
            return LocalMatches(np.empty((0, 2)), np.empty((0, 2)), np.empty(0))

    class Verifier:
        def verify(self, *args):
            return GeometryEvidence(reason="insufficient_matches")

    c = Config(
        warmup=False,
        tracking={"enabled": True, "mode": "verify"},
        patch_verification={"enabled": True},
        retrieval={"max_geometry_pairs": budget},
        geometry={
            "reserved_rotation_calls": 0 if budget == 1 else 2,
            "candidate_rotation_fallbacks": [-45, 45],
        },
    )
    image = np.random.default_rng(1).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    rec = Recorder(tmp_path, c)
    engine = Pipeline(c, Proposals(), Encoder(), Matcher(), Verifier())
    engine.run([FrameRecord(i, i, i * 0.13, 100, 100, image) for i in range(2)], image, "test", rec)
    rec.finish("completed")
    rows = list(map(json.loads, (rec.path / "predictions.jsonl").read_text().splitlines()))
    assert all(x["status"] == expected for x in rows)
    assert all(len(x["verified_pairs"]) <= budget for x in rows)
    events = list(map(json.loads, (rec.path / "memory_events.jsonl").read_text().splitlines()))
    assert not any(x["event_type"] == "ACCEPTED" for x in events)
    if expected == "FOUND":
        assert all(x["verified_pairs"][-1]["method"] == "dino_patch" for x in rows)
        assert any(
            x.get("evidence", {}).get("failed_gate") == "dense_evidence_requires_fine_confirmation"
            for x in events
        )


def test_coarse_memory_matching_preserves_identity_floor():
    from uav_refmatch.decision.identity import dense_identity_gate
    from uav_refmatch.config import Tracking
    from uav_refmatch.core.types import Pair

    config = Tracking(identity_policy="verified_bank")
    reference = SimpleNamespace(
        memory_id="mem_0001", origin="dynamic", acceptance_evidence={"geometry_reference": "R0"}
    )
    assert not dense_identity_gate(Pair(0, "mem_0001", 0.37), reference, config)[0]
    assert dense_identity_gate(Pair(0, "mem_0001", 0.8), reference, config)[0]
    assert not dense_identity_gate(Pair(0, "mem_0001", 0.8), None, config)[0]
    anchor = SimpleNamespace(memory_id="R0", origin="anchor")
    assert dense_identity_gate(Pair(0, "R0", 0.4), anchor, config)[0]
    assert not dense_identity_gate(Pair(0, "mem_0001", 0.8), anchor, config)[0]
