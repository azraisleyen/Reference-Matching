"""Recovery must not turn propagation or unrelated bank IDs into identity evidence."""

from types import SimpleNamespace
import numpy as np
import pytest
from uav_refmatch.config import Tracking
from uav_refmatch.tracking.controller import TrackController
from uav_refmatch.tracking.flow import TrackEstimate
from uav_refmatch.decision.identity import identity_gate
from uav_refmatch.core.types import Pair, Proposal


class Tracker:
    def initialize(self, *args):
        pass

    def reset(self):
        pass


def test_inconclusive_hold_has_hard_deadline_and_does_not_refresh_identity():
    c = TrackController(Tracking(max_unverified_sec=3), Tracker())
    e = np.array([1.0, 0.0])
    c.confirm(None, (0, 0, 10, 10), e, 0)
    estimate = TrackEstimate((0, 0, 10, 10), True)
    assert not c.can_propagate(estimate, e, 1.1)[0]
    assert c.defer(estimate, e, 1.1)
    assert c.state == "UNCERTAIN"
    assert c.can_propagate(estimate, e, 1.2)[0]
    assert not c.can_propagate(estimate, e, 2.2)[0]
    assert c.defer(estimate, e, 2.2)
    assert c.verified_at == 0
    assert not c.defer(estimate, e, 3.0)
    assert not c.can_propagate(estimate, e, 3.1)[0]


@pytest.mark.parametrize("bad", ["conflict", "appearance", "flow"])
def test_hold_never_ignores_contradiction(bad):
    c = TrackController(Tracking(max_unverified_sec=3), Tracker())
    c.confirm(None, (0, 0, 10, 10), np.array([1.0, 0.0]), 0)
    estimate = TrackEstimate((0, 0, 10, 10), bad != "flow")
    embedding = np.array([0.0, 1.0]) if bad == "appearance" else np.array([1.0, 0.0])
    assert not c.defer(estimate, embedding, 1.1, bad == "conflict")


def test_bank_evidence_requires_actual_admitted_reference():
    c = TrackController(Tracking(identity_policy="verified_bank"), Tracker())
    anchor = SimpleNamespace(embedding=np.array([1.0, 0.0]))
    pair, prop = Pair(0, "mem_0001", 0.9), Proposal(0, (0, 0, 10, 10), "test")
    emb = np.array([0.0, 1.0])

    def check(ref):
        return identity_gate(pair, prop, emb, anchor, c, TrackEstimate(), ref)

    assert not check(None)[0]
    ref = SimpleNamespace(memory_id="mem_0001", origin="dynamic", acceptance_evidence={})
    assert not check(ref)[0]
    ref.acceptance_evidence = {"geometry_reference": "R0"}
    accepted, evidence = check(ref)
    assert accepted and evidence["support"] == "verified_bank"
    ref.memory_id = "other"
    assert not check(ref)[0]


def test_invalid_horizon():
    with pytest.raises(ValueError):
        Tracking(reverify_interval_sec=2, max_unverified_sec=1)


def test_retry_is_not_stopped_by_geometry_below_detection_threshold(tmp_path):
    import json
    from uav_refmatch.config import Config
    from uav_refmatch.core.types import FrameRecord, GeometryEvidence, LocalMatches
    from uav_refmatch.pipeline import Pipeline
    from uav_refmatch.outputs.recorder import Recorder

    class Proposals:
        def propose(self, image):
            return [Proposal(0, (0, 0, 40, 40), "test"), Proposal(1, (50, 50, 90, 90), "test")]

    class Encoder:
        def encode(self, images):
            if len(images) == 1:
                return np.array([[1.0, 0.0]])
            return np.array([[0.8, 0.6], [0.4, np.sqrt(0.84)]])

    class Matcher:
        def match(self, *args):
            return LocalMatches(np.empty((0, 2)), np.empty((0, 2)), np.empty(0))

    class Verifier:
        calls = 0

        def verify(self, *args):
            self.calls += 1
            return GeometryEvidence(40, 40, 1.0, 0.0, 0.5, True, self.calls >= 2, "test")

    config = Config(
        warmup=False,
        retrieval={"detect_min_cosine": 0.7, "geometry_min_cosine": 0.3, "max_geometry_pairs": 4},
        geometry={
            "candidate_rotation_fallbacks": [-45, 45],
            "reserved_rotation_calls": 2,
            "rotation_retry_policy": "cosine",
        },
    )
    verifier = Verifier()
    engine = Pipeline(config, Proposals(), Encoder(), Matcher(), verifier)
    image = np.zeros((100, 100, 3), np.uint8)
    rec = Recorder(tmp_path, config)
    engine.run([FrameRecord(0, 0, 0, 100, 100, image)], image, "test", rec)
    rec.finish("completed")
    row = json.loads((rec.path / "predictions.jsonl").read_text())
    assert row["status"] == "FOUND"
    assert row["bbox_xyxy"] == [0, 0, 40, 40]
    assert verifier.calls == 3
    assert row["verified_pairs"][-1]["candidate_rotation_degrees"] == -45
