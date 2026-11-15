import numpy as np
import pytest
from uav_refmatch.config import Tracking
from uav_refmatch.tracking.flow import FlowTracker, TrackEstimate
from uav_refmatch.tracking.controller import TrackController


def textured():
    import cv2

    image = np.random.default_rng(12).integers(0, 256, (160, 240, 3), dtype=np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0)


def test_motion_and_discontinuity():
    import cv2

    image = textured()
    tracker = FlowTracker(Tracking())
    assert not tracker.update(image, 0).reliable
    tracker.initialize(image, (40, 40, 130, 120), 0.0)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 5], [0, 1, 3]]), (240, 160))
    result = tracker.update(shifted, 0.13)
    assert result.reliable
    assert np.allclose(result.bbox, (45, 43, 135, 123), atol=2)
    assert not tracker.update(shifted, 10).reliable
    assert not tracker.update(shifted, 10.1).reliable


def test_blank_target_does_not_propagate():
    tracker = FlowTracker(Tracking())
    image = np.zeros((100, 100, 3), np.uint8)
    tracker.initialize(image, (20, 20, 80, 80), 0)
    assert not tracker.update(image, 0.1).reliable


def test_appearance_age_and_memory_are_independent():
    controller = TrackController(Tracking(reverify_interval_sec=1))
    image = textured()
    embedding = np.array([1.0, 0.0])
    controller.confirm(image, (40, 40, 130, 120), embedding, 0)
    embedding[:] = 0  # caller mutation cannot overwrite identity template
    estimate = TrackEstimate((40, 40, 130, 120), True)
    assert controller.can_propagate(estimate, np.array([1.0, 0.0]), 0.5)[0]
    assert not controller.can_propagate(estimate, np.array([0.0, 1.0]), 0.6)[0]
    assert not controller.can_propagate(estimate, np.array([1.0, 0.0]), 1.0)[0]
    assert controller.verified_at == 0
    controller.reject()
    assert controller.state == "LOST"
    assert not controller.predict(image, 1.1).reliable


def test_point_config_validation():
    with pytest.raises(ValueError):
        Tracking(min_points=20, max_points=10)


@pytest.mark.parametrize("distinct_view", [False, True])
def test_propagation_never_updates_dynamic_memory(tmp_path, distinct_view):
    import json
    from uav_refmatch.config import Config
    from uav_refmatch.core.types import FrameRecord, Proposal, GeometryEvidence, LocalMatches
    from uav_refmatch.pipeline import Pipeline
    from uav_refmatch.outputs.recorder import Recorder

    class Proposals:
        calls = 0

        def propose(self, image):
            self.calls += 1
            return [Proposal(0, (40, 40, 130, 120), "test")]

    class Encoder:
        calls = 0

        def encode(self, images):
            self.calls += 1
            vector = (
                [0.85, np.sqrt(1 - 0.85**2)] if distinct_view and self.calls > 1 else [1.0, 0.0]
            )
            return np.tile(vector, (len(images), 1))

    class Matcher:
        def match(self, *args):
            return LocalMatches(np.empty((0, 2)), np.empty((0, 2)), np.empty(0))

    class Verifier:
        def verify(self, *args):
            return GeometryEvidence(40, 40, 1.0, 0.0, 0.5, True, True, "test")

    image = textured()
    config = Config(tracking=Tracking(enabled=True), warmup=False)
    proposals = Proposals()
    engine = Pipeline(config, proposals, Encoder(), Matcher(), Verifier())
    recorder = Recorder(tmp_path, config)
    frames = [FrameRecord(i, i, t, 240, 160, image) for i, t in enumerate([0.0, 0.13, 0.26, 1.1])]
    engine.run(frames, image, "test", recorder)
    recorder.finish("completed")
    rows = [
        json.loads(line) for line in (recorder.path / "predictions.jsonl").read_text().splitlines()
    ]
    expected = (
        ["FOUND", "FOUND", "TRACKED", "FOUND"]
        if distinct_view
        else ["FOUND", "TRACKED", "TRACKED", "FOUND"]
    )
    assert [r["status"] for r in rows] == expected
    assert proposals.calls == (3 if distinct_view else 2)
    assert not rows[2]["verified_pairs"]
    events = [
        json.loads(line)
        for line in (recorder.path / "memory_events.jsonl").read_text().splitlines()
    ]
    assert not any(e["event_type"] == "ACCEPTED" and e["frame_index"] == 2 for e in events)
    if distinct_view:
        assert any(e["event_type"] == "ACCEPTED" and e["frame_index"] == 1 for e in events)


def test_tracking_annotation_is_not_green():
    from uav_refmatch.outputs.video_renderer import draw_prediction

    image = np.zeros((50, 50, 3), np.uint8)
    result = draw_prediction(image, dict(status="TRACKED", bbox_xyxy=[10, 10, 30, 30]), False)
    assert tuple(result[10, 10]) == (0, 180, 255)


def test_identity_gate_does_not_borrow_another_objects_track():
    from types import SimpleNamespace
    from uav_refmatch.decision.identity import identity_gate
    from uav_refmatch.core.types import Pair, Proposal

    controller = TrackController(Tracking())
    controller.embedding = np.array([0.0, 1.0])
    anchor = SimpleNamespace(embedding=np.array([1.0, 0.0]))
    estimate = TrackEstimate((10, 10, 30, 30), True)
    pair = Pair(0, "mem_0001", 0.9)
    embedding = np.array([0.0, 1.0])
    accepted, _ = identity_gate(
        pair, Proposal(0, (60, 60, 90, 90), "test"), embedding, anchor, controller, estimate
    )
    assert not accepted
    accepted, evidence = identity_gate(
        pair, Proposal(0, (10, 10, 30, 30), "test"), embedding, anchor, controller, estimate
    )
    assert accepted and evidence["support"] == "temporal"
    assert not identity_gate(
        pair, Proposal(0, (10, 10, 30, 30), "test"), embedding, anchor, controller, TrackEstimate()
    )[0]


def test_reference_region_is_applied_once(tmp_path):
    from uav_refmatch.pipeline import Pipeline
    from uav_refmatch.config import Config

    class Proposals:
        def prepare_reference(self, image):
            return {"shape": list(image.shape)}

    config = Config(reference={"bbox_xyxy": (10, 20, 40, 60)})
    engine = Pipeline(config, Proposals(), None, None, None)
    image = textured()
    assert engine.prepare_reference(image)["shape"] == [40, 30, 3]
    assert engine.reference_view(image).shape == (40, 30, 3)
    with pytest.raises(ValueError):
        engine.reference_view(np.zeros((30, 30, 3), np.uint8))


def test_mil_adapter_requires_motion_support_and_resets_on_gap():
    controller = TrackController(Tracking(backend="mil"))
    image = textured()
    assert not controller.predict(image, 0).reliable
    controller.confirm(image, (40, 40, 130, 120), np.array([1.0, 0.0]), 0)
    estimate = controller.predict(image, 0.1)
    assert estimate.reliable
    assert estimate.diagnostics["mil_flow_iou"] >= 0.1
    assert not controller.predict(image, 10.0).reliable
    assert controller.state == "LOST"
    assert controller.array_bytes() == controller.embedding.nbytes


def test_inconclusive_pipeline_hold_never_updates_memory(tmp_path):
    distinct_view = False
    import json
    from uav_refmatch.config import Config
    from uav_refmatch.core.types import FrameRecord, Proposal, GeometryEvidence, LocalMatches
    from uav_refmatch.pipeline import Pipeline
    from uav_refmatch.outputs.recorder import Recorder

    class Proposals:
        calls = 0

        def propose(self, image):
            self.calls += 1
            return [Proposal(0, (40, 40, 130, 120), "test")]

    class Encoder:
        calls = 0

        def encode(self, images):
            self.calls += 1
            vector = (
                [0.85, np.sqrt(1 - 0.85**2)] if distinct_view and self.calls > 1 else [1.0, 0.0]
            )
            return np.tile(vector, (len(images), 1))

    class Matcher:
        def match(self, *args):
            return LocalMatches(np.empty((0, 2)), np.empty((0, 2)), np.empty(0))

    class Verifier:
        calls = 0

        def verify(self, *args):
            self.calls += 1
            passed = self.calls == 1
            return GeometryEvidence(40, 40, 1.0, 0.0, 0.5, True, passed, "test")

    image = textured()
    config = Config(
        tracking=Tracking(enabled=True, max_gap_sec=2, max_unverified_sec=2), warmup=False
    )
    proposals = Proposals()
    engine = Pipeline(config, proposals, Encoder(), Matcher(), Verifier())
    recorder = Recorder(tmp_path, config)
    frames = [
        FrameRecord(i, i, t, 240, 160, image) for i, t in enumerate([0.0, 0.13, 1.1, 1.2, 2.1])
    ]
    engine.run(frames, image, "test", recorder)
    recorder.finish("completed")
    rows = [
        json.loads(line) for line in (recorder.path / "predictions.jsonl").read_text().splitlines()
    ]
    assert [r["status"] for r in rows] == ["FOUND", "TRACKED", "TRACKED", "TRACKED", "NOT_FOUND"]
    assert rows[2]["tracking"]["state"] == "UNCERTAIN"
    assert rows[3]["tracking"]["last_verified_sec"] == 0
    assert rows[4]["bbox_xyxy"] is None
    events = [
        json.loads(line)
        for line in (recorder.path / "memory_events.jsonl").read_text().splitlines()
    ]
    assert not any(e["event_type"] == "ACCEPTED" for e in events)
