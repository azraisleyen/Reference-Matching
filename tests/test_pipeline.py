"""Deterministic test doubles are ONLY in tests; these are not model integration results."""

import json
import numpy as np
import pytest
from uav_refmatch.config import Config
from uav_refmatch.core.types import FrameRecord, Proposal, LocalMatches
from uav_refmatch.geometry.verifier import AffineRansacVerifier
from uav_refmatch.outputs.recorder import Recorder
from uav_refmatch.pipeline import Pipeline


class Proposals:
    def __init__(self, empty=False, fail=False):
        self.empty, self.fail = empty, fail

    def propose(self, image):
        if self.fail:
            raise ValueError("test_inference_error")
        return [] if self.empty else [Proposal(0, (0, 0, 64, 64), "TEST_DOUBLE")]


class Encoder:
    def __init__(self):
        self.calls = 0

    def encode(self, images):
        self.calls += 1
        vector = [1.0, 0.0, 0.0] if self.calls == 1 else [0.85, 0.5267827, 0.0]
        return np.tile(np.array(vector, np.float32), (len(images), 1))


class Matcher:
    def __init__(self):
        self.calls = 0

    def match(self, reference, candidate):
        self.calls += 1
        points = np.random.default_rng(42).uniform(1, 62, (40, 2)).astype(np.float32)
        return LocalMatches(points, points, np.ones(40))


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_end_to_end_records_and_memory(tmp_path, texture, mode):
    config = Config(mode=mode, warmup=False)
    recorder = Recorder(tmp_path, config)
    encoder, matcher = Encoder(), Matcher()
    pipeline = Pipeline(
        config, Proposals(), encoder, matcher, AffineRansacVerifier(config.geometry)
    )
    frames = [FrameRecord(i, i * 4, i / 7.5, 64, 64, texture) for i in range(4)]
    status = pipeline.run(frames, texture, "target", recorder)
    recorder.finish(status)
    rows = [
        json.loads(line) for line in (recorder.path / "predictions.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 4 and all(p["status"] == "FOUND" for p in rows)
    assert rows[0]["proposals"][0]["bbox_xyxy"] == [0, 0, 64, 64]
    assert rows[0]["proposals"][0]["crop_bounds"] == [0, 0, 64, 64]
    assert encoder.calls == 5  # R0 once, candidates once per frame, never reference re-encoding
    assert matcher.calls <= 4 * 3
    assert [p["active_reference_count"] for p in rows] == (
        [1, 1, 2, 2] if mode == "dynamic" else [1, 1, 1, 1]
    )
    assert not list(recorder.path.glob("*.png")) and not list(recorder.path.glob("*.jpg"))


def test_not_found_record(tmp_path, texture):
    config = Config()
    recorder = Recorder(tmp_path, config)
    pipeline = Pipeline(
        config, Proposals(empty=True), Encoder(), Matcher(), AffineRansacVerifier(config.geometry)
    )
    pipeline.run([FrameRecord(0, 0, 0, 64, 64, texture)], texture, "t", recorder)
    recorder.finish("completed")
    row = json.loads((recorder.path / "predictions.jsonl").read_text())
    assert row["status"] == "NOT_FOUND" and row["bbox_xyxy"] is None


def test_rejected_candidates_are_inspectable(tmp_path, texture):
    config = Config(retrieval={"detect_min_cosine": 0.9}, memory={"memory_min_cosine": 0.95})
    recorder = Recorder(tmp_path, config)
    matcher = Matcher()
    pipeline = Pipeline(
        config, Proposals(), Encoder(), matcher, AffineRansacVerifier(config.geometry)
    )
    pipeline.run([FrameRecord(0, 0, 0, 64, 64, texture)], texture, "target", recorder)
    recorder.finish("completed")
    row = json.loads((recorder.path / "predictions.jsonl").read_text())
    assert row["status"] == "NOT_FOUND" and row["bbox_xyxy"] is None
    assert row["proposals"][0]["candidate_index"] == 0
    assert row["retrieval_candidates"][0]["cosine"] < 0.9
    assert matcher.calls == 0


def test_diagnostic_geometry_never_bypasses_final_gate(tmp_path, texture):
    config = Config(
        retrieval={"detect_min_cosine": 0.9, "geometry_min_cosine": 0.3},
        memory={"memory_min_cosine": 0.95},
    )
    recorder = Recorder(tmp_path, config)
    matcher = Matcher()
    pipeline = Pipeline(
        config, Proposals(), Encoder(), matcher, AffineRansacVerifier(config.geometry)
    )
    pipeline.run([FrameRecord(0, 0, 0, 64, 64, texture)], texture, "target", recorder)
    recorder.finish("completed")
    row = json.loads((recorder.path / "predictions.jsonl").read_text())
    assert row["status"] == "NOT_FOUND" and row["bbox_xyxy"] is None
    assert row["geometry"]["accepted"]
    assert row["decision_reason"] == "geometry_passed_below_visual_threshold"
    assert matcher.calls == 1


def test_error_is_not_not_found(tmp_path, texture):
    config = Config()
    recorder = Recorder(tmp_path, config)
    pipeline = Pipeline(
        config, Proposals(fail=True), Encoder(), Matcher(), AffineRansacVerifier(config.geometry)
    )
    with pytest.raises(RuntimeError):
        pipeline.run([FrameRecord(0, 0, 0, 64, 64, texture)], texture, "t", recorder)
    recorder.finish("failed")
    row = json.loads((recorder.path / "predictions.jsonl").read_text())
    assert row["status"] == "ERROR" and row["bbox_xyxy"] is None


def test_cancel(tmp_path, texture):
    config = Config()
    recorder = Recorder(tmp_path, config)
    pipeline = Pipeline(
        config, Proposals(), Encoder(), Matcher(), AffineRansacVerifier(config.geometry)
    )
    assert (
        pipeline.run([FrameRecord(0, 0, 0, 64, 64, texture)], texture, "t", recorder, lambda: True)
        == "cancelled"
    )
    recorder.finish("cancelled")
    assert (recorder.path / "predictions.jsonl").read_text() == ""
