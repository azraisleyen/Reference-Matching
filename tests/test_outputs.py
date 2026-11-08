from dataclasses import replace
import csv
import json
import av
import numpy as np
import pytest
from uav_refmatch.config import Config
from uav_refmatch.outputs.recorder import Recorder, file_hash
from uav_refmatch.outputs.video_renderer import draw_prediction, render_video, inspect_frame
from test_frame_sampler import make_video


def test_serialization_and_empty_not_found(tmp_path, found):
    recorder = Recorder(tmp_path, Config())
    recorder.prediction(found)
    recorder.prediction(replace(found, frame_index=1, status="NOT_FOUND", bbox_xyxy=None))
    recorder.finish("completed")
    rows = list(csv.DictReader((recorder.path / "detections.csv").open()))
    assert rows[0]["x_max"] == "64" and rows[1]["x_max"] == ""
    assert json.loads((recorder.path / "summary.json").read_text())["accuracy_metrics"] is None


def test_draw_coordinates_and_no_false_box(texture):
    image = np.zeros_like(texture)
    result = draw_prediction(image, {"status": "FOUND", "bbox_xyxy": [10, 20, 30, 40]}, False)
    assert tuple(result[20, 10]) == (0, 255, 0)
    np.testing.assert_array_equal(
        draw_prediction(image, {"status": "NOT_FOUND", "bbox_xyxy": None}), image
    )


def test_render_without_inference_preserves_pts(tmp_path, monkeypatch, found):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    source = tmp_path / "source.mp4"
    make_video(source)
    recorder = Recorder(tmp_path / "runs", Config())
    recorder.manifest["source_video"] = {"name": source.name, "sha256": file_hash(source)}
    recorder.prediction(replace(found, bbox_xyxy=(10, 20, 30, 40)))
    recorder.finish("completed")
    rendered = render_video(recorder.path, source)

    def timestamps(path):
        with av.open(str(path)) as container:
            return [float(f.pts * f.time_base) for f in container.decode(video=0)]

    assert timestamps(rendered) == timestamps(source)
    preview, row = inspect_frame(recorder.path, 0, source)
    assert tuple(preview[20, 10]) == (0, 255, 0)
    assert row["prediction"]["status"] == "FOUND"
    other = tmp_path / "other.mp4"
    make_video(other, count=8)
    with pytest.raises(ValueError, match="SHA256"):
        render_video(recorder.path, other)


def test_vfr_render_preserves_actual_presentation_times(tmp_path, monkeypatch, found):
    from fractions import Fraction

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    source = tmp_path / "vfr_source.mp4"
    times = [0, 20, 145, 150, 285, 500, 550, 800]
    with av.open(str(source), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for stamp in times:
            frame = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), np.uint8), format="rgb24")
            frame.pts, frame.time_base = stamp, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    recorder = Recorder(tmp_path / "runs", Config())
    recorder.manifest["source_video"] = {"name": source.name, "sha256": file_hash(source)}
    recorder.prediction(found)
    recorder.finish("completed")
    output = render_video(recorder.path, source)

    def pts(path):
        with av.open(str(path)) as container:
            return [float(f.pts * f.time_base) for f in container.decode(video=0)]

    assert pts(source) == pts(output)


def test_report_explains_unused_dynamic_memory(tmp_path, found):
    recorder = Recorder(tmp_path, Config(mode="dynamic"))
    recorder.prediction(found)
    recorder.finish("completed")
    summary = json.loads((recorder.path / "summary.json").read_text())
    assert summary["mode"] == "dynamic"
    assert summary["memory_diagnostics"]["accepted"] == 0
    assert summary["memory_diagnostics"]["warning"]
    assert summary["memory_diagnostics"]["max_found_cosine"] == found.dino_cosine


def test_candidate_colors_scores_and_detection_precedence(monkeypatch):
    from uav_refmatch.outputs.video_renderer import draw_annotations
    import cv2

    image = np.zeros((160, 320, 3), dtype=np.uint8)
    prediction = dict(
        status="NOT_FOUND",
        bbox_xyxy=None,
        proposals=[dict(candidate_index=0, bbox_xyxy=[20, 30, 100, 80])],
        retrieval_candidates=[dict(candidate_index=0, cosine=0.7234)],
    )
    labels = []
    original = cv2.putText

    def capture(image, text, *args, **kwargs):
        labels.append(text)
        return original(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture)
    result = draw_annotations(image, prediction)
    assert tuple(result[30, 20]) == (255, 165, 0)
    assert "candidate 0 cos=0.723" in labels
    assert not np.any(np.all(result == (0, 255, 0), axis=2))
    prediction.update(
        status="FOUND", bbox_xyxy=[20, 30, 100, 80], matched_memory_id="R0", timestamp_sec=0.0
    )
    result = draw_annotations(image, prediction)
    assert tuple(result[30, 20]) == (0, 255, 0)
    assert np.any(np.all(result[81:] == (255, 165, 0), axis=2))
    np.testing.assert_array_equal(draw_annotations(image, None), image)
    np.testing.assert_array_equal(draw_annotations(image, {"status": "NOT_FOUND"}), image)


def test_export_includes_orange_candidates_only_on_sampled_frames(tmp_path, monkeypatch, found):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    source = tmp_path / "source.mp4"
    make_video(source, count=3)
    recorder = Recorder(tmp_path / "runs", Config())
    recorder.manifest["source_video"] = {"name": source.name, "sha256": file_hash(source)}
    recorder.prediction(
        replace(
            found,
            status="NOT_FOUND",
            bbox_xyxy=None,
            proposals=[dict(candidate_index=0, bbox_xyxy=[10, 20, 40, 40])],
            retrieval_candidates=[dict(candidate_index=0, cosine=0.7)],
        )
    )
    recorder.finish("completed")
    output = render_video(recorder.path, source)
    with av.open(str(output)) as container:
        images = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]

    def orange(image):
        return (
            (image[:, :, 0] > 180)
            & (image[:, :, 1] > 80)
            & (image[:, :, 1] < 220)
            & (image[:, :, 2] < 60)
        )

    assert orange(images[0]).sum() > 20
    assert not orange(images[1]).any()
    manifest = json.loads((recorder.path / "render_manifest.json").read_text())
    assert manifest["annotation_policy"] == "accepted_green_candidates_orange"
