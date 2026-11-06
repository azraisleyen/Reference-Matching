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
