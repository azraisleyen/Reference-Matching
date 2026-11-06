from dataclasses import replace
import av
import json
from uav_refmatch.config import Config
from uav_refmatch.outputs.recorder import Recorder, file_hash
from uav_refmatch.outputs.video_renderer import render_video
from test_frame_sampler import make_video


def test_partial_run_render_stops_at_last_processed_frame(tmp_path, monkeypatch, found):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    source = tmp_path / "source.mp4"
    make_video(source, count=16)
    recorder = Recorder(tmp_path / "runs", Config(max_frames=2))
    recorder.manifest["source_video"] = {"name": source.name, "sha256": file_hash(source)}
    recorder.prediction(found)
    recorder.prediction(replace(found, frame_index=1, source_frame_index=4, timestamp_sec=4 / 30))
    recorder.finish("completed")
    output = render_video(recorder.path, source)
    with av.open(str(output)) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 5
    assert json.loads((recorder.path / "render_manifest.json").read_text())[
        "truncated_to_last_processed_frame"
    ]
