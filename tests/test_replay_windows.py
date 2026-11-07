import importlib.util
from pathlib import Path
from test_frame_sampler import make_video


def test_cached_window_seek_preserves_recorded_indices(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "replay", Path(__file__).parents[1] / "scripts/benchmark_memory_replay.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    video = tmp_path / "source.mp4"
    make_video(video, count=12)
    import av

    with av.open(str(video)) as container:
        stamps = [frame.time for frame in container.decode(video=0)]
    selected = {
        i: dict(frame_index=i, source_frame_index=i, timestamp_sec=stamps[i]) for i in [0, 1, 8, 9]
    }
    adapter = module.RecordedProposals()
    frames = list(module.frames(video, selected, adapter))
    assert [f.frame_index for f in frames] == [0, 1, 8, 9]
    assert [f.source_frame_index for f in frames] == [0, 1, 8, 9]
    assert [f.timestamp_sec for f in frames] == [stamps[i] for i in [0, 1, 8, 9]]
