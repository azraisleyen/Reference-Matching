from fractions import Fraction
import av
import numpy as np
import pytest
from uav_refmatch.data.frame_sampler import TimestampSampler
from uav_refmatch.data.video_reader import VideoReader, probe_video


@pytest.mark.parametrize("fps", [25, 30, 60])
def test_cfr_sampling(fps):
    sampler = TimestampSampler(7.5)
    accepted = [i for i in range(fps * 2) if sampler.accept(i / fps)]
    assert len(accepted) == 15
    assert len(set(accepted)) == len(accepted)
    for tick, index in enumerate(accepted):
        assert tick / 7.5 <= index / fps + 1e-8
        assert index / fps < tick / 7.5 + 1 / fps + 1e-8


def test_vfr_and_gaps():
    sampler = TimestampSampler(7.5)
    pts = [2.0, 2.01, 2.15, 2.2, 2.51, 2.52, 2.6]
    assert [x for x in pts if sampler.accept(x)] == [2.0, 2.15, 2.51, 2.6]


def test_low_fps_no_duplicates():
    sampler = TimestampSampler(7.5)
    assert all(sampler.accept(i / 3) for i in range(10))


@pytest.mark.parametrize("values", [[0.0, -0.1], [float("nan")], [float("inf")]])
def test_invalid_pts(values):
    sampler = TimestampSampler(7.5)
    with pytest.raises(ValueError):
        for value in values:
            sampler.accept(value)


def make_video(path, count=16, fps=30):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        for i in range(count):
            image = np.full((64, 64, 3), i * 5, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_real_decode(tmp_path):
    path = tmp_path / "synthetic.mp4"
    make_video(path)
    meta = probe_video(path)
    assert meta["source_fps"] == 30
    frames = list(VideoReader(path))
    assert [f.source_frame_index for f in frames] == [0, 4, 8, 12]
    assert [f.frame_index for f in frames] == list(range(4))
    assert all(f.width == 64 and f.height == 64 for f in frames)


def test_actual_vfr_video_pts(tmp_path):
    path = tmp_path / "vfr.mp4"
    timestamps = [0, 20, 145, 150, 285, 500, 550, 800]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        for stamp in timestamps:
            frame = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), np.uint8), format="rgb24")
            frame.pts, frame.time_base = stamp, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    frames = list(VideoReader(path))
    assert [f.source_frame_index for f in frames] == [0, 2, 4, 5, 6, 7]
    assert [round(f.timestamp_sec, 3) for f in frames] == [0, 0.145, 0.285, 0.5, 0.55, 0.8]
