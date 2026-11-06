"""Exercise missing/backward PTS policy without corrupting real media files."""

from fractions import Fraction
import numpy as np
import pytest
from uav_refmatch.data.video_reader import VideoReader


class FakeFrame:
    width = height = 32
    time_base = Fraction(1, 30)

    def __init__(self, pts):
        self.pts = pts

    def to_ndarray(self, format):
        return np.zeros((32, 32, 3), np.uint8)


class FakeContainer:
    def __init__(self, pts):
        self.frames = [FakeFrame(p) for p in pts]
        self.streams = type("Streams", (), {"video": [None]})()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def decode(self, stream):
        return iter(self.frames)


def test_missing_pts_requires_explicit_verified_cfr(monkeypatch):
    monkeypatch.setattr("av.open", lambda _: FakeContainer([0, None, 2]))
    with pytest.raises(ValueError, match="verified CFR"):
        list(VideoReader("not-a-real-file"))


def test_controlled_fallback_is_observable(monkeypatch):
    monkeypatch.setattr("av.open", lambda _: FakeContainer([None] * 9))
    reader = VideoReader("not-a-real-file", verified_cfr_fps=30)
    frames = list(reader)
    assert reader.fallback_used and reader.fallback_frames == 9
    assert [f.source_frame_index for f in frames] == [0, 4, 8]
    assert [f.timestamp_sec for f in frames] == [0, 4 / 30, 8 / 30]
