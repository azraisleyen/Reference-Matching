"""Streaming PTS decode. Pixel coordinates are coded, unrotated video coordinates."""

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from time import perf_counter
import av
from .frame_sampler import TimestampSampler
from ..core.types import FrameRecord


def probe_video(path: str | Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    raw = json.loads(result.stdout)
    if not raw.get("streams"):
        raise ValueError("Video stream not found")
    s = raw["streams"][0]

    def rate(name):
        try:
            return float(Fraction(s.get(name, "0/1")))
        except (ValueError, ZeroDivisionError):
            return None

    rotation = float(s.get("tags", {}).get("rotate", 0))
    for side in s.get("side_data_list", []):
        rotation = float(side.get("rotation", rotation))
    duration = s.get("duration", raw.get("format", {}).get("duration"))
    return dict(
        codec=s.get("codec_name"),
        width=s["width"],
        height=s["height"],
        source_fps=rate("r_frame_rate"),
        average_fps=rate("avg_frame_rate"),
        r_frame_rate=s.get("r_frame_rate"),
        avg_frame_rate=s.get("avg_frame_rate"),
        duration_sec=float(duration) if duration else None,
        time_base=s.get("time_base"),
        frame_count=int(s["nb_frames"]) if s.get("nb_frames", "").isdigit() else None,
        rotation_degrees=rotation,
        coordinate_system="coded_unrotated",
    )


class VideoReader:
    def __init__(self, path, target_fps=7.5, verified_cfr_fps=None):
        self.path = Path(path)
        self.target_fps = target_fps
        self.verified_cfr_fps = verified_cfr_fps
        self.fallback_used = False
        self.fallback_frames = 0

    def __iter__(self):
        sampler = TimestampSampler(self.target_fps)
        processed = 0
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            tick = perf_counter()
            previous = None
            origin = None
            for source_index, frame in enumerate(container.decode(stream)):
                timestamp = float(frame.pts * frame.time_base) if frame.pts is not None else None
                if timestamp is None or (previous is not None and timestamp < previous):
                    if self.verified_cfr_fps is None:
                        raise ValueError(
                            "Missing/non-monotonic PTS; explicitly verified CFR is required"
                        )
                    if origin is None:
                        origin = 0.0
                    timestamp = origin + source_index / self.verified_cfr_fps
                    self.fallback_used = True
                    self.fallback_frames += 1
                if origin is None:
                    origin = timestamp - source_index / (self.verified_cfr_fps or 1)
                previous = timestamp
                if not sampler.accept(timestamp):
                    continue
                rgb = frame.to_ndarray(format="rgb24")
                elapsed = (perf_counter() - tick) * 1000
                yield FrameRecord(
                    processed, source_index, timestamp, frame.width, frame.height, rgb, elapsed
                )
                processed += 1
                tick = perf_counter()
