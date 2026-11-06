"""First actual frame at/after each deadline, with no duplicated source frames."""

import math


class TimestampSampler:
    def __init__(self, target_fps: float):
        if not math.isfinite(target_fps) or target_fps <= 0:
            raise ValueError("target_fps must be positive and finite")
        self.fps = target_fps
        self.origin = None
        self.next_tick = 0
        self.previous = None

    def accept(self, timestamp: float) -> bool:
        if not math.isfinite(timestamp):
            raise ValueError("Non-finite presentation timestamp")
        if self.previous is not None and timestamp < self.previous:
            raise ValueError("Non-monotonic presentation timestamps")
        self.previous = timestamp
        if self.origin is None:
            self.origin = timestamp
        elapsed = timestamp - self.origin
        if elapsed + 1e-9 < self.next_tick / self.fps:
            return False
        self.next_tick = math.floor(elapsed * self.fps + 1e-9) + 1
        return True
