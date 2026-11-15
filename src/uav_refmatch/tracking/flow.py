"""Sparse pyramidal LK with forward/backward and robust motion checks.

Points originate inside the last accepted box. No checkpoint or online training.
This estimates motion, not physical instance identity.
"""

from dataclasses import dataclass, field
import cv2
import numpy as np


@dataclass
class TrackEstimate:
    bbox: tuple | None = None
    reliable: bool = False
    reason: str = "uninitialized"
    diagnostics: dict = field(default_factory=dict)


class FlowTracker:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.gray = self.points = self.box = self.time = None

    def _gray(self, image):
        h, w = image.shape[:2]
        scale = min(1.0, self.config.max_side / max(h, w))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (round(w * scale), round(h * scale)))
        return gray, np.array([gray.shape[1] / w, gray.shape[0] / h])

    def initialize(self, image, bbox, timestamp):
        self.gray, self.scale = self._gray(image)
        self.box = np.asarray(bbox, dtype=float).reshape(2, 2) * self.scale
        self.time = timestamp
        self._points()

    def _points(self):
        mask = np.zeros_like(self.gray)
        h, w = mask.shape
        a, b = np.floor(self.box[0]).astype(int)
        c, d = np.ceil(self.box[1]).astype(int)
        mask[max(0, b) : min(h, d), max(0, a) : min(w, c)] = 255
        self.points = cv2.goodFeaturesToTrack(self.gray, self.config.max_points, 0.01, 3, mask=mask)

    def update(self, image, timestamp):
        if self.gray is None:
            return TrackEstimate()
        gray, scale = self._gray(image)
        if (
            not 0 < timestamp - self.time <= self.config.max_gap_sec
            or gray.shape != self.gray.shape
            or not np.allclose(scale, self.scale)
        ):
            self.reset()
            return TrackEstimate(reason="discontinuity")
        if self.points is None or len(self.points) < self.config.min_points:
            self.reset()
            return TrackEstimate(reason="insufficient_points")
        nxt, ok, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.points, None)
        if nxt is None:
            self.reset()
            return TrackEstimate(reason="flow_failed")
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, nxt, None)
        if back is None:
            self.reset()
            return TrackEstimate(reason="reverse_flow_failed")
        diag = max(1.0, float(np.linalg.norm(self.box[1] - self.box[0])))
        error = np.linalg.norm(back - self.points, axis=2).ravel() / diag
        valid = (ok.ravel() > 0) & (reverse.ravel() > 0) & np.isfinite(error)
        valid &= error <= self.config.max_fb_error_fraction
        diagnostics = dict(points=len(valid), consistent_points=int(valid.sum()))
        if valid.sum() < self.config.min_points:
            self.reset()
            return TrackEstimate(reason="forward_backward_failed", diagnostics=diagnostics)
        old = self.points[valid].reshape(-1, 2)
        new = nxt[valid].reshape(-1, 2)
        transform, inliers = cv2.estimateAffinePartial2D(
            old,
            new,
            method=cv2.RANSAC,
            ransacReprojThreshold=max(1.0, diag * self.config.max_fb_error_fraction),
        )
        ratio = float(inliers.sum() / len(valid)) if inliers is not None else 0.0
        diagnostics["support_ratio"] = ratio
        if transform is None or ratio < self.config.min_support_ratio:
            self.reset()
            return TrackEstimate(reason="motion_support_failed", diagnostics=diagnostics)
        motion_scale = float(np.linalg.norm(transform[0, :2]))
        if not 1 / self.config.max_scale_change <= motion_scale <= self.config.max_scale_change:
            self.reset()
            return TrackEstimate(reason="scale_jump", diagnostics=diagnostics)
        x1, y1 = self.box[0]
        x2, y2 = self.box[1]
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)
        warped = corners @ transform[:, :2].T + transform[:, 2]
        low, high = warped.min(0), warped.max(0)
        low = np.maximum(low, 0)
        high = np.minimum(high, [gray.shape[1], gray.shape[0]])
        if np.any(high - low < 2):
            self.reset()
            return TrackEstimate(reason="outside_frame", diagnostics=diagnostics)
        self.box = np.array([low, high])
        self.gray = gray
        self.time = timestamp
        self._points()
        return TrackEstimate(
            tuple((self.box / self.scale).ravel()), True, "flow_supported", diagnostics
        )
