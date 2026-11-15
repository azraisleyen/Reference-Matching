"""MIL localization guarded by forward/backward motion consistency.

MIL's boolean output is not calibrated confidence. Flow and DINO checks remain
mandatory; this adapter does not weaken identity or long-term memory gates.
"""

import cv2
from .flow import FlowTracker, TrackEstimate
from ..proposals.boxes import iou


class GuardedMILTracker:
    def __init__(self, config):
        self.config = config
        self.flow = FlowTracker(config)
        self.model = None

    def reset(self):
        self.flow.reset()
        self.model = None

    def initialize(self, image, bbox, timestamp):
        self.flow.initialize(image, bbox, timestamp)
        self.model = cv2.TrackerMIL_create()
        x1, y1, x2, y2 = map(round, bbox)
        self.model.init(image, (x1, y1, x2 - x1, y2 - y1))

    def update(self, image, timestamp):
        estimate = self.flow.update(image, timestamp)
        if not estimate.reliable or self.model is None:
            self.reset()
            return estimate
        ok, (x, y, w, h) = self.model.update(image)
        box = (max(0.0, x), max(0.0, y), min(image.shape[1], x + w), min(image.shape[0], y + h))
        overlap = iou(box, estimate.bbox)
        diagnostics = {**estimate.diagnostics, "mil_flow_iou": overlap}
        if not ok or overlap < self.config.association_min_iou or min(w, h) <= 0:
            self.reset()
            return TrackEstimate(reason="mil_motion_disagreement", diagnostics=diagnostics)
        return TrackEstimate(box, True, "mil_motion_supported", diagnostics)
