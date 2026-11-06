import math
import numpy as np
from ..core.types import Proposal


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def clean_proposals(boxes, width, height, config, source):
    """Stable model-order NMS: Florence has no calibrated proposal confidences."""
    kept = []
    for index, box in enumerate(boxes):
        if len(box) != 4 or not np.isfinite(box).all():
            continue
        x1, y1, x2, y2 = map(float, box)
        box = (
            max(0.0, min(width, x1)),
            max(0.0, min(height, y1)),
            max(0.0, min(width, x2)),
            max(0.0, min(height, y2)),
        )
        if min(box[2] - box[0], box[3] - box[1]) < config.min_box_side:
            continue
        if any(iou(box, p.bbox_xyxy) >= config.nms_iou for p in kept):
            continue
        kept.append(Proposal(index, box, source))
        if len(kept) == config.max_proposals:
            break
    return kept


def crop(image, bbox, padding):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    bounds = (
        max(0, math.floor(x1 - dx)),
        max(0, math.floor(y1 - dy)),
        min(w, math.ceil(x2 + dx)),
        min(h, math.ceil(y2 + dy)),
    )
    a, b, c, d = bounds
    if c <= a or d <= b:
        raise ValueError("Empty crop")
    return image[b:d, a:c].copy(), bounds
