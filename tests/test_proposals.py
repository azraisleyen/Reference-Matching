import numpy as np
from uav_refmatch.config import Proposals
from uav_refmatch.proposals.boxes import clean_proposals, crop


def test_clip_invalid_duplicate_nms():
    boxes = [
        [-10, -10, 20, 20],
        [0, 0, 20, 20],
        [30, 30, 10, 10],
        [0, 0, 2, 2],
        [0, 0, np.nan, 20],
        [25, 25, 100, 100],
    ]
    result = clean_proposals(boxes, 64, 64, Proposals(), "florence")
    assert [p.bbox_xyxy for p in result] == [(0.0, 0.0, 20.0, 20.0), (25.0, 25.0, 64.0, 64.0)]


def test_padding(texture):
    image, bounds = crop(texture, (10, 10, 30, 30), 0.1)
    assert bounds == (8, 8, 32, 32)
    assert image.shape == (24, 24, 3)


def test_empty():
    assert clean_proposals([], 64, 64, Proposals(), "f") == []
