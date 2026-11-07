"""Preprocessing contract tests, independent of checkpoint access."""

import numpy as np
from uav_refmatch.matching.efficientloftr import EfficientLoFTRMatcher
import torch


def test_loftr_resize_preserves_crop_coordinate_scale():
    adapter = object.__new__(EfficientLoFTRMatcher)
    adapter.max_side = 640
    adapter.resize_to_max = False
    adapter.device = "cpu"
    adapter.dtype = torch.float32
    image = np.zeros((103, 257, 3), np.uint8)
    tensor, scale = adapter.prepare(image)
    assert tensor.shape == (1, 1, 96, 256)
    np.testing.assert_allclose(scale, [257 / 256, 103 / 96])
    assert tensor.min() == 0 and tensor.max() == 0


def test_tiny_crop_resizes_to_supported_shape():
    adapter = object.__new__(EfficientLoFTRMatcher)
    adapter.max_side = 640
    adapter.resize_to_max = False
    adapter.device = "cpu"
    adapter.dtype = torch.float32
    tensor, scale = adapter.prepare(np.full((12, 18, 3), 255, np.uint8))
    assert tensor.shape == (1, 1, 32, 32)
    assert tensor.min() == 1
    np.testing.assert_allclose(scale, [18 / 32, 12 / 32])


def test_optional_loftr_upscale_retains_original_pixel_coordinates():
    adapter = object.__new__(EfficientLoFTRMatcher)
    adapter.max_side = 640
    adapter.resize_to_max = True
    adapter.device = "cpu"
    adapter.dtype = torch.float32
    tensor, scale = adapter.prepare(np.zeros((80, 100, 3), np.uint8))
    assert tensor.shape == (1, 1, 512, 640)
    np.testing.assert_allclose(scale, [100 / 640, 80 / 512])
