import numpy as np
from uav_refmatch.config import Geometry
from uav_refmatch.core.types import LocalMatches
from uav_refmatch.geometry.verifier import AffineRansacVerifier


def test_valid_affine():
    rng = np.random.default_rng(2)
    src = rng.uniform(5, 75, (60, 2)).astype(np.float32)
    dst = src * 1.1 + 3
    dst[:8] = rng.uniform(0, 99, (8, 2))
    result = AffineRansacVerifier(Geometry()).verify(
        LocalMatches(src, dst, np.ones(60)), (100, 100), (100, 100)
    )
    assert (
        result.accepted and result.num_inliers >= 50 and result.median_reprojection_error_px < 0.01
    )


def test_insufficient():
    points = np.zeros((2, 2), np.float32)
    result = AffineRansacVerifier(Geometry()).verify(
        LocalMatches(points, points, np.ones(2)), (100, 100), (100, 100)
    )
    assert not result.accepted and result.reason == "insufficient_matches"


def test_degenerate():
    points = np.ones((20, 2), np.float32)
    assert (
        AffineRansacVerifier(Geometry())
        .verify(LocalMatches(points, points, np.ones(20)), (100, 100), (100, 100))
        .reason
        == "degenerate_points"
    )


def test_low_coverage():
    points = np.random.default_rng(1).uniform(1, 3, (20, 2)).astype(np.float32)
    result = AffineRansacVerifier(Geometry()).verify(
        LocalMatches(points, points, np.ones(20)), (100, 100), (100, 100)
    )
    assert not result.accepted and result.reason == "low_spatial_coverage"
