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


def test_geometry_records_all_failed_checks_without_relaxing_acceptance():
    import numpy as np
    from uav_refmatch.config import Geometry
    from uav_refmatch.core.types import LocalMatches
    from uav_refmatch.geometry.verifier import AffineRansacVerifier

    points = np.random.default_rng(9).uniform(10, 20, (20, 2)).astype(np.float32)
    result = AffineRansacVerifier(Geometry(min_inliers=25, min_spatial_coverage=0.1)).verify(
        LocalMatches(points, points, np.ones(20)), (100, 100, 3), (100, 100, 3)
    )
    assert not result.accepted
    assert result.reason == "insufficient_inliers"
    assert set(result.diagnostics["failed_checks"]) == {
        "insufficient_inliers",
        "low_spatial_coverage",
    }
    assert len(result.diagnostics["transform"]) == 2
    assert result.diagnostics["raw_matches"] == 20
