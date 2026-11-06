import cv2
import numpy as np
from ..config import Geometry
from ..core.types import GeometryEvidence, LocalMatches


class AffineRansacVerifier:
    def __init__(self, config: Geometry, seed: int = 42) -> None:
        self.config = config
        self.seed = seed

    def verify(
        self,
        matches: LocalMatches,
        reference_shape: tuple[int, ...],
        candidate_shape: tuple[int, ...],
    ) -> GeometryEvidence:
        c = self.config
        src = np.asarray(matches.reference_points, dtype=np.float32)
        dst = np.asarray(matches.candidate_points, dtype=np.float32)
        n = len(src)
        if src.shape != (n, 2) or dst.shape != (n, 2) or len(matches.confidence) != n:
            return GeometryEvidence(num_matches=n, reason="invalid_match_shape")
        if (
            not np.isfinite(src).all()
            or not np.isfinite(dst).all()
            or not np.isfinite(matches.confidence).all()
        ):
            return GeometryEvidence(num_matches=n, reason="nonfinite_matches")
        for points, shape in [(src, reference_shape), (dst, candidate_shape)]:
            if (
                np.any(points < 0)
                or np.any(points[:, 0] >= shape[1])
                or np.any(points[:, 1] >= shape[0])
            ):
                return GeometryEvidence(num_matches=n, reason="out_of_bounds_matches")
        if n < c.min_matches:
            return GeometryEvidence(num_matches=n)
        if (
            np.linalg.matrix_rank(src - src.mean(axis=0)) < 2
            or np.linalg.matrix_rank(dst - dst.mean(axis=0)) < 2
        ):
            return GeometryEvidence(num_matches=n, reason="degenerate_points")
        cv2.setRNGSeed(self.seed)
        transform, mask = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=c.ransac_reprojection_threshold_px,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )
        if transform is None or mask is None or not np.isfinite(transform).all():
            return GeometryEvidence(num_matches=n, reason="invalid_transform")
        selected = mask.ravel().astype(bool)
        count = int(selected.sum())
        scale = float(np.linalg.norm(transform[:, 0]))
        valid = bool(c.min_scale <= scale <= c.max_scale and np.linalg.det(transform[:, :2]) > 0)
        # Explicit two-coordinate arithmetic avoids unnecessary BLAS dispatch for Nx2.
        projected = np.column_stack(
            (
                src[:, 0] * transform[0, 0] + src[:, 1] * transform[0, 1] + transform[0, 2],
                src[:, 0] * transform[1, 0] + src[:, 1] * transform[1, 1] + transform[1, 2],
            )
        )
        if not np.isfinite(projected).all():
            return GeometryEvidence(num_matches=n, reason="nonfinite_reprojection")
        error = (
            float(np.median(np.linalg.norm(projected[selected] - dst[selected], axis=1)))
            if count
            else None
        )

        def coverage(points, shape):
            if len(points) < 3:
                return 0.0
            return float(cv2.contourArea(cv2.convexHull(points))) / (shape[0] * shape[1])

        coverage_value = min(
            coverage(src[selected], reference_shape), coverage(dst[selected], candidate_shape)
        )
        reason = "geometry_passed"
        for condition, failure in [
            (valid, "invalid_scale_or_transform"),
            (count >= c.min_inliers, "insufficient_inliers"),
            (count / n >= c.min_inlier_ratio, "low_inlier_ratio"),
            (
                error is not None and error <= c.max_median_reprojection_error_px,
                "high_reprojection_error",
            ),
            (coverage_value >= c.min_spatial_coverage, "low_spatial_coverage"),
        ]:
            if not condition:
                reason = failure
                break
        return GeometryEvidence(
            n, count, count / n, error, coverage_value, valid, reason == "geometry_passed", reason
        )
