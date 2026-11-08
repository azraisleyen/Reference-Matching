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
        raw_count = n
        selected_confidence = np.asarray(matches.confidence) >= c.min_match_confidence
        src, dst = src[selected_confidence], dst[selected_confidence]
        n = len(src)
        if n < c.min_matches:
            return GeometryEvidence(
                num_matches=n, diagnostics={"raw_matches": raw_count, "filtered_matches": n}
            )
        if (
            np.linalg.matrix_rank(src - src.mean(axis=0)) < 2
            or np.linalg.matrix_rank(dst - dst.mean(axis=0)) < 2
        ):
            return GeometryEvidence(num_matches=n, reason="degenerate_points")
        cv2.setRNGSeed(self.seed)
        estimate = (
            cv2.estimateAffinePartial2D if c.model == "partial_affine" else cv2.estimateAffine2D
        )
        transform, mask = estimate(
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
        scales = np.linalg.svd(transform[:, :2], compute_uv=False)
        valid = bool(
            scales.min() >= c.min_scale
            and scales.max() <= c.max_scale
            and scales.max() / max(scales.min(), 1e-12) <= c.max_anisotropy
            and np.linalg.det(transform[:, :2]) > 0
        )
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

        reference_coverage = coverage(src[selected], reference_shape)
        candidate_coverage = coverage(dst[selected], candidate_shape)
        coverage_value = min(reference_coverage, candidate_coverage)
        checks = [
            (valid, "invalid_scale_or_transform"),
            (count >= c.min_inliers, "insufficient_inliers"),
            (count / n >= c.min_inlier_ratio, "low_inlier_ratio"),
            (
                error is not None and error <= c.max_median_reprojection_error_px,
                "high_reprojection_error",
            ),
            (coverage_value >= c.min_spatial_coverage, "low_spatial_coverage"),
        ]
        failures = [failure for condition, failure in checks if not condition]
        reason = failures[0] if failures else "geometry_passed"
        return GeometryEvidence(
            n,
            count,
            count / n,
            error,
            coverage_value,
            valid,
            not failures,
            reason,
            dict(
                raw_matches=raw_count,
                filtered_matches=n,
                transform=transform.tolist(),
                singular_values=scales.tolist(),
                determinant=float(np.linalg.det(transform[:, :2])),
                reference_coverage=reference_coverage,
                candidate_coverage=candidate_coverage,
                failed_checks=failures,
            ),
        )
