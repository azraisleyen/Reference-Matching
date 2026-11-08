"""Anchor-controlled admission without additional, hidden geometry calls."""

import cv2
import numpy as np


def select_anchor_geometry(verified, candidate_index, config):
    """Reuse only accepted R0 evidence belonging to the winning candidate."""
    evidence = [
        v.geometry
        for v in verified
        if v.pair.candidate_index == candidate_index
        and v.pair.memory_id == "R0"
        and v.geometry.accepted
    ]
    return max(
        evidence,
        key=lambda g: (
            g.num_inliers >= config.memory_min_inliers
            and g.inlier_ratio >= config.memory_min_inlier_ratio,
            g.num_inliers,
            g.inlier_ratio,
        ),
        default=None,
    )


class MemoryAdmissionPolicy:
    def __init__(self, config):
        self.config = config

    def evaluate(self, prediction, entry, anchor_cosine):
        c = self.config
        evidence = {"anchor_cosine": anchor_cosine, "dino_cosine": prediction.dino_cosine}
        if prediction.status != "FOUND" or entry is None:
            evidence["failed_gate"] = "not_found_or_missing_entry"
            return "REJECTED_LOW_CONFIDENCE", evidence
        if (
            prediction.dino_cosine < c.memory_min_cosine
            or anchor_cosine is None
            or anchor_cosine < c.anchor_min_cosine
        ):
            evidence.update(
                failed_gate="similarity",
                required_cosine=c.memory_min_cosine,
                required_anchor_cosine=c.anchor_min_cosine,
            )
            return "REJECTED_LOW_CONFIDENCE", evidence
        anchor_geometry = prediction.anchor_geometry if c.use_verified_anchor else None
        if (
            c.anchor_only_admission
            and prediction.matched_memory_id != "R0"
            and (anchor_geometry is None or not anchor_geometry.accepted)
        ):
            evidence["failed_gate"] = "anchor_geometry_required"
            return "REJECTED_GEOMETRY", evidence
        if c.reject_border_boxes and prediction.bbox_xyxy is not None:
            x1, y1, x2, y2 = prediction.bbox_xyxy
            # Account for Florence's quantized image-border coordinates.
            mx, my = max(2, prediction.width * 0.002), max(2, prediction.height * 0.002)
            if x1 <= mx or y1 <= my or x2 >= prediction.width - mx or y2 >= prediction.height - my:
                evidence["failed_gate"] = "border_truncated_candidate"
                return "REJECTED_QUALITY", evidence
        # The pipeline supplies R0 evidence for the winning candidate only.
        # Do not borrow geometry from a different box or make an extra matcher call.
        g = anchor_geometry if c.anchor_only_admission and anchor_geometry else prediction.geometry
        evidence["geometry_reference"] = (
            "R0" if anchor_geometry and c.anchor_only_admission else prediction.matched_memory_id
        )
        if g:
            evidence.update(
                num_inliers=g.num_inliers,
                inlier_ratio=g.inlier_ratio,
                required_inliers=c.memory_min_inliers,
                required_inlier_ratio=c.memory_min_inlier_ratio,
            )
        if (
            not g
            or not g.accepted
            or g.num_inliers < c.memory_min_inliers
            or g.inlier_ratio < c.memory_min_inlier_ratio
        ):
            evidence["failed_gate"] = "geometry_support"
            return "REJECTED_GEOMETRY", evidence
        variance = float(
            cv2.Laplacian(cv2.cvtColor(entry.crop_image, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        )
        evidence.update(
            laplacian_variance=variance, num_inliers=g.num_inliers, inlier_ratio=g.inlier_ratio
        )
        if (
            min(entry.crop_image.shape[:2]) < c.min_crop_side
            or not np.isfinite(variance)
            or variance < c.min_laplacian_variance
        ):
            return "REJECTED_QUALITY", evidence
        return None, evidence
