"""Anchor-controlled admission without additional, hidden geometry calls."""

import cv2
import numpy as np


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
        if c.anchor_only_admission and prediction.matched_memory_id != "R0":
            evidence["failed_gate"] = "anchor_geometry_required"
            return "REJECTED_GEOMETRY", evidence
        if c.reject_border_boxes and prediction.bbox_xyxy is not None:
            x1, y1, x2, y2 = prediction.bbox_xyxy
            # Account for Florence's quantized image-border coordinates.
            mx, my = max(2, prediction.width * 0.002), max(2, prediction.height * 0.002)
            if x1 <= mx or y1 <= my or x2 >= prediction.width - mx or y2 >= prediction.height - my:
                evidence["failed_gate"] = "border_truncated_candidate"
                return "REJECTED_QUALITY", evidence
        g = prediction.geometry
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
