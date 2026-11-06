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
            return "REJECTED_LOW_CONFIDENCE", evidence
        if (
            prediction.dino_cosine < c.memory_min_cosine
            or anchor_cosine is None
            or anchor_cosine < c.anchor_min_cosine
        ):
            return "REJECTED_LOW_CONFIDENCE", evidence
        g = prediction.geometry
        if (
            not g
            or not g.accepted
            or g.num_inliers < c.memory_min_inliers
            or g.inlier_ratio < c.memory_min_inlier_ratio
        ):
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
