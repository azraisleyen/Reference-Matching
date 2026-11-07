"""Bounded orientation retries; correspondences return to original crop pixels."""

import numpy as np
from PIL import Image

from ..core.types import LocalMatches


def rotate_crop(image: np.ndarray, degrees: int) -> np.ndarray:
    return np.array(
        Image.fromarray(image).rotate(degrees, resample=Image.Resampling.BILINEAR, expand=True)
    )


def restore_matches(matches: LocalMatches, original_shape, rotated_shape, degrees):
    angle = np.deg2rad(degrees)
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    old_center = (np.array(original_shape[:2][::-1], dtype=float) - 1) / 2
    new_center = (np.array(rotated_shape[:2][::-1], dtype=float) - 1) / 2
    points = (matches.candidate_points - new_center) @ rotation + old_center
    h, w = original_shape[:2]
    inside = (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
    # Expanded black padding is not image evidence. Never clip its matches into the object.
    return LocalMatches(
        matches.reference_points[inside], points[inside], matches.confidence[inside]
    )


def rotation_attempts(pairs, degrees, anchor_first=False):
    if anchor_first:
        # Preserve the anchor's orientation rescue before spending retries on memory.
        ranked = sorted(pairs, key=lambda p: p.memory_id != "R0")
        return [(pair, angle) for pair in ranked for angle in degrees]
    return [(pair, angle) for angle in degrees for pair in pairs]
