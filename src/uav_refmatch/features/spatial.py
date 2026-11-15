"""Patch coordinates stay in the original crop's coordinate system."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpatialFeatures:
    descriptors: np.ndarray
    grid: tuple[int, int]
    image_shape: tuple[int, int]

    def __post_init__(self):
        if min(*self.grid, *self.image_shape) <= 0 or self.descriptors.ndim != 2:
            raise ValueError("Invalid spatial feature geometry")
        if (
            len(self.descriptors) != self.grid[0] * self.grid[1]
            or not np.isfinite(self.descriptors).all()
        ):
            raise ValueError("Patch count or finite descriptor check failed")

    def points(self, indices):
        h, w = self.grid
        ih, iw = self.image_shape
        return np.column_stack(
            ((indices % w + 0.5) * iw / w, (indices // w + 0.5) * ih / h)
        ).astype(np.float32)
