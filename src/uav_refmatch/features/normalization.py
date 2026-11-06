import numpy as np


def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    if (
        values.ndim != 2
        or not values.shape[0]
        or not values.shape[1]
        or not np.isfinite(values).all()
    ):
        raise ValueError("Embeddings must be a finite nonempty [batch, dimension] matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Zero-norm embedding")
    return values / norms
