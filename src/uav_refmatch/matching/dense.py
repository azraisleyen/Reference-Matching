"""Optional frozen DINO patch verification; no new weights and no hidden LoFTR calls."""

from collections import OrderedDict
from dataclasses import replace
from hashlib import sha256
import math
import torch
from ..core.types import LocalMatches
from ..geometry.verifier import AffineRansacVerifier


def mutual_matches(reference, candidate):
    # Torch avoids platform-specific NumPy/Accelerate floating-status warnings.
    a = torch.from_numpy(reference.descriptors.copy())
    b = torch.from_numpy(candidate.descriptors.copy())
    with torch.inference_mode():
        scores = a @ b.T
        if not torch.isfinite(scores).all():
            raise ValueError("Nonfinite patch similarity")
        forward = scores.argmax(1)
        backward = scores.argmax(0)
        indices = torch.arange(len(a))
        keep = backward[forward] == indices
        ri, ci = indices[keep].numpy(), forward[keep].numpy()
        confidence = scores[indices[keep], forward[keep]].clamp(-1, 1).numpy()
    return LocalMatches(reference.points(ri), candidate.points(ci), confidence)


class DenseVerificationBackend:
    def __init__(self, encoder, config, geometry):
        if not callable(getattr(encoder, "encode_spatial", None)):
            raise ValueError("Patch verification requires a spatial feature encoder")
        self.encoder, self.config, self.geometry = encoder, config, geometry
        self.cache = OrderedDict()

    def features(self, image):
        key = (image.shape, image.dtype.str, sha256(image.tobytes()).digest())
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        result = self.encoder.encode_spatial(image, self.config.image_size)
        self.cache[key] = result
        while len(self.cache) > self.config.cache_entries:
            self.cache.popitem(last=False)
        return result

    def verify(self, reference, candidate):
        matches = mutual_matches(self.features(reference), self.features(candidate))
        tolerance = self.config.reprojection_fraction * math.hypot(*candidate.shape[:2])
        config = self.geometry.model_copy(
            update={
                "min_match_confidence": self.config.min_cosine,
                "ransac_reprojection_threshold_px": tolerance,
                "max_median_reprojection_error_px": tolerance,
            }
        )
        evidence = AffineRansacVerifier(config).verify(matches, reference.shape, candidate.shape)
        return replace(
            evidence,
            diagnostics={
                **evidence.diagnostics,
                "backend": "dino_patch",
                "input_size": self.config.image_size,
                "effective_reprojection_px": tolerance,
                "min_patch_cosine": self.config.min_cosine,
                "cache_bytes": sum(v.descriptors.nbytes for v in self.cache.values()),
            },
        )
