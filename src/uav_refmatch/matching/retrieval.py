import numpy as np
from ..core.types import Pair
from ..features.normalization import normalize


def retrieve(embeddings, references, config):
    if not len(embeddings):
        return []
    scores = normalize(embeddings) @ normalize(np.stack([r.embedding for r in references])).T
    # Top K candidates by their best reference; all eligible reference pairs share one budget.
    candidates = sorted(range(len(scores)), key=lambda i: (-float(scores[i].max()), i))[
        : config.top_k_candidates
    ]
    pairs = [
        Pair(i, ref.memory_id, float(np.clip(scores[i, j], -1, 1)))
        for i in candidates
        for j, ref in enumerate(references)
        if scores[i, j] >= config.detect_min_cosine
    ]
    return sorted(pairs, key=lambda p: (-p.cosine, p.candidate_index, p.memory_id))[
        : config.max_geometry_pairs
    ]
