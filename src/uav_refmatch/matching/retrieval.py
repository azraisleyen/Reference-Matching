from dataclasses import dataclass
import numpy as np
from collections.abc import Sequence
from ..config import Retrieval
from ..core.types import Pair, ReferenceEntry
from ..features.normalization import normalize


@dataclass(frozen=True)
class RetrievalResult:
    pairs: list[Pair]
    candidate_best_matches: list[Pair]


def retrieve(
    embeddings: np.ndarray, references: Sequence[ReferenceEntry], config: Retrieval
) -> RetrievalResult:
    if not len(embeddings):
        return RetrievalResult([], [])
    scores = np.einsum(
        "id,jd->ij",
        normalize(embeddings),
        normalize(np.stack([r.embedding for r in references])),
        optimize=False,
    )
    # Keep below-threshold scores for diagnosis/calibration; they never bypass the gate.
    best = []
    for i in range(len(scores)):
        j = min(
            range(len(references)), key=lambda j: (-float(scores[i, j]), references[j].memory_id)
        )
        best.append(Pair(i, references[j].memory_id, float(np.clip(scores[i, j], -1, 1))))
    best.sort(key=lambda pair: (-pair.cosine, pair.candidate_index, pair.memory_id))
    candidates = [pair.candidate_index for pair in best[: config.top_k_candidates]]
    pairs = [
        Pair(i, ref.memory_id, float(np.clip(scores[i, j], -1, 1)))
        for i in candidates
        for j, ref in enumerate(references)
        if scores[i, j]
        >= (
            config.geometry_min_cosine
            if config.geometry_min_cosine is not None
            else config.detect_min_cosine
        )
    ]
    pairs.sort(key=lambda pair: (-pair.cosine, pair.candidate_index, pair.memory_id))
    selected = pairs[: config.max_geometry_pairs]
    if config.reserve_anchor_pair and not any(p.memory_id == "R0" for p in selected):
        anchor = next((p for p in pairs if p.memory_id == "R0"), None)
        if anchor is not None:
            selected = selected[: max(0, config.max_geometry_pairs - 1)] + [anchor]
    return RetrievalResult(selected, best)
