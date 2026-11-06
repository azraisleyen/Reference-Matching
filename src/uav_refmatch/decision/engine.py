import numpy as np
from ..core.types import VerifiedPair, Proposal


def choose(
    verified_pairs: list[VerifiedPair], proposals: list[Proposal], threshold: float
) -> VerifiedPair | None:
    accepted = []
    for item in verified_pairs:
        b = proposals[item.pair.candidate_index].bbox_xyxy
        if (
            item.pair.cosine >= threshold
            and item.geometry.accepted
            and np.isfinite(b).all()
            and b[2] > b[0]
            and b[3] > b[1]
        ):
            accepted.append(item)
    return min(
        accepted,
        key=lambda x: (
            -x.pair.cosine,
            -x.geometry.num_inliers,
            x.pair.candidate_index,
            x.pair.memory_id,
        ),
        default=None,
    )
