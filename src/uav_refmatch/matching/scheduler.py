"""Bounded verification planning; policies do not change acceptance thresholds."""

from .orientation import rotation_attempts


def initial_pairs(pairs, retrieval, geometry):
    if not geometry.candidate_rotation_fallbacks or not geometry.reserved_rotation_calls:
        return pairs
    limit = max(1, retrieval.max_geometry_pairs - geometry.reserved_rotation_calls)
    # Cover distinct candidates before spending more calls on their memory views.
    selected = []
    seen = set()
    for pair in pairs:
        if pair.candidate_index not in seen:
            selected.append(pair)
            seen.add(pair.candidate_index)
    selected.extend(p for p in pairs if p not in selected)
    selected = selected[:limit]
    if retrieval.reserve_anchor_pair and not any(p.memory_id == "R0" for p in selected):
        anchor = next((p for p in pairs if p.memory_id == "R0"), None)
        if anchor is not None and limit > 1:
            selected[-1] = anchor
    return selected


def retry_attempts(verified, geometry):
    pairs = [v.pair for v in verified]
    if geometry.rotation_retry_policy == "legacy":
        return rotation_attempts(
            pairs, geometry.candidate_rotation_fallbacks, geometry.rotation_retry_anchor_first
        )
    if geometry.rotation_retry_policy == "cosine":
        ranked = sorted(
            verified, key=lambda v: (-v.pair.cosine, v.pair.candidate_index, v.pair.memory_id)
        )
    else:
        ranked = sorted(
            verified,
            key=lambda v: (
                -int(v.geometry.transform_valid),
                -v.geometry.num_inliers,
                -v.geometry.inlier_ratio,
                -v.pair.cosine,
                v.pair.candidate_index,
                v.pair.memory_id,
            ),
        )
    return [(v.pair, angle) for v in ranked for angle in geometry.candidate_rotation_fallbacks]
