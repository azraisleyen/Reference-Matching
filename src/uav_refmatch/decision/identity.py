"""Auditable gates, not an uncalibrated weighted confidence score."""

import numpy as np


from ..proposals.boxes import iou as box_iou


def identity_gate(pair, proposal, embedding, anchor, controller, estimate, reference=None):
    config = controller.config
    anchor_cosine = float(np.clip(embedding @ anchor.embedding, -1, 1))
    temporal = False
    similarity = None
    overlap = None
    if estimate is not None and estimate.reliable and controller.embedding is not None:
        overlap = box_iou(proposal.bbox_xyxy, estimate.bbox)
        similarity = float(np.clip(embedding @ controller.embedding, -1, 1))
        temporal = (
            overlap >= config.association_min_iou and similarity >= config.appearance_min_cosine
        )
    # Only an actual, previously admitted entry may supply bank identity evidence.
    # A high score tagged with an arbitrary memory ID is not sufficient.
    bank = bool(
        config.identity_policy == "verified_bank"
        and reference is not None
        and reference.memory_id == pair.memory_id
        and reference.origin == "dynamic"
        and reference.acceptance_evidence
        and pair.cosine >= config.verified_bank_min_cosine
    )
    accepted = temporal or bank or anchor_cosine >= config.acquisition_anchor_min_cosine
    return accepted, dict(
        candidate_index=pair.candidate_index,
        memory_id=pair.memory_id,
        anchor_cosine=anchor_cosine,
        temporal_cosine=similarity,
        association_iou=overlap,
        accepted=accepted,
        support="temporal"
        if temporal
        else "verified_bank"
        if bank
        else "anchor"
        if accepted
        else "none",
    )


def dense_identity_gate(pair, reference, config):
    """Dense geometry cannot bypass the trusted-bank identity floor.

    Called only for a geometrically accepted pair at decision time. Temporal
    similarity alone must never authorize a coarse match against a bank entry.
    """
    anchor = (
        reference is not None
        and reference.origin == "anchor"
        and pair.memory_id == "R0"
        and reference.memory_id == "R0"
    )
    bank = bool(
        config.identity_policy == "verified_bank"
        and reference is not None
        and reference.memory_id == pair.memory_id
        and reference.origin == "dynamic"
        and reference.acceptance_evidence
        and pair.cosine >= config.verified_bank_min_cosine
    )
    accepted = bool(anchor or bank)
    return accepted, dict(
        candidate_index=pair.candidate_index,
        memory_id=pair.memory_id,
        cosine=pair.cosine,
        accepted=accepted,
        support="dense_anchor"
        if anchor
        else "dense_verified_bank"
        if bank
        else "dense_bank_identity_rejected",
    )
