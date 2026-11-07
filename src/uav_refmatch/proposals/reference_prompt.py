"""Deterministic caption shortening; no target-class dictionary or video access."""

import re


def subject_phrase(caption: str, policy: str = "caption_subject_v1") -> str:
    """Keep the subject before common scene-description verbs, else keep caption.

    This is a deliberately small heuristic, not a general syntactic parser.
    The full caption and resulting phrase are always retained in provenance.
    """
    text = " ".join(caption.split()).strip(" .")
    if not text:
        raise ValueError("Florence returned an empty reference caption")
    phrase = re.split(
        r"\s+(?:sitting|standing|parked|located|positioned|resting)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" .")

    if policy == "caption_subject_v2":
        phrase = re.split(
            r"\s+(?:on|in|near|beside|against)\s+(?:a|an|the)\s+",
            phrase,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
    elif policy != "caption_subject_v1":
        raise ValueError("Unknown reference prompt policy")
    return phrase
