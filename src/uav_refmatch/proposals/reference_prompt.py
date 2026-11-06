"""Deterministic caption shortening; no target-class dictionary or video access."""

import re


def subject_phrase(caption: str) -> str:
    """Keep the subject before common scene-description verbs, else keep caption.

    This is a deliberately small heuristic, not a general syntactic parser.
    The full caption and resulting phrase are always retained in provenance.
    """
    text = " ".join(caption.split()).strip(" .")
    if not text:
        raise ValueError("Florence returned an empty reference caption")
    return re.split(
        r"\s+(?:sitting|standing|parked|located|positioned|resting)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" .")
