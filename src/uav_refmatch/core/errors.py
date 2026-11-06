"""Actionable persisted diagnostics without source paths, credentials or query strings."""

import re


def safe_error(exc):
    message = str(exc)
    message = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED_TOKEN]", message)
    message = re.sub(
        r"(?i)(authorization|token|password)([=: ]+)[^\s,]+", r"\1\2[REDACTED]", message
    )
    message = re.sub(r"https?://[^\s]+", "[UPSTREAM_URL]", message)
    message = re.sub(r"(?:/[\w.~-]+){2,}", "[LOCAL_PATH]", message)
    message = re.sub(r"[A-Za-z]:\\[^\s]+", "[LOCAL_PATH]", message)
    return f"{type(exc).__name__}: {message[:1600]}"
