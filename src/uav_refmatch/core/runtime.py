"""Shared hardware policy and immutable upstream revisions."""

import os
import re
from pathlib import Path
import torch
from huggingface_hub import HfApi


def resolve_revision(model_id, revision):
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    # Resolve before any remote code executes; record exact commit in the manifest.
    return HfApi().model_info(model_id, revision=revision).sha


def device_for(config):
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if config.precision != "fp32" and device != "cuda":
        raise RuntimeError("fp16 requires CUDA; select fp32 on CPU/MPS")
    return device


def lock_path():
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "uav-refmatch"
    root.mkdir(parents=True, exist_ok=True)
    return root / "inference.lock"
