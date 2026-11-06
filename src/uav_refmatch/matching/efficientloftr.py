"""Official full outdoor implementation, isolated behind crop-coordinate outputs."""

from copy import deepcopy
import importlib
import os
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np
import torch
from ..core.types import LocalMatches
from ..config import Models
from ..outputs.recorder import file_hash


class EfficientLoFTRMatcher:
    def __init__(self, config: Models, device: str) -> None:
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "uav-refmatch"
        repo_value = config.eloftr_repo or os.getenv("ELOFTR_REPO") or str(cache / "EfficientLoFTR")
        checkpoint_value = (
            config.eloftr_checkpoint
            or os.getenv("ELOFTR_CHECKPOINT")
            or str(cache / "eloftr_outdoor.ckpt")
        )
        if not repo_value or not checkpoint_value:
            raise RuntimeError(
                "Set ELOFTR_REPO and ELOFTR_CHECKPOINT to the official repository and outdoor checkpoint"
            )
        repo, checkpoint = (
            Path(repo_value).expanduser().resolve(),
            Path(checkpoint_value).expanduser().resolve(),
        )
        if not (repo / "src/loftr/__init__.py").is_file() or not checkpoint.is_file():
            raise FileNotFoundError(
                "Official EfficientLoFTR source or outdoor checkpoint is missing"
            )
        self.commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if config.eloftr_commit and self.commit != config.eloftr_commit:
            raise RuntimeError("EfficientLoFTR commit differs from configured revision")
        if subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip():
            raise RuntimeError("EfficientLoFTR tracked source has unrecorded changes")
        sys.path.insert(0, str(repo))
        try:
            module = importlib.import_module("src.loftr")
            if not Path(module.__file__).resolve().is_relative_to(repo):
                raise RuntimeError("Conflicting src.loftr module in Python environment")
        finally:
            sys.path.remove(str(repo))
        cfg = deepcopy(module.full_default_cfg)
        cfg["mp"] = False
        cfg["half"] = config.precision == "fp16"
        matcher = module.LoFTR(config=cfg)
        self.checkpoint_sha256 = file_hash(checkpoint)
        # The official Lightning checkpoint contains ModelCheckpoint as a dictionary key.
        # Allow exactly that installed class only for the byte-verified official artifact.
        official_sha = "0af6291141c736e75e94b7f8aae4399b77c7731b3f08758212b2cfe370188878"
        if self.checkpoint_sha256 == official_sha:
            from pytorch_lightning.callbacks import ModelCheckpoint

            with torch.serialization.safe_globals([ModelCheckpoint]):
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        else:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        matcher.load_state_dict(state["state_dict"], strict=True)
        self.model = module.reparameter(matcher).eval().to(device)
        self.model.requires_grad_(False)
        # Upstream FineMatching otherwise enables CUDA autocast even for full fp32.
        # Its validation flag disables that context and only retains extra local matrices.
        self.model.fine_matching.validate = config.precision == "fp32"
        self.dtype = torch.float16 if config.precision == "fp16" else torch.float32
        self.model.to(dtype=self.dtype)
        self.device = device
        self.max_side = config.loftr_max_side

    def prepare(self, rgb):
        h, w = rgb.shape[:2]
        factor = min(1.0, self.max_side / max(h, w))
        new_w, new_h = max(32, int(w * factor) // 32 * 32), max(32, int(h * factor) // 32 * 32)
        gray = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (new_w, new_h))
        tensor = torch.from_numpy(gray.copy())[None, None].to(self.device, dtype=self.dtype) / 255.0
        return tensor, np.array([w / new_w, h / new_h], dtype=np.float32)

    def match(self, reference: np.ndarray, candidate: np.ndarray) -> LocalMatches:
        ref, ref_scale = self.prepare(reference)
        cand, cand_scale = self.prepare(candidate)
        batch = {"image0": ref, "image1": cand}
        with torch.inference_mode():
            self.model(batch)
        return LocalMatches(
            batch["mkpts0_f"].float().cpu().numpy() * ref_scale,
            batch["mkpts1_f"].float().cpu().numpy() * cand_scale,
            batch["mconf"].float().cpu().numpy(),
        )
