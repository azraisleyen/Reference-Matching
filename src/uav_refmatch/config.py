"""Validated, serializable experiment configuration (unknown keys are errors)."""

from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

LVD = "facebook/dinov3-vitb16-pretrain-lvd1689m"
SAT = "facebook/dinov3-vitl16-pretrain-sat493m"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Models(StrictModel):
    florence_id: Literal["microsoft/Florence-2-large-ft", "microsoft/Florence-2-base-ft"] = (
        "microsoft/Florence-2-large-ft"
    )
    florence_revision: str | None = None
    encoder_id: Literal[LVD, SAT] = LVD
    encoder_revision: str | None = None
    eloftr_repo: str | None = None
    eloftr_checkpoint: str | None = None
    eloftr_commit: str | None = "07e9c1401e71e9c556b1fda9d86d060af36ba176"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    precision: Literal["fp32", "fp16"] = "fp32"
    batch_size: int = Field(16, ge=1)
    max_new_tokens: int = Field(1024, ge=64)
    florence_num_beams: int = Field(3, ge=1, le=8)
    florence_use_cache: bool = True
    florence_precision: Literal["fp32", "fp16"] | None = None
    loftr_max_side: int = Field(640, ge=32)

    @model_validator(mode="after")
    def pin_default_revisions(self):
        pins = {
            "microsoft/Florence-2-large-ft": "4a12a2b54b7016a48a22037fbd62da90cd566f2a",
            "microsoft/Florence-2-base-ft": "f6c1a25888ffc1d945ee8a1a77ac833c7303d46e",
            LVD: "5931719e67bbdb9737e363e781fb0c67687896bc",
            SAT: "f692fa42da72c6797b67cd73494a168d1120d3ee",
        }
        if self.florence_revision is None:
            object.__setattr__(self, "florence_revision", pins[self.florence_id])
        if self.encoder_revision is None:
            object.__setattr__(self, "encoder_revision", pins[self.encoder_id])
        return self


class Proposals(StrictModel):
    max_proposals: int = Field(30, ge=1)
    nms_iou: float = Field(0.85, gt=0, le=1)
    crop_padding: float = Field(0.10, ge=0, le=1)
    min_box_side: int = Field(12, ge=1)
    tiling: Literal[False] = False


class Retrieval(StrictModel):
    top_k_candidates: int = Field(5, ge=1)
    max_geometry_pairs: int = Field(3, ge=1)
    detect_min_cosine: float = Field(0.70, ge=-1, le=1)
    geometry_min_cosine: float | None = Field(None, ge=-1, le=1)


class Geometry(StrictModel):
    model: Literal["partial_affine", "affine"] = "partial_affine"
    min_match_confidence: float = Field(0.0, ge=0, le=1)
    max_anisotropy: float = Field(10.0, ge=1)
    min_matches: int = Field(12, ge=3)
    min_inliers: int = Field(8, ge=3)
    min_inlier_ratio: float = Field(0.25, ge=0, le=1)
    ransac_reprojection_threshold_px: float = Field(3.0, gt=0)
    min_spatial_coverage: float = Field(0.08, ge=0, le=1)
    min_scale: float = Field(0.05, gt=0)
    max_scale: float = Field(20.0, gt=0)
    max_median_reprojection_error_px: float = Field(3.0, gt=0)


class Memory(StrictModel):
    capacity: int = Field(8, ge=1)
    memory_min_cosine: float = Field(0.80, ge=-1, le=1)
    memory_min_inliers: int = Field(12, ge=3)
    memory_min_inlier_ratio: float = Field(0.40, ge=0, le=1)
    duplicate_cosine: float = Field(0.96, ge=-1, le=1)
    cooldown_frames: int = Field(8, ge=0)
    confirmation_frames: int = Field(2, ge=1)
    anchor_min_cosine: float = Field(0.75, ge=-1, le=1)
    confirmation_cosine: float = Field(0.85, ge=-1, le=1)
    min_crop_side: int = Field(24, ge=1)
    min_laplacian_variance: float = Field(10.0, ge=0)
    max_pending: Literal[1] = 1


class Config(StrictModel):
    mode: Literal["static", "dynamic"] = "dynamic"
    target_fps: float = Field(7.5, gt=0)
    max_frames: int | None = Field(None, ge=1)
    seed: int = 42
    models: Models = Models()
    proposals: Proposals = Proposals()
    retrieval: Retrieval = Retrieval()
    geometry: Geometry = Geometry()
    memory: Memory = Memory()
    verified_cfr_fps: float | None = Field(None, gt=0)
    warmup: bool = True

    @model_validator(mode="after")
    def thresholds(self):
        if (
            self.retrieval.geometry_min_cosine is not None
            and self.retrieval.geometry_min_cosine > self.retrieval.detect_min_cosine
        ):
            raise ValueError("Geometry screening threshold must not exceed detection threshold")
        if self.memory.memory_min_cosine < self.retrieval.detect_min_cosine:
            raise ValueError("Memory threshold must be at least the detection threshold")
        if self.geometry.max_scale <= self.geometry.min_scale:
            raise ValueError("Invalid scale interval")
        if self.models.precision != "fp32" and self.models.device not in ("cuda", "auto"):
            raise ValueError("Mixed/half precision is only supported on CUDA")
        if self.models.florence_precision == "fp16" and self.models.device == "cpu":
            raise ValueError("Florence fp16 requires an explicitly available GPU")
        return self


def load_config(path: str | Path | None = None, **overrides) -> Config:
    data = yaml.safe_load(Path(path).read_text()) if path else {}
    data = data or {}
    for key, value in overrides.items():
        if value is not None:
            data[key] = value
    return Config.model_validate(data)
