from dataclasses import dataclass, field
from typing import Literal
import numpy as np

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class FrameRecord:
    frame_index: int
    source_frame_index: int
    timestamp_sec: float
    width: int
    height: int
    image: np.ndarray = field(repr=False)
    decode_ms: float = 0.0


@dataclass(frozen=True)
class Proposal:
    proposal_id: int
    bbox_xyxy: BBox
    source_model: str


@dataclass(frozen=True)
class LocalMatches:
    reference_points: np.ndarray
    candidate_points: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class GeometryEvidence:
    num_matches: int = 0
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    median_reprojection_error_px: float | None = None
    spatial_coverage: float = 0.0
    transform_valid: bool = False
    accepted: bool = False
    reason: str = "insufficient_matches"


@dataclass(frozen=True)
class Pair:
    candidate_index: int
    memory_id: str
    cosine: float


@dataclass(frozen=True)
class VerifiedPair:
    pair: Pair
    geometry: GeometryEvidence


@dataclass
class Prediction:
    frame_index: int
    source_frame_index: int
    timestamp_sec: float
    width: int
    height: int
    status: Literal["FOUND", "NOT_FOUND", "ERROR"]
    target_reference_id: str
    bbox_xyxy: BBox | None = None
    matched_memory_id: str | None = None
    dino_cosine: float | None = None
    geometry: GeometryEvidence | None = None
    candidate_count: int = 0
    active_reference_count: int = 1
    decision_reason: str = "no_candidates"
    processing_ms: float = 0.0
    verified_pairs: list[dict] = field(default_factory=list)
    retrieval_candidates: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class ReferenceEntry:
    memory_id: str
    origin: Literal["anchor", "dynamic"]
    source_frame_index: int | None
    source_timestamp_sec: float | None
    source_bbox_xyxy: BBox | None
    crop_image: np.ndarray = field(repr=False)
    embedding: np.ndarray = field(repr=False)
    matched_parent_id: str | None = None
    acceptance_evidence: dict = field(default_factory=dict)
    created_at_frame: int = -1
    quality_metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryEvent:
    frame_index: int
    timestamp_sec: float | None
    event_type: str
    source_reference: str | None
    new_reference_id: str | None
    reason: str
    memory_size: int
    evidence: dict = field(default_factory=dict)
