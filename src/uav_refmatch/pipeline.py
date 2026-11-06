"""One causal inference engine for CLI/UI and static/dynamic experiments."""

from dataclasses import asdict
from collections.abc import Callable, Iterable
from .config import Config
from .core.interfaces import ProposalGenerator, FeatureEncoder, LocalMatcher, GeometricVerifier
from .outputs.recorder import Recorder
from time import perf_counter
import numpy as np
from .core.errors import safe_error
from .core.types import FrameRecord, Prediction, ReferenceEntry, VerifiedPair
from .proposals.boxes import crop
from .matching.retrieval import retrieve
from .matching.orientation import rotate_crop, restore_matches
from .decision.engine import choose
from .memory.dynamic import DynamicReferenceMemory
from .memory.static import StaticReferenceMemory
from .evaluation.profiler import Profiler


class Pipeline:
    def __init__(
        self,
        config: Config,
        proposals: ProposalGenerator,
        encoder: FeatureEncoder,
        matcher: LocalMatcher,
        verifier: GeometricVerifier,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.proposals = proposals
        self.encoder = encoder
        self.matcher = matcher
        self.verifier = verifier
        self.device = device

    def prepare_reference(self, image: np.ndarray) -> dict:
        prepare = getattr(self.proposals, "prepare_reference", None)
        return prepare(image) if prepare else {}

    def run(
        self,
        frames: Iterable[FrameRecord],
        reference_image: np.ndarray,
        reference_id: str,
        recorder: Recorder,
        cancel: Callable[[], bool] = lambda: False,
        progress: Callable[[dict], None] = lambda p: None,
    ) -> str:
        prepare_timer = perf_counter()
        recorder.manifest["proposal_reference_context"] = self.prepare_reference(reference_image)
        Profiler(self.device).sync()
        recorder.manifest.setdefault(
            "proposal_reference_prepare_ms", (perf_counter() - prepare_timer) * 1000
        )
        anchor_timer = perf_counter()
        anchor = ReferenceEntry(
            "R0",
            "anchor",
            None,
            None,
            None,
            reference_image,
            self.encoder.encode([reference_image])[0],
        )
        recorder.manifest["reference_encoding_ms"] = (perf_counter() - anchor_timer) * 1000
        memory_type = (
            DynamicReferenceMemory if self.config.mode == "dynamic" else StaticReferenceMemory
        )
        memory = memory_type(anchor, self.config.memory)
        recorder.events(memory.drain_events())
        counts = {"FOUND": 0, "NOT_FOUND": 0, "ERROR": 0}
        try:
            for frame in frames:
                if cancel():
                    return "cancelled"
                profiler = Profiler(self.device)
                profiler.values["decode_ms"] = frame.decode_ms
                start = perf_counter()
                refs = memory.active(frame.frame_index)
                lookup = {r.memory_id: r for r in refs}
                prediction = Prediction(
                    frame.frame_index,
                    frame.source_frame_index,
                    frame.timestamp_sec,
                    frame.width,
                    frame.height,
                    "NOT_FOUND",
                    reference_id,
                    active_reference_count=len(refs),
                )
                entry = None
                anchor_cosine = None
                try:
                    with profiler.measure("proposal_ms"):
                        proposals = self.proposals.propose(frame.image)
                        prepared = [
                            crop(frame.image, p.bbox_xyxy, self.config.proposals.crop_padding)
                            for p in proposals
                        ]
                        crops = [item[0] for item in prepared]
                        prediction.proposals = [
                            dict(candidate_index=i, **asdict(p), crop_bounds=prepared[i][1])
                            for i, p in enumerate(proposals)
                        ]
                        prediction.proposal_diagnostics = dict(
                            getattr(self.proposals, "last_diagnostics", {})
                        )
                    prediction.candidate_count = len(proposals)
                    verified = []
                    if crops:
                        with profiler.measure("dinov3_ms"):
                            embeddings = self.encoder.encode(crops)
                        with profiler.measure("retrieval_ms"):
                            retrieval = retrieve(embeddings, refs, self.config.retrieval)
                            pairs = retrieval.pairs
                            prediction.retrieval_candidates = [
                                asdict(pair) for pair in retrieval.candidate_best_matches
                            ]
                            if retrieval.candidate_best_matches:
                                prediction.dino_cosine = retrieval.candidate_best_matches[0].cosine
                        prediction.decision_reason = (
                            "no_visual_match" if not pairs else "geometry_rejected"
                        )
                        for pair in pairs:
                            ref = lookup[pair.memory_id]
                            with profiler.measure("efficientloftr_ms"):
                                matches = self.matcher.match(
                                    ref.crop_image, crops[pair.candidate_index]
                                )
                            with profiler.measure("ransac_ms"):
                                evidence = self.verifier.verify(
                                    matches, ref.crop_image.shape, crops[pair.candidate_index].shape
                                )
                            verified.append(VerifiedPair(pair, evidence))
                        # Original candidate pairs have priority. Retries spend the SAME
                        # total LoFTR-call budget, not an additional per-pair budget.
                        if not any(v.geometry.accepted for v in verified):
                            for degrees in self.config.geometry.candidate_rotation_fallbacks:
                                for pair in pairs:
                                    if len(verified) >= self.config.retrieval.max_geometry_pairs:
                                        break
                                    ref = lookup[pair.memory_id]
                                    original = crops[pair.candidate_index]
                                    with profiler.measure("efficientloftr_ms"):
                                        rotated = rotate_crop(original, degrees)
                                        matches = restore_matches(
                                            self.matcher.match(ref.crop_image, rotated),
                                            original.shape,
                                            rotated.shape,
                                            degrees,
                                        )
                                    with profiler.measure("ransac_ms"):
                                        evidence = self.verifier.verify(
                                            matches, ref.crop_image.shape, original.shape
                                        )
                                    verified.append(VerifiedPair(pair, evidence, degrees))
                                    if evidence.accepted:
                                        break
                                if any(v.geometry.accepted for v in verified):
                                    break
                    with profiler.measure("decision_ms"):
                        winner = choose(
                            verified, proposals, self.config.retrieval.detect_min_cosine
                        )
                        prediction.verified_pairs = [asdict(v) for v in verified]
                        if verified:
                            strongest = verified[0]
                            prediction.dino_cosine = strongest.pair.cosine
                            prediction.geometry = strongest.geometry
                        if not winner and any(v.geometry.accepted for v in verified):
                            prediction.decision_reason = "geometry_passed_below_visual_threshold"
                        if winner:
                            pair = winner.pair
                            index = pair.candidate_index
                            prediction.status = "FOUND"
                            prediction.bbox_xyxy = proposals[index].bbox_xyxy
                            prediction.matched_memory_id = pair.memory_id
                            prediction.dino_cosine = pair.cosine
                            prediction.geometry = winner.geometry
                            prediction.decision_reason = "visual_and_geometry_passed"
                            if self.config.mode == "dynamic":
                                anchor_cosine = float(
                                    np.clip(embeddings[index] @ anchor.embedding, -1, 1)
                                )
                                entry = ReferenceEntry(
                                    "",
                                    "dynamic",
                                    frame.source_frame_index,
                                    frame.timestamp_sec,
                                    prediction.bbox_xyxy,
                                    crops[index],
                                    embeddings[index],
                                    pair.memory_id,
                                    created_at_frame=frame.frame_index,
                                )
                except Exception as exc:
                    prediction.status = "ERROR"
                    prediction.bbox_xyxy = None
                    prediction.decision_reason = safe_error(exc)
                    recorder.log(
                        "frame_error", frame_index=frame.frame_index, error=safe_error(exc)
                    )
                # Prediction is committed before any memory update. Its duration excludes memory/recording.
                prediction.processing_ms = (perf_counter() - start) * 1000 + frame.decode_ms
                recorder.prediction(prediction)
                with profiler.measure("memory_ms"):
                    if self.config.mode == "dynamic":
                        memory.update(prediction, entry, anchor_cosine)
                recorder.events(memory.drain_events())
                profiler.values["total_frame_ms"] = (
                    perf_counter() - start
                ) * 1000 + frame.decode_ms
                recorder.csv(
                    "frame_timings.csv", dict(frame_index=frame.frame_index, **profiler.values)
                )
                recorder.csv(
                    "resource_usage.csv",
                    dict(frame_index=frame.frame_index, **profiler.resources(memory)),
                )
                counts[prediction.status] += 1
                recorder.save_manifest()
                progress(
                    dict(
                        processed_frames=frame.frame_index + 1,
                        counts=counts.copy(),
                        active_references=len(memory.entries),
                        last_frame_ms=profiler.values["total_frame_ms"],
                        device=self.device,
                        stage_timings_ms=profiler.values.copy(),
                        candidate_count=prediction.candidate_count,
                    )
                )
                if prediction.status == "ERROR":
                    # Stop on model/runtime failures; do not manufacture subsequent NOT_FOUND records.
                    raise RuntimeError(prediction.decision_reason)
            return "completed"
        finally:
            memory.reset()
            recorder.events(memory.drain_events())
