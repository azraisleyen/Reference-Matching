"""One causal inference engine for CLI/UI and static/dynamic experiments."""

from dataclasses import asdict
from time import perf_counter
import numpy as np
from .core.errors import safe_error
from .core.types import Prediction, ReferenceEntry, VerifiedPair
from .proposals.boxes import crop
from .matching.retrieval import retrieve
from .decision.engine import choose
from .memory.dynamic import DynamicReferenceMemory
from .memory.static import StaticReferenceMemory
from .evaluation.profiler import Profiler


class Pipeline:
    def __init__(self, config, proposals, encoder, matcher, verifier, device="cpu"):
        self.config = config
        self.proposals = proposals
        self.encoder = encoder
        self.matcher = matcher
        self.verifier = verifier
        self.device = device

    def run(
        self,
        frames,
        reference_image,
        reference_id,
        recorder,
        cancel=lambda: False,
        progress=lambda p: None,
    ):
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
                        crops = [
                            crop(frame.image, p.bbox_xyxy, self.config.proposals.crop_padding)[0]
                            for p in proposals
                        ]
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
                    with profiler.measure("decision_ms"):
                        winner = choose(
                            verified, proposals, self.config.retrieval.detect_min_cosine
                        )
                        prediction.verified_pairs = [asdict(v) for v in verified]
                        if verified:
                            strongest = verified[0]
                            prediction.dino_cosine = strongest.pair.cosine
                            prediction.geometry = strongest.geometry
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
                    )
                )
                if prediction.status == "ERROR":
                    # Stop on model/runtime failures; do not manufacture subsequent NOT_FOUND records.
                    raise RuntimeError(prediction.decision_reason)
            return "completed"
        finally:
            memory.reset()
            recorder.events(memory.drain_events())
