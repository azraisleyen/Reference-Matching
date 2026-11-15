"""One causal inference engine for CLI/UI and static/dynamic experiments."""

from dataclasses import asdict
from collections.abc import Callable, Iterable
from .config import Config
from .tracking.controller import TrackController
from .core.interfaces import ProposalGenerator, FeatureEncoder, LocalMatcher, GeometricVerifier
from .outputs.recorder import Recorder
from time import perf_counter
import numpy as np
from .core.errors import safe_error
from .core.types import FrameRecord, Prediction, ReferenceEntry, VerifiedPair
from .proposals.boxes import crop
from .matching.retrieval import retrieve
from .matching.orientation import rotate_crop, restore_matches
from .matching.scheduler import initial_pairs, retry_attempts
from .matching.dense import DenseVerificationBackend
from .decision.engine import choose
from .decision.identity import identity_gate, dense_identity_gate, box_iou
from .core.types import Proposal
from .memory.dynamic import DynamicReferenceMemory
from .memory.policies import select_anchor_geometry
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

    def reference_view(self, image: np.ndarray) -> np.ndarray:
        box = self.config.reference.bbox_xyxy
        if box is None:
            return image
        a, b, c, d = box
        if c > image.shape[1] or d > image.shape[0]:
            raise ValueError("Reference box exceeds image bounds")
        return image[b:d, a:c].copy()

    def prepare_reference(self, image: np.ndarray) -> dict:
        prepare = getattr(self.proposals, "prepare_reference", None)
        return prepare(self.reference_view(image)) if prepare else {}

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
        reference_image = self.reference_view(reference_image)
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
        dense = (
            DenseVerificationBackend(
                self.encoder, self.config.patch_verification, self.config.geometry
            )
            if self.config.patch_verification.enabled
            else None
        )
        counts = {"FOUND": 0, "TRACKED": 0, "NOT_FOUND": 0, "ERROR": 0}
        track = TrackController(self.config.tracking) if self.config.tracking.enabled else None
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
                    propagated = False
                    estimate = None
                    track_embedding = None
                    if track:
                        with profiler.measure("tracking_ms"):
                            estimate = track.predict(frame.image, frame.timestamp_sec)
                            if estimate.reliable:
                                track_crop, _ = crop(
                                    frame.image, estimate.bbox, self.config.proposals.crop_padding
                                )
                                track_embedding = self.encoder.encode([track_crop])[0]
                                propagated, similarity = track.can_propagate(
                                    estimate, track_embedding, frame.timestamp_sec
                                )
                                prediction.tracking["appearance_cosine"] = similarity
                                # Complete independent consecutive identity checks before
                                # allowing propagation; tracking must not starve bank admission.
                                if self.config.tracking.mode == "verify":
                                    propagated = False
                                    prediction.tracking["verification_trigger"] = (
                                        "verify_each_frame"
                                    )
                                if memory.pending is not None:
                                    propagated = False
                                    prediction.tracking["verification_trigger"] = (
                                        "memory_confirmation"
                                    )
                            if propagated:
                                prediction.status = "TRACKED"
                                prediction.bbox_xyxy = estimate.bbox
                                prediction.decision_reason = (
                                    "temporal_propagation_not_identity_detection"
                                )
                    if track:
                        prediction.tracking["state_before_verification"] = track.state
                        prediction.tracking["verification_trigger"] = prediction.tracking.get(
                            "verification_trigger", track.last_reason
                        )
                    if not propagated:
                        with profiler.measure("proposal_ms"):
                            proposals = list(self.proposals.propose(frame.image))
                            if (
                                estimate
                                and estimate.reliable
                                and len(proposals) < self.config.proposals.max_proposals
                                and not any(
                                    box_iou(p.bbox_xyxy, estimate.bbox)
                                    >= self.config.tracking.proposal_duplicate_iou
                                    for p in proposals
                                )
                            ):
                                proposals.append(
                                    Proposal(
                                        max((p.proposal_id for p in proposals), default=-1) + 1,
                                        estimate.bbox,
                                        "temporal_motion_proposal",
                                    )
                                )
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
                                    prediction.dino_cosine = retrieval.candidate_best_matches[
                                        0
                                    ].cosine
                            prediction.decision_reason = (
                                "no_visual_match" if not pairs else "geometry_rejected"
                            )

                            def admissible(item):
                                if (
                                    not item.geometry.accepted
                                    or item.pair.cosine < self.config.retrieval.detect_min_cosine
                                ):
                                    return False
                                if item.method == "dino_patch":
                                    return dense_identity_gate(
                                        item.pair, lookup[item.pair.memory_id], self.config.tracking
                                    )[0]
                                if track is None:
                                    return True
                                idx = item.pair.candidate_index
                                return identity_gate(
                                    item.pair,
                                    proposals[idx],
                                    embeddings[idx],
                                    anchor,
                                    track,
                                    estimate,
                                    lookup[item.pair.memory_id],
                                )[0]

                            for pair in initial_pairs(
                                pairs, self.config.retrieval, self.config.geometry
                            ):
                                ref = lookup[pair.memory_id]
                                with profiler.measure("efficientloftr_ms"):
                                    matches = self.matcher.match(
                                        ref.crop_image, crops[pair.candidate_index]
                                    )
                                with profiler.measure("ransac_ms"):
                                    evidence = self.verifier.verify(
                                        matches,
                                        ref.crop_image.shape,
                                        crops[pair.candidate_index].shape,
                                    )
                                verified.append(VerifiedPair(pair, evidence))
                            if dense is not None and not any(admissible(v) for v in verified):
                                # R0 evidence helps acquisition; bank views remain available.
                                dense_pairs = sorted(
                                    (
                                        p
                                        for p in pairs
                                        if p.cosine >= self.config.retrieval.detect_min_cosine
                                        and dense_identity_gate(
                                            p, lookup[p.memory_id], self.config.tracking
                                        )[0]
                                    ),
                                    key=lambda p: (p.memory_id != "R0", -p.cosine),
                                )
                                for pair in dense_pairs:
                                    if len(verified) >= self.config.retrieval.max_geometry_pairs:
                                        break
                                    with profiler.measure("dense_matching_ms"):
                                        evidence = dense.verify(
                                            lookup[pair.memory_id].crop_image,
                                            crops[pair.candidate_index],
                                        )
                                    verified.append(
                                        VerifiedPair(pair, evidence, method="dino_patch")
                                    )
                                    if admissible(verified[-1]):
                                        break
                            # Original candidate pairs have priority. Retries spend the SAME
                            # total LoFTR-call budget, not an additional per-pair budget.
                            if not any(admissible(v) for v in verified):
                                attempts = retry_attempts(verified, self.config.geometry)
                                for pair, degrees in attempts:
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
                                    if admissible(verified[-1]):
                                        break
                        with profiler.measure("decision_ms"):
                            eligible = [
                                v
                                for v in verified
                                if v.method != "dino_patch"
                                or dense_identity_gate(
                                    v.pair, lookup[v.pair.memory_id], self.config.tracking
                                )[0]
                            ]
                            if track:
                                eligible = []
                                identity_evidence = []
                                for item in verified:
                                    if not item.geometry.accepted:
                                        continue
                                    idx = item.pair.candidate_index
                                    accepted, evidence = identity_gate(
                                        item.pair,
                                        proposals[idx],
                                        embeddings[idx],
                                        anchor,
                                        track,
                                        estimate,
                                        lookup[item.pair.memory_id],
                                    )
                                    if item.method == "dino_patch":
                                        accepted, dense_evidence = dense_identity_gate(
                                            item.pair,
                                            lookup[item.pair.memory_id],
                                            self.config.tracking,
                                        )
                                        evidence.update(dense_evidence)
                                    identity_evidence.append(evidence)
                                    if accepted:
                                        eligible.append(item)
                                prediction.tracking["identity_checks"] = identity_evidence
                            winner = choose(
                                eligible, proposals, self.config.retrieval.detect_min_cosine
                            )
                            prediction.verified_pairs = [asdict(v) for v in verified]
                            if verified:
                                strongest = verified[0]
                                prediction.dino_cosine = strongest.pair.cosine
                                prediction.geometry = strongest.geometry
                            if not winner and any(v.geometry.accepted for v in verified):
                                prediction.decision_reason = (
                                    "geometry_passed_below_visual_threshold"
                                )
                            if (
                                track
                                and not eligible
                                and any(v.geometry.accepted for v in verified)
                            ):
                                prediction.decision_reason = "identity_unconfirmed"
                            if winner:
                                pair = winner.pair
                                index = pair.candidate_index
                                prediction.status = "FOUND"
                                prediction.bbox_xyxy = proposals[index].bbox_xyxy
                                prediction.matched_memory_id = pair.memory_id
                                prediction.dino_cosine = pair.cosine
                                prediction.geometry = winner.geometry
                                prediction.anchor_geometry = select_anchor_geometry(
                                    verified, index, self.config.memory
                                )
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
                    if track:
                        if prediction.status == "FOUND":
                            track.confirm(
                                frame.image,
                                prediction.bbox_xyxy,
                                embeddings[index],
                                frame.timestamp_sec,
                            )
                        elif prediction.status != "TRACKED":
                            conflicting = prediction.decision_reason == "identity_unconfirmed"
                            if (
                                self.config.tracking.mode == "propagate"
                                and estimate
                                and track.defer(
                                    estimate, track_embedding, frame.timestamp_sec, conflicting
                                )
                            ):
                                prediction.status = "TRACKED"
                                prediction.bbox_xyxy = estimate.bbox
                                prediction.tracking["verification_failure"] = (
                                    prediction.decision_reason
                                )
                                prediction.decision_reason = (
                                    "inconclusive_verification_bounded_hold"
                                )
                            else:
                                track.reject()
                        prediction.tracking.update(
                            state=track.state,
                            reason=track.last_reason,
                            last_verified_sec=track.verified_at,
                            last_verification_attempt_sec=track.last_attempt_at,
                            unverified_age_sec=(
                                frame.timestamp_sec - track.verified_at
                                if track.verified_at is not None
                                else None
                            ),
                            diagnostics=estimate.diagnostics if estimate else {},
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
                    dict(
                        frame_index=frame.frame_index,
                        **profiler.resources(memory),
                        tracking_array_bytes=track.array_bytes() if track else 0,
                    ),
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
