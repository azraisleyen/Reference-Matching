from dataclasses import replace
from .base import ReferenceMemory
from .policies import MemoryAdmissionPolicy
from ..core.types import MemoryEvent


class DynamicReferenceMemory(ReferenceMemory):
    def __init__(self, anchor, config):
        super().__init__(anchor, config)
        self.policy = MemoryAdmissionPolicy(config)
        self.last_accepted = -config.cooldown_frames
        self.next_id = 1

    def reset(self):
        super().reset()
        self.last_accepted = -self.config.cooldown_frames
        self.next_id = 1

    def update(self, prediction, entry=None, anchor_cosine=None):
        c = self.config

        def event(kind, reason, new_id=None, evidence=None):
            self.events.append(
                MemoryEvent(
                    prediction.frame_index,
                    prediction.timestamp_sec,
                    kind,
                    prediction.matched_memory_id,
                    new_id,
                    reason,
                    len(self.entries),
                    evidence or {},
                )
            )

        rejected, evidence = self.policy.evaluate(prediction, entry, anchor_cosine)
        if rejected:
            self.pending = None
            event(rejected, "admission_gate_failed", evidence=evidence)
            return
        if prediction.frame_index - self.last_accepted < c.cooldown_frames:
            self.pending = None
            event("REJECTED_COOLDOWN", "cooldown", evidence=evidence)
            return
        entry = self.freeze(entry)
        duplicate = max(float(entry.embedding @ ref.embedding) for ref in self.entries)
        if duplicate >= c.duplicate_cosine:
            self.pending = None
            event(
                "REJECTED_DUPLICATE",
                "embedding_duplicate",
                evidence={**evidence, "duplicate_cosine": duplicate},
            )
            return
        count = 1
        if self.pending:
            previous, previous_count = self.pending
            if (
                previous.created_at_frame == prediction.frame_index - 1
                and float(previous.embedding @ entry.embedding) >= c.confirmation_cosine
            ):
                count = previous_count + 1
        if count < c.confirmation_frames:
            self.pending = (entry, count)
            event(
                "PENDING",
                "awaiting_consecutive_confirmation",
                evidence={**evidence, "confirmation_count": count},
            )
            return
        self.pending = None
        if len(self.entries) >= c.capacity + 1:
            removed = self.entries.pop(1)
            self.events.append(
                MemoryEvent(
                    prediction.frame_index,
                    prediction.timestamp_sec,
                    "EVICTED",
                    removed.memory_id,
                    None,
                    "fifo_capacity",
                    len(self.entries),
                    {"evicted_created_at_frame": removed.created_at_frame},
                )
            )
        memory_id = f"mem_{self.next_id:04d}"
        self.next_id += 1
        entry = replace(
            entry,
            memory_id=memory_id,
            acceptance_evidence={**evidence, "confirmation_count": count},
            quality_metadata={"laplacian_variance": evidence["laplacian_variance"]},
        )
        self.entries.append(entry)
        self.last_accepted = prediction.frame_index
        event("ACCEPTED", "anchor_geometry_temporal_passed", memory_id, entry.acceptance_evidence)
