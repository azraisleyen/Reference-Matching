from abc import ABC, abstractmethod
from dataclasses import replace
import numpy as np
from ..core.types import MemoryEvent
from ..features.normalization import normalize


class ReferenceMemory(ABC):
    def __init__(self, anchor, config):
        self.config = config
        self.anchor = self.freeze(anchor)
        self.entries = [self.anchor]
        self.events = [MemoryEvent(-1, None, "INITIALIZED", None, "R0", "pinned_anchor", 1)]
        self.pending = None

    @staticmethod
    def freeze(entry):
        embedding = normalize(np.asarray(entry.embedding)[None])[0].copy()
        image = entry.crop_image.copy()
        embedding.flags.writeable = False
        image.flags.writeable = False
        return replace(entry, embedding=embedding, crop_image=image)

    def active(self, frame_index):
        return tuple(e for e in self.entries if e.created_at_frame < frame_index)

    def drain_events(self):
        events, self.events = self.events, []
        return events

    def reset(self):
        self.entries = [self.anchor]
        self.pending = None
        self.events.append(MemoryEvent(-1, None, "RESET", None, None, "run_finished", 1))

    def usage(self):
        entries = self.entries + ([self.pending[0]] if self.pending else [])
        return dict(
            active_reference_count=len(self.entries),
            pending_count=int(self.pending is not None),
            image_bytes=sum(e.crop_image.nbytes for e in entries),
            embedding_bytes=sum(e.embedding.nbytes for e in entries),
        )

    @abstractmethod
    def update(self, prediction, entry=None, anchor_cosine=None): ...
