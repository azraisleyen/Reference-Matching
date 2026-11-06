from .base import ReferenceMemory


class StaticReferenceMemory(ReferenceMemory):
    def update(self, prediction, entry=None, anchor_cosine=None):
        return None
