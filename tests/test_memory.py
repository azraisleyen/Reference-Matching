from dataclasses import replace
import numpy as np
import pytest
from uav_refmatch.config import Memory
from uav_refmatch.memory.dynamic import DynamicReferenceMemory
from uav_refmatch.memory.static import StaticReferenceMemory


def advance(memory, prediction, entry, frame):
    memory.update(
        replace(prediction, frame_index=frame), replace(entry, created_at_frame=frame), 0.85
    )


def test_static_never_updates(anchor, found, entry):
    memory = StaticReferenceMemory(anchor, Memory())
    for i in range(10):
        advance(memory, found, entry, i)
    assert len(memory.entries) == 1 and memory.pending is None
    assert not memory.anchor.embedding.flags.writeable


def test_pending_causality(anchor, found, entry):
    memory = DynamicReferenceMemory(anchor, Memory())
    advance(memory, found, entry, 0)
    assert memory.pending and len(memory.entries) == 1
    advance(memory, found, entry, 1)
    assert len(memory.entries) == 2
    assert len(memory.active(1)) == 1 and len(memory.active(2)) == 2
    assert {"PENDING", "ACCEPTED"} <= {e.event_type for e in memory.events}


def test_capacity_fifo_pinned(anchor, found, entry):
    memory = DynamicReferenceMemory(
        anchor, Memory(capacity=2, confirmation_frames=1, cooldown_frames=0, duplicate_cosine=0.999)
    )
    for i, angle in enumerate([0.0, 1.0, 2.0, 3.0]):
        vector = np.array(
            [0.85, 0.5267827 * np.cos(angle), 0.5267827 * np.sin(angle)], dtype=np.float32
        )
        advance(memory, found, replace(entry, embedding=vector), i)
    assert [e.memory_id for e in memory.entries] == ["R0", "mem_0003", "mem_0004"]
    assert sum(e.event_type == "EVICTED" for e in memory.events) == 2


@pytest.mark.parametrize("status", ["NOT_FOUND", "ERROR"])
def test_not_found_clears_pending(anchor, found, entry, status):
    memory = DynamicReferenceMemory(anchor, Memory())
    advance(memory, found, entry, 0)
    memory.update(replace(found, status=status), None, None)
    assert memory.pending is None and len(memory.entries) == 1


def test_duplicate(anchor, found, entry):
    memory = DynamicReferenceMemory(anchor, Memory(confirmation_frames=1, cooldown_frames=0))
    advance(memory, found, entry, 0)
    advance(memory, found, entry, 1)
    assert len(memory.entries) == 2 and memory.events[-1].event_type == "REJECTED_DUPLICATE"


def test_anchor_gate_is_separate_from_found(anchor, found, entry):
    memory = DynamicReferenceMemory(anchor, Memory())
    memory.update(found, entry, 0.6)
    assert found.status == "FOUND" and len(memory.entries) == 1
    assert memory.events[-1].event_type == "REJECTED_LOW_CONFIDENCE"


def test_quality_gate(anchor, found, entry):
    memory = DynamicReferenceMemory(anchor, Memory())
    memory.update(found, replace(entry, crop_image=np.zeros((64, 64, 3), np.uint8)), 0.85)
    assert memory.events[-1].event_type == "REJECTED_QUALITY"


def test_geometry_gate(anchor, found, entry):
    memory = DynamicReferenceMemory(anchor, Memory())
    memory.update(replace(found, geometry=replace(found.geometry, num_inliers=8)), entry, 0.85)
    assert memory.events[-1].event_type == "REJECTED_GEOMETRY"


def test_cooldown_reset_new_video(anchor, found, entry):
    memory = DynamicReferenceMemory(anchor, Memory(confirmation_frames=1))
    advance(memory, found, entry, 0)
    advance(memory, found, entry, 1)
    assert memory.events[-1].event_type == "REJECTED_COOLDOWN"
    memory.reset()
    assert len(memory.entries) == 1 and memory.pending is None
    assert memory.events[-1].event_type == "RESET"
    other = DynamicReferenceMemory(anchor, Memory())
    assert len(other.entries) == 1


def test_pending_not_bbox_iou(anchor, found, entry):
    memory = DynamicReferenceMemory(anchor, Memory())
    advance(memory, found, entry, 0)
    advance(memory, replace(found, bbox_xyxy=(200, 200, 264, 264)), entry, 1)
    assert len(memory.entries) == 2


def test_pending_nonconsecutive_restarts(anchor, found, entry):
    memory = DynamicReferenceMemory(anchor, Memory())
    advance(memory, found, entry, 0)
    advance(memory, found, entry, 3)
    assert len(memory.entries) == 1 and memory.pending[1] == 1
