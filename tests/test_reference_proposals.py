from types import SimpleNamespace

import numpy as np
import pytest

from uav_refmatch.config import Proposals
from uav_refmatch.proposals.florence2 import Florence2Proposals
from uav_refmatch.proposals.reference_prompt import subject_phrase


def adapter(task, generate):
    obj = object.__new__(Florence2Proposals)
    obj.config = Proposals(task=task)
    obj.model_config = SimpleNamespace(florence_id="test")
    obj.reference_context = {}
    obj._generate = generate
    return obj


def test_reference_caption_is_cached_but_changes_with_reference():
    calls = []

    def generate(image, task):
        calls.append((image.copy(), task))
        return {"<CAPTION>": "A blue vehicle parked beside a building."}, {}

    obj = adapter("reference_grounding", generate)
    reference = np.zeros((20, 20, 3), np.uint8)
    first = obj.prepare_reference(reference)
    assert first["grounding_text"] == "A blue vehicle"
    assert obj.prepare_reference(reference.copy()) == first
    assert len(calls) == 1
    obj.prepare_reference(reference + 1)
    assert len(calls) == 2
    assert all(task == "<CAPTION>" for _, task in calls)


def test_grounding_uses_reference_phrase_and_original_image_dimensions():
    calls = []

    def generate(image, task, text):
        calls.append((image.shape, task, text))
        return {task: {"bboxes": [[-5, 10, 45, 60]], "labels": [text]}}, {"task": task}

    obj = adapter("reference_grounding", generate)
    with pytest.raises(RuntimeError, match="Prepare"):
        obj.propose(np.zeros((80, 120, 3), np.uint8))
    obj.reference_context = {"grounding_text": "A blue vehicle"}
    boxes = obj.propose(np.zeros((80, 120, 3), np.uint8))
    assert calls == [((80, 120, 3), "<CAPTION_TO_PHRASE_GROUNDING>", "A blue vehicle")]
    assert boxes[0].bbox_xyxy == (0, 10, 45, 60)
    assert obj.last_diagnostics["raw_labels"] == ["A blue vehicle"]


@pytest.mark.parametrize("task", ["<OD>", "<REGION_PROPOSAL>"])
def test_unconditioned_tasks_need_no_reference_caption(task):
    def generate(image, used_task, text):
        assert used_task == task and text == ""
        return {task: {"bboxes": []}}, {"task": task}

    obj = adapter(task, generate)
    assert obj.prepare_reference(np.zeros((20, 20, 3), np.uint8)) == {}
    assert obj.propose(np.zeros((80, 120, 3), np.uint8)) == []


def test_subject_phrase_has_no_target_class_dictionary():
    assert subject_phrase("  A red suitcase sitting on a floor. ") == "A red suitcase"
    assert subject_phrase("A person riding a bicycle.") == "A person riding a bicycle"
    with pytest.raises(ValueError, match="empty"):
        subject_phrase("  . ")


def test_rotation_matches_return_to_original_coordinates_and_discard_padding():
    from uav_refmatch.core.types import LocalMatches
    from uav_refmatch.matching.orientation import restore_matches

    original = np.array([[10.0, 12.0], [30.0, 25.0]])
    angle = np.deg2rad(-45)
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    old_center = (np.array([60, 40]) - 1) / 2
    new_center = (np.array([72, 72]) - 1) / 2
    rotated = (original - old_center) @ rotation.T + new_center
    matches = LocalMatches(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        np.vstack([rotated, [-100.0, -100.0]]),
        np.array([0.8, 0.9, 0.99]),
    )
    restored = restore_matches(matches, (40, 60, 3), (72, 72, 3), -45)
    np.testing.assert_allclose(restored.candidate_points, original, atol=1e-10)
    np.testing.assert_allclose(restored.confidence, [0.8, 0.9])
    assert len(restored.reference_points) == 2


def test_subject_v2_removes_scene_locative_without_object_dictionary():
    assert (
        subject_phrase("A soccer goal post on a grassy field.", "caption_subject_v2")
        == "A soccer goal post"
    )
    assert subject_phrase("A red car near the building.", "caption_subject_v2") == "A red car"
    assert (
        subject_phrase("A person riding a bicycle.", "caption_subject_v2")
        == "A person riding a bicycle"
    )
    assert subject_phrase("A goal on a field.") == "A goal on a field"


def test_anchor_retry_schedule_preserves_both_anchor_orientations():
    from uav_refmatch.core.types import Pair
    from uav_refmatch.matching.orientation import rotation_attempts

    pairs = [Pair(0, "mem_1", 0.9), Pair(0, "R0", 0.6)]
    attempts = rotation_attempts(pairs, [-45, 45], True)
    assert [(p.memory_id, a) for p, a in attempts[:2]] == [("R0", -45), ("R0", 45)]
    assert rotation_attempts(pairs, [-45, 45], False)[0][0].memory_id == "mem_1"
