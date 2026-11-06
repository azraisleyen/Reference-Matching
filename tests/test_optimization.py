from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from uav_refmatch.config import Geometry, Models, Retrieval
from uav_refmatch.core.types import LocalMatches, ReferenceEntry
from uav_refmatch.geometry.verifier import AffineRansacVerifier
from uav_refmatch.matching.retrieval import retrieve
from uav_refmatch.proposals.cache_compat import enable_legacy_cache


def test_legacy_cache_bridge_is_instance_local():
    class Florence2LanguageForConditionalGeneration:
        @classmethod
        def _supports_default_dynamic_cache(cls):
            return True

    one, two = (
        Florence2LanguageForConditionalGeneration(),
        Florence2LanguageForConditionalGeneration(),
    )
    enable_legacy_cache(SimpleNamespace(language_model=one))
    assert not one._supports_default_dynamic_cache()
    assert two._supports_default_dynamic_cache()


def test_cache_bridge_rejects_unknown_model():
    with pytest.raises(RuntimeError, match="Unexpected Florence"):
        enable_legacy_cache(SimpleNamespace(language_model=object()))


def test_geometry_gate_can_diagnose_below_detection_threshold():
    anchor = ReferenceEntry(
        "R0", "anchor", None, None, None, np.zeros((8, 8, 3), np.uint8), np.array([1.0, 0.0])
    )
    embeddings = np.array([[0.4, np.sqrt(0.84)]])
    assert not retrieve(embeddings, [anchor], Retrieval()).pairs
    result = retrieve(embeddings, [anchor], Retrieval(geometry_min_cosine=0.3))
    assert len(result.pairs) == 1
    assert result.pairs[0].cosine < Retrieval().detect_min_cosine


def test_full_affine_handles_anisotropic_view_and_rejects_reflection():
    src = np.random.default_rng(4).uniform(10, 70, (60, 2)).astype(np.float32)
    dst = src @ np.array([[1.7, 0.2], [0.3, 0.7]], np.float32) + 10
    matches = LocalMatches(src, dst, np.ones(60))
    config = Geometry(model="affine", min_spatial_coverage=0.02)
    assert AffineRansacVerifier(config).verify(matches, (100, 100, 3), (160, 160, 3)).accepted
    reflected = dst.copy()
    reflected[:, 0] = 159 - reflected[:, 0]
    evidence = AffineRansacVerifier(config).verify(
        replace(matches, candidate_points=reflected), (100, 100, 3), (160, 160, 3)
    )
    assert not evidence.accepted
    assert evidence.reason == "invalid_scale_or_transform"


def test_confidence_filter_does_not_manufacture_support():
    src = np.random.default_rng(4).uniform(10, 80, (30, 2)).astype(np.float32)
    scores = np.zeros(30)
    scores[:5] = 0.9
    matches = LocalMatches(src, src, scores)
    ev = AffineRansacVerifier(Geometry(min_match_confidence=0.5)).verify(
        matches, (100, 100, 3), (100, 100, 3)
    )
    assert not ev.accepted and ev.num_matches == 5


def test_generation_controls_validated():
    with pytest.raises(ValueError):
        Models(florence_num_beams=0)
    assert Models().florence_use_cache
