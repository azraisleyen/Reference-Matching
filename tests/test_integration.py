"""Explicit real-checkpoint tests; opt in with UAV_REAL_MODELS=1. No substitutes."""

import os
import numpy as np
import pytest
from uav_refmatch.config import Config
from uav_refmatch.core.runtime import device_for

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("UAV_REAL_MODELS") != "1",
        reason="Real model tests require UAV_REAL_MODELS=1 and checkpoint access",
    ),
]


def test_real_dinov3(texture):
    from uav_refmatch.features.dinov3 import DinoV3Encoder

    config = Config()
    model = DinoV3Encoder(config.models, device_for(config.models))
    embeddings = model.encode([texture])
    assert embeddings.shape == (1, model.dimension)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5)


def test_real_florence(texture):
    from uav_refmatch.proposals.florence2 import Florence2Proposals

    config = Config()
    model = Florence2Proposals(config.models, config.proposals, device_for(config.models))
    assert isinstance(model.propose(texture), list)


def test_real_efficientloftr(texture):
    from uav_refmatch.matching.efficientloftr import EfficientLoFTRMatcher

    config = Config()
    model = EfficientLoFTRMatcher(config.models, device_for(config.models))
    matches = model.match(texture, texture)
    assert matches.reference_points.shape[1] == 2
