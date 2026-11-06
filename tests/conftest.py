from dataclasses import replace
import numpy as np
import pytest
from uav_refmatch.core.types import ReferenceEntry, Prediction, GeometryEvidence


@pytest.fixture
def texture():
    return np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def anchor(texture):
    return ReferenceEntry(
        "R0", "anchor", None, None, None, texture, np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )


@pytest.fixture
def evidence():
    return GeometryEvidence(30, 24, 0.8, 0.1, 0.5, True, True, "geometry_passed")


@pytest.fixture
def found(evidence):
    return Prediction(0, 0, 0.0, 64, 64, "FOUND", "target", (0, 0, 64, 64), "R0", 0.9, evidence)


@pytest.fixture
def entry(anchor):
    return replace(
        anchor,
        memory_id="",
        origin="dynamic",
        embedding=np.array([0.85, 0.5267827, 0.0], dtype=np.float32),
        matched_parent_id="R0",
        created_at_frame=0,
    )
