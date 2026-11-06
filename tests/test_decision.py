from dataclasses import replace
from uav_refmatch.core.types import Pair, VerifiedPair, Proposal
from uav_refmatch.decision.engine import choose


def test_not_found():
    assert choose([], [], 0.7) is None


def test_visual_geometry_and_tie(evidence):
    proposals = [Proposal(0, (0, 0, 20, 20), "f")]
    good = VerifiedPair(Pair(0, "R0", 0.8), evidence)
    bad = VerifiedPair(Pair(0, "R0", 0.95), replace(evidence, accepted=False))
    assert choose([bad, good], proposals, 0.7) is good
    assert choose([good], proposals, 0.9) is None
