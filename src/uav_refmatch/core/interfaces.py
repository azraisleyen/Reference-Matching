from typing import Protocol
import numpy as np
from .types import Proposal, LocalMatches, GeometryEvidence


class ProposalGenerator(Protocol):
    def propose(self, image: np.ndarray) -> list[Proposal]: ...


class FeatureEncoder(Protocol):
    def encode(self, images: list[np.ndarray]) -> np.ndarray: ...


class LocalMatcher(Protocol):
    def match(self, reference: np.ndarray, candidate: np.ndarray) -> LocalMatches: ...


class GeometricVerifier(Protocol):
    def verify(
        self, matches: LocalMatches, reference_shape: tuple, candidate_shape: tuple
    ) -> GeometryEvidence: ...
