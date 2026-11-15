"""Explicit lifecycle, frozen verified appearance and bounded propagation age."""

import numpy as np
from .flow import FlowTracker
from .mil import GuardedMILTracker


class TrackController:
    def __init__(self, config, tracker=None):
        self.config = config
        self.tracker = tracker or (
            GuardedMILTracker(config) if config.backend == "mil" else FlowTracker(config)
        )
        self.state = "SEARCHING"
        self.embedding = None
        self.verified_at = None
        self.last_attempt_at = None
        self.last_reason = "uninitialized"

    def predict(self, image, timestamp):
        estimate = self.tracker.update(image, timestamp)
        if self.verified_at is not None and not estimate.reliable:
            self.state = "LOST"
        self.last_reason = estimate.reason
        return estimate

    def can_propagate(self, estimate, embedding, timestamp):
        if not estimate.reliable or self.embedding is None:
            return False, None
        similarity = float(np.clip(embedding @ self.embedding, -1, 1))
        age = timestamp - self.verified_at
        horizon = self.config.max_unverified_sec or self.config.reverify_interval_sec
        last_attempt = (
            self.last_attempt_at if self.last_attempt_at is not None else self.verified_at
        )
        due = timestamp - last_attempt >= self.config.reverify_interval_sec
        valid = 0 <= age < horizon and not due and similarity >= self.config.appearance_min_cosine
        if valid:
            # An inconclusive check never refreshes verified_at or the identity template.
            self.last_reason = (
                "bounded_uncertain_propagation"
                if self.state == "UNCERTAIN"
                else "appearance_supported"
            )
        else:
            self.state = "UNCERTAIN"
            self.last_reason = "identity_check_required"
        return valid, similarity

    def defer(self, estimate, embedding, timestamp, conflicting=False):
        """Retain a supported position briefly after an inconclusive check, not a detection."""
        self.last_attempt_at = timestamp
        if (
            conflicting
            or self.config.max_unverified_sec == 0
            or not estimate.reliable
            or embedding is None
            or self.embedding is None
            or self.verified_at is None
        ):
            return False
        age = timestamp - self.verified_at
        similarity = float(np.clip(embedding @ self.embedding, -1, 1))
        if not (
            0 <= age < self.config.max_unverified_sec
            and similarity >= self.config.appearance_min_cosine
        ):
            return False
        self.state = "UNCERTAIN"
        self.last_reason = "inconclusive_verification_bounded_hold"
        return True

    def confirm(self, image, bbox, embedding, timestamp):
        self.tracker.initialize(image, bbox, timestamp)
        self.embedding = embedding.copy()
        self.verified_at = timestamp
        self.last_attempt_at = timestamp
        self.state = "TRACKING"
        self.last_reason = "identity_verified"

    def reject(self):
        self.state = "LOST" if self.verified_at is not None else "SEARCHING"
        self.last_reason = "identity_not_verified"
        self.tracker.reset()

    def array_bytes(self):
        """Known NumPy state only; excludes OpenCV internal MIL allocations."""
        flow = getattr(self.tracker, "flow", self.tracker)
        return sum(
            value.nbytes
            for value in (flow.gray, flow.points, flow.box, self.embedding)
            if value is not None
        )
