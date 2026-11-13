# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-owned admission lease; independent of scheduler pause/resume."""

import math
import time


class MigrationAdmissionGate:
    def __init__(self):
        self.owner = None
        self.deadline = 0.0
        self.started = 0.0
        self.admitted = set()
        self.last_duration = 0.0
        self.reason = "initial"
        self.queue_seconds = 0.0
        self.waiting = 0
        self.observed = 0.0

    def status(self):
        now = time.monotonic()
        if self.owner is not None and now >= self.deadline:
            self._close("lease_expired")
        return {
            "enabled": self.owner is not None,
            "owner": self.owner,
            "remaining_seconds": max(0, self.deadline - now),
            "duration_seconds": now - self.started
            if self.owner
            else self.last_duration,
            "reason": self.reason,
            "queue_seconds": self.queue_seconds,
            "blocked_waiting": self.waiting,
        }

    def observe(self, waiting):
        now = min(time.monotonic(), self.deadline)
        if self.owner is not None:
            self.queue_seconds += max(0, now - self.observed) * self.waiting
            self.observed = now
            self.waiting = waiting

    def _close(self, reason):
        self.observe(0)
        self.last_duration = min(time.monotonic(), self.deadline) - self.started
        self.owner = None
        self.deadline = 0.0
        self.admitted.clear()
        self.reason = reason

    def update(self, action, owner, timeout=10, requests=()):
        self.status()
        if not isinstance(owner, str) or not owner:
            raise ValueError("admission lease requires an owner")
        if action == "open":
            if not math.isfinite(timeout) or not 0 < timeout <= 600:
                raise ValueError("admission timeout must be in (0, 600]")
            if self.owner is not None:
                raise ValueError("migration admission gate already owned")
            self.owner = owner
            self.started = time.monotonic()
            self.deadline = self.started + timeout
            self.observed = self.started
            self.queue_seconds = 0.0
            self.waiting = 0
            self.admitted = {
                r.request_id
                for r in requests
                if r.status.name == "RUNNING"
                or r.num_preemptions > 0
                or r.num_computed_tokens > 0
            }
            self.reason = "opened"
        elif action == "renew":
            if (
                self.owner != owner
                or not math.isfinite(timeout)
                or not 0 < timeout <= 600
            ):
                raise ValueError("invalid admission lease renewal")
            self.deadline = time.monotonic() + timeout
        elif action == "close":
            if self.owner is not None and self.owner != owner:
                raise ValueError("migration admission gate owner mismatch")
            if self.owner is not None:
                self._close("released")
        else:
            raise ValueError("unknown admission action")
        return self.status()

    def allows(self, request):
        return not self.status()["enabled"] or request.request_id in self.admitted
