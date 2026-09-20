"""Cancellable confirmation with a visible countdown.

Some actions should not happen because a perception pipeline was briefly
confident. Anything destructive or system-level gets an explicit delay during
which the operator can see what is about to happen and stop it.

The design rule is that **cancelling must be easier than confirming**. There is
no "confirm" gesture to perform - the action proceeds if nothing happens, and
nearly any deliberate signal stops it: the panic key, a new gesture, or the
operator simply moving out of frame. A confirmation that required a second
gesture would be vulnerable to the same false positive that triggered the first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from ..logsetup import get_logger

log = get_logger("control.confirm")


class ConfirmState(str, Enum):
    IDLE = "idle"
    COUNTING = "counting"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"


@dataclass
class Confirmation:
    action: str
    description: str
    started_at: float
    duration: float
    state: ConfirmState = ConfirmState.COUNTING
    cancel_reason: str = ""

    def remaining(self, now: float) -> float:
        return max(0.0, self.duration - (now - self.started_at))

    def progress(self, now: float) -> float:
        if self.duration <= 0:
            return 1.0
        return min(1.0, (now - self.started_at) / self.duration)


class Confirmer:
    """Holds at most one pending confirmation at a time.

    A second request while one is already counting down *replaces* nothing and
    is refused: queueing destructive actions behind one another is never what a
    user wants, and it would let a burst of false positives stack up.
    """

    def __init__(self, duration_s: float = 5.0) -> None:
        self.duration_s = duration_s
        self.pending: Confirmation | None = None
        self.history: list[Confirmation] = []

    # ------------------------------------------------------------------ #
    @property
    def is_counting(self) -> bool:
        return self.pending is not None and self.pending.state is ConfirmState.COUNTING

    def request(
        self,
        action: str,
        description: str = "",
        now: float | None = None,
        duration: float | None = None,
    ) -> bool:
        """Begin a countdown. Returns False if one is already running."""
        now = time.perf_counter() if now is None else now
        if self.is_counting:
            log.debug("ignoring %s: %s is already awaiting confirmation",
                      action, self.pending.action)
            return False
        self.pending = Confirmation(
            action=action,
            description=description or action,
            started_at=now,
            duration=self.duration_s if duration is None else duration,
        )
        log.warning(
            "%s in %.0fs - cancel with Esc or any gesture",
            self.pending.description, self.pending.duration,
        )
        return True

    def cancel(self, reason: str = "cancelled", now: float | None = None) -> bool:
        """Stop the pending countdown. Returns True if one was stopped."""
        if not self.is_counting:
            return False
        now = time.perf_counter() if now is None else now
        self.pending.state = ConfirmState.CANCELLED
        self.pending.cancel_reason = reason
        log.info("%s cancelled: %s", self.pending.action, reason)
        self.history.append(self.pending)
        self.pending = None
        return True

    def update(self, now: float | None = None) -> Confirmation | None:
        """Advance the countdown. Returns the confirmation when it fires."""
        if not self.is_counting:
            return None
        now = time.perf_counter() if now is None else now
        if self.pending.remaining(now) > 0.0:
            return None
        fired = self.pending
        fired.state = ConfirmState.CONFIRMED
        self.history.append(fired)
        self.pending = None
        log.warning("%s confirmed - executing", fired.action)
        return fired

    def hud_lines(self, now: float | None = None) -> list[str]:
        if not self.is_counting:
            return []
        now = time.perf_counter() if now is None else now
        return [
            self.pending.description.upper(),
            f"in {self.pending.remaining(now):0.1f}s",
            "Esc or any gesture cancels",
        ]
