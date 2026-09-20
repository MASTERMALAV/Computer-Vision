"""Authenticated operator sessions.

Gating every action on a *fresh* face match would be both unaffordable and
unusable on this hardware. Unaffordable because detection plus embedding costs
far more than the per-frame budget left over from hand tracking. Unusable
because faces are occluded constantly in normal work - you look down at the
keyboard, turn to a second monitor, rub your eye - and a per-frame gate would
drop the cursor every time, including mid-drag.

So identity is a *session*. One good match opens it; it stays open while the
operator keeps being seen, expires after a period with no confirmation, and is
re-confirmed cheaply in the background. Actions that genuinely matter can still
demand a match within the last few seconds, which is a different and stricter
question than "is the session open".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..config import SecurityConfig
from ..control.dispatcher import IdentityStatus
from ..logsetup import get_logger

log = get_logger("face.session")


@dataclass
class SessionEvent:
    kind: str  # opened | renewed | expired | rejected | switched
    name: str
    timestamp: float
    detail: str = ""

    def __str__(self) -> str:
        extra = f" - {self.detail}" if self.detail else ""
        return f"{self.kind}:{self.name or 'unknown'}{extra}"


@dataclass
class IdentitySession:
    """Tracks who is currently authenticated, and how recently that was proven."""

    security: SecurityConfig = field(default_factory=SecurityConfig)

    name: str = ""
    opened_at: float = 0.0
    last_match_at: float = 0.0
    last_confidence: float = 0.0
    matches: int = 0
    _consecutive_failures: int = 0

    # How many consecutive non-matching observations end a session. Without
    # this, one bad frame while turning your head would log you out.
    reject_streak_to_close: int = 5

    # ------------------------------------------------------------------ #
    @property
    def is_open(self) -> bool:
        return bool(self.name)

    def is_valid(self, now: float | None = None) -> bool:
        """Open and not yet timed out."""
        if not self.is_open:
            return False
        now = time.perf_counter() if now is None else now
        return (now - self.last_match_at) <= self.security.session_timeout_s

    def is_fresh(self, now: float | None = None) -> bool:
        """Matched recently enough for a destructive action."""
        if not self.is_open:
            return False
        now = time.perf_counter() if now is None else now
        return (now - self.last_match_at) <= self.security.fresh_match_within_s

    def needs_reverify(self, now: float | None = None) -> bool:
        """Due for a cheap background re-confirmation."""
        now = time.perf_counter() if now is None else now
        if not self.is_open:
            return True
        return (now - self.last_match_at) >= self.security.reverify_interval_s

    def age(self, now: float | None = None) -> float:
        now = time.perf_counter() if now is None else now
        return now - self.last_match_at if self.is_open else 0.0

    # ------------------------------------------------------------------ #
    def observe_match(self, name: str, confidence: float, now: float) -> SessionEvent | None:
        """Record a successful recognition."""
        self._consecutive_failures = 0

        if not self.is_open:
            self.name = name
            self.opened_at = now
            self.last_match_at = now
            self.last_confidence = confidence
            self.matches = 1
            log.info("session opened for %s (%.3f)", name, confidence)
            return SessionEvent("opened", name, now)

        if name != self.name:
            # A different enrolled person stepped in. Treat that as a new
            # session rather than silently inheriting the previous one.
            previous = self.name
            self.name = name
            self.opened_at = now
            self.last_match_at = now
            self.last_confidence = confidence
            self.matches = 1
            log.warning("operator switched: %s -> %s", previous, name)
            return SessionEvent("switched", name, now, detail=f"was {previous}")

        self.last_match_at = now
        self.last_confidence = confidence
        self.matches += 1
        return None

    def observe_no_match(self, now: float) -> SessionEvent | None:
        """Record an observation that did not match the open session."""
        if not self.is_open:
            return None
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.reject_streak_to_close:
            name = self.name
            self.close()
            log.info("session closed for %s: unrecognised for %d observations",
                     name, self._consecutive_failures)
            return SessionEvent("rejected", name, now, detail="unrecognised")
        return None

    def tick(self, now: float) -> SessionEvent | None:
        """Expire the session if it has gone too long without confirmation."""
        if self.is_open and not self.is_valid(now):
            name = self.name
            self.close()
            log.info("session expired for %s", name)
            return SessionEvent("expired", name, now)
        return None

    def close(self) -> None:
        self.name = ""
        self.opened_at = 0.0
        self.last_match_at = 0.0
        self.last_confidence = 0.0
        self.matches = 0
        self._consecutive_failures = 0

    # ------------------------------------------------------------------ #
    def status(self, now: float | None = None) -> IdentityStatus:
        """The view of identity that the action dispatcher consumes."""
        now = time.perf_counter() if now is None else now

        if not self.security.require_identity:
            return IdentityStatus(
                authenticated=True, name="(gating disabled)",
                last_match_time=now, confidence=1.0,
            )

        valid = self.is_valid(now)
        expected = self.security.operator.strip()
        if valid and expected and self.name != expected:
            # Someone enrolled, but not the designated operator.
            return IdentityStatus(
                authenticated=False, name=self.name,
                last_match_time=self.last_match_at, confidence=self.last_confidence,
            )
        return IdentityStatus(
            authenticated=valid,
            name=self.name or "unknown",
            last_match_time=self.last_match_at,
            confidence=self.last_confidence,
        )

    def describe(self, now: float | None = None) -> str:
        now = time.perf_counter() if now is None else now
        if not self.security.require_identity:
            return "identity gating off"
        if not self.is_open:
            return "not authenticated"
        state = "valid" if self.is_valid(now) else "EXPIRED"
        return (
            f"{self.name} {state}  seen {self.age(now):.0f}s ago  "
            f"conf {self.last_confidence:.2f}"
        )
