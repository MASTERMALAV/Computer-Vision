"""Turns gesture events into mouse actions, under an identity gate.

This is the only place in the system that is allowed to affect the machine, so
every safety rule lives here rather than being scattered through the pipeline:

* **Disarmed by default.** Actions are logged, counted and shown on the HUD but
  not executed until the operator explicitly arms the system.
* **Identity gating.** When enabled, actions require a live authenticated
  session belonging to the enrolled operator.
* **Freshness.** Destructive actions additionally require a face match within
  the last few seconds - a long-lived session is not enough.
* **Cooldowns.** An action cannot repeat faster than a human could mean it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..config import SecurityConfig
from ..gestures.fsm import GestureEvent, GestureType
from ..logsetup import get_logger
from .injector import MouseInjector

log = get_logger("control.dispatcher")


@dataclass
class IdentityStatus:
    """What the face stage currently believes about who is present.

    Kept deliberately small so the dispatcher never reaches into the face
    pipeline; the face stage pushes this in, and a system with face recognition
    disabled just leaves it at its permissive default.
    """

    authenticated: bool = True
    name: str = "(gating disabled)"
    last_match_time: float = 0.0
    confidence: float = 1.0

    def is_fresh(self, within_s: float, now: float) -> bool:
        return self.authenticated and (now - self.last_match_time) <= within_s


@dataclass
class ActionRecord:
    action: str
    timestamp: float
    executed: bool
    reason: str = ""


@dataclass
class DispatcherStats:
    executed: int = 0
    blocked_disarmed: int = 0
    blocked_identity: int = 0
    blocked_cooldown: int = 0
    recent: list[ActionRecord] = field(default_factory=list)

    def note(self, record: ActionRecord, keep: int = 12) -> None:
        self.recent.append(record)
        if len(self.recent) > keep:
            del self.recent[: len(self.recent) - keep]

    def as_dict(self) -> dict:
        return {
            "executed": self.executed,
            "blocked_disarmed": self.blocked_disarmed,
            "blocked_identity": self.blocked_identity,
            "blocked_cooldown": self.blocked_cooldown,
        }


class ActionDispatcher:
    """Executes mouse actions for gesture events, subject to the safety rules."""

    # Actions that change the world irreversibly enough to want a fresh face
    # match rather than merely a valid session.
    DESTRUCTIVE: frozenset[str] = frozenset()

    def __init__(
        self,
        injector: MouseInjector,
        security: SecurityConfig | None = None,
        min_interval_s: float = 0.12,
    ) -> None:
        self.injector = injector
        self.security = security or SecurityConfig()
        self.min_interval_s = min_interval_s
        self.identity = IdentityStatus()
        self.stats = DispatcherStats()
        self._last_action_at: dict[str, float] = {}
        self._drag_active = False

    # ------------------------------------------------------------------ #
    def set_identity(self, status: IdentityStatus) -> None:
        self._maybe_end_drag_on_identity_loss(status)
        self.identity = status

    def _maybe_end_drag_on_identity_loss(self, new_status: IdentityStatus) -> None:
        """Never leave a button held when the operator stops being recognised."""
        if self._drag_active and self.identity.authenticated and not new_status.authenticated:
            log.warning("identity lost mid-drag; releasing the mouse button")
            self.injector.button_up("left")
            self._drag_active = False

    def _allowed(self, action: str, now: float, destructive: bool = False) -> tuple[bool, str]:
        if not self.injector.armed:
            self.stats.blocked_disarmed += 1
            return False, "disarmed"

        if self.security.require_identity:
            if not self.identity.authenticated:
                self.stats.blocked_identity += 1
                return False, "no authenticated operator"
            if destructive and not self.identity.is_fresh(
                self.security.fresh_match_within_s, now
            ):
                self.stats.blocked_identity += 1
                return False, "identity match is stale"

        last = self._last_action_at.get(action, -999.0)
        # A drag release must never be rate-limited: dropping it would leave the
        # button stuck down.
        if action != "drag_end" and (now - last) < self.min_interval_s:
            self.stats.blocked_cooldown += 1
            return False, "cooldown"
        return True, ""

    # ------------------------------------------------------------------ #
    def dispatch(self, events: list[GestureEvent], now: float | None = None) -> list[ActionRecord]:
        now = time.perf_counter() if now is None else now
        records: list[ActionRecord] = []

        for event in events:
            action = self._action_for(event)
            if action is None:
                continue

            destructive = action in self.DESTRUCTIVE
            allowed, reason = self._allowed(action, now, destructive)
            if not allowed:
                record = ActionRecord(action, now, executed=False, reason=reason)
                # A suppressed drag_start must not leave the FSM believing a
                # drag is running, or its drag_end will release a button that
                # was never pressed.
                if action == "drag_start":
                    self._drag_active = False
                records.append(record)
                self.stats.note(record)
                log.debug("blocked %s: %s", action, reason)
                continue

            self._execute(action)
            self._last_action_at[action] = now
            self.stats.executed += 1
            record = ActionRecord(action, now, executed=True)
            records.append(record)
            self.stats.note(record)
            log.info("action: %s", action)

        return records

    @staticmethod
    def _action_for(event: GestureEvent) -> str | None:
        return {
            GestureType.CLICK: "click",
            GestureType.DOUBLE_CLICK: "double_click",
            GestureType.RIGHT_CLICK: "right_click",
            GestureType.DRAG_START: "drag_start",
            GestureType.DRAG_END: "drag_end",
        }.get(event.type)

    def _execute(self, action: str) -> None:
        if action == "click":
            self.injector.click("left")
        elif action == "double_click":
            self.injector.double_click("left")
        elif action == "right_click":
            self.injector.click("right")
        elif action == "drag_start":
            self.injector.button_down("left")
            self._drag_active = True
        elif action == "drag_end":
            self.injector.button_up("left")
            self._drag_active = False

    # ------------------------------------------------------------------ #
    @property
    def drag_active(self) -> bool:
        return self._drag_active

    def shutdown(self) -> None:
        """Release anything held. Must run on every exit path."""
        if self._drag_active:
            self.injector.button_up("left")
            self._drag_active = False
        self.injector.release_all()
