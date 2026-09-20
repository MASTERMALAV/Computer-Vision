"""Gesture recognition: noisy landmarks in, reliable discrete events out.

Three mechanisms do the work, and all three are necessary:

**Hysteresis.** A single threshold oscillates: hold a pinch exactly at the
boundary and the signal chatters open/closed many times a second, emitting a
burst of clicks. Each detector therefore has two thresholds - it closes at a
tighter distance than it opens (a Schmitt trigger), so the boundary region is
stable.

**Temporal debouncing.** A single bad frame must not produce an event. A state
change is only accepted once it has held for a few consecutive frames, which
costs a little latency and removes essentially all single-frame false positives.

**Scale normalisation.** Every distance is in units of the user's own hand span
(see :mod:`argus.hands.landmarks`), so one set of thresholds works whether the
hand is near the camera or far from it.

The clutch is a *pose*, not a pinch. Holding a pinch to keep the cursor alive
would be both exhausting and impossible to combine with clicking, so instead:
index extended means the cursor is live, and relaxing the hand parks it. That
leaves pinches free to mean exactly one thing each.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from ..hands.landmarks import Hand
from ..logsetup import get_logger

log = get_logger("gestures")


class GestureType(str, Enum):
    CLUTCH_ENGAGE = "clutch_engage"
    CLUTCH_RELEASE = "clutch_release"
    # Fired the moment the fingers begin closing, before any click is certain.
    # The pointer engine uses it to freeze the cursor so the pinch cannot drag
    # it off target while it completes.
    PINCH_APPROACH = "pinch_approach"
    PINCH_ABORT = "pinch_abort"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG_START = "drag_start"
    DRAG_END = "drag_end"
    SCROLL = "scroll"
    HAND_LOST = "hand_lost"


@dataclass
class GestureEvent:
    type: GestureType
    timestamp: float
    hand: str = "Unknown"
    value: float = 0.0
    detail: str = ""

    def __str__(self) -> str:
        extra = f" {self.detail}" if self.detail else ""
        return f"{self.type.value}({self.hand}{extra})"


# --------------------------------------------------------------------------- #
class SchmittTrigger:
    """Boolean signal with separate enter and exit thresholds.

    ``enter`` must be the tighter (smaller) threshold for a distance-based
    signal: the state turns on when the distance drops below ``enter`` and only
    turns off again once it rises above the looser ``exit``.
    """

    __slots__ = ("enter", "exit", "state")

    def __init__(self, enter: float, exit: float, state: bool = False) -> None:
        if exit <= enter:
            raise ValueError(
                f"exit threshold ({exit}) must be looser than enter ({enter}) "
                "or the trigger has no hysteresis"
            )
        self.enter = enter
        self.exit = exit
        self.state = state

    def update(self, value: float) -> bool:
        if self.state:
            if value > self.exit:
                self.state = False
        elif value < self.enter:
            self.state = True
        return self.state

    def reset(self, state: bool = False) -> None:
        self.state = state


class Debouncer:
    """Accepts a state change only after it holds for N consecutive frames."""

    __slots__ = ("frames", "state", "_pending", "_count")

    def __init__(self, frames: int = 3, initial: bool = False) -> None:
        self.frames = max(1, frames)
        self.state = initial
        self._pending = initial
        self._count = 0

    def update(self, raw: bool) -> bool:
        if raw == self.state:
            self._count = 0
            self._pending = raw
            return self.state
        if raw == self._pending:
            self._count += 1
        else:
            self._pending = raw
            self._count = 1
        if self._count >= self.frames:
            self.state = raw
            self._count = 0
        return self.state

    def reset(self, state: bool = False) -> None:
        self.state = state
        self._pending = state
        self._count = 0


# --------------------------------------------------------------------------- #
class PinchState(str, Enum):
    OPEN = "open"
    APPROACHING = "approaching"
    CLOSED = "closed"


class PinchDetector:
    """One thumb-to-fingertip pinch, with click / drag discrimination.

    A quick close-and-release is a click. Holding the pinch past
    ``drag_dwell_s`` converts it into a press-and-hold drag, and the click that
    would otherwise have fired is suppressed - so one gesture never produces
    both.
    """

    def __init__(
        self,
        finger: str,
        close_at: float,
        open_at: float,
        approach_at: float,
        debounce_frames: int = 2,
        drag_dwell_s: float = 0.35,
        cooldown_s: float = 0.25,
        double_click_s: float = 0.40,
    ) -> None:
        self.finger = finger
        self.trigger = SchmittTrigger(enter=close_at, exit=open_at)
        self.approach_at = approach_at
        self.debounce = Debouncer(debounce_frames)
        self.drag_dwell_s = drag_dwell_s
        self.cooldown_s = cooldown_s
        self.double_click_s = double_click_s

        self.state = PinchState.OPEN
        self.dragging = False
        self._closed_at: float | None = None
        self._last_click_at: float = -999.0
        self._last_event_at: float = -999.0
        self._approach_sent = False

    @property
    def is_closed(self) -> bool:
        return self.state is PinchState.CLOSED

    def reset(self) -> None:
        self.trigger.reset(False)
        self.debounce.reset(False)
        self.state = PinchState.OPEN
        self.dragging = False
        self._closed_at = None
        self._approach_sent = False

    def update(self, distance: float, now: float, hand_label: str) -> list[GestureEvent]:
        events: list[GestureEvent] = []
        closed = self.debounce.update(self.trigger.update(distance))

        # --- anticipatory freeze ------------------------------------------- #
        # Emitted on the way in, before the click is confirmed, so the pointer
        # can be locked before the fingers deform the hand enough to move it.
        approaching = distance < self.approach_at
        if approaching and not self._approach_sent and not closed:
            self._approach_sent = True
            self.state = PinchState.APPROACHING
            events.append(
                GestureEvent(GestureType.PINCH_APPROACH, now, hand_label, distance, self.finger)
            )
        elif not approaching and self._approach_sent and not closed:
            self._approach_sent = False
            if self.state is PinchState.APPROACHING:
                self.state = PinchState.OPEN
                events.append(
                    GestureEvent(GestureType.PINCH_ABORT, now, hand_label, distance, self.finger)
                )

        # --- close --------------------------------------------------------- #
        if closed and self.state is not PinchState.CLOSED:
            self.state = PinchState.CLOSED
            self._closed_at = now
            self._approach_sent = True

        # --- held long enough to become a drag ------------------------------ #
        elif closed and self.state is PinchState.CLOSED and not self.dragging:
            if self._closed_at is not None and (now - self._closed_at) >= self.drag_dwell_s:
                self.dragging = True
                events.append(
                    GestureEvent(GestureType.DRAG_START, now, hand_label, distance, self.finger)
                )

        # --- release -------------------------------------------------------- #
        elif not closed and self.state is PinchState.CLOSED:
            held = now - (self._closed_at or now)
            self.state = PinchState.OPEN
            self._approach_sent = False
            if self.dragging:
                self.dragging = False
                events.append(
                    GestureEvent(GestureType.DRAG_END, now, hand_label, held, self.finger)
                )
            elif now - self._last_event_at >= self.cooldown_s:
                if now - self._last_click_at <= self.double_click_s:
                    events.append(
                        GestureEvent(
                            GestureType.DOUBLE_CLICK, now, hand_label, held, self.finger
                        )
                    )
                    self._last_click_at = -999.0  # a triple pinch is not a double-double
                else:
                    events.append(
                        GestureEvent(GestureType.CLICK, now, hand_label, held, self.finger)
                    )
                    self._last_click_at = now
                self._last_event_at = now
            self._closed_at = None

        return events


# --------------------------------------------------------------------------- #
@dataclass
class GestureThresholds:
    """All tunable gesture geometry, in hand-scale units.

    Defaults are starting points. ``argus calibrate`` measures your
    own hand and writes values fitted to it, which matters because thumb length
    relative to palm span varies a lot between people.
    """

    pinch_close: float = 0.34
    pinch_open: float = 0.50
    pinch_approach: float = 0.62

    # A finger counts as extended above this tip-to-MCP ratio, curled below.
    finger_extended: float = 1.05
    finger_curled: float = 0.85

    debounce_frames: int = 2
    clutch_debounce_frames: int = 3
    drag_dwell_s: float = 0.35
    click_cooldown_s: float = 0.25
    double_click_s: float = 0.40

    # Above this hand speed (hand-scale units/sec) landmarks are least reliable,
    # so gesture transitions are ignored until the hand settles.
    motion_gate_speed: float = 2.6

    # Extending the middle finger alongside the index switches to scroll,
    # matching the one-finger-move / two-finger-scroll trackpad convention.
    scroll_middle_extended: float = 1.05
    scroll_debounce_frames: int = 3


@dataclass
class GestureState:
    """What the recogniser currently believes about the hand."""

    clutch_engaged: bool = False
    dragging: bool = False
    pinch_index: float = 1.0
    pinch_middle: float = 1.0
    hand_speed: float = 0.0
    hand_present: bool = False
    extensions: dict[str, float] = field(default_factory=dict)
    suppressed_by_motion: bool = False
    # "point" while one finger is out, "scroll" while two are.
    mode: str = "point"



class GestureEngine:
    """Turns a stream of :class:`Hand` observations into gesture events."""

    def __init__(self, thresholds: GestureThresholds | None = None) -> None:
        self.t = thresholds or GestureThresholds()
        self.state = GestureState()

        self.clutch_debounce = Debouncer(self.t.clutch_debounce_frames)
        self.scroll_debounce = Debouncer(self.t.scroll_debounce_frames)
        self.left_click = PinchDetector(
            "index",
            close_at=self.t.pinch_close,
            open_at=self.t.pinch_open,
            approach_at=self.t.pinch_approach,
            debounce_frames=self.t.debounce_frames,
            drag_dwell_s=self.t.drag_dwell_s,
            cooldown_s=self.t.click_cooldown_s,
            double_click_s=self.t.double_click_s,
        )
        self.right_click = PinchDetector(
            "middle",
            close_at=self.t.pinch_close,
            open_at=self.t.pinch_open,
            approach_at=self.t.pinch_approach,
            debounce_frames=self.t.debounce_frames,
            # A right click has no drag mode; setting the dwell out of reach
            # keeps it a pure click.
            drag_dwell_s=9999.0,
            cooldown_s=self.t.click_cooldown_s,
            double_click_s=0.0,
        )

        self._prev_palm = None
        self._prev_time: float | None = None
        self._speed_window: deque[float] = deque(maxlen=5)
        self._missing_frames = 0

    # ------------------------------------------------------------------ #
    def _clutch_pose(self, hand: Hand) -> bool:
        """True when the hand is in the pointing pose that enables the cursor.

        Requires the index extended. The other fingers are deliberately *not*
        required to be curled: insisting on a strict pose makes the cursor drop
        out whenever the hand relaxes slightly, and the user is holding this
        pose continuously while working.
        """
        index = hand.finger_extension("index")
        if self.state.clutch_engaged:
            # Looser test to stay engaged - hysteresis on the pose itself.
            return index > self.t.finger_curled
        return index > self.t.finger_extended

    def _hand_speed(self, hand: Hand, now: float) -> float:
        """Palm speed in hand-scale units per second."""
        palm = hand.palm_center
        scale = max(hand.scale, 1e-6)
        if self._prev_palm is None or self._prev_time is None:
            speed = 0.0
        else:
            dt = now - self._prev_time
            if dt <= 1e-6 or dt > 0.25:
                speed = 0.0
            else:
                dist = float(((palm - self._prev_palm) ** 2).sum() ** 0.5) / scale
                speed = dist / dt
        self._prev_palm = palm
        self._prev_time = now
        self._speed_window.append(speed)
        return sum(self._speed_window) / len(self._speed_window)

    # ------------------------------------------------------------------ #
    def update(self, hand: Hand | None, now: float) -> list[GestureEvent]:
        """Advance the state machine by one frame."""
        events: list[GestureEvent] = []

        # ---- no hand ---------------------------------------------------- #
        if hand is None or not hand.is_reliable:
            self._missing_frames += 1
            # Tolerate brief dropouts: a single missed frame mid-drag must not
            # release the button and abandon whatever is being dragged.
            if self._missing_frames <= 4:
                return events
            if self.state.clutch_engaged:
                self.state.clutch_engaged = False
                events.append(GestureEvent(GestureType.CLUTCH_RELEASE, now, detail="hand lost"))
            if self.left_click.dragging:
                self.left_click.dragging = False
                events.append(GestureEvent(GestureType.DRAG_END, now, detail="hand lost"))
            if self.state.hand_present:
                events.append(GestureEvent(GestureType.HAND_LOST, now))
            self.state.hand_present = False
            self.state.dragging = False
            self.clutch_debounce.reset(False)
            self.left_click.reset()
            self.right_click.reset()
            self._prev_palm = None
            self._prev_time = None
            return events

        self._missing_frames = 0
        self.state.hand_present = True
        label = hand.handedness

        speed = self._hand_speed(hand, now)
        self.state.hand_speed = speed
        self.state.pinch_index = hand.pinch("index")
        self.state.pinch_middle = hand.pinch("middle")
        self.state.extensions = hand.extensions()

        # ---- clutch ------------------------------------------------------ #
        engaged = self.clutch_debounce.update(self._clutch_pose(hand))
        if engaged != self.state.clutch_engaged:
            self.state.clutch_engaged = engaged
            events.append(
                GestureEvent(
                    GestureType.CLUTCH_ENGAGE if engaged else GestureType.CLUTCH_RELEASE,
                    now,
                    label,
                )
            )
            if not engaged and self.left_click.dragging:
                self.left_click.dragging = False
                events.append(
                    GestureEvent(GestureType.DRAG_END, now, label, detail="clutch released")
                )

        # ---- mode: pointing or scrolling --------------------------------- #
        scrolling = self.scroll_debounce.update(
            self.state.clutch_engaged
            and hand.finger_extension("middle") > self.t.scroll_middle_extended
        )
        self.state.mode = "scroll" if scrolling else "point"

        # ---- pinches ----------------------------------------------------- #
        # Landmarks are least trustworthy while the hand is moving fast, and a
        # fast-moving hand is not trying to click. Gate new transitions on that,
        # but never abandon a drag already in progress.
        fast = speed > self.t.motion_gate_speed
        self.state.suppressed_by_motion = fast

        # Clicking is disabled while scrolling: the hand shape for two-finger
        # scroll brings the thumb close to the middle finger, which would
        # otherwise read as a right click on almost every scroll.
        if (
            self.state.clutch_engaged
            and self.state.mode == "point"
            and (not fast or self.left_click.dragging)
        ):
            events += self.left_click.update(self.state.pinch_index, now, label)
            # Only consider a right click when the index pinch is clearly open,
            # so the two cannot fire from one ambiguous hand shape.
            if self.state.pinch_index > self.t.pinch_open:
                # The detector is generic, so its CLICK is relabelled here -
                # otherwise a thumb-middle pinch would dispatch a left click.
                for ev in self.right_click.update(self.state.pinch_middle, now, label):
                    if ev.type in (GestureType.CLICK, GestureType.DOUBLE_CLICK):
                        ev = GestureEvent(
                            GestureType.RIGHT_CLICK, ev.timestamp, ev.hand, ev.value, ev.detail
                        )
                    elif ev.type in (GestureType.DRAG_START, GestureType.DRAG_END):
                        continue  # right button has no drag mode
                    events.append(ev)

        self.state.dragging = self.left_click.dragging
        return events

    def reset(self) -> None:
        self.state = GestureState()
        self.clutch_debounce.reset(False)
        self.scroll_debounce.reset(False)
        self.left_click.reset()
        self.right_click.reset()
        self._prev_palm = None
        self._prev_time = None
        self._speed_window.clear()
        self._missing_frames = 0
