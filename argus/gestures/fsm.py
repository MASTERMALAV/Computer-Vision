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
index extended means the cursor is live, and relaxing the hand parks it.

That leaves the two pinches free, and each carries the same tap/hold pair:

    index pinch    tap -> left click     hold + move -> drag
    middle pinch   tap -> right click    hold + move -> scroll

One rule to learn rather than four, and no extra hand shape for scrolling - an
earlier design used two extended fingers, which required finger precision and a
pose change at the same moment as a movement.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from ..hands.landmarks import Hand
from ..logsetup import get_logger
from .poses import (
    KnobConfig,
    KnobTracker,
    Pose,
    PoseThresholds,
    PoseTracker,
    classify,
)

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
    SCROLL_START = "scroll_start"
    SCROLL_END = "scroll_end"
    # The action layer: a pose selects what is being adjusted, and turning the
    # hand adjusts it. See argus/gestures/poses.py.
    KNOB_START = "knob_start"
    KNOB_STEP = "knob_step"
    KNOB_END = "knob_end"
    LAUNCH = "launch"
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

    # Holding the thumb-middle pinch past this turns it into a scroll, exactly
    # as holding the thumb-index pinch turns it into a drag. Slightly shorter
    # than the drag dwell, because a right click is a rarer intent than a
    # scroll and waiting to find out feels sluggish.
    scroll_dwell_s: float = 0.30

    # ---- action layer ---- #
    actions_enabled: bool = True
    # How long the thumbs-up must be held before the app launches. Long enough
    # that a thumbs-up meant for a person in the room does not start something.
    launch_hold_s: float = 0.90
    launch_cooldown_s: float = 2.50
    thumb_out: float = 1.05
    knob: KnobConfig = field(default_factory=KnobConfig)


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
    # point | scroll | volume | brightness. Anything other than "point" means
    # the cursor is deliberately held still.
    mode: str = "point"
    pose: str = "unknown"
    knob_value: int = 0  # signed steps emitted this frame
    # True as soon as the hand *looks* like an action pose, before the debounce
    # commits. The cursor stops here; the action still waits.
    action_pending: bool = False
    # 0..1 while a thumbs-up is being held, so the operator can see it counting
    # rather than guessing whether the pose registered at all.
    launch_progress: float = 0.0



class GestureEngine:
    """Turns a stream of :class:`Hand` observations into gesture events."""

    def __init__(self, thresholds: GestureThresholds | None = None) -> None:
        self.t = thresholds or GestureThresholds()
        self.state = GestureState()

        self.clutch_debounce = Debouncer(self.t.clutch_debounce_frames)
        self.poses = PoseTracker(window=5, required=4)
        self.knob = KnobTracker(self.t.knob)
        self._pose_thresholds = PoseThresholds(
            extended=self.t.finger_extended,
            curled=self.t.finger_curled,
            thumb_out=self.t.thumb_out,
        )
        # Which pose adjusts what. Poses rather than more pinches, because the
        # two pinches already carry a tap and a hold each.
        self._knobs = {Pose.OPEN_PALM: "volume", Pose.V_SIGN: "brightness"}
        self._active_knob = ""
        self._launched_at = -1e9
        self._launch_armed = True
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
            # Held rather than tapped, this becomes a scroll. The detector's
            # "drag" machinery models it exactly - a press, a period of holding,
            # and a release - so it is reused and the events are relabelled.
            drag_dwell_s=self.t.scroll_dwell_s,
            cooldown_s=self.t.click_cooldown_s,
            # No double right click; a second tap is just another right click.
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
            if self.right_click.dragging:
                self.right_click.dragging = False
                events.append(GestureEvent(GestureType.SCROLL_END, now, detail="hand lost"))
            if self.state.hand_present:
                events.append(GestureEvent(GestureType.HAND_LOST, now))
            self.state.hand_present = False
            if self._active_knob:
                events.append(
                    GestureEvent(GestureType.KNOB_END, now, detail=self._active_knob)
                )
                self._active_knob = ""
            self.state.dragging = False
            self.state.mode = "point"
            self.state.action_pending = False
            self.state.pose = "unknown"
            self.poses.reset()
            self.knob.reset()
            self._launch_armed = True
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
            if not engaged:
                if self.left_click.dragging:
                    self.left_click.dragging = False
                    events.append(
                        GestureEvent(GestureType.DRAG_END, now, label, detail="clutch released")
                    )
                if self.right_click.dragging:
                    self.right_click.dragging = False
                    events.append(
                        GestureEvent(GestureType.SCROLL_END, now, label, detail="clutch released")
                    )

        # ---- action layer ------------------------------------------------ #
        # Evaluated before the pinches, because an action pose takes the hand
        # out of cursor duty entirely and must win any ambiguity.
        events += self._update_actions(hand, now, label)
        if self._active_knob:
            self.state.dragging = self.left_click.dragging
            self.state.mode = self._active_knob
            return events

        # ---- pinches ----------------------------------------------------- #
        # Landmarks are least trustworthy while the hand is moving fast, and a
        # fast-moving hand is not trying to click. Gate new gestures on that,
        # but never abandon one already in progress.
        fast = speed > self.t.motion_gate_speed
        self.state.suppressed_by_motion = fast
        scrolling = self.right_click.dragging

        if self.state.clutch_engaged:
            # The index pinch is ignored while scrolling. The hand is already
            # holding a pinch and sweeping, and a stray index pinch there would
            # click whatever the page had just scrolled under the cursor.
            if not scrolling and (not fast or self.left_click.dragging):
                events += self.left_click.update(self.state.pinch_index, now, label)

            # The middle pinch keeps being evaluated once a scroll is running,
            # including during fast motion - scrolling *is* fast motion, and
            # gating it would strand the gesture with no way to end. Starting
            # one still requires a still-ish hand and a clearly open index
            # pinch, so the two pinches can never be confused for each other.
            if scrolling or (
                not fast
                and not self.left_click.dragging
                and self.state.pinch_index > self.t.pinch_open
            ):
                for event in self.right_click.update(self.state.pinch_middle, now, label):
                    events.append(self._relabel_middle(event))

        self.state.dragging = self.left_click.dragging
        self.state.mode = "scroll" if self.right_click.dragging else "point"
        return events

    def _update_actions(self, hand: Hand, now: float, label: str) -> list[GestureEvent]:
        """Poses and rotation: volume, brightness, and launching an app."""
        events: list[GestureEvent] = []
        if not self.t.actions_enabled:
            return events

        raw = classify(hand, self._pose_thresholds)
        # Freeze early, commit late - the same shape as the click freeze. An
        # open palm keeps the index extended, so without this the clutch would
        # keep steering the cursor for the whole debounce window.
        self.state.action_pending = raw in self._knobs or raw is Pose.THUMBS_UP

        pose = self.poses.update(raw, now)
        self.state.pose = pose.value
        self.state.knob_value = 0

        # A pinch already in progress owns the hand; a pose must not hijack it.
        busy = self.left_click.dragging or self.right_click.dragging
        knob = "" if busy else self._knobs.get(pose, "")

        if knob and knob != self._active_knob:
            # Entering a knob stops the pinch detectors being updated at all,
            # so any half-open pinch is cleared rather than left to fire a
            # click when the pose is released.
            self.left_click.reset()
            self.right_click.reset()

        if knob != self._active_knob:
            if self._active_knob:
                events.append(
                    GestureEvent(GestureType.KNOB_END, now, label, detail=self._active_knob)
                )
            self.knob.reset()
            self._active_knob = knob
            if knob:
                events.append(GestureEvent(GestureType.KNOB_START, now, label, detail=knob))

        if self._active_knob:
            steps = self.knob.update(hand, active=True)
            if steps:
                self.state.knob_value = steps
                events.append(
                    GestureEvent(
                        GestureType.KNOB_STEP, now, label,
                        value=float(steps), detail=self._active_knob,
                    )
                )
            return events

        # ---- launch ---- #
        # Requires a deliberate hold, and re-arms only once the pose is
        # released, so holding a thumbs-up cannot launch repeatedly.
        if pose is Pose.THUMBS_UP:
            held = self.poses.held_for(now)
            self.state.launch_progress = (
                min(1.0, held / max(self.t.launch_hold_s, 1e-6))
                if self._launch_armed else 1.0
            )
            if (
                self._launch_armed
                and held >= self.t.launch_hold_s
                and (now - self._launched_at) >= self.t.launch_cooldown_s
            ):
                self._launch_armed = False
                self._launched_at = now
                events.append(GestureEvent(GestureType.LAUNCH, now, label, value=held))
        else:
            self._launch_armed = True
            self.state.launch_progress = 0.0

        return events

    @staticmethod
    def _relabel_middle(event: GestureEvent) -> GestureEvent:
        """Give the middle-finger detector's generic events their meaning.

        A tap is a right click. Held past the dwell it is a scroll, which the
        detector models as a drag - press, hold, release - because the
        mechanics are identical and only the effect differs.
        """
        mapping = {
            GestureType.CLICK: GestureType.RIGHT_CLICK,
            GestureType.DOUBLE_CLICK: GestureType.RIGHT_CLICK,
            GestureType.DRAG_START: GestureType.SCROLL_START,
            GestureType.DRAG_END: GestureType.SCROLL_END,
        }
        replacement = mapping.get(event.type)
        if replacement is None:
            return event
        return GestureEvent(
            replacement, event.timestamp, event.hand, event.value, event.detail
        )

    def reset(self) -> None:
        self.state = GestureState()
        self.clutch_debounce.reset(False)
        self.left_click.reset()
        self.right_click.reset()
        self.poses.reset()
        self.knob.reset()
        self._active_knob = ""
        self._launch_armed = True
        self._prev_palm = None
        self._prev_time = None
        self._speed_window.clear()
        self._missing_frames = 0
