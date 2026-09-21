"""Whole-hand poses, and turning the hand as a control.

The pinches are full: index and middle each carry a tap and a hold, which is
four actions and as much as two fingers can unambiguously say. Anything else
needs a different channel, so the action layer uses *poses* - the overall shape
of the hand - and rotation rather than more pinching.

Poses are chosen to be far apart in the measurement space rather than merely
different to a human. ``OPEN_PALM`` needs four extended fingers, ``V_SIGN``
needs exactly two with the other two curled, ``THUMBS_UP`` needs all four curled
with the thumb clear of the palm. No two of them are one noisy landmark apart.

Rotation is a good fit for a continuous quantity. A knob has no travel limit -
you can keep turning - which is exactly the property hand *translation* lacks,
and it is why volume and brightness are turned rather than swiped.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..hands.landmarks import MIDDLE_MCP, WRIST, Hand


class Pose(str, Enum):
    UNKNOWN = "unknown"
    POINT = "point"  # index out - the cursor pose
    OPEN_PALM = "open_palm"  # all four fingers out
    V_SIGN = "v_sign"  # index + middle out, ring + pinky curled
    THUMBS_UP = "thumbs_up"  # all four curled, thumb clear
    FIST = "fist"  # everything curled


# Tip-to-knuckle span of each finger, relative to the index. Calibration only
# ever measures the index - it is the pointing finger - so applying that one
# number to the others asks the pinky to be as long as the index. It is not.
FINGER_LENGTH_RATIO: dict[str, float] = {
    "index": 1.00,
    "middle": 1.06,
    "ring": 0.95,
    "pinky": 0.78,
}

FINGERS_IN_ORDER = ("index", "middle", "ring", "pinky")


@dataclass
class PoseThresholds:
    """Where each finger counts as out, and where it counts as down.

    Both are derived from the single calibrated index-finger threshold rather
    than being set independently, so a calibration stays meaningful.
    """

    extended: float = 1.05
    curled: float = 0.85
    # The thumb is measured from the wrist rather than its own knuckle, which
    # barely moves; below this the thumb is tucked against the palm.
    thumb_out: float = 0.95

    def up_threshold(self, finger: str) -> float:
        return self.extended * FINGER_LENGTH_RATIO.get(finger, 1.0)

    def down_threshold(self, finger: str) -> float:
        """Halfway between clenched and extended, for that finger.

        The calibrated ``curled`` value comes from a tight fist. Poses like a V
        sign or a thumbs up fold the spare fingers loosely, nowhere near a fist,
        so requiring them to reach it meant those poses essentially never
        registered. Nearer-folded-than-extended is the question that actually
        matters.
        """
        return (self.curled + self.up_threshold(finger)) / 2.0


def classify(hand: Hand, t: PoseThresholds | None = None) -> Pose:
    """Classify the overall hand shape.

    Returns ``UNKNOWN`` rather than guessing when the hand sits between poses -
    a wrong pose acts on the system, so ambiguity must not resolve to an action.
    """
    t = t or PoseThresholds()
    ext = hand.extensions()

    out = [ext[f] > t.up_threshold(f) for f in FINGERS_IN_ORDER]
    down = [ext[f] < t.down_threshold(f) for f in FINGERS_IN_ORDER]
    thumb = ext["thumb"]

    if all(out):
        return Pose.OPEN_PALM
    if out[0] and out[1] and down[2] and down[3]:
        return Pose.V_SIGN
    if all(down):
        return Pose.THUMBS_UP if thumb > t.thumb_out else Pose.FIST
    if out[0] and not out[1]:
        return Pose.POINT
    return Pose.UNKNOWN


class PoseTracker:
    """Majority vote over a short window, so one bad frame changes nothing.

    A pose here triggers a *system* action - changing volume, launching an
    application - so it is debounced harder than the pinches, which only move a
    cursor that can be moved back.
    """

    def __init__(self, window: int = 5, required: int = 4) -> None:
        self.window = max(1, window)
        self.required = min(max(1, required), self.window)
        self._history: deque[Pose] = deque(maxlen=self.window)
        self.pose = Pose.UNKNOWN
        self.held_since: float | None = None

    def update(self, pose: Pose, now: float) -> Pose:
        self._history.append(pose)
        counts: dict[Pose, int] = {}
        for p in self._history:
            counts[p] = counts.get(p, 0) + 1

        winner, votes = max(counts.items(), key=lambda kv: kv[1])
        if votes >= self.required and winner != self.pose:
            self.pose = winner
            self.held_since = now
        elif winner != self.pose and self.pose not in counts:
            # The current pose has fallen out of the window entirely.
            self.pose = Pose.UNKNOWN
            self.held_since = None
        return self.pose

    def held_for(self, now: float) -> float:
        return 0.0 if self.held_since is None else now - self.held_since

    def reset(self) -> None:
        self._history.clear()
        self.pose = Pose.UNKNOWN
        self.held_since = None


# --------------------------------------------------------------------------- #
# Rotation
# --------------------------------------------------------------------------- #
def hand_angle(hand: Hand) -> float:
    """Roll of the hand in the image plane, in degrees.

    Measured along the wrist-to-middle-knuckle axis, which is a rigid bone
    span: it does not move when fingers open or close, so the angle reflects
    the wrist turning and nothing else.
    """
    vector = hand.pixels[MIDDLE_MCP] - hand.pixels[WRIST]
    return math.degrees(math.atan2(float(vector[1]), float(vector[0])))


@dataclass
class RotationConfig:
    # Degrees of turn per emitted step. Small enough to feel continuous,
    # large enough that hand tremor cannot produce a step.
    degrees_per_step: float = 9.0
    # Ignore jitter below this per frame.
    dead_zone_deg: float = 0.8
    # One landmark glitch must not dump twenty steps into the volume.
    max_steps_per_frame: int = 3
    # The preview is mirrored, which reverses apparent rotation. With this set,
    # turning the hand clockwise *as the operator sees it* increases the value.
    invert: bool = True


class RotationTracker:
    """Accumulates hand rotation and emits discrete, signed steps."""

    def __init__(self, config: RotationConfig | None = None) -> None:
        self.config = config or RotationConfig()
        self._previous: float | None = None
        self._accumulator = 0.0
        self.total_degrees = 0.0

    def reset(self) -> None:
        self._previous = None
        self._accumulator = 0.0

    def update(self, angle_deg: float, active: bool) -> int:
        """Feed the current hand angle. Returns whole steps to apply."""
        if not active:
            self.reset()
            return 0

        if self._previous is None:
            self._previous = angle_deg
            return 0

        # Unwrap: atan2 jumps by 360 at the seam, which would otherwise read as
        # an enormous instantaneous rotation.
        delta = angle_deg - self._previous
        while delta > 180.0:
            delta -= 360.0
        while delta < -180.0:
            delta += 360.0
        self._previous = angle_deg

        if abs(delta) < self.config.dead_zone_deg:
            return 0
        # A single frame cannot legitimately contain a large turn; anything
        # that big is a tracking glitch, not a wrist.
        if abs(delta) > 60.0:
            return 0

        self.total_degrees += delta
        direction = -1.0 if self.config.invert else 1.0
        self._accumulator += direction * delta / max(self.config.degrees_per_step, 1e-6)

        steps = int(self._accumulator)
        if not steps:
            return 0
        self._accumulator -= steps
        limit = self.config.max_steps_per_frame
        return max(-limit, min(limit, steps))


@dataclass
class KnobConfig:
    """How a held pose turns into steps of a value.

    ``vertical`` is the default. Volume and brightness are *bounded* - nought to
    a hundred - and a slider is the shape people reach for, matching both the
    on-screen widget and what an operator does without being told. Rotation was
    the first choice because a turn has no travel limit, but that property only
    matters for unbounded things like a long document; for a bounded one it buys
    nothing and costs familiarity.
    """

    mode: str = "vertical"  # vertical | rotate

    # vertical: hand travel, in hand-scale units, per step
    units_per_step: float = 0.07
    dead_zone_units: float = 0.004

    # rotate: degrees of wrist turn per step
    degrees_per_step: float = 9.0
    dead_zone_deg: float = 0.8

    max_steps_per_frame: int = 3
    invert: bool = False


class KnobTracker:
    """Turns a held pose plus hand movement into signed steps."""

    def __init__(self, config: KnobConfig | None = None) -> None:
        self.config = config or KnobConfig()
        self._previous: float | None = None
        self._accumulator = 0.0

    def reset(self) -> None:
        self._previous = None
        self._accumulator = 0.0

    def update(self, hand: Hand, active: bool) -> int:
        if not active:
            self.reset()
            return 0
        if self.config.mode == "rotate":
            return self._step(hand_angle(hand), rotational=True)
        # Screen y grows downward, so moving the hand *up* must increase.
        scale = max(hand.scale, 1e-6)
        return self._step(-float(hand.palm_center[1]) / scale, rotational=False)

    def _step(self, value: float, rotational: bool) -> int:
        cfg = self.config
        if self._previous is None:
            self._previous = value
            return 0

        delta = value - self._previous
        if rotational:
            # atan2 wraps at 180; unwrap or the seam reads as a huge turn.
            while delta > 180.0:
                delta -= 360.0
            while delta < -180.0:
                delta += 360.0
        self._previous = value

        dead = cfg.dead_zone_deg if rotational else cfg.dead_zone_units
        per_step = cfg.degrees_per_step if rotational else cfg.units_per_step
        limit = 60.0 if rotational else 0.5
        if abs(delta) < dead:
            return 0
        if abs(delta) > limit:
            return 0  # a tracking glitch, not a hand

        direction = -1.0 if cfg.invert else 1.0
        self._accumulator += direction * delta / max(per_step, 1e-6)
        steps = int(self._accumulator)
        if not steps:
            return 0
        self._accumulator -= steps
        cap = cfg.max_steps_per_frame
        return max(-cap, min(cap, steps))


def angular_span(angles: list[float]) -> float:
    """Total unwrapped rotation across a sequence of angles, in degrees."""
    if len(angles) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(angles, angles[1:]):
        delta = b - a
        while delta > 180.0:
            delta -= 360.0
        while delta < -180.0:
            delta += 360.0
        total += delta
    return total
