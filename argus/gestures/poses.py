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


@dataclass
class PoseThresholds:
    extended: float = 1.05
    curled: float = 0.85
    # The thumb is measured from the wrist rather than its own knuckle, which
    # barely moves; below this the thumb is tucked against the palm.
    thumb_out: float = 1.05


def classify(hand: Hand, t: PoseThresholds | None = None) -> Pose:
    """Classify the overall hand shape.

    Returns ``UNKNOWN`` rather than guessing when the hand sits between poses -
    a wrong pose acts on the system, so ambiguity must not resolve to an action.
    """
    t = t or PoseThresholds()
    ext = hand.extensions()
    index, middle = ext["index"], ext["middle"]
    ring, pinky = ext["ring"], ext["pinky"]
    thumb = ext["thumb"]

    out = [f > t.extended for f in (index, middle, ring, pinky)]
    tucked = [f < t.curled for f in (index, middle, ring, pinky)]

    if all(out):
        return Pose.OPEN_PALM
    if out[0] and out[1] and tucked[2] and tucked[3]:
        return Pose.V_SIGN
    if all(tucked):
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
