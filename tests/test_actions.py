"""Poses, rotation, and the action layer built on them."""

from __future__ import annotations

import pytest

from argus.gestures.fsm import GestureEngine, GestureThresholds, GestureType
from argus.gestures.poses import (
    Pose,
    PoseThresholds,
    PoseTracker,
    RotationConfig,
    RotationTracker,
    angular_span,
    classify,
    hand_angle,
)

from conftest import make_hand, pointing_hand

DT = 1.0 / 30.0


def hand_with(index=1.3, middle=0.6, ring=0.6, pinky=0.6, thumb_far=False, **kw):
    """A hand with each finger's extension set independently."""
    h = make_hand(index_extension=index, middle_extension=middle, **kw)
    import numpy as np

    from argus.hands.landmarks import (
        PINKY_MCP, PINKY_TIP, RING_MCP, RING_TIP, THUMB_TIP, WRIST,
    )

    scale = h.scale
    # Place ring and pinky tips at the requested extension from their knuckles.
    for mcp, tip, ext in ((RING_MCP, RING_TIP, ring), (PINKY_MCP, PINKY_TIP, pinky)):
        direction = np.array([0.0, -1.0]) if ext >= 1.0 else np.array([0.0, 1.0])
        h.pixels[tip] = h.pixels[mcp] + direction * ext * scale
    # The thumb is placed explicitly rather than inherited from the pinch
    # geometry. A real fist folds the thumb across the palm; leaving it where a
    # thumb-to-index pinch would put it leaves it sticking out sideways, which
    # reads as a thumbs-up and is not a fist at all.
    if thumb_far:
        h.pixels[THUMB_TIP] = h.pixels[WRIST] + np.array(
            [0.0, -1.4 * scale], dtype="float32"
        )
    else:
        h.pixels[THUMB_TIP] = h.pixels[WRIST] + np.array(
            [0.30 * scale, -0.45 * scale], dtype="float32"
        )
    return h


# --------------------------------------------------------------------------- #
# Pose classification
# --------------------------------------------------------------------------- #
def test_open_palm():
    assert classify(hand_with(1.3, 1.3, 1.3, 1.3)) is Pose.OPEN_PALM


def test_v_sign():
    assert classify(hand_with(1.3, 1.3, 0.5, 0.5)) is Pose.V_SIGN


def test_point():
    assert classify(hand_with(1.3, 0.6, 0.6, 0.6)) is Pose.POINT


def test_fist_versus_thumbs_up():
    """The thumb is the only difference, so it must be the only thing checked."""
    assert classify(hand_with(0.5, 0.5, 0.5, 0.5, thumb_far=False)) is Pose.FIST
    assert classify(hand_with(0.5, 0.5, 0.5, 0.5, thumb_far=True)) is Pose.THUMBS_UP


def test_ambiguous_hands_are_unknown_not_guessed():
    """A wrong pose acts on the machine, so ambiguity must not become an action."""
    # Three fingers out: neither a V nor an open palm.
    assert classify(hand_with(1.3, 1.3, 1.3, 0.5)) is Pose.UNKNOWN


def test_poses_are_far_apart_in_measurement_space():
    """Each pose should need more than one landmark to move to become another."""
    palm = hand_with(1.3, 1.3, 1.3, 1.3)
    v = hand_with(1.3, 1.3, 0.5, 0.5)
    assert classify(palm) is not classify(v)
    # Nudging one finger must not flip open palm into V.
    nudged = hand_with(1.3, 1.3, 1.3, 0.95)
    assert classify(nudged) is not Pose.V_SIGN


# --------------------------------------------------------------------------- #
# Pose debouncing
# --------------------------------------------------------------------------- #
def test_a_single_bad_frame_does_not_change_the_pose():
    t = PoseTracker(window=5, required=4)
    for i in range(6):
        t.update(Pose.OPEN_PALM, i * DT)
    assert t.pose is Pose.OPEN_PALM
    t.update(Pose.V_SIGN, 7 * DT)
    assert t.pose is Pose.OPEN_PALM, "one frame must not switch a system action"


def test_a_sustained_change_is_accepted():
    t = PoseTracker(window=5, required=4)
    for i in range(6):
        t.update(Pose.OPEN_PALM, i * DT)
    for i in range(5):
        t.update(Pose.V_SIGN, (7 + i) * DT)
    assert t.pose is Pose.V_SIGN


def test_held_for_measures_from_the_change():
    t = PoseTracker(window=3, required=2)
    for i in range(3):
        t.update(Pose.THUMBS_UP, 100.0 + i * DT)
    assert t.held_for(101.0) > 0.9


# --------------------------------------------------------------------------- #
# Rotation
# --------------------------------------------------------------------------- #
def test_hand_angle_follows_the_wrist():
    import numpy as np

    h = pointing_hand()
    base = hand_angle(h)
    # Rotate the whole hand 30 degrees about the wrist.
    from argus.hands.landmarks import WRIST

    theta = np.radians(30.0)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    wrist = h.pixels[WRIST].copy()
    h.pixels[:] = (h.pixels - wrist) @ rot.T + wrist
    assert hand_angle(h) == pytest.approx(base + 30.0, abs=0.5)


def test_rotation_emits_steps_in_both_directions():
    r = RotationTracker(RotationConfig(degrees_per_step=10.0, invert=False))
    up = sum(r.update(a, True) for a in range(0, 60, 3))
    r.reset()
    down = sum(r.update(-a, True) for a in range(0, 60, 3))
    assert up > 0
    assert down < 0


def test_invert_reverses_the_direction():
    plain = RotationTracker(RotationConfig(degrees_per_step=10.0, invert=False))
    flipped = RotationTracker(RotationConfig(degrees_per_step=10.0, invert=True))
    a = sum(plain.update(x, True) for x in range(0, 60, 3))
    b = sum(flipped.update(x, True) for x in range(0, 60, 3))
    assert a == -b


def test_a_still_hand_emits_nothing():
    r = RotationTracker()
    assert sum(r.update(42.0, True) for _ in range(40)) == 0


def test_the_360_degree_seam_does_not_produce_a_huge_jump():
    """atan2 wraps at 180; unwrapping must treat that as a small step."""
    r = RotationTracker(RotationConfig(degrees_per_step=5.0, invert=False))
    steps = [r.update(a, True) for a in (170.0, 175.0, 179.0, -179.0, -175.0)]
    assert all(abs(s) <= 3 for s in steps), f"seam produced {steps}"


def test_a_tracking_glitch_is_rejected():
    r = RotationTracker(RotationConfig(degrees_per_step=5.0))
    r.update(0.0, True)
    assert r.update(120.0, True) == 0, "an impossible jump must not turn the knob"


def test_releasing_resets_the_accumulator():
    r = RotationTracker(RotationConfig(degrees_per_step=10.0, invert=False))
    for a in range(0, 30, 3):
        r.update(a, True)
    r.update(0.0, False)
    assert r._previous is None


def test_angular_span_unwraps():
    assert angular_span([170.0, 179.0, -179.0, -170.0]) == pytest.approx(20.0, abs=0.01)


# --------------------------------------------------------------------------- #
# The action layer in the engine
# --------------------------------------------------------------------------- #
def feed(engine, hand, frames, t0=100.0):
    events = []
    for i in range(frames):
        events += engine.update(hand, t0 + i * DT)
    return events


def types_of(events):
    return [e.type for e in events]


def test_open_palm_starts_the_volume_knob():
    e = GestureEngine()
    events = feed(e, hand_with(1.3, 1.3, 1.3, 1.3), 8)
    assert GestureType.KNOB_START in types_of(events)
    assert e.state.mode == "volume"


def test_v_sign_starts_the_brightness_knob():
    e = GestureEngine()
    feed(e, hand_with(1.3, 1.3, 0.5, 0.5), 8)
    assert e.state.mode == "brightness"


def test_turning_an_open_palm_emits_volume_steps():
    import numpy as np

    from argus.hands.landmarks import WRIST

    e = GestureEngine()
    now = 100.0
    events = []
    for i in range(24):
        h = hand_with(1.3, 1.3, 1.3, 1.3)
        theta = np.radians(i * 4.0)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        wrist = h.pixels[WRIST].copy()
        h.pixels[:] = (h.pixels - wrist) @ rot.T + wrist
        events += e.update(h, now)
        now += DT
    steps = [ev for ev in events if ev.type is GestureType.KNOB_STEP]
    assert steps, "turning the hand must move the knob"
    assert all(ev.detail == "volume" for ev in steps)


def test_leaving_the_pose_ends_the_knob():
    e = GestureEngine()
    feed(e, hand_with(1.3, 1.3, 1.3, 1.3), 8)
    events = feed(e, pointing_hand(pinch_index=0.95), 8, t0=101.0)
    assert GestureType.KNOB_END in types_of(events)
    assert e.state.mode == "point"


def test_thumbs_up_launches_only_after_a_deliberate_hold():
    t = GestureThresholds(launch_hold_s=0.5)
    e = GestureEngine(t)
    early = feed(e, hand_with(0.5, 0.5, 0.5, 0.5, thumb_far=True), 6)
    assert GestureType.LAUNCH not in types_of(early)
    later = feed(e, hand_with(0.5, 0.5, 0.5, 0.5, thumb_far=True), 25, t0=100.2)
    assert GestureType.LAUNCH in types_of(later)


def test_holding_thumbs_up_launches_once_not_repeatedly():
    t = GestureThresholds(launch_hold_s=0.3)
    e = GestureEngine(t)
    events = feed(e, hand_with(0.5, 0.5, 0.5, 0.5, thumb_far=True), 90)
    launches = [x for x in types_of(events) if x is GestureType.LAUNCH]
    assert len(launches) == 1, f"held pose launched {len(launches)} times"


def test_a_plain_fist_does_not_launch():
    t = GestureThresholds(launch_hold_s=0.3)
    e = GestureEngine(t)
    events = feed(e, hand_with(0.5, 0.5, 0.5, 0.5, thumb_far=False), 60)
    assert GestureType.LAUNCH not in types_of(events)


def test_actions_can_be_disabled():
    t = GestureThresholds(actions_enabled=False)
    e = GestureEngine(t)
    events = feed(e, hand_with(1.3, 1.3, 1.3, 1.3), 10)
    assert GestureType.KNOB_START not in types_of(events)


def test_a_knob_does_not_hijack_a_drag_in_progress():
    """Opening the hand mid-drag must not silently become a volume change."""
    e = GestureEngine()
    feed(e, pointing_hand(pinch_index=0.95), 8)
    feed(e, pointing_hand(pinch_index=0.15), 20, t0=101.0)
    assert e.state.dragging
    events = feed(e, hand_with(1.3, 1.3, 1.3, 1.3), 10, t0=102.0)
    # The drag ends first (the pinch opened); a knob may then start, but never
    # while the drag was still held.
    kinds = types_of(events)
    if GestureType.KNOB_START in kinds and GestureType.DRAG_END in kinds:
        assert kinds.index(GestureType.DRAG_END) < kinds.index(GestureType.KNOB_START)


# --------------------------------------------------------------------------- #
# Reported from real use, and reproduced here
#
# With the operator's own calibration (extended 0.958 from the index finger,
# curled 0.267 from a tight fist) the V sign never registered at all and the
# thumbs up worked roughly half the time. Both came from applying one finger's
# thresholds to every finger.
# --------------------------------------------------------------------------- #
REAL_EXTENDED = 0.958
REAL_CURLED = 0.267


def real_thresholds():
    return PoseThresholds(extended=REAL_EXTENDED, curled=REAL_CURLED, thumb_out=0.95)


# Measured spans for a hand held naturally, not posed for the machine.
OUT = {"index": 1.15, "middle": 1.25, "ring": 1.10, "pinky": 0.88}
FOLDED = {"index": 0.45, "middle": 0.48, "ring": 0.45, "pinky": 0.40}


def natural_hand(index, middle, ring, pinky, thumb_far=False):
    return hand_with(index, middle, ring, pinky, thumb_far=thumb_far)


def test_a_short_pinky_no_longer_breaks_the_open_palm():
    """The pinky is about 78% of the index. One threshold asked it to be as
    long, so an open palm depended on luck - and when it failed the clutch
    stayed engaged and the cursor kept moving."""
    hand = natural_hand(OUT["index"], OUT["middle"], OUT["ring"], OUT["pinky"])
    assert classify(hand, real_thresholds()) is Pose.OPEN_PALM


def test_loosely_folded_fingers_make_a_v_sign():
    """Calibrated 'curled' comes from a clenched fist. A V sign folds the spare
    fingers loosely, and requiring a fist meant it never registered."""
    hand = natural_hand(OUT["index"], OUT["middle"], FOLDED["ring"], FOLDED["pinky"])
    assert classify(hand, real_thresholds()) is Pose.V_SIGN


def test_a_loose_thumbs_up_registers():
    hand = natural_hand(
        FOLDED["index"], FOLDED["middle"], FOLDED["ring"], FOLDED["pinky"], thumb_far=True
    )
    assert classify(hand, real_thresholds()) is Pose.THUMBS_UP


def test_a_loose_fist_is_still_not_a_thumbs_up():
    """Relaxing the thumb must not launch an application."""
    hand = natural_hand(
        FOLDED["index"], FOLDED["middle"], FOLDED["ring"], FOLDED["pinky"], thumb_far=False
    )
    assert classify(hand, real_thresholds()) is Pose.FIST


def test_pointing_is_unaffected_by_the_change():
    hand = natural_hand(OUT["index"], FOLDED["middle"], FOLDED["ring"], FOLDED["pinky"])
    assert classify(hand, real_thresholds()) is Pose.POINT


def test_each_finger_gets_its_own_threshold():
    from argus.gestures.poses import FINGER_LENGTH_RATIO

    t = real_thresholds()
    assert t.up_threshold("pinky") < t.up_threshold("index")
    assert t.up_threshold("middle") > t.up_threshold("index")
    assert FINGER_LENGTH_RATIO["pinky"] < FINGER_LENGTH_RATIO["index"]


def test_down_sits_between_folded_and_extended():
    t = real_thresholds()
    for finger in ("index", "middle", "ring", "pinky"):
        assert REAL_CURLED < t.down_threshold(finger) < t.up_threshold(finger)
