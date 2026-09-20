"""Gesture state machine: hysteresis, debouncing, click/drag discrimination."""

from __future__ import annotations

import pytest

from argus.gestures.fsm import (
    Debouncer,
    GestureEngine,
    GestureThresholds,
    GestureType,
    PinchDetector,
    SchmittTrigger,
)

from conftest import make_hand

DT = 1.0 / 30.0


def types_of(events):
    return [e.type for e in events]


class Clock:
    """Monotonic test clock.

    Tests must not jump time between phases. A gap longer than drag_dwell_s
    turns a quick pinch into a held one - correct behaviour, but not what a
    click test means to assert.
    """

    def __init__(self, t0: float = 100.0, dt: float = DT):
        self.t = t0
        self.dt = dt

    def tick(self) -> float:
        now = self.t
        self.t += self.dt
        return now


def feed(engine: GestureEngine, hand, frames: int, clock=None, dt: float = DT):
    """Run the engine for N frames on a constant hand; collect all events."""
    clock = clock if clock is not None else Clock(dt=dt)
    events = []
    for _ in range(frames):
        events += engine.update(hand, clock.tick())
    return events


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def test_schmitt_requires_hysteresis():
    with pytest.raises(ValueError, match="hysteresis"):
        SchmittTrigger(enter=0.5, exit=0.4)
    with pytest.raises(ValueError):
        SchmittTrigger(enter=0.5, exit=0.5)


def test_schmitt_does_not_chatter_in_the_deadband():
    t = SchmittTrigger(enter=0.3, exit=0.5)
    assert t.update(0.6) is False
    assert t.update(0.25) is True  # closed
    # Sitting in the band between the thresholds must hold the state.
    for value in (0.35, 0.45, 0.4, 0.49):
        assert t.update(value) is True, "must stay closed inside the deadband"
    assert t.update(0.55) is False  # only opens past the looser threshold


def test_single_threshold_would_chatter_but_schmitt_does_not():
    t = SchmittTrigger(enter=0.3, exit=0.5)
    t.update(0.2)
    flips = 0
    prev = t.state
    # A signal oscillating right at 0.4 - between the thresholds.
    for value in [0.38, 0.42, 0.39, 0.41] * 10:
        if t.update(value) != prev:
            flips += 1
            prev = t.state
    assert flips == 0


def test_debouncer_ignores_single_frame_glitches():
    d = Debouncer(frames=3)
    assert d.update(True) is False
    assert d.update(False) is False  # glitch gone
    assert d.update(True) is False
    assert d.update(True) is False
    assert d.update(True) is True  # held long enough


def test_debouncer_resets_pending_on_flip_flop():
    d = Debouncer(frames=3)
    for _ in range(10):
        d.update(True)
        d.update(False)
    assert d.state is False


# --------------------------------------------------------------------------- #
# Pinch detector
# --------------------------------------------------------------------------- #
def detector(**kw):
    defaults = dict(
        finger="index", close_at=0.34, open_at=0.50, approach_at=0.62,
        debounce_frames=2, drag_dwell_s=0.35, cooldown_s=0.25, double_click_s=0.40,
    )
    defaults.update(kw)
    return PinchDetector(**defaults)


def test_quick_pinch_is_a_click():
    d = detector()
    t = 0.0
    events = []
    for _ in range(4):  # close
        events += d.update(0.20, t, "Right")
        t += DT
    for _ in range(4):  # release quickly
        events += d.update(0.90, t, "Right")
        t += DT
    assert GestureType.CLICK in types_of(events)
    assert GestureType.DRAG_START not in types_of(events)


def test_held_pinch_becomes_a_drag_and_emits_no_click():
    d = detector()
    t = 0.0
    events = []
    for _ in range(20):  # hold well past the dwell
        events += d.update(0.20, t, "Right")
        t += DT
    for _ in range(4):
        events += d.update(0.90, t, "Right")
        t += DT
    kinds = types_of(events)
    assert GestureType.DRAG_START in kinds
    assert GestureType.DRAG_END in kinds
    assert GestureType.CLICK not in kinds, "a drag must not also emit a click"


def test_drag_start_precedes_drag_end():
    d = detector()
    t, events = 0.0, []
    for _ in range(20):
        events += d.update(0.20, t, "Right"); t += DT
    for _ in range(4):
        events += d.update(0.90, t, "Right"); t += DT
    kinds = types_of(events)
    assert kinds.index(GestureType.DRAG_START) < kinds.index(GestureType.DRAG_END)


def test_approach_fires_before_the_click():
    """The freeze signal must arrive before the click, or it is useless."""
    d = detector()
    t, events = 0.0, []
    events += d.update(0.58, t, "Right")  # entering the approach band
    t += DT
    for _ in range(4):
        events += d.update(0.20, t, "Right")
        t += DT
    for _ in range(4):
        events += d.update(0.90, t, "Right")
        t += DT
    kinds = types_of(events)
    assert GestureType.PINCH_APPROACH in kinds
    assert kinds.index(GestureType.PINCH_APPROACH) < kinds.index(GestureType.CLICK)


def test_aborted_approach_emits_abort_and_no_click():
    d = detector()
    t, events = 0.0, []
    events += d.update(0.58, t, "Right"); t += DT
    for _ in range(4):
        events += d.update(0.95, t, "Right"); t += DT
    kinds = types_of(events)
    assert GestureType.PINCH_ABORT in kinds
    assert GestureType.CLICK not in kinds


def test_two_quick_pinches_make_a_double_click():
    d = detector(cooldown_s=0.05)
    t, events = 0.0, []
    for _ in range(2):
        for _ in range(4):
            events += d.update(0.20, t, "Right"); t += DT
        for _ in range(4):
            events += d.update(0.90, t, "Right"); t += DT
    kinds = types_of(events)
    assert GestureType.CLICK in kinds
    assert GestureType.DOUBLE_CLICK in kinds


def test_cooldown_suppresses_a_machine_gun_of_clicks():
    d = detector(cooldown_s=1.0)
    t, events = 0.0, []
    for _ in range(5):
        for _ in range(3):
            events += d.update(0.20, t, "Right"); t += DT
        for _ in range(3):
            events += d.update(0.90, t, "Right"); t += DT
    clicks = [k for k in types_of(events) if k in (GestureType.CLICK, GestureType.DOUBLE_CLICK)]
    assert len(clicks) == 1, f"cooldown should allow one click, got {len(clicks)}"


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
def test_extended_index_engages_the_clutch():
    e = GestureEngine()
    events = feed(e, make_hand(index_extension=1.3), 6)
    assert GestureType.CLUTCH_ENGAGE in types_of(events)
    assert e.state.clutch_engaged


def test_curled_index_does_not_engage():
    e = GestureEngine()
    feed(e, make_hand(index_extension=0.6), 10)
    assert not e.state.clutch_engaged


def test_clutch_has_pose_hysteresis():
    """Once engaged, a slight relaxation must not drop the cursor."""
    e = GestureEngine()
    c = Clock()
    feed(e, make_hand(index_extension=1.3), 6, c)
    assert e.state.clutch_engaged
    # Between finger_curled (0.85) and finger_extended (1.05).
    feed(e, make_hand(index_extension=0.95), 10, c)
    assert e.state.clutch_engaged, "clutch must not drop in the hysteresis band"
    feed(e, make_hand(index_extension=0.5), 10, c)
    assert not e.state.clutch_engaged


def test_no_clicks_while_clutch_is_released():
    e = GestureEngine()
    events = feed(e, make_hand(index_extension=0.5, pinch_index=0.10), 15)
    assert GestureType.CLICK not in types_of(events)


def test_click_requires_engaged_clutch():
    e = GestureEngine()
    c = Clock()
    feed(e, make_hand(index_extension=1.3, pinch_index=0.95), 6, c)
    events = feed(e, make_hand(index_extension=1.3, pinch_index=0.15), 5, c)
    events += feed(e, make_hand(index_extension=1.3, pinch_index=0.95), 5, c)
    assert GestureType.CLICK in types_of(events)
    assert GestureType.DRAG_START not in types_of(events)


def test_middle_pinch_produces_a_right_click_not_a_left_one():
    e = GestureEngine()
    c = Clock()
    feed(e, make_hand(index_extension=1.3, pinch_index=0.95, pinch_middle=0.95), 6, c)
    events = feed(e, make_hand(index_extension=1.3, pinch_index=0.95, pinch_middle=0.15), 5, c)
    events += feed(e, make_hand(index_extension=1.3, pinch_index=0.95, pinch_middle=0.95), 5, c)
    kinds = types_of(events)
    assert GestureType.RIGHT_CLICK in kinds
    assert GestureType.CLICK not in kinds, "thumb-middle must never emit a left click"


def test_thresholds_are_depth_invariant():
    """The same gesture at half the apparent size must behave identically."""
    def sequence(scale: float):
        e = GestureEngine()
        c = Clock()
        feed(e, make_hand(scale=scale, index_extension=1.3, pinch_index=0.95), 6, c)
        out = feed(e, make_hand(scale=scale, index_extension=1.3, pinch_index=0.15), 5, c)
        out += feed(e, make_hand(scale=scale, index_extension=1.3, pinch_index=0.95), 5, c)
        return types_of(out)

    near = sequence(200.0)  # hand close to the camera
    far = sequence(50.0)  # identical gesture, four times smaller in frame
    assert near == far
    assert GestureType.CLICK in near


def test_tiny_hand_is_rejected_as_unreliable():
    e = GestureEngine()
    feed(e, make_hand(scale=12.0, index_extension=1.3), 12)
    assert not e.state.clutch_engaged


def test_brief_dropout_does_not_end_a_drag():
    e = GestureEngine()
    c = Clock()
    feed(e, make_hand(index_extension=1.3, pinch_index=0.95), 6, c)
    feed(e, make_hand(index_extension=1.3, pinch_index=0.15), 20, c)
    assert e.state.dragging
    # Two missing frames - a normal tracking hiccup.
    e.update(None, c.tick())
    e.update(None, c.tick())
    assert e.state.dragging, "a brief dropout must not abandon a drag"


def test_sustained_hand_loss_ends_the_drag_and_clutch():
    e = GestureEngine()
    c = Clock()
    feed(e, make_hand(index_extension=1.3, pinch_index=0.95), 6, c)
    feed(e, make_hand(index_extension=1.3, pinch_index=0.15), 20, c)
    assert e.state.dragging
    events = []
    for _ in range(8):
        events += e.update(None, c.tick())
    kinds = types_of(events)
    assert GestureType.DRAG_END in kinds
    assert GestureType.CLUTCH_RELEASE in kinds
    assert not e.state.dragging


def test_releasing_the_clutch_ends_a_drag():
    e = GestureEngine()
    c = Clock()
    feed(e, make_hand(index_extension=1.3, pinch_index=0.95), 6, c)
    feed(e, make_hand(index_extension=1.3, pinch_index=0.15), 20, c)
    assert e.state.dragging
    events = feed(e, make_hand(index_extension=0.4, pinch_index=0.15), 10, c)
    assert GestureType.DRAG_END in types_of(events)


def test_fast_motion_gates_new_gestures():
    t = GestureThresholds(motion_gate_speed=0.5)
    e = GestureEngine(t)
    c = Clock()
    feed(e, make_hand(index_extension=1.3, pinch_index=0.95), 6, c)
    # Move the palm a long way each frame -> high measured speed.
    events = []
    for i in range(10):
        h = make_hand(palm=(300.0 + i * 90.0, 360.0), index_extension=1.3, pinch_index=0.15)
        events += e.update(h, c.tick())
    assert e.state.suppressed_by_motion
    assert GestureType.CLICK not in types_of(events)
