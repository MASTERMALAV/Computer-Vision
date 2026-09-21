"""The action layer end to end: gesture -> event -> effect, and every guard.

The unit tests cover poses and rotation separately. This covers the path that
actually matters: a rotating hand in front of the camera must move the volume,
and must *not* move it when the system is disarmed, when the operator is not
recognised, or while the keyboard is in use. Those three guards are the only
thing standing between a perception glitch and an unwanted change to the
machine, so each is asserted rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from argus.app import _run_system_actions
from argus.config import ArgusConfig
from argus.control.dispatcher import ActionDispatcher, IdentityStatus
from argus.control.injector import MouseInjector
from argus.control.screens import Monitor, VirtualDesktop
from argus.gestures.fsm import GestureEngine, GestureType
from argus.hands.landmarks import WRIST

from conftest import pointing_hand
from test_actions import hand_with

DT = 1.0 / 30.0


def desktop() -> VirtualDesktop:
    return VirtualDesktop(0, 0, 1920, 1080, (Monitor(0, 0, 0, 1920, 1080, True, 96),))


class FakeSystem:
    """Records what would have happened, without touching the machine."""

    def __init__(self) -> None:
        self.volume: list[int] = []
        self.brightness: list[int] = []
        self.launches = 0
        self.level = 50

    def volume_step(self, steps: int) -> None:
        self.volume.append(steps)

    def brightness_step(self, steps: int, percent: int) -> int:
        self.brightness.append(steps * percent)
        self.level = max(0, min(100, self.level + steps * percent))
        return self.level

    def launch_app(self, now: float) -> bool:
        self.launches += 1
        return True


def rotating_palm(index: int, degrees_per_frame: float = 4.0):
    """An open palm turned progressively further."""
    h = hand_with(1.3, 1.3, 1.3, 1.3)
    theta = np.radians(index * degrees_per_frame)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    wrist = h.pixels[WRIST].copy()
    h.pixels[:] = (h.pixels - wrist) @ rot.T + wrist
    return h


def run(engine, frames, builder, t0=100.0):
    events = []
    for i in range(frames):
        events += engine.update(builder(i), t0 + i * DT)
    return events


@pytest.fixture
def rig():
    cfg = ArgusConfig()
    injector = MouseInjector(desktop(), armed=True)
    dispatcher = ActionDispatcher(injector, cfg.security)
    system = FakeSystem()
    return cfg, injector, dispatcher, system


def dispatch(events, rig, now=100.0, blocked=False):
    cfg, injector, dispatcher, system = rig
    notes: list[str] = []
    _run_system_actions(events, system, cfg, injector, dispatcher, now, blocked, notes)
    return notes


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #
def test_turning_an_open_palm_changes_the_volume(rig):
    engine = GestureEngine()
    events = run(engine, 26, rotating_palm)
    notes = dispatch(events, rig)
    _, _, _, system = rig
    assert system.volume, "a turned open palm must change the volume"
    assert any("volume" in n for n in notes)


def sliding_palm(index: int, pixels_per_frame: float):
    """An open palm moved vertically, without turning."""
    h = hand_with(1.3, 1.3, 1.3, 1.3)
    h.pixels[:, 1] += index * pixels_per_frame
    return h


def test_moving_the_palm_up_and_down_goes_opposite_ways(rig):
    """The default knob is a slider: up raises, down lowers."""
    _, _, _, system = rig

    engine = GestureEngine()
    dispatch(run(engine, 26, lambda i: sliding_palm(i, -6.0)), rig)  # upwards
    raised = sum(system.volume)

    system.volume.clear()
    engine2 = GestureEngine()
    dispatch(run(engine2, 26, lambda i: sliding_palm(i, +6.0)), rig)  # downwards
    lowered = sum(system.volume)

    assert raised > 0, "moving the hand up must raise the volume"
    assert lowered < 0, "moving the hand down must lower it"


def test_rotation_mode_still_works_and_is_signed():
    """The turn-a-dial mode remains available for anyone who prefers it."""
    from argus.gestures.poses import KnobConfig, KnobTracker

    clockwise = KnobTracker(KnobConfig(mode="rotate", degrees_per_step=8.0))
    anticlockwise = KnobTracker(KnobConfig(mode="rotate", degrees_per_step=8.0))

    forward = sum(clockwise.update(rotating_palm(i, +4.0), True) for i in range(26))
    backward = sum(anticlockwise.update(rotating_palm(i, -4.0), True) for i in range(26))

    assert forward != 0 and backward != 0
    assert (forward > 0) != (backward > 0), "opposite turns must go opposite ways"


def test_brightness_uses_the_full_step_count(rig):
    """A three-step turn must move brightness three steps, not one.

    Volume is a repeated key tap so a multi-step turn naturally sends several;
    brightness sets a level, and an earlier version collapsed any turn to a
    single increment, making the two knobs feel completely different.
    """
    cfg, injector, dispatcher, system = rig
    events = [
        type(
            "E", (), {"type": GestureType.KNOB_STEP, "value": 3.0, "detail": "brightness"}
        )()
    ]
    _run_system_actions(events, system, cfg, injector, dispatcher, 100.0, False, [])
    assert system.brightness == [3 * cfg.actions.brightness_percent]


def test_thumbs_up_launches(rig):
    engine = GestureEngine()
    events = run(engine, 40, lambda i: hand_with(0.5, 0.5, 0.5, 0.5, thumb_far=True))
    dispatch(events, rig)
    _, _, _, system = rig
    assert system.launches == 1


# --------------------------------------------------------------------------- #
# The guards
# --------------------------------------------------------------------------- #
def test_nothing_happens_while_disarmed(rig):
    cfg, injector, dispatcher, system = rig
    injector.disarm()
    events = run(GestureEngine(), 26, rotating_palm)
    _run_system_actions(events, system, cfg, injector, dispatcher, 100.0, False, [])
    assert system.volume == []
    assert system.launches == 0


def test_nothing_happens_while_typing(rig):
    cfg, injector, dispatcher, system = rig
    events = run(GestureEngine(), 26, rotating_palm)
    notes: list[str] = []
    _run_system_actions(events, system, cfg, injector, dispatcher, 100.0, True, notes)
    assert system.volume == []
    assert any("typing" in n for n in notes)


def test_nothing_happens_without_a_recognised_operator(rig):
    cfg, injector, dispatcher, system = rig
    cfg.security.require_identity = True
    dispatcher.set_identity(IdentityStatus(authenticated=False, name="unknown"))
    events = run(GestureEngine(), 26, rotating_palm)
    notes: list[str] = []
    _run_system_actions(events, system, cfg, injector, dispatcher, 100.0, False, notes)
    assert system.volume == []
    assert any("operator" in n for n in notes)


def test_a_recognised_operator_is_allowed(rig):
    cfg, injector, dispatcher, system = rig
    cfg.security.require_identity = True
    dispatcher.set_identity(
        IdentityStatus(authenticated=True, name="malav", last_match_time=99.9)
    )
    events = run(GestureEngine(), 26, rotating_palm)
    _run_system_actions(events, system, cfg, injector, dispatcher, 100.0, False, [])
    assert system.volume


# --------------------------------------------------------------------------- #
# Interaction with the cursor
# --------------------------------------------------------------------------- #
def test_the_cursor_does_not_move_while_turning_a_knob():
    from argus.control.pointer import PointerConfig, PointerEngine

    injector = MouseInjector(desktop(), armed=True)
    pointer = PointerEngine(injector, PointerConfig())
    engine = GestureEngine()

    # Settle pointing first so the cursor has an anchor.
    for i in range(8):
        h = pointing_hand(palm=(640.0, 360.0), pinch_index=0.95)
        ev = engine.update(h, 100.0 + i * DT)
        pointer.update(h, engine.state, ev, 100.0 + i * DT)
    before = pointer.state.position

    now = 101.0
    for i in range(26):
        h = rotating_palm(i)
        h.pixels[:, 0] += i * 6.0  # and drifting sideways while turning
        ev = engine.update(h, now)
        pointer.update(h, engine.state, ev, now)
        now += DT

    assert engine.state.mode == "volume"
    assert pointer.state.position == before, "turning a knob must not drag the cursor"


def test_a_knob_cannot_start_mid_drag_and_leaves_no_stale_pinch():
    """Entering a knob must not strand the pinch detectors.

    While a knob is active the pinch detectors are not updated at all. If one
    were left believing it was closed, releasing the pose would emit a click
    nobody asked for.
    """
    engine = GestureEngine()
    for i in range(8):
        engine.update(pointing_hand(pinch_index=0.95), 100.0 + i * DT)
    # Close the index pinch, but not long enough to become a drag.
    for i in range(3):
        engine.update(pointing_hand(pinch_index=0.15), 101.0 + i * DT)

    events = []
    now = 102.0
    for i in range(12):
        events += engine.update(rotating_palm(i), now)
        now += DT
    assert engine.state.mode == "volume"

    # Leaving the pose must not produce a click from the stale pinch.
    for i in range(10):
        events += engine.update(pointing_hand(pinch_index=0.95), now)
        now += DT
    kinds = [e.type for e in events]
    assert GestureType.CLICK not in kinds, "a stale pinch fired a click after the knob"
