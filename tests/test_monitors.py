"""Jumping the cursor between displays."""

from __future__ import annotations

import pytest

from argus.control.injector import MouseInjector
from argus.control.pointer import PointerConfig, PointerEngine
from argus.control.screens import (
    Monitor,
    VirtualDesktop,
    absolute_position,
    jump_target,
    relative_position,
)


def mixed_desktop() -> VirtualDesktop:
    """The development machine: different sizes, offsets and DPI.

    Monitor 0 is 2560x1440 at 96 dpi sitting left of the origin and slightly
    higher; monitor 1 is 1920x1080 at 144 dpi. Nothing about them matches, which
    is exactly why a fixed pixel offset would be useless.
    """
    left = Monitor(0, -2560, -158, 0, 1282, is_primary=False, dpi=96)
    right = Monitor(1, 0, 0, 1920, 1080, is_primary=True, dpi=144)
    return VirtualDesktop(-2560, -158, 4480, 1440, (left, right))


def single_desktop() -> VirtualDesktop:
    only = Monitor(0, 0, 0, 1920, 1080, True, 96)
    return VirtualDesktop(0, 0, 1920, 1080, (only,))


# --------------------------------------------------------------------------- #
def test_relative_position_round_trips():
    monitor = mixed_desktop().monitors[0]
    for point in ((-2560, -158), (-1280, 500), (-1, 1281)):
        fx, fy = relative_position(monitor, *point)
        back = absolute_position(monitor, fx, fy)
        assert back[0] == pytest.approx(point[0], abs=1.5)
        assert back[1] == pytest.approx(point[1], abs=1.5)


def test_relative_position_is_clamped():
    monitor = mixed_desktop().monitors[1]
    assert relative_position(monitor, -99999, -99999) == (0.0, 0.0)
    assert relative_position(monitor, 99999, 99999) == (1.0, 1.0)


def test_jump_lands_on_the_other_monitor():
    desktop = mixed_desktop()
    x, y, target = jump_target(desktop, -1280, 562)
    assert target.index == 1
    assert desktop.monitor_at(x, y).index == 1


def test_jump_is_reversible():
    desktop = mixed_desktop()
    x, y, _ = jump_target(desktop, -1280, 562)
    back_x, back_y, target = jump_target(desktop, x, y)
    assert target.index == 0
    assert back_x == pytest.approx(-1280, abs=3)
    assert back_y == pytest.approx(562, abs=3)


def test_relative_position_is_preserved_not_centred():
    """Landing in the middle every time would discard where the pointer was."""
    desktop = mixed_desktop()
    # Top-left corner of the left monitor.
    x, y, target = jump_target(desktop, -2560, -158)
    fx, fy = relative_position(target, x, y)
    assert fx == pytest.approx(0.0, abs=0.01)
    assert fy == pytest.approx(0.0, abs=0.01)

    # Bottom-right corner of the same monitor.
    x, y, target = jump_target(desktop, -1, 1281)
    fx, fy = relative_position(target, x, y)
    assert fx == pytest.approx(1.0, abs=0.01)
    assert fy == pytest.approx(1.0, abs=0.01)


def test_the_landing_point_is_always_inside_the_desktop():
    desktop = mixed_desktop()
    for start in ((-2560, -158), (-1, 1281), (0, 0), (1919, 1079), (960, 540)):
        x, y, _ = jump_target(desktop, *start)
        assert desktop.left <= x <= desktop.right - 1
        assert desktop.top <= y <= desktop.bottom - 1


def test_a_single_monitor_jump_is_a_no_op():
    desktop = single_desktop()
    x, y, target = jump_target(desktop, 500, 400)
    assert (x, y) == (500, 400)
    assert target.index == 0


# --------------------------------------------------------------------------- #
def test_jump_to_clears_the_anchor():
    """Otherwise the next frame computes its displacement against the position
    before the jump and immediately undoes it."""
    injector = MouseInjector(mixed_desktop(), armed=True)
    pointer = PointerEngine(injector, PointerConfig())

    from argus.gestures.fsm import GestureState

    from conftest import pointing_hand

    state = GestureState(clutch_engaged=True)
    for i in range(6):
        hand = pointing_hand(palm=(640.0 + i * 5, 360.0))
        pointer.update(hand, state, [], 100.0 + i / 30.0)

    pointer.jump_to(-1280, 562)
    assert pointer._prev_source is None
    assert pointer.state.position == (-1280.0, 562.0)
    assert pointer.state.monitor == 0


def test_jump_to_is_clamped_to_the_desktop():
    injector = MouseInjector(mixed_desktop(), armed=True)
    pointer = PointerEngine(injector, PointerConfig())
    pointer.jump_to(99999, 99999)
    x, y = pointer.state.position
    assert x <= mixed_desktop().right - 1
    assert y <= mixed_desktop().bottom - 1


def test_a_jump_does_not_drag_the_cursor_back_next_frame():
    from argus.gestures.fsm import GestureState

    from conftest import pointing_hand

    injector = MouseInjector(mixed_desktop(), armed=True)
    pointer = PointerEngine(injector, PointerConfig())
    state = GestureState(clutch_engaged=True)

    for i in range(6):
        pointer.update(pointing_hand(palm=(640.0, 360.0)), state, [], 100.0 + i / 30.0)
    pointer.jump_to(-1280, 562)

    # The hand has not moved, so nothing should pull the cursor away.
    pointer.update(pointing_hand(palm=(640.0, 360.0)), state, [], 101.0)
    pointer.update(pointing_hand(palm=(640.0, 360.0)), state, [], 101.0 + 1 / 30.0)
    x, _ = pointer.state.position
    assert x == pytest.approx(-1280, abs=5)
