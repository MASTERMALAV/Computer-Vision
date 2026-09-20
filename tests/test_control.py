"""Screen geometry, injection safety, pointer mapping and action dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from argus.config import SecurityConfig
from argus.control.dispatcher import ActionDispatcher, IdentityStatus
from argus.control.injector import MouseInjector
from argus.control.pointer import PointerConfig, PointerEngine
from argus.control.screens import Monitor, VirtualDesktop
from argus.gestures.fsm import GestureEngine, GestureEvent, GestureType

from conftest import make_hand, pointing_hand, scrolling_hand

DT = 1.0 / 30.0


def dual_desktop() -> VirtualDesktop:
    """The development machine: 2560x1440 to the LEFT of a 1920x1080 primary.

    The left monitor therefore occupies negative X - the case that breaks naive
    single-screen coordinate maths.
    """
    left = Monitor(0, -2560, -158, 0, 1282, is_primary=False, dpi=96)
    right = Monitor(1, 0, 0, 1920, 1080, is_primary=True, dpi=144)
    return VirtualDesktop(left=-2560, top=-158, width=4480, height=1440,
                          monitors=(left, right))


# --------------------------------------------------------------------------- #
# Virtual desktop geometry
# --------------------------------------------------------------------------- #
def test_corners_map_to_the_full_absolute_range():
    vd = dual_desktop()
    assert vd.to_absolute(vd.left, vd.top) == (0, 0)
    # width-1 / height-1: using width would make the last column unreachable,
    # so a maximise button on the right edge could never be clicked.
    assert vd.to_absolute(vd.right - 1, vd.bottom - 1) == (65535, 65535)


def test_negative_coordinates_are_reachable():
    """A monitor left of the primary must be addressable at all."""
    vd = dual_desktop()
    x, y = vd.to_absolute(-2000, 500)
    assert 0 < x < 65535
    # Naive code that clamps to [0, width) would map every negative x to 0.
    assert x > 100


def test_absolute_mapping_round_trips():
    vd = dual_desktop()
    for px in range(vd.left, vd.right, 137):
        for py in range(vd.top, vd.bottom, 91):
            nx, ny = vd.to_absolute(px, py)
            rx = vd.left + nx * (vd.width - 1) / 65535.0
            ry = vd.top + ny * (vd.height - 1) / 65535.0
            assert abs(rx - px) < 1.0 and abs(ry - py) < 1.0


def test_clamp_keeps_the_cursor_on_the_desktop():
    vd = dual_desktop()
    assert vd.clamp(-99999, -99999) == (vd.left, vd.top)
    assert vd.clamp(99999, 99999) == (vd.right - 1, vd.bottom - 1)


def test_monitor_lookup():
    vd = dual_desktop()
    assert vd.monitor_at(-1000, 500).index == 0
    assert vd.monitor_at(960, 540).index == 1
    assert vd.monitor_at(99999, 99999).index == vd.primary.index


def test_primary_is_identified():
    assert dual_desktop().primary.index == 1


# --------------------------------------------------------------------------- #
# Injector safety
# --------------------------------------------------------------------------- #
def test_injector_starts_disarmed():
    inj = MouseInjector(dual_desktop())
    assert not inj.armed
    inj.move_to(100, 100)
    inj.click("left")
    assert inj.stats.suppressed > 0


def test_disarm_releases_held_buttons():
    inj = MouseInjector(dual_desktop())
    inj.button_down("left")
    assert "left" in inj.buttons_down
    inj.disarm()
    assert inj.buttons_down == frozenset()


def test_release_all_is_idempotent():
    inj = MouseInjector(dual_desktop())
    inj.release_all()
    inj.release_all()
    assert inj.buttons_down == frozenset()


def test_unknown_button_is_rejected():
    inj = MouseInjector(dual_desktop())
    with pytest.raises(ValueError):
        inj.button_down("fourth")


def test_simulated_position_tracks_moves_while_disarmed():
    inj = MouseInjector(dual_desktop())
    inj.move_to(-1000, 400)
    assert inj.position() == (-1000, 400)


# --------------------------------------------------------------------------- #
# Pointer engine
# --------------------------------------------------------------------------- #
def engaged_state(speed: float = 0.5):
    from argus.gestures.fsm import GestureState

    return GestureState(clutch_engaged=True, hand_speed=speed)


def build_pointer(**kw):
    inj = MouseInjector(dual_desktop())
    return PointerEngine(inj, PointerConfig(**kw)), inj


def drive(pointer, xs, scale=100.0, t0=100.0, state=None, events=None):
    """Feed a sequence of palm x-positions; return cursor positions."""
    state = state or engaged_state()
    out = []
    for i, x in enumerate(xs):
        hand = make_hand(palm=(x, 360.0), scale=scale, index_extension=1.3)
        pointer.update(hand, state, events if i == 0 and events else [], t0 + i * DT)
        out.append(pointer.state.position)
    return out


def test_cursor_does_not_move_when_clutch_is_released():
    from argus.gestures.fsm import GestureState

    p, _ = build_pointer()
    start = p.state.position
    for i in range(10):
        hand = make_hand(palm=(500 + i * 30, 360.0), index_extension=1.3)
        p.update(hand, GestureState(clutch_engaged=False), [], 100.0 + i * DT)
    assert p.state.position == start, "a released clutch must park the cursor"


def test_moving_the_hand_right_moves_the_cursor_right():
    p, _ = build_pointer()
    out = drive(p, [500 + i * 12 for i in range(25)])
    assert out[-1][0] > out[0][0]


def test_stationary_hand_produces_a_stationary_cursor():
    p, _ = build_pointer()
    out = drive(p, [500.0] * 30)
    assert out[-1] == pytest.approx(out[5], abs=0.51)


def test_mapping_is_depth_invariant():
    """The same gesture at a different distance must move the cursor equally.

    The hand travels the same number of *hand widths* in both cases, just at
    different apparent pixel sizes - the cursor must not care.
    """
    near, _ = build_pointer()
    far, _ = build_pointer()
    near_out = drive(near, [500 + i * 20.0 for i in range(25)], scale=200.0)
    far_out = drive(far, [500 + i * 5.0 for i in range(25)], scale=50.0)
    near_travel = near_out[-1][0] - near_out[0][0]
    far_travel = far_out[-1][0] - far_out[0][0]
    assert near_travel == pytest.approx(far_travel, rel=0.05)


def test_freeze_stops_cursor_motion():
    p, _ = build_pointer()
    drive(p, [500 + i * 12 for i in range(10)])
    frozen_at = p.state.position
    approach = [GestureEvent(GestureType.PINCH_APPROACH, 200.0, "Right")]
    p.update(make_hand(palm=(700.0, 360.0), index_extension=1.3), engaged_state(),
             approach, 200.0)
    for i in range(1, 8):
        p.update(make_hand(palm=(700.0 + i * 25, 360.0), index_extension=1.3),
                 engaged_state(), [], 200.0 + i * DT)
    assert p.state.frozen
    assert p.state.position == frozen_at, "cursor must not move while frozen"


def test_freeze_is_released_by_the_click():
    p, _ = build_pointer()
    drive(p, [500 + i * 12 for i in range(10)])
    p.update(make_hand(palm=(700.0, 360.0), index_extension=1.3), engaged_state(),
             [GestureEvent(GestureType.PINCH_APPROACH, 200.0, "Right")], 200.0)
    assert p.state.frozen
    p.update(make_hand(palm=(700.0, 360.0), index_extension=1.3), engaged_state(),
             [GestureEvent(GestureType.CLICK, 200.1, "Right")], 200.1)
    assert not p.state.frozen


def test_freeze_expires_if_the_pinch_is_just_held_halfway():
    p, _ = build_pointer(freeze_timeout_s=0.2)
    p.update(make_hand(palm=(700.0, 360.0), index_extension=1.3), engaged_state(),
             [GestureEvent(GestureType.PINCH_APPROACH, 200.0, "Right")], 200.0)
    assert p.state.frozen
    p.update(make_hand(palm=(700.0, 360.0), index_extension=1.3), engaged_state(), [], 200.5)
    assert not p.state.frozen, "a stuck freeze would leave the cursor dead"


def test_motion_during_freeze_is_discarded_not_deferred():
    """Movement while frozen must be dropped, not released in a lump afterwards."""
    p, _ = build_pointer()
    drive(p, [500.0] * 6)
    before = p.state.position
    p.update(make_hand(palm=(500.0, 360.0), index_extension=1.3), engaged_state(),
             [GestureEvent(GestureType.PINCH_APPROACH, 200.0, "Right")], 200.0)
    for i in range(1, 6):
        p.update(make_hand(palm=(500.0 + i * 40, 360.0), index_extension=1.3),
                 engaged_state(), [], 200.0 + i * DT)
    p.update(make_hand(palm=(700.0, 360.0), index_extension=1.3), engaged_state(),
             [GestureEvent(GestureType.CLICK, 200.5, "Right")], 200.5)
    after = p.state.position
    assert abs(after[0] - before[0]) < 60.0, "frozen motion must not be replayed"


def test_reengaging_the_clutch_does_not_jump_the_cursor():
    p, _ = build_pointer()
    drive(p, [500 + i * 10 for i in range(10)])
    parked = p.state.position
    # Release, move the hand a long way, re-engage.
    p.update(None, engaged_state(), [GestureEvent(GestureType.CLUTCH_RELEASE, 201.0)], 201.0)
    p.update(make_hand(palm=(1200.0, 360.0), index_extension=1.3), engaged_state(),
             [GestureEvent(GestureType.CLUTCH_ENGAGE, 202.0, "Right")], 202.0)
    assert p.state.position == pytest.approx(parked, abs=1.0), (
        "re-engaging must resume from where the cursor was, not teleport"
    )


def test_cursor_is_clamped_to_the_desktop():
    p, _ = build_pointer()
    drive(p, [500 - i * 60 for i in range(60)])
    x, y = p.state.position
    vd = dual_desktop()
    assert vd.left <= x <= vd.right - 1
    assert vd.top <= y <= vd.bottom - 1


def test_single_frame_glitch_cannot_fling_the_cursor():
    p, _ = build_pointer(max_jump_px=100.0)
    drive(p, [500.0] * 8)
    before = p.state.position
    p.update(make_hand(palm=(500.0, 360.0), index_extension=1.3), engaged_state(), [], 200.0)
    p.update(make_hand(palm=(3000.0, 360.0), index_extension=1.3), engaged_state(), [], 200.0 + DT)
    moved = abs(p.state.position[0] - before[0])
    assert moved <= 101.0, f"a glitch moved the cursor {moved:.0f}px despite the clamp"


def test_faster_movement_yields_more_cursor_travel_per_hand_unit():
    """The gain curve must actually do something.

    Both runs move leftwards by amounts small enough that neither reaches a
    desktop edge - if either saturated, clamping rather than gain would decide
    the comparison, which is what makes a naive version of this test pass for
    the wrong reason.
    """
    vd = dual_desktop()
    slow, _ = build_pointer()
    fast, _ = build_pointer()
    slow_out = drive(slow, [500 - i * 1.0 for i in range(14)])
    fast_out = drive(fast, [500 - i * 8.0 for i in range(14)])

    for out in (slow_out, fast_out):
        assert out[-1][0] > vd.left + 5, "run saturated; the test would be meaningless"

    slow_per_unit = abs(slow_out[-1][0] - slow_out[0][0]) / (13 * 1.0)
    fast_per_unit = abs(fast_out[-1][0] - fast_out[0][0]) / (13 * 8.0)
    assert fast_per_unit > slow_per_unit * 1.5, (
        f"gain curve too weak: slow {slow_per_unit:.2f} px per hand px, "
        f"fast {fast_per_unit:.2f}"
    )


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def click_event(t=100.0):
    return GestureEvent(GestureType.CLICK, t, "Right")


def test_disarmed_dispatcher_executes_nothing():
    inj = MouseInjector(dual_desktop())
    d = ActionDispatcher(inj)
    records = d.dispatch([click_event()], 100.0)
    assert records[0].executed is False
    assert records[0].reason == "disarmed"
    assert d.stats.executed == 0


def test_armed_dispatcher_executes():
    inj = MouseInjector(dual_desktop(), armed=True)
    d = ActionDispatcher(inj)
    records = d.dispatch([click_event()], 100.0)
    assert records[0].executed is True


def test_identity_gate_blocks_when_unauthenticated():
    inj = MouseInjector(dual_desktop(), armed=True)
    d = ActionDispatcher(inj, SecurityConfig(require_identity=True))
    d.set_identity(IdentityStatus(authenticated=False, name="unknown"))
    records = d.dispatch([click_event()], 100.0)
    assert records[0].executed is False
    assert "operator" in records[0].reason


def test_identity_gate_allows_the_enrolled_operator():
    inj = MouseInjector(dual_desktop(), armed=True)
    d = ActionDispatcher(inj, SecurityConfig(require_identity=True))
    d.set_identity(IdentityStatus(authenticated=True, name="malav", last_match_time=99.9))
    assert d.dispatch([click_event()], 100.0)[0].executed is True


def test_cooldown_blocks_a_repeat():
    inj = MouseInjector(dual_desktop(), armed=True)
    d = ActionDispatcher(inj, min_interval_s=0.5)
    assert d.dispatch([click_event(100.0)], 100.0)[0].executed is True
    assert d.dispatch([click_event(100.1)], 100.1)[0].executed is False
    assert d.dispatch([click_event(101.0)], 101.0)[0].executed is True


def test_drag_end_is_never_rate_limited():
    """Dropping a drag_end would leave the mouse button stuck down."""
    inj = MouseInjector(dual_desktop(), armed=True)
    d = ActionDispatcher(inj, min_interval_s=5.0)
    d.dispatch([GestureEvent(GestureType.DRAG_START, 100.0, "Right")], 100.0)
    rec = d.dispatch([GestureEvent(GestureType.DRAG_END, 100.05, "Right")], 100.05)
    assert rec[0].executed is True
    assert inj.buttons_down == frozenset()


def test_losing_identity_mid_drag_releases_the_button():
    inj = MouseInjector(dual_desktop(), armed=True)
    d = ActionDispatcher(inj, SecurityConfig(require_identity=True))
    d.set_identity(IdentityStatus(authenticated=True, name="malav", last_match_time=100.0))
    d.dispatch([GestureEvent(GestureType.DRAG_START, 100.0, "Right")], 100.0)
    assert inj.buttons_down == {"left"}
    d.set_identity(IdentityStatus(authenticated=False, name="unknown"))
    assert inj.buttons_down == frozenset(), "a lost operator must not leave a held button"


def test_blocked_drag_start_does_not_leave_phantom_state():
    inj = MouseInjector(dual_desktop())  # disarmed
    d = ActionDispatcher(inj)
    d.dispatch([GestureEvent(GestureType.DRAG_START, 100.0, "Right")], 100.0)
    assert d.drag_active is False


def test_shutdown_releases_everything():
    inj = MouseInjector(dual_desktop(), armed=True)
    d = ActionDispatcher(inj)
    d.dispatch([GestureEvent(GestureType.DRAG_START, 100.0, "Right")], 100.0)
    d.shutdown()
    assert inj.buttons_down == frozenset()
    assert d.drag_active is False


def test_right_click_maps_to_the_right_button():
    inj = MouseInjector(dual_desktop(), armed=True)
    d = ActionDispatcher(inj)
    before = inj.stats.clicks
    d.dispatch([GestureEvent(GestureType.RIGHT_CLICK, 100.0, "Right")], 100.0)
    assert inj.stats.clicks == before + 1


# --------------------------------------------------------------------------- #
# End-to-end: landmarks -> gestures -> pointer -> dispatch
# --------------------------------------------------------------------------- #
def test_full_chain_click_moves_nothing_and_fires_once():
    """A pinch-click must not drag the cursor - the drift problem, end to end."""
    inj = MouseInjector(dual_desktop(), armed=True)
    pointer = PointerEngine(inj, PointerConfig())
    gestures = GestureEngine()
    dispatch = ActionDispatcher(inj)

    now = 100.0
    # Settle with the clutch engaged and the hand still.
    for _ in range(8):
        h = make_hand(palm=(640.0, 360.0), index_extension=1.3, pinch_index=0.95)
        ev = gestures.update(h, now)
        pointer.update(h, gestures.state, ev, now)
        dispatch.dispatch(ev, now)
        now += DT
    before = pointer.state.position

    # Pinch: the thumb closes and, as it does, the palm drifts slightly - which
    # is exactly what happens on a real hand.
    executed = []

    def step(hand):
        nonlocal now
        ev = gestures.update(hand, now)
        pointer.update(hand, gestures.state, ev, now)
        executed.extend(r for r in dispatch.dispatch(ev, now) if r.executed)
        now += DT

    # Fingers close over a few frames, and the palm drifts slightly as they do -
    # exactly what a real hand does, and the source of click-induced drift.
    for i, pinch in enumerate([0.70, 0.55, 0.40, 0.28, 0.20, 0.18]):
        step(make_hand(palm=(640.0 + i * 3.0, 360.0), index_extension=1.3, pinch_index=pinch))
    # Release quickly, well inside drag_dwell_s, so this is a click not a drag.
    for i in range(4):
        step(make_hand(palm=(640.0 + 18.0, 360.0), index_extension=1.3, pinch_index=0.95))

    after = pointer.state.position
    drift = abs(after[0] - before[0])
    assert [r.action for r in executed].count("click") == 1, "exactly one click"
    assert drift < 40.0, f"click induced {drift:.0f}px of cursor drift"


# --------------------------------------------------------------------------- #
# Scroll
# --------------------------------------------------------------------------- #
def scroll_state():
    from argus.gestures.fsm import GestureState

    return GestureState(clutch_engaged=True, mode="scroll", hand_speed=0.4)


def test_scroll_mode_emits_wheel_and_does_not_move_the_cursor():
    inj = MouseInjector(dual_desktop(), armed=True)
    p = PointerEngine(inj, PointerConfig())
    p.sync_from_system()
    before = p.state.position

    notches = 0
    for i in range(16):
        hand = scrolling_hand(palm=(640.0, 360.0 - i * 9.0))
        p.update(hand, scroll_state(), [], 100.0 + i * DT)
        notches += p.state.scroll_notches
    assert notches != 0, "vertical movement in scroll mode must emit wheel notches"
    assert p.state.position == before, "scrolling must not move the pointer"
    assert inj.stats.scrolls > 0


def test_scroll_direction_follows_hand_direction():
    inj = MouseInjector(dual_desktop(), armed=True)
    up, down = PointerEngine(inj, PointerConfig()), PointerEngine(inj, PointerConfig())

    def run(engine, step):
        total = 0
        for i in range(16):
            engine.update(scrolling_hand(palm=(640.0, 360.0 + i * step)),
                          scroll_state(), [], 100.0 + i * DT)
            total += engine.state.scroll_notches
        return total

    assert run(up, -9.0) > 0
    assert run(down, +9.0) < 0


def test_a_still_hand_does_not_scroll():
    inj = MouseInjector(dual_desktop(), armed=True)
    p = PointerEngine(inj, PointerConfig())
    total = 0
    for i in range(30):
        p.update(scrolling_hand(palm=(640.0, 360.0)), scroll_state(), [], 100.0 + i * DT)
        total += p.state.scroll_notches
    assert total == 0


def test_scroll_is_depth_invariant():
    """Same movement in hand-widths must scroll the same amount at any distance."""
    inj = MouseInjector(dual_desktop(), armed=True)

    def run(scale, step):
        engine = PointerEngine(inj, PointerConfig())
        total = 0
        for i in range(20):
            engine.update(scrolling_hand(palm=(640.0, 360.0 - i * step), scale=scale),
                          scroll_state(), [], 100.0 + i * DT)
            total += engine.state.scroll_notches
        return total

    assert run(200.0, 18.0) == run(50.0, 4.5)


def test_leaving_scroll_mode_resets_the_accumulator():
    inj = MouseInjector(dual_desktop(), armed=True)
    p = PointerEngine(inj, PointerConfig())
    from argus.gestures.fsm import GestureState

    for i in range(6):
        p.update(scrolling_hand(palm=(640.0, 360.0 - i * 4.0)), scroll_state(), [],
                 100.0 + i * DT)
    # Disengaging must not leave a partial notch that fires later.
    p.update(None, GestureState(clutch_engaged=False), [], 101.0)
    p.update(scrolling_hand(palm=(640.0, 360.0)), scroll_state(), [], 101.1)
    assert p.state.scroll_notches == 0
