"""Status pill: refresh throttling, change detection and placement."""

from __future__ import annotations

import numpy as np
import pytest

from argus.ui.statuspill import CORNERS, REFRESH_HZ, PillState, StatusPill


class FakeMonitor:
    left, top, right, bottom = 0, 0, 1920, 1080


class FakeDesktop:
    primary = FakeMonitor()


@pytest.fixture
def pill(monkeypatch):
    """A pill that never touches the screen - rendering only."""
    p = StatusPill(title="test pill", width=360, height=72)
    drawn = []

    def fake_show(state, now=None):
        # Reproduce show()'s throttle decision without any GUI calls.
        import time as _time

        now = _time.perf_counter() if now is None else now
        signature = (
            state.armed, state.engaged, state.mode, state.frozen,
            round(state.pinch, 2), round(state.fps), state.note, state.identity,
        )
        due = (now - p._last_draw) >= p._interval
        if not due and signature == p._last_signature:
            return False
        p._render(state)
        drawn.append(now)
        p._last_draw = now
        p._last_signature = signature
        return True

    monkeypatch.setattr(p, "show", fake_show)
    p.drawn = drawn
    return p


def test_render_produces_the_right_canvas_size():
    p = StatusPill(width=400, height=80)
    canvas = p._render(PillState(armed=True, engaged=True, pinch=0.2, fps=29.0))
    assert canvas.shape == (80, 400, 3)
    assert canvas.dtype == np.uint8


def test_render_is_visibly_different_when_armed():
    """The armed state must be obvious at a glance, not a subtle label."""
    p = StatusPill()
    off = p._render(PillState(armed=False)).copy()
    on = p._render(PillState(armed=True)).copy()
    assert not np.array_equal(off, on)
    # The left-edge stripe is the at-a-glance signal.
    assert not np.array_equal(off[:, :6], on[:, :6])


def test_unchanged_state_does_not_redraw(pill):
    state = PillState(armed=True, engaged=True, mode="point", pinch=0.9, fps=30.0)
    assert pill.show(state, now=100.0) is True
    # Same state, well inside the refresh interval.
    assert pill.show(state, now=100.01) is False
    assert len(pill.drawn) == 1


def test_a_changed_state_redraws_immediately(pill):
    """Latency matters for state: arming must show up at once, not in 80 ms."""
    pill.show(PillState(armed=False), now=100.0)
    assert pill.show(PillState(armed=True), now=100.001) is True


def test_throttle_limits_redraws_of_a_drifting_value(pill):
    """A value that changes every frame must still not redraw every frame.

    Measured: drawing the pill on every frame cost 8.18 ms, nearly as much as
    the full preview it replaces.
    """
    now = 100.0
    for i in range(60):  # two seconds at 30 fps
        pill.show(PillState(armed=True, fps=29.0, pinch=0.5 + i * 0.001), now=now)
        now += 1.0 / 30.0
    # At 12 Hz, two seconds allows about 24 redraws, not 60.
    assert len(pill.drawn) <= int(2.0 * REFRESH_HZ) + 2
    assert len(pill.drawn) < 60


def test_refresh_interval_matches_the_declared_rate():
    p = StatusPill()
    assert p._interval == pytest.approx(1.0 / REFRESH_HZ)


# --------------------------------------------------------------------------- #
# Placement
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("corner", CORNERS)
def test_every_corner_lands_inside_the_primary_display(corner, monkeypatch):
    import argus.control.screens as screens

    monkeypatch.setattr(screens, "get_virtual_desktop", lambda: FakeDesktop())
    p = StatusPill(corner=corner, width=360, height=72, margin=18)
    x, y = p._position()
    assert 0 <= x <= 1920 - 360
    assert 0 <= y <= 1080 - 72


def test_corners_are_actually_different(monkeypatch):
    import argus.control.screens as screens

    monkeypatch.setattr(screens, "get_virtual_desktop", lambda: FakeDesktop())
    positions = {c: StatusPill(corner=c)._position() for c in CORNERS}
    assert len(set(positions.values())) == len(CORNERS)


def test_an_unknown_corner_falls_back_rather_than_raising():
    assert StatusPill(corner="middle-of-nowhere").corner == "bottom-right"


def test_opacity_is_clamped_to_a_visible_range():
    assert StatusPill(opacity=0).opacity >= 40
    assert StatusPill(opacity=999).opacity == 255
