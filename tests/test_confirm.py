"""Cancellable confirmation countdown."""

from __future__ import annotations

from argus.control.confirm import ConfirmState, Confirmer


def test_nothing_pending_initially():
    c = Confirmer(duration_s=5.0)
    assert not c.is_counting
    assert c.update(now=100.0) is None


def test_countdown_fires_only_after_the_full_duration():
    c = Confirmer(duration_s=5.0)
    assert c.request("shutdown", "Shut down", now=100.0)
    assert c.is_counting
    assert c.update(now=104.9) is None, "must not fire early"
    fired = c.update(now=105.0)
    assert fired is not None
    assert fired.action == "shutdown"
    assert fired.state is ConfirmState.CONFIRMED
    assert not c.is_counting


def test_cancelling_stops_it_permanently():
    c = Confirmer(duration_s=5.0)
    c.request("shutdown", now=100.0)
    assert c.cancel("panic key", now=102.0) is True
    assert not c.is_counting
    assert c.update(now=200.0) is None, "a cancelled action must never fire"
    assert c.history[-1].state is ConfirmState.CANCELLED
    assert c.history[-1].cancel_reason == "panic key"


def test_cancel_with_nothing_pending_is_harmless():
    assert Confirmer().cancel("nothing") is False


def test_a_second_request_is_refused_not_queued():
    """Queued destructive actions are never what anyone wants."""
    c = Confirmer(duration_s=5.0)
    assert c.request("shutdown", now=100.0) is True
    assert c.request("reboot", now=101.0) is False
    assert c.pending.action == "shutdown"


def test_remaining_and_progress():
    c = Confirmer(duration_s=4.0)
    c.request("x", now=100.0)
    assert c.pending.remaining(101.0) == 3.0
    assert c.pending.progress(102.0) == 0.5
    assert c.pending.remaining(999.0) == 0.0
    assert c.pending.progress(999.0) == 1.0


def test_hud_lines_appear_only_while_counting():
    c = Confirmer(duration_s=5.0)
    assert c.hud_lines(now=100.0) == []
    c.request("shutdown", "Shut down the computer", now=100.0)
    lines = c.hud_lines(now=101.0)
    assert any("SHUT DOWN" in line for line in lines)
    assert any("cancel" in line.lower() for line in lines)


def test_zero_duration_fires_immediately():
    c = Confirmer(duration_s=0.0)
    c.request("x", now=100.0)
    assert c.update(now=100.0) is not None


def test_a_new_request_can_follow_a_completed_one():
    c = Confirmer(duration_s=1.0)
    c.request("first", now=100.0)
    c.update(now=101.0)
    assert c.request("second", now=102.0) is True
    assert len(c.history) == 1
