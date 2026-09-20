"""Rolling metrics behaviour."""

from __future__ import annotations

import time

import pytest

from argus.metrics import Metrics, RollingWindow


def test_rolling_window_evicts_oldest():
    w = RollingWindow(capacity=3)
    for v in (1.0, 2.0, 3.0, 4.0):
        w.add(v)
    assert sorted(w.values().tolist()) == [2.0, 3.0, 4.0]
    assert w.total == 4  # total counts everything ever added
    assert w.mean() == pytest.approx(3.0)


def test_rolling_window_partial_fill():
    w = RollingWindow(capacity=10)
    w.add(5.0)
    assert w.values().size == 1
    assert w.mean() == pytest.approx(5.0)


def test_empty_window_is_safe():
    w = RollingWindow(capacity=4)
    assert w.mean() == 0.0
    assert w.percentile(95) == 0.0
    s = w.summary("x")
    assert s.count == 0 and s.p95 == 0.0


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        RollingWindow(capacity=0)


def test_timer_records_elapsed():
    m = Metrics(window=10)
    with m.timer("work"):
        time.sleep(0.02)
    stat = m.get("work")
    assert stat is not None
    assert stat.count == 1
    assert stat.mean >= 15.0  # ~20 ms, with slack for timer granularity


def test_timer_records_even_when_body_raises():
    m = Metrics(window=10)
    with pytest.raises(RuntimeError):
        with m.timer("boom"):
            raise RuntimeError("x")
    assert m.get("boom").count == 1


def test_fps_from_marked_frames():
    m = Metrics(window=30)
    for _ in range(5):
        m.mark_frame()
        time.sleep(0.01)
    # 5 frames at ~10 ms apart -> roughly 100 fps; assert the order of magnitude.
    assert 30 < m.fps < 400
    assert m.frames == 5


def test_unknown_timer_returns_none():
    m = Metrics()
    assert m.get("never-used") is None
    assert m.mean_ms("never-used") == 0.0


def test_report_and_dict_are_serialisable():
    import json

    m = Metrics(window=10)
    with m.timer("stage"):
        pass
    m.mark_frame()
    m.mark_frame()
    assert "fps_window" in m.report()
    payload = json.dumps(m.to_dict())
    assert "stage" in payload
