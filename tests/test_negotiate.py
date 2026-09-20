"""Backend negotiation logic, with the camera measurement stubbed out."""

from __future__ import annotations

import pytest

from argus.capture import negotiate as neg
from argus.capture.negotiate import CaptureProfile
from argus.config import CameraConfig


@pytest.fixture
def cfg():
    return CameraConfig(width=1280, height=720, fps=30, fourcc="MJPG", backend="auto")


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Never touch the real capture profile cache during tests."""
    monkeypatch.setattr(neg, "CACHE_PATH", tmp_path / "profiles.json")


def profile(backend: str, fps: float, w: int = 1280, h: int = 720) -> CaptureProfile:
    return CaptureProfile(
        backend=backend, width=w, height=h, measured_fps=fps, fourcc="", requested_fps=30
    )


def test_ratio_and_describe():
    p = profile("msmf", 15.0)
    assert p.ratio == pytest.approx(0.5)
    assert "msmf" in p.describe() and "15.0" in p.describe()


def test_picks_the_backend_that_actually_delivers(cfg, monkeypatch):
    """The motivating case: DirectShow reports success but delivers 5 fps."""
    measured = {"dshow": profile("dshow", 5.0), "msmf": profile("msmf", 30.0)}
    monkeypatch.setattr(neg, "WINDOWS_CANDIDATES", ("dshow", "msmf"))
    monkeypatch.setattr(neg, "_measure", lambda i, c, b: measured.get(b))
    monkeypatch.setattr(neg.sys, "platform", "win32")

    assert neg.negotiate(1, "Brio 100", cfg).backend == "msmf"


def test_stops_early_when_first_candidate_is_good_enough(cfg, monkeypatch):
    tried: list[str] = []

    def fake(index, c, backend):
        tried.append(backend)
        return profile(backend, 30.0)

    monkeypatch.setattr(neg, "WINDOWS_CANDIDATES", ("msmf", "dshow"))
    monkeypatch.setattr(neg, "_measure", fake)
    monkeypatch.setattr(neg.sys, "platform", "win32")

    assert neg.negotiate(1, "Cam", cfg).backend == "msmf"
    assert tried == ["msmf"], "should not probe a second backend after a good result"


def test_skips_backends_that_cannot_open(cfg, monkeypatch):
    monkeypatch.setattr(neg, "WINDOWS_CANDIDATES", ("msmf", "dshow"))
    monkeypatch.setattr(
        neg, "_measure", lambda i, c, b: None if b == "msmf" else profile("dshow", 28.0)
    )
    monkeypatch.setattr(neg.sys, "platform", "win32")
    assert neg.negotiate(1, "Cam", cfg).backend == "dshow"


def test_raises_when_nothing_works(cfg, monkeypatch):
    monkeypatch.setattr(neg, "_measure", lambda i, c, b: None)
    with pytest.raises(RuntimeError, match="no frames on any backend"):
        neg.negotiate(1, "Cam", cfg)


def test_explicit_backend_is_honoured_even_if_slow(cfg, monkeypatch):
    cfg.backend = "dshow"
    monkeypatch.setattr(neg, "_measure", lambda i, c, b: profile(b, 5.0))
    result = neg.negotiate(1, "Cam", cfg)
    assert result.backend == "dshow"
    assert result.measured_fps == 5.0


def test_explicit_backend_failure_is_reported(cfg, monkeypatch):
    cfg.backend = "msmf"
    monkeypatch.setattr(neg, "_measure", lambda i, c, b: None)
    with pytest.raises(RuntimeError, match="msmf backend"):
        neg.negotiate(1, "Cam", cfg)


def test_result_is_cached_and_reused(cfg, monkeypatch):
    calls = {"n": 0}

    def fake(index, c, backend):
        calls["n"] += 1
        return profile("msmf", 30.0)

    monkeypatch.setattr(neg, "WINDOWS_CANDIDATES", ("msmf",))
    monkeypatch.setattr(neg, "_measure", fake)
    monkeypatch.setattr(neg.sys, "platform", "win32")

    neg.negotiate(1, "Brio 100", cfg)
    neg.negotiate(1, "Brio 100", cfg)
    assert calls["n"] == 1, "second call should have been served from the cache"

    # A different requested format is a different cache entry.
    cfg.width, cfg.height = 640, 360
    neg.negotiate(1, "Brio 100", cfg)
    assert calls["n"] == 2


def test_cache_can_be_bypassed(cfg, monkeypatch):
    calls = {"n": 0}

    def fake(index, c, backend):
        calls["n"] += 1
        return profile("msmf", 30.0)

    monkeypatch.setattr(neg, "WINDOWS_CANDIDATES", ("msmf",))
    monkeypatch.setattr(neg, "_measure", fake)
    monkeypatch.setattr(neg.sys, "platform", "win32")

    neg.negotiate(1, "Cam", cfg)
    neg.negotiate(1, "Cam", cfg, use_cache=False)
    assert calls["n"] == 2


def test_corrupt_cache_is_ignored(cfg, monkeypatch):
    neg.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    neg.CACHE_PATH.write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(neg, "WINDOWS_CANDIDATES", ("msmf",))
    monkeypatch.setattr(neg, "_measure", lambda i, c, b: profile("msmf", 30.0))
    monkeypatch.setattr(neg.sys, "platform", "win32")
    assert neg.negotiate(1, "Cam", cfg).backend == "msmf"


def test_ties_break_toward_higher_resolution(cfg, monkeypatch):
    monkeypatch.setattr(neg, "WINDOWS_CANDIDATES", ("a", "b"))
    results = {
        "a": profile("a", 20.0, 640, 360),
        "b": profile("b", 20.0, 1280, 720),
    }
    monkeypatch.setattr(neg, "_measure", lambda i, c, bk: results[bk])
    monkeypatch.setattr(neg.sys, "platform", "win32")
    # Neither reaches GOOD_ENOUGH, so both are probed and the better one wins.
    assert neg.negotiate(1, "Cam", cfg).backend == "b"
