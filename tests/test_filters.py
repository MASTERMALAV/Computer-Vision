"""One Euro filter, gain curve and position history."""

from __future__ import annotations

import numpy as np
import pytest

from argus.control.filters import (
    GainConfig,
    OneEuroConfig,
    OneEuroFilter,
    PositionHistory,
    smoothstep,
)

DT = 1.0 / 30.0


def run(filt: OneEuroFilter, samples, dt: float = DT, start: float = 100.0):
    """Feed samples at a fixed rate; return the filtered outputs."""
    out = []
    t = start
    for s in samples:
        out.append(np.array(filt(s, t), dtype=np.float64))
        t += dt
    return out


def test_first_sample_passes_through():
    f = OneEuroFilter()
    assert f(np.array([5.0, 7.0]), 0.0).tolist() == [5.0, 7.0]


def test_suppresses_jitter_on_a_stationary_signal():
    rng = np.random.default_rng(0)
    truth = np.array([500.0, 300.0])
    noise = [truth + rng.normal(0, 2.0, 2) for _ in range(120)]

    f = OneEuroFilter(OneEuroConfig(min_cutoff=0.6, beta=0.01))
    out = run(f, noise)[30:]  # discard warm-up

    raw_spread = np.std([n for n in noise[30:]], axis=0).mean()
    filtered_spread = np.std(out, axis=0).mean()
    assert filtered_spread < raw_spread * 0.5, (
        f"filter should cut stationary jitter by at least half "
        f"(raw {raw_spread:.2f} -> filtered {filtered_spread:.2f})"
    )


def test_tracks_a_fast_ramp_without_falling_far_behind():
    """Speed coupling must keep lag small when the signal really moves."""
    ramp = [np.array([i * 20.0, 0.0]) for i in range(60)]
    f = OneEuroFilter(OneEuroConfig(min_cutoff=1.0, beta=0.05))
    out = run(f, ramp)
    # After the ramp settles, the filter must be close to the true value.
    lag = abs(out[-1][0] - ramp[-1][0])
    assert lag < 60.0, f"lag of {lag:.1f}px at the end of a fast ramp is too high"


def test_beta_reduces_lag():
    """Higher beta must track a moving signal more closely - the core property."""
    ramp = [np.array([i * 20.0, 0.0]) for i in range(60)]
    low = run(OneEuroFilter(OneEuroConfig(min_cutoff=0.4, beta=0.0)), ramp)
    high = run(OneEuroFilter(OneEuroConfig(min_cutoff=0.4, beta=0.2)), ramp)
    assert abs(high[-1][0] - ramp[-1][0]) < abs(low[-1][0] - ramp[-1][0])


def test_lower_min_cutoff_is_steadier_at_rest():
    rng = np.random.default_rng(1)
    noise = [np.array([100.0, 100.0]) + rng.normal(0, 3.0, 2) for _ in range(120)]
    steady = run(OneEuroFilter(OneEuroConfig(min_cutoff=0.3, beta=0.0)), noise)[40:]
    loose = run(OneEuroFilter(OneEuroConfig(min_cutoff=5.0, beta=0.0)), noise)[40:]
    assert np.std(steady, axis=0).mean() < np.std(loose, axis=0).mean()


def test_handles_jittery_sample_rate():
    """Real frames arrive irregularly; the filter must use real timestamps.

    Feeding the same signal at an irregular rate must not blow up or produce a
    wildly different result from the regular case.
    """
    rng = np.random.default_rng(2)
    signal = [np.array([i * 3.0, 0.0]) for i in range(80)]

    regular = OneEuroFilter(OneEuroConfig(min_cutoff=1.0, beta=0.02))
    out_reg = run(regular, signal)

    jittery = OneEuroFilter(OneEuroConfig(min_cutoff=1.0, beta=0.02))
    t = 100.0
    out_jit = []
    for s in signal:
        out_jit.append(np.array(jittery(s, t), dtype=np.float64))
        t += DT * rng.uniform(0.7, 1.6)  # matches measured p95 gap spread

    assert np.all(np.isfinite(out_jit))
    assert abs(out_jit[-1][0] - out_reg[-1][0]) < 40.0


def test_repeated_timestamp_does_not_divide_by_zero():
    f = OneEuroFilter()
    f(np.array([1.0, 1.0]), 10.0)
    result = f(np.array([2.0, 2.0]), 10.0)
    assert np.all(np.isfinite(result))


def test_backwards_timestamp_is_survived():
    f = OneEuroFilter()
    f(np.array([1.0, 1.0]), 10.0)
    assert np.all(np.isfinite(f(np.array([2.0, 2.0]), 9.0)))


def test_long_gap_resets_instead_of_inventing_velocity():
    """A hand that left the frame and came back must not produce a huge jump."""
    f = OneEuroFilter()
    f(np.array([0.0, 0.0]), 0.0)
    f(np.array([1.0, 0.0]), DT)
    out = f(np.array([900.0, 0.0]), 5.0)  # 5 s later, far away
    assert out[0] == pytest.approx(900.0), "after a long gap the filter should re-anchor"


def test_scalar_signals_work():
    f = OneEuroFilter()
    assert f(1.0, 0.0) == 1.0
    assert np.isfinite(f(2.0, DT))


def test_reset_clears_state():
    f = OneEuroFilter()
    f(np.array([10.0, 10.0]), 0.0)
    f.reset()
    assert f(np.array([99.0, 99.0]), 1.0).tolist() == [99.0, 99.0]


# --------------------------------------------------------------------------- #
def test_smoothstep_bounds_and_midpoint():
    assert smoothstep(0, 1, -5) == 0.0
    assert smoothstep(0, 1, 5) == 1.0
    assert smoothstep(0, 1, 0.5) == pytest.approx(0.5)


def test_smoothstep_degenerate_range():
    assert smoothstep(1.0, 1.0, 0.5) == 0.0
    assert smoothstep(1.0, 1.0, 1.5) == 1.0


def test_gain_curve_is_monotonic_and_bounded():
    g = GainConfig()
    speeds = np.linspace(0, 8, 60)
    gains = [g.gain(float(s)) for s in speeds]
    assert gains[0] == pytest.approx(g.min_gain)
    assert gains[-1] == pytest.approx(g.max_gain)
    assert all(b >= a - 1e-9 for a, b in zip(gains, gains[1:])), "gain must never decrease"


def test_gain_is_low_for_precision_and_high_for_travel():
    g = GainConfig()
    assert g.gain(0.05) < 1.0, "slow movements must be precise"
    assert g.gain(6.0) > 2.0, "fast movements must cover ground"


# --------------------------------------------------------------------------- #
def test_position_history_recalls_the_past():
    h = PositionHistory(seconds=0.5)
    for i in range(20):
        h.add(np.array([float(i), 0.0]), i * DT)
    got = h.before(seconds_ago=5 * DT, now=19 * DT)
    assert got is not None
    assert got[0] == pytest.approx(14.0, abs=1.0)


def test_position_history_empty_is_safe():
    assert PositionHistory().before(0.1, 1.0) is None


def test_position_history_is_bounded():
    h = PositionHistory(seconds=0.2, expected_hz=30)
    for i in range(1000):
        h.add(np.array([float(i), 0.0]), i * DT)
    assert len(h) <= 30, "history must not grow without bound"
