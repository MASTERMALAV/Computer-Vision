"""Fitts's law harness: task layout and throughput maths."""

from __future__ import annotations

import math

import numpy as np
import pytest

from argus.config import ArgusConfig
from argus.practice import (
    CONDITIONS,
    TARGETS_PER_RING,
    Report,
    Trial,
    advise,
    ring_positions,
    target_order,
)


def trial(deviation=0.0, distance=500.0, width=60.0, mt=0.8, condition=0, hit=True):
    """A trial whose endpoint error along the movement axis is exactly `deviation`."""
    start = (0.0, 0.0)
    target = (distance, 0.0)
    click = (distance + deviation, 0.0)
    return Trial(condition, distance, width, start, target, click, mt, hit)


# --------------------------------------------------------------------------- #
# Task layout
# --------------------------------------------------------------------------- #
def test_the_zigzag_visits_every_target_once():
    order = target_order(TARGETS_PER_RING)
    assert sorted(order) == list(range(TARGETS_PER_RING))


def test_movements_cross_the_ring_rather_than_hopping_around_it():
    """The standard task alternates across the circle, so each movement is
    roughly the ring diameter. Hopping to a neighbour would make the distance
    - and therefore the index of difficulty - far smaller than intended."""
    diameter = 600
    points = ring_positions((0, 0), diameter, TARGETS_PER_RING)
    order = target_order(TARGETS_PER_RING)
    first = np.linalg.norm(np.array(points[order[0]]) - np.array(points[order[1]]))
    assert first > diameter * 0.85


def test_ring_positions_sit_on_the_circle():
    centre, diameter = (100, 200), 400
    for x, y in ring_positions(centre, diameter, TARGETS_PER_RING):
        radius = math.hypot(x - centre[0], y - centre[1])
        assert radius == pytest.approx(diameter / 2, abs=0.01)


def test_conditions_differ_in_difficulty():
    ids = [math.log2(d / w + 1) for d, w in CONDITIONS]
    assert len(set(round(i, 2) for i in ids)) == len(CONDITIONS)
    assert max(ids) - min(ids) > 1.0, "conditions must span a useful range"


# --------------------------------------------------------------------------- #
# Deviation
# --------------------------------------------------------------------------- #
def test_deviation_is_signed_along_the_movement_axis():
    assert trial(deviation=+20).deviation == pytest.approx(20.0)
    assert trial(deviation=-20).deviation == pytest.approx(-20.0)


def test_sideways_error_does_not_count():
    """Fitts's law is one-dimensional; error across the axis does not affect
    whether the target was acquired along it."""
    t = Trial(0, 500.0, 60.0, (0.0, 0.0), (500.0, 0.0), (500.0, 40.0), 0.8, True)
    assert t.deviation == pytest.approx(0.0, abs=1e-6)


def test_zero_length_movement_is_safe():
    t = Trial(0, 0.0, 60.0, (10.0, 10.0), (10.0, 10.0), (12.0, 10.0), 0.5, True)
    assert t.deviation == 0.0


# --------------------------------------------------------------------------- #
# Throughput
# --------------------------------------------------------------------------- #
def test_throughput_matches_the_shannon_formulation():
    rng = np.random.default_rng(0)
    deviations = rng.normal(0.0, 10.0, 40)
    trials = [trial(deviation=float(d), distance=500.0, mt=0.8) for d in deviations]

    result = Report.throughput(trials)
    expected_we = 4.133 * float(deviations.std(ddof=1))
    expected_ide = math.log2(500.0 / expected_we + 1.0)

    # The report rounds for display, so compare at the reported precision.
    assert result["effective_width_px"] == pytest.approx(expected_we, abs=0.05)
    assert result["index_of_difficulty_bits"] == pytest.approx(expected_ide, abs=0.001)
    assert result["throughput_bits_per_s"] == pytest.approx(expected_ide / 0.8, abs=0.001)


def test_effective_width_rewards_accuracy_not_just_speed():
    """Two people taking the same time: the accurate one must score higher.

    Using the nominal width instead would score them identically, which is the
    whole reason effective width exists.
    """
    rng = np.random.default_rng(1)
    tight = [trial(deviation=float(d)) for d in rng.normal(0, 5.0, 40)]
    sloppy = [trial(deviation=float(d)) for d in rng.normal(0, 25.0, 40)]
    assert (
        Report.throughput(tight)["throughput_bits_per_s"]
        > Report.throughput(sloppy)["throughput_bits_per_s"]
    )


def test_faster_is_better_at_equal_accuracy():
    rng = np.random.default_rng(2)
    deviations = rng.normal(0, 10.0, 40)
    quick = [trial(deviation=float(d), mt=0.5) for d in deviations]
    slow = [trial(deviation=float(d), mt=1.0) for d in deviations]
    assert (
        Report.throughput(quick)["throughput_bits_per_s"]
        > Report.throughput(slow)["throughput_bits_per_s"]
    )


def test_too_few_trials_produces_nothing_rather_than_a_wrong_number():
    assert Report.throughput([trial(), trial()]) == {}


def test_error_and_overshoot_rates():
    trials = [trial(deviation=+10, hit=True) for _ in range(6)]
    trials += [trial(deviation=-10, hit=False) for _ in range(4)]
    result = Report.throughput(trials)
    assert result["error_rate_pct"] == pytest.approx(40.0)
    assert result["overshoot_pct"] == pytest.approx(60.0)


def test_overall_averages_the_conditions():
    report = Report()
    rng = np.random.default_rng(3)
    for condition in range(2):
        for d in rng.normal(0, 8.0, 20):
            report.trials.append(trial(deviation=float(d), condition=condition))
    overall = report.overall()
    assert len(overall["conditions"]) == 2
    assert overall["throughput_bits_per_s"] > 0


# --------------------------------------------------------------------------- #
# Advice
# --------------------------------------------------------------------------- #
def test_persistent_overshoot_suggests_lowering_the_gain():
    cfg = ArgusConfig()
    notes = advise(
        {"throughput_bits_per_s": 2.5, "error_rate_pct": 8, "overshoot_pct": 85}, cfg
    )
    joined = " ".join(notes)
    assert "pixels_per_unit" in joined
    suggested = int(cfg.control.gain.pixels_per_unit * 0.8)
    assert str(suggested) in joined


def test_persistent_undershoot_suggests_raising_the_gain():
    cfg = ArgusConfig()
    notes = advise(
        {"throughput_bits_per_s": 2.5, "error_rate_pct": 8, "overshoot_pct": 10}, cfg
    )
    suggested = int(cfg.control.gain.pixels_per_unit * 1.2)
    assert str(suggested) in " ".join(notes)


def test_balanced_overshoot_says_so():
    notes = advise(
        {"throughput_bits_per_s": 2.5, "error_rate_pct": 5, "overshoot_pct": 50},
        ArgusConfig(),
    )
    assert any("about right" in n for n in notes)


def test_a_high_error_rate_is_called_out():
    notes = advise(
        {"throughput_bits_per_s": 2.5, "error_rate_pct": 30, "overshoot_pct": 50},
        ArgusConfig(),
    )
    assert any("Error rate" in n for n in notes)


def test_advice_on_no_data_is_honest():
    assert "Not enough" in advise({}, ArgusConfig())[0]
