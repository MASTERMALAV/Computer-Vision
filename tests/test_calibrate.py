"""Threshold derivation from measured hand poses."""

from __future__ import annotations

from argus.config import ArgusConfig
from argus.gestures.calibrate import derive_thresholds


def stats(p05, p50, p95, n=45):
    return {"n": n, "mean": p50, "std": (p95 - p05) / 4, "p05": p05, "p50": p50, "p95": p95}


SEPARATED = {
    "open": stats(0.82, 0.90, 0.98),
    "pinch": stats(0.15, 0.20, 0.26),
    "pinch_middle": stats(0.16, 0.22, 0.28),
    "extended": stats(1.22, 1.30, 1.38),
    "curled": stats(0.52, 0.60, 0.68),
}


def test_well_separated_poses_produce_clean_thresholds():
    t, warnings = derive_thresholds(SEPARATED)
    assert not warnings
    assert t["pinch_close"] < t["pinch_open"] < t["pinch_approach"]
    assert t["finger_curled"] < t["finger_extended"]


def test_derived_thresholds_sit_between_the_measured_poses():
    t, _ = derive_thresholds(SEPARATED)
    # The close threshold must be above every observed pinch and below the
    # open hand, or it either never fires or fires constantly.
    assert t["pinch_close"] > SEPARATED["pinch"]["p95"]
    assert t["pinch_close"] < SEPARATED["open"]["p05"]


def test_overlapping_pinch_is_reported():
    data = dict(SEPARATED)
    data["pinch"] = stats(0.70, 0.80, 0.90)  # indistinguishable from an open hand
    t, warnings = derive_thresholds(data)
    assert warnings, "overlapping poses must be flagged, not silently fitted"
    assert any("overlap" in w for w in warnings)
    # Even on bad input the ordering invariant must hold, so the config the
    # user might save still passes validation.
    assert t["pinch_close"] < t["pinch_open"] < t["pinch_approach"]


def test_overlapping_clutch_poses_are_reported():
    data = dict(SEPARATED)
    data["curled"] = stats(1.15, 1.25, 1.35)  # never actually curled
    _, warnings = derive_thresholds(data)
    assert any("clutch" in w for w in warnings)


def test_derived_thresholds_pass_config_validation():
    """Whatever calibration emits must be loadable - no invalid configs."""
    t, _ = derive_thresholds(SEPARATED)
    overrides = [f"gestures.{k}={v}" for k, v in t.items()]
    cfg = ArgusConfig.load(None, overrides)
    cfg.validate()
    assert cfg.gestures.pinch_close == t["pinch_close"]


def test_overlapping_input_still_yields_a_valid_config():
    data = dict(SEPARATED)
    data["pinch"] = stats(0.70, 0.80, 0.90)
    data["curled"] = stats(1.15, 1.25, 1.35)
    t, _ = derive_thresholds(data)
    ArgusConfig.load(None, [f"gestures.{k}={v}" for k, v in t.items()]).validate()


def test_missing_poses_are_tolerated():
    t, _ = derive_thresholds({"open": SEPARATED["open"], "pinch": SEPARATED["pinch"]})
    assert "pinch_close" in t
    assert "finger_extended" not in t


def test_no_data_yields_nothing():
    t, warnings = derive_thresholds({})
    assert t == {}
    assert not warnings
