"""Config layering, and the health checks that judge rather than just report."""

from __future__ import annotations

import pytest

from argus.config import ArgusConfig
from argus.doctor import Status, check_thresholds


# --------------------------------------------------------------------------- #
# Layering
#
# `argus calibrate --write` produces a file of gesture thresholds only. Loading
# it with -c used to replace the whole configuration, silently dropping the
# camera choice, the HUD mode and the app launch target.
# --------------------------------------------------------------------------- #
def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_a_later_file_wins_key_by_key(tmp_path):
    base = write(
        tmp_path, "base.yaml",
        "capture:\n  camera:\n    device: msmf:0\n    width: 1280\n"
        "gestures:\n  pinch_close: 0.34\n  drag_dwell_s: 0.35\n",
    )
    overlay = write(tmp_path, "cal.yaml", "gestures:\n  pinch_close: 0.17\n")

    cfg = ArgusConfig.load([base, overlay])
    assert cfg.gestures.pinch_close == 0.17, "the overlay must win"
    assert cfg.gestures.drag_dwell_s == 0.35, "untouched keys must survive"
    assert cfg.capture.camera.device == "msmf:0", "other sections must survive"
    assert cfg.capture.camera.width == 1280


def test_an_overlay_does_not_wipe_its_own_section(tmp_path):
    """Merging per section rather than per key would drop drag_dwell_s here."""
    base = write(tmp_path, "base.yaml", "gestures:\n  pinch_close: 0.34\n  drag_dwell_s: 0.5\n")
    overlay = write(tmp_path, "cal.yaml", "gestures:\n  pinch_open: 0.6\n")
    cfg = ArgusConfig.load([base, overlay])
    assert cfg.gestures.drag_dwell_s == 0.5
    assert cfg.gestures.pinch_open == 0.6
    assert cfg.gestures.pinch_close == 0.34


def test_three_layers_stack_in_order(tmp_path):
    a = write(tmp_path, "a.yaml", "runtime:\n  log_level: INFO\n")
    b = write(tmp_path, "b.yaml", "runtime:\n  log_level: DEBUG\n")
    c = write(tmp_path, "c.yaml", "runtime:\n  log_level: WARNING\n")
    assert ArgusConfig.load([a, b, c]).runtime.log_level == "WARNING"


def test_a_single_path_still_works(tmp_path):
    path = write(tmp_path, "one.yaml", "capture:\n  camera:\n    device: brio\n")
    assert ArgusConfig.load(path).capture.camera.device == "brio"


def test_a_missing_layer_is_reported(tmp_path):
    from argus.config import ConfigError

    good = write(tmp_path, "good.yaml", "runtime:\n  log_level: INFO\n")
    with pytest.raises(ConfigError, match="not found"):
        ArgusConfig.load([good, tmp_path / "nope.yaml"])


def test_overrides_still_beat_every_file(tmp_path):
    base = write(tmp_path, "base.yaml", "capture:\n  camera:\n    width: 1280\n")
    overlay = write(tmp_path, "over.yaml", "capture:\n  camera:\n    width: 640\n")
    cfg = ArgusConfig.load([base, overlay], ["capture.camera.width=1920"])
    assert cfg.capture.camera.width == 1920


# --------------------------------------------------------------------------- #
# Threshold margins
#
# The check that earns its place: a configuration can be entirely valid and
# still be fragile, which presents as "unreliable" rather than as a fault.
# --------------------------------------------------------------------------- #
def status_of(group, name):
    return next(c.status for c in group.checks if c.name == name)


def test_healthy_thresholds_pass():
    cfg = ArgusConfig()
    group = check_thresholds(cfg)
    assert group.worst is Status.PASS


def test_a_narrow_pinch_band_is_flagged():
    cfg = ArgusConfig.load(None, ["gestures.pinch_close=0.40", "gestures.pinch_open=0.44",
                                  "gestures.pinch_approach=0.54"])
    assert status_of(check_thresholds(cfg), "pinch hysteresis") is Status.WARN


def test_a_clutch_threshold_with_no_headroom_fails():
    """The real case: an engage threshold at 0.958 against a pointing finger
    that measures about 1.0 leaves almost nothing, and the cursor becomes
    reluctant to engage - while every ordering check still passes."""
    cfg = ArgusConfig.load(None, ["gestures.finger_extended=1.30",
                                  "gestures.finger_curled=0.40"])
    assert status_of(check_thresholds(cfg), "clutch headroom") is Status.FAIL


def test_a_thin_clutch_margin_warns():
    cfg = ArgusConfig.load(None, ["gestures.finger_extended=1.15",
                                  "gestures.finger_curled=0.40"])
    assert status_of(check_thresholds(cfg), "clutch headroom") is Status.WARN


def test_the_measured_calibration_passes():
    """The operator's real calibrated values, as a regression guard."""
    cfg = ArgusConfig.load(None, [
        "gestures.pinch_close=0.168",
        "gestures.pinch_open=0.468",
        "gestures.pinch_approach=0.568",
        "gestures.finger_extended=0.958",
        "gestures.finger_curled=0.267",
    ])
    group = check_thresholds(cfg)
    assert group.worst is not Status.FAIL


def test_a_missing_freeze_margin_warns():
    cfg = ArgusConfig.load(None, ["gestures.pinch_close=0.20", "gestures.pinch_open=0.50",
                                  "gestures.pinch_approach=0.52"])
    assert status_of(check_thresholds(cfg), "click freeze margin") is Status.WARN


def test_every_non_pass_check_carries_a_fix():
    """A health check that reports a problem without saying what to do about it
    is just an alarm."""
    cfg = ArgusConfig.load(None, ["gestures.finger_extended=1.30",
                                  "gestures.finger_curled=0.40"])
    for check in check_thresholds(cfg).checks:
        if check.status in (Status.WARN, Status.FAIL):
            assert check.fix, f"{check.name} reports a problem but suggests nothing"
