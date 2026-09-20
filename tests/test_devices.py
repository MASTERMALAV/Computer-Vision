"""Device scoring and selection - the pure logic, with no camera attached."""

from __future__ import annotations

import pytest

from argus.capture.devices import CameraDevice, DeviceError, resolve_device

# Mirrors the real machine this was developed on.
FIXTURE = [
    CameraDevice(index=0, name="Integrated Camera"),
    CameraDevice(index=1, name="Brio 100"),
    CameraDevice(index=2, name="Integrated IR Camera", is_ir=True),
]


def test_ir_camera_is_flagged():
    from argus.capture.devices import _matches_any, _IR_PATTERNS

    assert _matches_any("Integrated IR Camera", _IR_PATTERNS)
    assert _matches_any("Windows Hello Face Software Device", _IR_PATTERNS)
    # "IR" as a word only - a camera called "Mirror Cam" is not infrared.
    assert not _matches_any("Mirror Cam", _IR_PATTERNS)
    assert not _matches_any("Brio 100", _IR_PATTERNS)


def test_virtual_camera_is_flagged():
    from argus.capture.devices import _matches_any, _VIRTUAL_PATTERNS

    assert _matches_any("OBS Virtual Camera", _VIRTUAL_PATTERNS)
    assert not _matches_any("Brio 100", _VIRTUAL_PATTERNS)


def test_external_beats_integrated():
    integrated, brio, ir = FIXTURE
    assert brio.score() > integrated.score()
    assert integrated.score() > ir.score()


def test_auto_selects_external_colour_camera():
    chosen = resolve_device("auto", FIXTURE)
    assert chosen.name == "Brio 100"


def test_auto_never_selects_ir():
    only_ir_and_integrated = [FIXTURE[0], FIXTURE[2]]
    assert resolve_device("auto", only_ir_and_integrated).index == 0


def test_auto_falls_back_to_ir_if_nothing_else():
    ir_only = [FIXTURE[2]]
    assert resolve_device("auto", ir_only).index == 2


def test_name_substring_is_case_insensitive():
    assert resolve_device("brio", FIXTURE).index == 1
    assert resolve_device("BRIO", FIXTURE).index == 1
    assert resolve_device("Brio 100", FIXTURE).index == 1


def test_explicit_index():
    assert resolve_device(0, FIXTURE).index == 0
    assert resolve_device("2", FIXTURE).index == 2


def test_explicit_index_outside_enumeration_is_honoured():
    dev = resolve_device("7", FIXTURE)
    assert dev.index == 7


def test_ambiguous_name_prefers_non_ir():
    devices = [
        CameraDevice(index=0, name="Integrated Camera"),
        CameraDevice(index=1, name="Integrated IR Camera", is_ir=True),
    ]
    assert resolve_device("integrated", devices).index == 0


def test_unmatched_name_lists_what_is_available():
    with pytest.raises(DeviceError) as exc:
        resolve_device("kinect", FIXTURE)
    message = str(exc.value)
    assert "Brio 100" in message and "Integrated Camera" in message


def test_no_devices_gives_actionable_error():
    with pytest.raises(DeviceError, match="privacy"):
        resolve_device("auto", [])


def test_unopenable_device_is_deprioritised():
    broken = CameraDevice(index=1, name="Brio 100", working=False)
    working = CameraDevice(index=0, name="Integrated Camera", working=True)
    assert resolve_device("auto", [working, broken]).index == 0
