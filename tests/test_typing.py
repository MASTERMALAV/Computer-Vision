"""Typing suppression: hold-off timing, and what must never be suppressed."""

from __future__ import annotations

import pytest

from argus.control import typing as typing_mod
from argus.control.typing import TYPING_KEYS, TypingConfig, TypingMonitor


@pytest.fixture
def keyboard(monkeypatch):
    """A fake keyboard whose held keys the test controls."""
    held: set[int] = set()
    monkeypatch.setattr(typing_mod, "key_down", lambda vk: vk in held)
    return held


def test_nothing_suppressed_when_idle(keyboard):
    m = TypingMonitor()
    m.update(100.0)
    assert not m.is_suppressing(100.0)
    assert not m.should_block_motion(100.0)


def test_a_keystroke_starts_suppression(keyboard):
    m = TypingMonitor(TypingConfig(hold_off_s=0.45))
    keyboard.add(ord("A"))
    assert m.update(100.0) is True
    assert m.is_suppressing(100.0)
    assert m.should_block_motion(100.2)


def test_suppression_expires(keyboard):
    m = TypingMonitor(TypingConfig(hold_off_s=0.45))
    keyboard.add(ord("A"))
    m.update(100.0)
    keyboard.clear()
    assert m.is_suppressing(100.4)
    assert not m.is_suppressing(100.5)


def test_continued_typing_keeps_extending_the_window(keyboard):
    """The window must bridge the gaps between keystrokes, not restart from
    the first one."""
    m = TypingMonitor(TypingConfig(hold_off_s=0.45))
    keyboard.add(ord("A"))
    now = 100.0
    for _ in range(10):  # a key every 150 ms, as in ordinary typing
        m.update(now)
        now += 0.15
    assert m.is_suppressing(now)
    keyboard.clear()
    assert not m.is_suppressing(now + 0.5)


def test_remaining_counts_down(keyboard):
    m = TypingMonitor(TypingConfig(hold_off_s=0.5))
    keyboard.add(ord("A"))
    m.update(100.0)
    assert m.remaining(100.2) == pytest.approx(0.3)
    assert m.remaining(999.0) == 0.0


# --------------------------------------------------------------------------- #
# What must NOT count as typing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "vk,name",
    [(0x11, "Ctrl"), (0x10, "Shift"), (0x12, "Alt")],
)
def test_modifiers_alone_are_not_typing(keyboard, vk, name):
    """Ctrl-click and Shift-click are deliberate combinations. Suppressing
    them would break exactly the gestures reached for while using a cursor."""
    m = TypingMonitor()
    keyboard.add(vk)
    assert m.update(100.0) is False
    assert not m.is_suppressing(100.0), f"{name} must not suppress the hand"


@pytest.mark.parametrize("vk,name", [(0x78, "F9"), (0x79, "F10"), (0x7A, "F11"), (0x1B, "Esc")])
def test_our_own_control_keys_are_not_typing(keyboard, vk, name):
    """Arming the system is not typing. Suppressing the hand the instant
    someone presses F9 to arm it would be perverse."""
    m = TypingMonitor()
    keyboard.add(vk)
    assert m.update(100.0) is False
    assert not m.is_suppressing(100.0), f"{name} must not suppress the hand"


def test_arrow_keys_are_not_typing(keyboard):
    for vk in (0x25, 0x26, 0x27, 0x28):
        assert vk not in TYPING_KEYS


# --------------------------------------------------------------------------- #
# Gestures already in progress
# --------------------------------------------------------------------------- #
def test_an_active_drag_is_never_interrupted(keyboard):
    """Releasing a drag because a key was pressed would drop whatever is held."""
    m = TypingMonitor(TypingConfig(hold_off_s=0.45))
    keyboard.add(ord("A"))
    m.update(100.0)
    assert m.should_block_actions(100.1, gesture_active=False) is True
    assert m.should_block_actions(100.1, gesture_active=True) is False
    assert m.should_block_motion(100.1, gesture_active=True) is False


def test_protection_can_be_turned_off(keyboard):
    m = TypingMonitor(TypingConfig(hold_off_s=0.45, protect_active_gestures=False))
    keyboard.add(ord("A"))
    m.update(100.0)
    assert m.should_block_actions(100.1, gesture_active=True) is True


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_disabled_monitor_never_suppresses(keyboard):
    m = TypingMonitor(TypingConfig(enabled=False))
    keyboard.add(ord("A"))
    assert m.update(100.0) is False
    assert not m.is_suppressing(100.0)


def test_motion_and_clicks_can_be_suppressed_independently(keyboard):
    m = TypingMonitor(TypingConfig(suppress_motion=False, suppress_clicks=True))
    keyboard.add(ord("A"))
    m.update(100.0)
    assert m.should_block_motion(100.1) is False
    assert m.should_block_actions(100.1) is True


def test_key_set_covers_ordinary_typing():
    for ch in "hello world 123":
        if ch == " ":
            assert 0x20 in TYPING_KEYS
        elif ch.isdigit():
            assert (0x30 + int(ch)) in TYPING_KEYS
        else:
            assert ord(ch.upper()) in TYPING_KEYS


def test_config_rejects_an_absurd_hold_off():
    from argus.config import ArgusConfig, ConfigError

    with pytest.raises(ConfigError, match="hold_off_s"):
        ArgusConfig.load(None, ["control.typing.hold_off_s=10"])
    with pytest.raises(ConfigError):
        ArgusConfig.load(None, ["control.typing.hold_off_s=-1"])


def test_stats_are_reported(keyboard):
    m = TypingMonitor()
    keyboard.add(ord("A"))
    m.update(100.0)
    m.note_frame(100.0)
    stats = m.stats()
    assert stats["keystrokes_seen"] == 1
    assert stats["suppressed_frames"] == 1
