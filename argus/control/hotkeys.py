"""Global key state, polled directly from the OS.

The preview window's own keyboard handling is not good enough for the controls
that matter. The moment ARGUS clicks anything, focus moves to whatever was
clicked, and a key handler attached to the preview window stops receiving
anything at all - including the key meant to stop it.

``GetAsyncKeyState`` reads the physical keyboard regardless of focus, so the
panic key works even when the system has driven focus somewhere else entirely.
That is the difference between a safety control and a decoration.
"""

from __future__ import annotations

import ctypes
import sys

IS_WINDOWS = sys.platform == "win32"

VK_ESCAPE = 0x1B
VK_F9 = 0x78
VK_F10 = 0x79
VK_CONTROL = 0x11
VK_SHIFT = 0x10

_KEY_NAMES = {
    VK_ESCAPE: "Esc",
    VK_F9: "F9",
    VK_F10: "F10",
}


def key_down(vk: int) -> bool:
    """True while the key is physically held, regardless of window focus."""
    if not IS_WINDOWS:
        return False
    # The high-order bit means "currently down". The low bit means "pressed
    # since last call", which is unreliable when several things poll it.
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def key_name(vk: int) -> str:
    return _KEY_NAMES.get(vk, f"VK_{vk:02X}")


class EdgeDetector:
    """Reports a key press once, on the transition from up to down."""

    __slots__ = ("vk", "_was_down")

    def __init__(self, vk: int) -> None:
        self.vk = vk
        self._was_down = False

    def pressed(self) -> bool:
        down = key_down(self.vk)
        fired = down and not self._was_down
        self._was_down = down
        return fired


class PanicSwitch:
    """Global emergency stop.

    Held for ``hold_s`` so that an incidental tap of the key does not disarm
    mid-task, but a deliberate hold always does - and unlike a window key
    handler, it cannot be starved of focus.
    """

    def __init__(self, vk: int = VK_ESCAPE, hold_s: float = 0.35) -> None:
        self.vk = vk
        self.hold_s = hold_s
        self._down_since: float | None = None

    def triggered(self, now: float) -> bool:
        if not key_down(self.vk):
            self._down_since = None
            return False
        if self._down_since is None:
            self._down_since = now
            return False
        return (now - self._down_since) >= self.hold_s

    def reset(self) -> None:
        self._down_since = None

    @property
    def description(self) -> str:
        return f"hold {key_name(self.vk)} for {self.hold_s:.1f}s"
