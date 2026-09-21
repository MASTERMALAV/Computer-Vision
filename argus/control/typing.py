"""Ignore the hand for a moment after each keystroke.

While you type, your hands sit over the keyboard - and an index finger resting
above the keys looks, to a camera, exactly like the pointing pose that drives the
cursor. The result is a cursor that twitches mid-sentence, and worse, a stray
pinch that clicks somewhere and moves the text caret.

Every laptop touchpad solves this the same way: suppress pointing input briefly
after each keystroke. The window is short enough to be invisible when you stop
typing and reach for the mouse, and long enough to cover the gaps between
keystrokes in normal typing.

Two details matter:

**Modifiers do not count as typing.** Holding Ctrl or Shift while clicking is a
deliberate combination, and suppressing the click would break Ctrl-click and
Shift-click - which are exactly the gestures a person reaches for when they are
already using the mouse.

**Our own control keys do not count either.** Pressing F9 to arm the system, or
F11 to change the display, is not typing, and suppressing the hand immediately
after arming would be perverse.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..logsetup import get_logger
from .hotkeys import key_down

log = get_logger("control.typing")


def _range(first: int, last: int) -> tuple[int, ...]:
    return tuple(range(first, last + 1))


# Keys whose use means "this person is typing". Deliberately excludes the
# modifiers, the function keys and the arrows: those either accompany mouse use
# or are our own controls.
TYPING_KEYS: tuple[int, ...] = (
    *_range(0x41, 0x5A),  # A-Z
    *_range(0x30, 0x39),  # 0-9 across the top
    *_range(0x60, 0x69),  # numpad 0-9
    0x20,  # space
    0x0D,  # enter
    0x08,  # backspace
    0x09,  # tab
    0x2E,  # delete
    *_range(0xBA, 0xC0),  # ; = , - . / `
    *_range(0xDB, 0xDE),  # [ \ ] '
)


@dataclass
class TypingConfig:
    enabled: bool = True
    # How long after a keystroke the hand stays ignored. Long enough to bridge
    # the gaps in ordinary typing, short enough that reaching for the cursor
    # immediately after a word does not feel blocked.
    hold_off_s: float = 0.45
    # Whether cursor motion is suppressed as well as clicks. Clicks are the
    # damaging case - a stray click while typing moves the caret and can land
    # anywhere - but a twitching cursor is the visible one.
    suppress_motion: bool = True
    suppress_clicks: bool = True
    # A gesture already in progress is never interrupted: releasing a drag
    # because a key was pressed would drop whatever is being dragged.
    protect_active_gestures: bool = True


class TypingMonitor:
    """Tracks whether the operator is currently typing."""

    def __init__(self, config: TypingConfig | None = None) -> None:
        self.config = config or TypingConfig()
        self._suppress_until = 0.0
        self._last_key_at = 0.0
        self.keystrokes = 0
        self.suppressed_frames = 0
        self._was_suppressing = False

    # ------------------------------------------------------------------ #
    def _any_typing_key_down(self) -> bool:
        for vk in TYPING_KEYS:
            if key_down(vk):
                return True
        return False

    def update(self, now: float) -> bool:
        """Poll the keyboard. Returns True while typing is being detected."""
        if not self.config.enabled:
            return False
        if self._any_typing_key_down():
            self._last_key_at = now
            self._suppress_until = now + self.config.hold_off_s
            self.keystrokes += 1
            return True
        return False

    # ------------------------------------------------------------------ #
    def is_suppressing(self, now: float) -> bool:
        return self.config.enabled and now < self._suppress_until

    def remaining(self, now: float) -> float:
        return max(0.0, self._suppress_until - now)

    def should_block_motion(self, now: float, gesture_active: bool = False) -> bool:
        if not self.config.suppress_motion or not self.is_suppressing(now):
            return False
        if gesture_active and self.config.protect_active_gestures:
            return False
        return True

    def should_block_actions(self, now: float, gesture_active: bool = False) -> bool:
        if not self.config.suppress_clicks or not self.is_suppressing(now):
            return False
        if gesture_active and self.config.protect_active_gestures:
            return False
        return True

    def note_frame(self, now: float) -> bool:
        """Count a suppressed frame and report whether the state just changed."""
        suppressing = self.is_suppressing(now)
        if suppressing:
            self.suppressed_frames += 1
        changed = suppressing != self._was_suppressing
        self._was_suppressing = suppressing
        if changed:
            log.debug("typing suppression %s", "on" if suppressing else "off")
        return changed

    def reset(self) -> None:
        self._suppress_until = 0.0
        self._was_suppressing = False

    def stats(self) -> dict:
        return {
            "keystrokes_seen": self.keystrokes,
            "suppressed_frames": self.suppressed_frames,
            "hold_off_s": self.config.hold_off_s,
        }
