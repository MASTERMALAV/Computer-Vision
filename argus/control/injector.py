"""Mouse input injection via the Win32 ``SendInput`` API.

Why ``SendInput`` and not the alternatives:

* ``pyautogui`` wraps the deprecated ``mouse_event`` and adds a default pause
  between calls. At 30 Hz that pause alone would dominate the latency budget.
* ``SetCursorPos`` teleports the pointer without generating input events. Many
  applications - games, drawing tools, anything using Raw Input or
  ``WM_MOUSEMOVE`` deltas - simply will not see it.
* ``SendInput`` posts into the same queue as a physical mouse, so every
  application reacts exactly as it would to real hardware.

Safety: the injector starts in **dry-run** mode. Nothing reaches the OS until
:meth:`MouseInjector.arm` is called. A perception bug should not be able to
click things on a machine someone is working on.
"""

from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass, field

from ..logsetup import get_logger
from .screens import VirtualDesktop, get_cursor_position, get_virtual_desktop

log = get_logger("control.injector")

IS_WINDOWS = sys.platform == "win32"

INPUT_MOUSE = 0

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x1000
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_ABSOLUTE = 0x8000

WHEEL_DELTA = 120

# Stamped into dwExtraInfo on every event we post, so our own injected input can
# be told apart from the user's real mouse ("ARGU" as a 32-bit value).
ARGUS_SIGNATURE = 0x41524755

# ULONG_PTR is pointer-sized: 8 bytes on 64-bit Python, 4 on 32-bit. Hardcoding
# DWORD here is a classic bug - the struct silently mis-aligns on x64 and
# SendInput rejects every event by returning 0.
ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


@dataclass
class InjectorStats:
    moves: int = 0
    clicks: int = 0
    scrolls: int = 0
    failures: int = 0
    suppressed: int = 0  # calls made while disarmed
    last_error: int = 0
    inject_ms: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        avg = sum(self.inject_ms) / len(self.inject_ms) if self.inject_ms else 0.0
        return {
            "moves": self.moves,
            "clicks": self.clicks,
            "scrolls": self.scrolls,
            "failures": self.failures,
            "suppressed": self.suppressed,
            "mean_inject_ms": round(avg, 4),
        }


class InjectorError(RuntimeError):
    pass


class MouseInjector:
    """Moves and clicks the real cursor.

    Disarmed by default: every call is counted and logged but nothing is sent.
    """

    def __init__(self, desktop: VirtualDesktop | None = None, armed: bool = False) -> None:
        self.desktop = desktop or get_virtual_desktop()
        self._armed = armed
        self.stats = InjectorStats()
        self._buttons_down: set[str] = set()
        self._last_position: tuple[float, float] | None = None

        if IS_WINDOWS:
            # If this ever fires, the struct layout is wrong and nothing would
            # work; better to know at construction than to debug silent no-ops.
            expected = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
            actual = ctypes.sizeof(INPUT)
            if actual != expected:  # pragma: no cover - platform sanity check
                raise InjectorError(
                    f"INPUT struct is {actual} bytes, expected {expected}. "
                    "ctypes layout is wrong for this interpreter."
                )
            self._send_input = ctypes.windll.user32.SendInput
            self._send_input.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
            self._send_input.restype = wintypes.UINT
        else:
            self._send_input = None

    # ------------------------------------------------------------------ #
    @property
    def armed(self) -> bool:
        return self._armed

    def arm(self) -> None:
        if not self._armed:
            log.warning("injector ARMED - cursor input is now going to the real desktop")
        self._armed = True

    def disarm(self) -> None:
        """Stop sending input, releasing anything currently held down."""
        self.release_all()
        if self._armed:
            log.info("injector disarmed")
        self._armed = False

    def refresh_desktop(self) -> None:
        """Re-read the display layout, e.g. after a monitor is plugged in."""
        self.desktop = get_virtual_desktop()
        log.info("display layout refreshed:\n%s", self.desktop.describe())

    # ------------------------------------------------------------------ #
    def _send(self, *events: MOUSEINPUT) -> bool:
        if not self._armed:
            self.stats.suppressed += 1
            return True
        if not IS_WINDOWS or self._send_input is None:
            self.stats.suppressed += 1
            return True

        count = len(events)
        array = (INPUT * count)()
        for i, mi in enumerate(events):
            array[i].type = INPUT_MOUSE
            array[i].mi = mi

        start = time.perf_counter()
        sent = self._send_input(count, array, ctypes.sizeof(INPUT))
        self.stats.inject_ms.append((time.perf_counter() - start) * 1000.0)
        if len(self.stats.inject_ms) > 500:
            del self.stats.inject_ms[:250]

        if sent != count:
            self.stats.failures += 1
            self.stats.last_error = ctypes.get_last_error() if IS_WINDOWS else 0
            # UIPI blocks a normal process from injecting into a window running
            # elevated, and nothing at all reaches the UAC secure desktop.
            log.debug(
                "SendInput sent %d of %d events (error %d); a focused elevated "
                "window or the UAC prompt will block injection",
                sent,
                count,
                self.stats.last_error,
            )
            return False
        return True

    @staticmethod
    def _mouse(flags: int, dx: int = 0, dy: int = 0, data: int = 0) -> MOUSEINPUT:
        return MOUSEINPUT(
            dx=dx,
            dy=dy,
            mouseData=ctypes.c_uint32(data).value,
            dwFlags=flags,
            time=0,
            dwExtraInfo=ARGUS_SIGNATURE,
        )

    # ------------------------------------------------------------------ #
    # Movement
    # ------------------------------------------------------------------ #
    def move_to(self, x: float, y: float) -> bool:
        """Move to an absolute virtual-desktop pixel coordinate."""
        nx, ny = self.desktop.to_absolute(x, y)
        ok = self._send(
            self._mouse(
                MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
                dx=nx,
                dy=ny,
            )
        )
        self.stats.moves += 1
        self._last_position = self.desktop.clamp(x, y)
        return ok

    def position(self) -> tuple[float, float]:
        """Where the cursor is - the real one when armed, the simulated one when not."""
        if self._armed and IS_WINDOWS:
            return get_cursor_position()
        if self._last_position is None:
            self._last_position = self.desktop.primary.center
        return self._last_position

    # ------------------------------------------------------------------ #
    # Buttons
    # ------------------------------------------------------------------ #
    _DOWN = {
        "left": MOUSEEVENTF_LEFTDOWN,
        "right": MOUSEEVENTF_RIGHTDOWN,
        "middle": MOUSEEVENTF_MIDDLEDOWN,
    }
    _UP = {
        "left": MOUSEEVENTF_LEFTUP,
        "right": MOUSEEVENTF_RIGHTUP,
        "middle": MOUSEEVENTF_MIDDLEUP,
    }

    def button_down(self, button: str = "left") -> bool:
        if button in self._buttons_down:
            return True
        flag = self._DOWN.get(button)
        if flag is None:
            raise ValueError(f"unknown button {button!r}")
        ok = self._send(self._mouse(flag))
        self._buttons_down.add(button)
        return ok

    def button_up(self, button: str = "left") -> bool:
        flag = self._UP.get(button)
        if flag is None:
            raise ValueError(f"unknown button {button!r}")
        ok = self._send(self._mouse(flag))
        self._buttons_down.discard(button)
        return ok

    def click(self, button: str = "left") -> bool:
        """Press and release as a single atomic SendInput call.

        Sending both events together guarantees no cursor movement can be
        processed between them, which is what would otherwise turn a click into
        an accidental one-pixel drag.
        """
        down, up = self._DOWN.get(button), self._UP.get(button)
        if down is None or up is None:
            raise ValueError(f"unknown button {button!r}")
        ok = self._send(self._mouse(down), self._mouse(up))
        self.stats.clicks += 1
        return ok

    def double_click(self, button: str = "left") -> bool:
        down, up = self._DOWN[button], self._UP[button]
        ok = self._send(
            self._mouse(down), self._mouse(up), self._mouse(down), self._mouse(up)
        )
        self.stats.clicks += 2
        return ok

    @property
    def buttons_down(self) -> frozenset[str]:
        return frozenset(self._buttons_down)

    def release_all(self) -> None:
        """Release every held button. Always called on shutdown.

        Without this, exiting mid-drag leaves the OS believing a button is still
        physically held, and the desktop becomes unusable until the user clicks.
        """
        for button in list(self._buttons_down):
            try:
                self.button_up(button)
            except Exception:  # pragma: no cover - best effort during teardown
                pass

    # ------------------------------------------------------------------ #
    # Wheel
    # ------------------------------------------------------------------ #
    def scroll(self, clicks: float, horizontal: bool = False) -> bool:
        """Scroll by notches; one notch is the standard three-line step."""
        amount = int(round(clicks * WHEEL_DELTA))
        if amount == 0:
            return True
        ok = self._send(
            self._mouse(
                MOUSEEVENTF_HWHEEL if horizontal else MOUSEEVENTF_WHEEL,
                data=amount,
            )
        )
        self.stats.scrolls += 1
        return ok

    # ------------------------------------------------------------------ #
    def __enter__(self) -> "MouseInjector":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release_all()
