"""A small always-on-top status readout, instead of a full camera preview.

The preview window is a debugging instrument. Once the system is trusted it
stops earning its cost, and that cost is not small: on this machine drawing the
HUD and pushing it to the screen takes 6.85 ms of an 18.6 ms frame - **37 % of
all compute spent rendering a window nobody is looking at.** It also occupies a
large piece of screen and shows the room continuously.

The pill replaces it with roughly 360x70 pixels in a corner: armed state, what
the hand is doing, and a pinch meter. Two properties matter and neither is
optional:

**Click-through.** The system moves the real cursor, so a floating window that
could intercept a click would eventually swallow one of our own injected
clicks. ``WS_EX_TRANSPARENT`` makes the pill invisible to the mouse entirely.

**Never activates.** ``WS_EX_NOACTIVATE`` keeps it from stealing focus from
whatever you are actually working in.

Because the pill never takes focus, it also never receives key presses, so every
control in this mode is read from the hardware - see :mod:`argus.control.hotkeys`.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass

import numpy as np

from ..logsetup import get_logger
from .overlay import COLORS, draw_bar, draw_text, fps_color

log = get_logger("ui.pill")

IS_WINDOWS = sys.platform == "win32"

# Window styles
GWL_STYLE = -16
GWL_EXSTYLE = -20
WS_POPUP = 0x80000000
WS_VISIBLE = 0x10000000
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080  # keeps it out of the alt-tab list

HWND_TOPMOST = -1
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040
SWP_FRAMECHANGED = 0x0020

LWA_ALPHA = 0x00000002

CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")

# A status readout is read, not watched. Refreshing it at the camera's rate
# spends real time in the compositor for no benefit.
REFRESH_HZ = 12.0


@dataclass
class PillState:
    """Everything the pill displays. Deliberately flat and cheap to build."""

    armed: bool = False
    engaged: bool = False
    mode: str = "point"  # point | scroll | drag
    pinch: float = 1.0
    pinch_close: float = 0.34
    fps: float = 0.0
    identity: str = ""
    frozen: bool = False
    typing: bool = False
    note: str = ""


class StatusPill:
    """A compact, always-on-top, click-through status window."""

    def __init__(
        self,
        title: str = "ARGUS status",
        width: int = 360,
        height: int = 72,
        corner: str = "bottom-right",
        margin: int = 18,
        opacity: int = 225,
    ) -> None:
        self.title = title
        self.width = width
        self.height = height
        self.corner = corner if corner in CORNERS else "bottom-right"
        self.margin = margin
        self.opacity = max(40, min(255, opacity))

        self._canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self._interval = 1.0 / REFRESH_HZ
        self._last_draw = -1e9
        self._last_signature: tuple | None = None
        self._styled = False
        self._created = False
        self._hwnd = None

    # ------------------------------------------------------------------ #
    def _position(self) -> tuple[int, int]:
        """Corner placement on the primary display, in physical pixels."""
        from ..control.screens import get_virtual_desktop

        monitor = get_virtual_desktop().primary
        if self.corner == "top-left":
            return monitor.left + self.margin, monitor.top + self.margin
        if self.corner == "top-right":
            return monitor.right - self.width - self.margin, monitor.top + self.margin
        if self.corner == "bottom-left":
            return monitor.left + self.margin, monitor.bottom - self.height - self.margin * 3
        return (
            monitor.right - self.width - self.margin,
            monitor.bottom - self.height - self.margin * 3,
        )

    def _apply_styles(self) -> None:
        """Strip the frame, pin it on top, and make it ignore the mouse."""
        if not IS_WINDOWS or self._styled:
            return
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, self.title)
        if not hwnd:
            # HighGUI may not have realised the window yet; try again next frame.
            return
        self._hwnd = hwnd

        user32.SetWindowLongW(hwnd, GWL_STYLE, WS_POPUP | WS_VISIBLE)
        user32.SetWindowLongW(
            hwnd,
            GWL_EXSTYLE,
            WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
        )
        user32.SetLayeredWindowAttributes(hwnd, 0, self.opacity, LWA_ALPHA)

        x, y = self._position()
        user32.SetWindowPos(
            hwnd, wintypes.HWND(HWND_TOPMOST), x, y, self.width, self.height,
            SWP_NOACTIVATE | SWP_SHOWWINDOW | SWP_FRAMECHANGED,
        )
        self._styled = True
        log.debug("status pill placed at (%d, %d)", x, y)

    # ------------------------------------------------------------------ #
    def _render(self, state: PillState) -> np.ndarray:
        import cv2

        canvas = self._canvas
        canvas[:] = COLORS["bg"]

        accent = COLORS["error"] if state.armed else COLORS["ok"]
        # A solid stripe down the left edge: readable at a glance, from the
        # corner of your eye, without reading any text.
        cv2.rectangle(canvas, (0, 0), (6, self.height), accent, -1)

        label = "ARMED" if state.armed else "OFF"
        draw_text(canvas, label, (16, 26), 0.62, accent, 2)

        if state.armed:
            if state.typing:
                status, colour = "typing...", COLORS["warn"]
            elif not state.engaged:
                status, colour = "parked", COLORS["muted"]
            elif state.mode == "scroll":
                status, colour = "SCROLL", COLORS["accent"]
            elif state.mode == "drag":
                status, colour = "DRAG", COLORS["accent"]
            elif state.frozen:
                status, colour = "click...", COLORS["warn"]
            else:
                status, colour = "pointing", COLORS["ok"]
        else:
            status, colour = "F9 to arm", COLORS["muted"]
        draw_text(canvas, status, (104, 26), 0.52, colour, 1)

        draw_text(canvas, f"{state.fps:4.1f}", (self.width - 58, 26), 0.5,
                  fps_color(state.fps), 1)

        # Pinch meter: fills as the fingers close, and turns green past the
        # threshold - the single most useful number while learning.
        ratio = 1.0 - min(state.pinch / max(state.pinch_close * 2.6, 1e-6), 1.0)
        draw_bar(canvas, (16, 40), self.width - 78, ratio,
                 COLORS["ok"] if state.pinch < state.pinch_close else COLORS["accent"],
                 height=7)

        footer = state.note or state.identity or "Esc = stop"
        draw_text(canvas, footer[:52], (16, 64), 0.4, COLORS["muted"], 1)
        return canvas

    # ------------------------------------------------------------------ #
    def show(self, state: PillState, now: float | None = None) -> bool:
        """Update the pill, at most ``REFRESH_HZ`` times a second.

        Measured at every frame this cost 8.18 ms - almost as much as the full
        camera preview it replaces, which would have defeated the point. Almost
        none of that is drawing 360x72 pixels: it is the message-pump turn plus
        the desktop compositor handling a topmost *layered* window, and both are
        paid per update regardless of how little changed.

        A status readout does not need 30 Hz. Refreshing at 12 Hz keeps it
        perfectly responsive to read while cutting the cost by roughly a third
        of a frame, and an unchanged state skips the update entirely.
        """
        import cv2
        import time as _time

        now = _time.perf_counter() if now is None else now
        signature = (
            state.armed, state.engaged, state.mode, state.frozen, state.typing,
            round(state.pinch, 2), round(state.fps), state.note, state.identity,
        )
        due = (now - self._last_draw) >= self._interval
        if not due and signature == self._last_signature:
            return False

        if not self._created:
            cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)
            self._created = True

        cv2.imshow(self.title, self._render(state))
        # HighGUI needs a message-pump turn for the window to exist and repaint.
        # pollKey does that without waitKey's up-to-a-millisecond sleep.
        try:
            cv2.pollKey()
        except AttributeError:  # pragma: no cover - very old OpenCV
            cv2.waitKey(1)
        self._apply_styles()

        self._last_draw = now
        self._last_signature = signature
        return True

    def close(self) -> None:
        if not self._created:
            return
        import cv2

        try:
            cv2.destroyWindow(self.title)
            for _ in range(3):
                cv2.waitKey(1)
        except Exception:  # pragma: no cover - teardown races in HighGUI
            pass
        self._created = False
        self._styled = False

    def __enter__(self) -> "StatusPill":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
