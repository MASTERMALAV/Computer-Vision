"""Windows display geometry, DPI awareness, and the virtual desktop.

Two things must be right or every cursor coordinate is silently wrong:

**DPI awareness.** On a display with scaling (125%, 150%, ...) an unaware
process is lied to: Windows reports virtualised, scaled-down metrics and then
stretches the result. A cursor computed from those numbers lands in the wrong
place, and the error grows with distance from the origin. Awareness must be
declared *before* any metric is read, which is why :func:`ensure_dpi_aware` runs
at import time.

**The virtual desktop origin is not (0, 0).** The primary display's top-left is
the origin, so a second monitor placed to its *left* occupies negative X. On the
development setup - external monitor left, laptop right - the monitor's pixels
start at roughly x = -1920. Code that assumes ``0 <= x < screen_width`` cannot
reach the left monitor at all, which is the single most common bug in webcam
mouse projects.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass

from ..logsetup import get_logger

log = get_logger("control.screens")

IS_WINDOWS = sys.platform == "win32"

# GetSystemMetrics indices for the virtual desktop bounding box.
SM_CXSCREEN, SM_CYSCREEN = 0, 1
SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79
SM_CMONITORS = 80

# SetProcessDpiAwarenessContext values (negative sentinels, not handles).
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE = ctypes.c_void_p(-3)

MONITOR_DEFAULTTONEAREST = 2
MDT_EFFECTIVE_DPI = 0

_dpi_state: str | None = None


def ensure_dpi_aware() -> str:
    """Declare per-monitor DPI awareness. Idempotent; returns what was achieved.

    Tries the modern API first and falls back through the older ones, because
    the call fails (rather than no-ops) if awareness was already set by a
    manifest or by an earlier call in the same process.
    """
    global _dpi_state
    if _dpi_state is not None:
        return _dpi_state
    if not IS_WINDOWS:
        _dpi_state = "not-windows"
        return _dpi_state

    user32 = ctypes.windll.user32
    # Windows 10 1703+: the only mode that keeps metrics correct when a window
    # moves between displays with different scaling.
    try:
        if user32.SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2):
            _dpi_state = "per-monitor-v2"
            log.debug("DPI awareness: per-monitor v2")
            return _dpi_state
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        _dpi_state = "per-monitor"
        return _dpi_state
    except (AttributeError, OSError):
        pass
    try:
        user32.SetProcessDPIAware()
        _dpi_state = "system"
        return _dpi_state
    except (AttributeError, OSError):
        pass

    # Already set elsewhere, or an OS too old to care. Not fatal, but the
    # coordinates may be virtualised on scaled displays, so say so loudly.
    _dpi_state = "unknown"
    log.warning(
        "could not set DPI awareness; cursor coordinates may be inaccurate on "
        "displays with scaling other than 100%%"
    )
    return _dpi_state


@dataclass(frozen=True)
class Monitor:
    """One physical display, in virtual-desktop pixel coordinates."""

    index: int
    left: int
    top: int
    right: int
    bottom: int
    is_primary: bool
    dpi: int = 96

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def scale(self) -> float:
        return self.dpi / 96.0

    @property
    def center(self) -> tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom

    def describe(self) -> str:
        tag = " (primary)" if self.is_primary else ""
        scale = f"  {int(self.scale * 100)}%" if self.dpi != 96 else ""
        return (
            f"[{self.index}] {self.width}x{self.height} at ({self.left}, {self.top})"
            f"{tag}{scale}"
        )


@dataclass(frozen=True)
class VirtualDesktop:
    """The bounding box that encloses every display.

    ``left``/``top`` are negative when displays sit left of or above the
    primary. All cursor maths happens in this space.
    """

    left: int
    top: int
    width: int
    height: int
    monitors: tuple[Monitor, ...]

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def primary(self) -> Monitor:
        for m in self.monitors:
            if m.is_primary:
                return m
        return self.monitors[0]

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        """Clamp to the desktop bounds, staying one pixel inside the far edges."""
        cx = min(max(x, self.left), self.right - 1)
        cy = min(max(y, self.top), self.bottom - 1)
        return cx, cy

    def monitor_at(self, x: float, y: float) -> Monitor:
        for m in self.monitors:
            if m.contains(x, y):
                return m
        return self.primary

    def to_absolute(self, x: float, y: float) -> tuple[int, int]:
        """Convert a virtual-desktop pixel to SendInput's 0..65535 range.

        ``MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK`` normalises across the
        whole virtual desktop, so the offset of the desktop origin must be
        subtracted first - this is what makes negative-X monitors reachable.

        The denominator is ``width - 1``, not ``width``. With ``width`` the
        rightmost column is unreachable, which shows up as a cursor that can
        never quite touch the right or bottom edge - and therefore cannot hit a
        maximise button or an auto-hidden taskbar.
        """
        cx, cy = self.clamp(x, y)
        denom_x = max(self.width - 1, 1)
        denom_y = max(self.height - 1, 1)
        nx = int(round((cx - self.left) * 65535.0 / denom_x))
        ny = int(round((cy - self.top) * 65535.0 / denom_y))
        return min(max(nx, 0), 65535), min(max(ny, 0), 65535)

    def describe(self) -> str:
        lines = [
            f"virtual desktop: {self.width}x{self.height} "
            f"origin ({self.left}, {self.top})  {len(self.monitors)} display(s)"
        ]
        lines += ["  " + m.describe() for m in self.monitors]
        if self.left < 0 or self.top < 0:
            lines.append(
                "  note: a display sits left of / above the primary, so part of "
                "the desktop has negative coordinates"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class _MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", _RECT),
        ("rcWork", _RECT),
        ("dwFlags", wintypes.DWORD),
    ]


MONITORINFOF_PRIMARY = 0x00000001


def _monitor_dpi(hmonitor) -> int:
    """Effective DPI of a monitor; 96 if the API is unavailable."""
    try:
        dpi_x, dpi_y = wintypes.UINT(), wintypes.UINT()
        if ctypes.windll.shcore.GetDpiForMonitor(
            hmonitor, MDT_EFFECTIVE_DPI, ctypes.byref(dpi_x), ctypes.byref(dpi_y)
        ) == 0:
            return int(dpi_x.value)
    except (AttributeError, OSError):
        pass
    return 96


def enumerate_monitors() -> list[Monitor]:
    """All displays, in virtual-desktop coordinates."""
    if not IS_WINDOWS:
        return []
    ensure_dpi_aware()

    monitors: list[Monitor] = []
    callback_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(_RECT), wintypes.LPARAM
    )

    def _callback(hmonitor, _hdc, _rect_ptr, _lparam):
        info = _MONITORINFO()
        info.cbSize = ctypes.sizeof(_MONITORINFO)
        if ctypes.windll.user32.GetMonitorInfoW(hmonitor, ctypes.byref(info)):
            r = info.rcMonitor
            monitors.append(
                Monitor(
                    index=len(monitors),
                    left=int(r.left),
                    top=int(r.top),
                    right=int(r.right),
                    bottom=int(r.bottom),
                    is_primary=bool(info.dwFlags & MONITORINFOF_PRIMARY),
                    dpi=_monitor_dpi(hmonitor),
                )
            )
        return True

    ctypes.windll.user32.EnumDisplayMonitors(None, None, callback_type(_callback), 0)
    # Left-to-right ordering matches how the displays are physically arranged,
    # which is what a person means by "the monitor on the left".
    monitors.sort(key=lambda m: (m.left, m.top))
    return [
        Monitor(i, m.left, m.top, m.right, m.bottom, m.is_primary, m.dpi)
        for i, m in enumerate(monitors)
    ]


def get_virtual_desktop() -> VirtualDesktop:
    """Current display layout. Cheap enough to re-read if displays change."""
    if not IS_WINDOWS:
        # Headless/test fallback: a single notional 1920x1080 display.
        mon = Monitor(0, 0, 0, 1920, 1080, True, 96)
        return VirtualDesktop(0, 0, 1920, 1080, (mon,))

    ensure_dpi_aware()
    metric = ctypes.windll.user32.GetSystemMetrics
    left = int(metric(SM_XVIRTUALSCREEN))
    top = int(metric(SM_YVIRTUALSCREEN))
    width = int(metric(SM_CXVIRTUALSCREEN))
    height = int(metric(SM_CYVIRTUALSCREEN))

    monitors = tuple(enumerate_monitors())
    if not monitors:
        monitors = (Monitor(0, left, top, left + width, top + height, True, 96),)
    if width <= 0 or height <= 0:  # pragma: no cover - defensive
        width = int(metric(SM_CXSCREEN))
        height = int(metric(SM_CYSCREEN))
        left = top = 0
    return VirtualDesktop(left, top, width, height, monitors)


def get_cursor_position() -> tuple[int, int]:
    """Current cursor position in virtual-desktop coordinates."""
    if not IS_WINDOWS:
        return (0, 0)
    point = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
    return int(point.x), int(point.y)


# --------------------------------------------------------------------------- #
# Moving between displays
# --------------------------------------------------------------------------- #
def relative_position(monitor: Monitor, x: float, y: float) -> tuple[float, float]:
    """Where a point sits inside a monitor, as fractions in 0..1."""
    fx = (x - monitor.left) / max(monitor.width, 1)
    fy = (y - monitor.top) / max(monitor.height, 1)
    return min(max(fx, 0.0), 1.0), min(max(fy, 0.0), 1.0)


def absolute_position(monitor: Monitor, fx: float, fy: float) -> tuple[float, float]:
    """The inverse of :func:`relative_position`, kept one pixel inside the edges."""
    x = monitor.left + fx * (monitor.width - 1)
    y = monitor.top + fy * (monitor.height - 1)
    return x, y


def jump_target(
    desktop: VirtualDesktop, x: float, y: float, step: int = 1
) -> tuple[float, float, Monitor]:
    """Where the cursor should land on the next display.

    The *relative* position is preserved rather than centring, so the cursor
    keeps the place it had: at the top-left of one screen it arrives at the
    top-left of the next. Centring would discard the intent behind where the
    pointer already was.

    This matters more than it sounds on a mixed setup - these two displays
    differ in size, in vertical offset and in DPI, so no fixed pixel offset
    would land sensibly on both.
    """
    monitors = desktop.monitors
    if len(monitors) < 2:
        return x, y, desktop.monitor_at(x, y)

    current = desktop.monitor_at(x, y)
    index = next((i for i, m in enumerate(monitors) if m.index == current.index), 0)
    target = monitors[(index + step) % len(monitors)]

    fx, fy = relative_position(current, x, y)
    nx, ny = absolute_position(target, fx, fy)
    return nx, ny, target
