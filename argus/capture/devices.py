"""Camera discovery and selection.

OpenCV identifies cameras only by integer index, which is useless on a machine
with three of them ("is the Brio 0, 1 or 2 today?"). This module recovers the
friendly names and keeps them aligned with the indices OpenCV will actually use.

Name recovery, in order of preference:

1. ``pygrabber`` - enumerates the DirectShow filter graph. The order it returns
   is exactly the order ``cv2.CAP_DSHOW`` uses, so index <-> name is exact.
2. Windows PnP via PowerShell - always available, but the order is *not*
   guaranteed to match OpenCV indices, so names are attached only as a hint.
3. Nothing - devices are reported as "Camera N" and selection falls back to
   plain indices.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field, replace
from typing import Iterable

from ..logsetup import get_logger

log = get_logger("capture.devices")

# Resolutions we probe for, widest-first. Deliberately short: every entry costs
# a camera open/close round trip (~0.3-1.5 s on USB).
COMMON_MODES: tuple[tuple[int, int], ...] = (
    (1920, 1080),
    (1280, 720),
    (960, 540),
    (848, 480),
    (640, 480),
    (640, 360),
    (320, 240),
)

# Substrings that mark an infrared / depth sensor rather than a colour camera.
# Windows Hello IR cameras enumerate like normal webcams but return a dark,
# washed-out greyscale image that face recognition models were never trained on.
_IR_PATTERNS = (
    r"\bir\b",
    r"infrared",
    r"\bdepth\b",
    r"\b3d\b",
    r"windows hello",
)

# Virtual cameras (OBS, NDI, Teams, ...) - real, but rarely what you want by
# default, so they score below physical devices during auto-selection.
_VIRTUAL_PATTERNS = (
    r"obs",
    r"virtual",
    r"ndi",
    r"droidcam",
    r"iriun",
    r"epoccam",
    r"snap camera",
    r"xsplit",
    r"manycam",
)


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    low = name.lower()
    return any(re.search(p, low) for p in patterns)


@dataclass
class CameraDevice:
    """A camera OpenCV can open.

    An index alone does not identify a camera. Each capture backend enumerates
    devices independently, and they can disagree: on the development machine
    DirectShow lists [Integrated, Brio] while Media Foundation's index 0 is the
    Brio. So identity is the (backend, index) *pair*, written "msmf:0".

    ``backend_locked`` marks a device whose backend was chosen deliberately -
    by the user, or by verifying that exact pair - and must not be swapped.
    """

    index: int
    name: str
    backend: str = "dshow"
    is_ir: bool = False
    is_virtual: bool = False
    name_is_exact: bool = True  # False when the name came from an unordered source
    backend_locked: bool = False
    modes: list[tuple[int, int]] = field(default_factory=list)
    probed: bool = False
    working: bool | None = None  # None = not yet verified

    @property
    def spec(self) -> str:
        """The unambiguous identifier for this camera."""
        return f"{self.backend}:{self.index}"

    @property
    def label(self) -> str:
        tags = [self.spec]
        if self.is_ir:
            tags.append("IR")
        if self.is_virtual:
            tags.append("virtual")
        if not self.name_is_exact:
            tags.append("name?")
        suffix = f" [{', '.join(tags)}]" if tags else ""
        return f"{self.index}: {self.name}{suffix}"

    def score(self) -> int:
        """Higher is a better default choice for auto-selection.

        Preference order: external colour camera > integrated colour camera >
        virtual camera > IR camera. An external USB webcam is almost always
        positioned deliberately by the user, so it wins over the built-in one.
        """
        if self.working is False:
            return -1000
        score = 100
        low = self.name.lower()
        if self.is_ir:
            score -= 500
        if self.is_virtual:
            score -= 200
        if "integrated" in low or "built-in" in low or "internal" in low:
            score -= 50
        else:
            score += 25  # looks external
        # Known-good external webcam families get a nudge.
        if any(k in low for k in ("brio", "logi", "logitech", "razer", "elgato", "c9", "c2")):
            score += 40
        if self.modes:
            score += min(len(self.modes), 5)
            best_w = max(w for w, _ in self.modes)
            if best_w >= 1920:
                score += 10
        if self.working is True:
            score += 20
        return score

    def to_dict(self) -> dict:
        return {
            "spec": self.spec,
            "index": self.index,
            "name": self.name,
            "backend": self.backend,
            "is_ir": self.is_ir,
            "is_virtual": self.is_virtual,
            "name_is_exact": self.name_is_exact,
            "modes": [list(m) for m in self.modes],
            "probed": self.probed,
            "working": self.working,
        }


class DeviceError(RuntimeError):
    """Raised when a requested camera cannot be found or opened."""


# --------------------------------------------------------------------------- #
# Name sources
# --------------------------------------------------------------------------- #
def _names_via_pygrabber() -> list[str] | None:
    """DirectShow device names, in OpenCV CAP_DSHOW index order."""
    if sys.platform != "win32":
        return None
    try:
        from pygrabber.dshow_graph import FilterGraph  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        log.debug("pygrabber unavailable (%s); falling back to PnP names", exc)
        return None
    try:
        names = list(FilterGraph().get_input_devices())
        log.debug("pygrabber found %d DirectShow device(s)", len(names))
        return names
    except Exception as exc:  # pragma: no cover - COM can fail in odd ways
        log.warning("pygrabber enumeration failed: %s", exc)
        return None


def _names_via_pnp() -> list[str]:
    """Camera-class device names from Windows PnP. Order is NOT authoritative."""
    if sys.platform != "win32":
        return []
    ps = (
        "Get-CimInstance Win32_PnPEntity | "
        "Where-Object { $_.PNPClass -eq 'Camera' -or $_.Service -eq 'usbvideo' } | "
        "Select-Object -ExpandProperty Name | ConvertTo-Json -Compress"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return []
        data = json.loads(out.stdout.strip())
        if isinstance(data, str):
            return [data]
        return [str(x) for x in data]
    except Exception as exc:  # pragma: no cover - environment dependent
        log.debug("PnP camera enumeration failed: %s", exc)
        return []


# --------------------------------------------------------------------------- #
# Enumeration
# --------------------------------------------------------------------------- #
def backend_flag(backend: str) -> int:
    """Map a backend name to the OpenCV VideoCapture API preference constant."""
    import cv2

    return {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "any": cv2.CAP_ANY,
        "auto": cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY,
    }[backend]


def _open_index(index: int, backend: str):
    """Open a VideoCapture, returning it (caller must release) or None."""
    import cv2

    cap = cv2.VideoCapture(index, backend_flag(backend))
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def enumerate_devices(
    backend: str = "auto",
    max_index: int = 10,
    probe: bool = False,
    verify: bool = False,
) -> list[CameraDevice]:
    """List cameras available to OpenCV.

    Args:
        backend: ``auto``/``dshow``/``msmf``/``any``.
        max_index: how far to scan when names are unavailable.
        probe: additionally test each device for supported resolutions (slow).
        verify: open each device and grab one frame to confirm it really works.
    """
    resolved_backend = "dshow" if backend == "auto" and sys.platform == "win32" else backend
    if resolved_backend == "auto":
        resolved_backend = "any"

    devices: list[CameraDevice] = []
    names = _names_via_pygrabber() if resolved_backend in ("dshow", "auto") else None

    if names:
        for i, name in enumerate(names):
            devices.append(
                CameraDevice(
                    index=i,
                    name=name.strip() or f"Camera {i}",
                    backend=resolved_backend,
                    is_ir=_matches_any(name, _IR_PATTERNS),
                    is_virtual=_matches_any(name, _VIRTUAL_PATTERNS),
                    name_is_exact=True,
                )
            )
    else:
        # No authoritative ordering: scan indices and attach PnP names as hints.
        hints = _names_via_pnp()
        found = _scan_indices(resolved_backend, max_index)
        for slot, i in enumerate(found):
            hint = hints[slot] if slot < len(hints) else f"Camera {i}"
            devices.append(
                CameraDevice(
                    index=i,
                    name=hint,
                    backend=resolved_backend,
                    is_ir=_matches_any(hint, _IR_PATTERNS),
                    is_virtual=_matches_any(hint, _VIRTUAL_PATTERNS),
                    name_is_exact=False,
                )
            )

    if verify or probe:
        for dev in devices:
            _verify_device(dev, probe=probe)

    return devices


def _scan_indices(backend: str, max_index: int) -> list[int]:
    """Brute-force scan for openable camera indices."""
    found: list[int] = []
    misses = 0
    for i in range(max_index):
        cap = _open_index(i, backend)
        if cap is None:
            misses += 1
            # Indices are usually contiguous; stop after a run of failures.
            if misses >= 3 and found:
                break
            continue
        misses = 0
        found.append(i)
        cap.release()
    return found


def _verify_device(dev: CameraDevice, probe: bool = False) -> None:
    """Open a device, confirm it delivers a frame, optionally probe modes."""
    cap = _open_index(dev.index, dev.backend)
    if cap is None:
        dev.working = False
        log.debug("device %d (%s) could not be opened", dev.index, dev.name)
        return
    try:
        ok, frame = cap.read()
        dev.working = bool(ok and frame is not None and frame.size > 0)
        if probe and dev.working:
            import cv2

            supported: list[tuple[int, int]] = []
            for w, h in COMMON_MODES:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if (aw, ah) == (w, h) and (aw, ah) not in supported:
                    supported.append((aw, ah))
            dev.modes = supported
            dev.probed = True
    finally:
        cap.release()


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def resolve_device(
    spec: str | int,
    devices: list[CameraDevice] | None = None,
    backend: str = "auto",
    exclude_ir: bool = True,
) -> CameraDevice:
    """Turn a user-supplied camera spec into a concrete :class:`CameraDevice`.

    ``spec`` may be:
        * ``"auto"``     - highest-scoring device
        * ``"2"`` / ``2``- an explicit OpenCV index
        * ``"brio"``     - case-insensitive substring of the device name
    """
    if devices is None:
        devices = enumerate_devices(backend=backend)

    spec_str = str(spec).strip()

    # -- explicit backend:index, the unambiguous form ----------------------- #
    match = re.fullmatch(r"(dshow|msmf|any)\s*:\s*(\d+)", spec_str, flags=re.IGNORECASE)
    if match:
        wanted_backend = match.group(1).lower()
        idx = int(match.group(2))
        for dev in devices:
            if dev.index == idx and dev.backend == wanted_backend:
                return replace(dev, backend_locked=True)
        # Not enumerated under that backend, but the user named it explicitly -
        # honour it and let the capture layer report any failure.
        return CameraDevice(
            index=idx,
            name=f"Camera {idx} ({wanted_backend})",
            backend=wanted_backend,
            backend_locked=True,
            name_is_exact=False,
        )

    # -- explicit index ---------------------------------------------------- #
    if re.fullmatch(r"\d+", spec_str):
        idx = int(spec_str)
        for dev in devices:
            if dev.index == idx:
                return dev
        # Not enumerated, but the user asked for it by number - honour that and
        # let the capture layer report the failure if it does not open.
        log.warning("camera index %d was not enumerated; trying it anyway", idx)
        return CameraDevice(index=idx, name=f"Camera {idx}", backend=backend, name_is_exact=False)

    if not devices:
        raise DeviceError(
            "No cameras were found. Check that the camera is connected and that "
            "Windows camera privacy settings allow desktop apps to use it."
        )

    # -- auto -------------------------------------------------------------- #
    if spec_str.lower() == "auto":
        pool = [d for d in devices if not (exclude_ir and d.is_ir)] or devices
        best = max(pool, key=lambda d: (d.score(), -d.index))
        log.info("auto-selected camera %s (score %d)", best.label, best.score())
        return best

    # -- name substring ---------------------------------------------------- #
    low = spec_str.lower()
    matches = [d for d in devices if low in d.name.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer a non-IR match, then the best-scoring one.
        pool = [d for d in matches if not d.is_ir] or matches
        best = max(pool, key=lambda d: (d.score(), -d.index))
        log.warning(
            "camera spec %r matched %d devices (%s); using %s",
            spec_str,
            len(matches),
            ", ".join(d.name for d in matches),
            best.label,
        )
        return best

    available = "\n".join(f"    {d.label}" for d in devices)
    raise DeviceError(
        f"No camera matches {spec_str!r}.\nAvailable cameras:\n{available}\n"
        "Use an index, a name substring (e.g. 'brio'), or 'auto'."
    )
