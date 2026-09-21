"""Pick the capture backend by measuring it, not by assuming.

The motivating case, measured on the development machine's Logitech Brio 100:

    backend    format         requested       measured
    DirectShow YUY2 1280x720  30 fps           5.0 fps
    MSMF       (internal)     30 fps          30.2 fps

The camera *does* advertise MJPG 1280x720 @ 30 fps, but OpenCV's DirectShow
backend silently negotiates uncompressed YUY2 instead, and the camera itself
caps YUY2 at 720p to 5 fps. ``cap.set(CAP_PROP_FOURCC, MJPG)`` returns ``True``
and changes nothing. Meanwhile MSMF, which refuses ``CAP_PROP_FOURCC`` outright
(``set`` returns ``False``), picks a sane format on its own and hits full rate.

No static rule predicts this - it is a property of the camera, its driver, the
requested resolution and the OpenCV build. So ARGUS measures: it opens each
candidate backend, times real frames, and keeps the one that actually delivers.
The verdict is cached per camera and format, so only the first run pays for it.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import DATA_DIR, CameraConfig
from ..logsetup import get_logger

log = get_logger("capture.negotiate")

CACHE_PATH = DATA_DIR / "capture_profiles.json"
CACHE_VERSION = 1

# Backends worth trying, best-first guess. Order only affects which one wins a
# tie; the measurement decides the rest.
WINDOWS_CANDIDATES = ("msmf", "dshow")
POSIX_CANDIDATES = ("any",)

# A backend that reaches this fraction of the requested rate is good enough to
# stop looking - no point spending 1.5 s probing an alternative that cannot win.
GOOD_ENOUGH = 0.85

WARMUP_FRAMES = 8
MEASURE_FRAMES = 25


@dataclass
class CaptureProfile:
    """The outcome of negotiating one camera into one usable capture mode."""

    backend: str
    width: int
    height: int
    measured_fps: float
    fourcc: str
    requested_fps: int

    @property
    def ratio(self) -> float:
        return self.measured_fps / self.requested_fps if self.requested_fps else 0.0

    def describe(self) -> str:
        return (
            f"{self.backend} {self.width}x{self.height} "
            f"{self.fourcc or '-'} @ {self.measured_fps:.1f} fps measured"
        )


def _fourcc_str(cap) -> str:
    """Readable FOURCC. Some backends report binary junk here; show it as blank."""
    import cv2

    raw = int(cap.get(cv2.CAP_PROP_FOURCC))
    if not raw:
        return ""
    chars = [chr((raw >> (8 * i)) & 0xFF) for i in range(4)]
    if not all(32 <= ord(c) < 127 for c in chars):
        return ""
    return "".join(chars).strip()


def apply_format(cap, cfg: CameraConfig, backend: str) -> None:
    """Apply resolution / codec / FPS to an open VideoCapture.

    FOURCC is only attempted on DirectShow. MSMF rejects the property, and
    calling it there just produces a confusing ``False`` return in the logs.
    """
    import cv2

    if backend == "dshow" and cfg.fourcc:
        # On DirectShow the codec must be requested before the frame size, or
        # the driver re-negotiates the format and discards the codec choice.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    if cfg.fps:
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:  # pragma: no cover - not every backend has it
        pass
    if cfg.autofocus is not None:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if cfg.autofocus else 0)
    if cfg.autoexposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if cfg.autoexposure else 0.25)


MEASURE_ATTEMPTS = 3
MEASURE_RETRY_DELAY_S = 0.7


def _measure(
    index: int, cfg: CameraConfig, backend: str, attempt: int = 0
) -> CaptureProfile | None:
    """Open one backend, apply the format, and time real frames.

    Retries a device that opens but yields nothing. Windows commonly needs a
    moment to finish tearing down a previous process's capture handle, and two
    ARGUS commands run back to back hit that window routinely. Without the
    retry a transient handle release looks identical to a dead camera - which
    matters more now that a chosen camera is pinned to one backend and cannot
    quietly fall back to another device.
    """
    import cv2

    from .devices import backend_flag

    cap = cv2.VideoCapture(index, backend_flag(backend))
    if not cap.isOpened():
        cap.release()
        log.debug("%s: could not open camera %d", backend, index)
        return None
    try:
        apply_format(cap, cfg, backend)

        for _ in range(WARMUP_FRAMES):
            cap.read()  # let exposure and the format change settle

        start = time.perf_counter()
        good = 0
        for _ in range(MEASURE_FRAMES):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                good += 1
        elapsed = time.perf_counter() - start

        if good == 0 or elapsed <= 0:
            if attempt + 1 < MEASURE_ATTEMPTS:
                log.debug(
                    "%s:%d delivered no frames; retrying in %.1fs",
                    backend, index, MEASURE_RETRY_DELAY_S,
                )
                cap.release()
                time.sleep(MEASURE_RETRY_DELAY_S)
                return _measure(index, cfg, backend, attempt + 1)
            log.debug("%s: delivered no frames", backend)
            return None

        return CaptureProfile(
            backend=backend,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            measured_fps=good / elapsed,
            fourcc=_fourcc_str(cap),
            requested_fps=cfg.fps or 30,
        )
    finally:
        cap.release()


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def _cache_key(device_name: str, cfg: CameraConfig) -> str:
    return f"{device_name}|{cfg.width}x{cfg.height}@{cfg.fps}|{cfg.fourcc}"


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        if data.get("version") != CACHE_VERSION:
            return {}
        return data.get("profiles", {})
    except Exception as exc:  # pragma: no cover - a corrupt cache is not fatal
        log.debug("ignoring unreadable capture profile cache: %s", exc)
        return {}


def _save_cache(profiles: dict) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(
            json.dumps({"version": CACHE_VERSION, "profiles": profiles}, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover - read-only disk, etc.
        log.debug("could not write capture profile cache: %s", exc)


def clear_cache() -> None:
    """Forget every negotiated profile - used by ``--renegotiate``."""
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()
        log.info("cleared capture profile cache")


# --------------------------------------------------------------------------- #
def negotiate(
    index: int,
    device_name: str,
    cfg: CameraConfig,
    use_cache: bool = True,
) -> CaptureProfile:
    """Choose the backend that measurably delivers the requested format.

    Raises:
        RuntimeError: if no candidate backend produced a single frame.
    """
    # An explicitly requested backend is honoured without argument.
    if cfg.backend not in ("auto", "any"):
        profile = _measure(index, cfg, cfg.backend)
        if profile is None:
            raise RuntimeError(
                f"Camera {index} ({device_name}) produced no frames using the "
                f"{cfg.backend} backend."
            )
        if profile.ratio < GOOD_ENOUGH:
            log.warning(
                "%s delivers only %.1f fps of the %d fps requested. "
                "Try capture.camera.backend=auto to let ARGUS pick.",
                cfg.backend,
                profile.measured_fps,
                cfg.fps,
            )
        return profile

    key = _cache_key(device_name, cfg)
    cache = _load_cache() if use_cache else {}
    if key in cache:
        try:
            cached = CaptureProfile(**cache[key])
            log.debug("using cached capture profile: %s", cached.describe())
            return cached
        except TypeError:
            pass  # schema drifted; fall through and re-measure

    candidates = WINDOWS_CANDIDATES if sys.platform == "win32" else POSIX_CANDIDATES
    log.info(
        "negotiating capture format for '%s' (%dx%d@%d) across %s ...",
        device_name,
        cfg.width,
        cfg.height,
        cfg.fps,
        "/".join(candidates),
    )

    results: list[CaptureProfile] = []
    for backend in candidates:
        profile = _measure(index, cfg, backend)
        if profile is None:
            log.debug("  %-6s unusable", backend)
            continue
        log.info(
            "  %-6s %dx%d %-5s %5.1f fps",
            backend,
            profile.width,
            profile.height,
            profile.fourcc or "-",
            profile.measured_fps,
        )
        results.append(profile)
        if profile.ratio >= GOOD_ENOUGH:
            break  # good enough; no reason to keep probing

    if not results:
        raise RuntimeError(
            f"Camera {index} ({device_name}) produced no frames on any backend "
            f"({', '.join(candidates)}).\n"
            "  - another application may be holding the camera\n"
            "  - Windows Settings > Privacy & security > Camera must allow desktop apps"
        )

    # Prefer the fastest; break ties by resolution actually delivered.
    best = max(results, key=lambda p: (round(p.measured_fps, 1), p.width * p.height))
    log.info("selected %s", best.describe())

    if best.ratio < GOOD_ENOUGH:
        log.warning(
            "best available is %.1f fps against %d fps requested. "
            "This camera may not support %dx%d at that rate - "
            "try a lower resolution (e.g. --set capture.camera.width=640 "
            "--set capture.camera.height=360).",
            best.measured_fps,
            cfg.fps,
            cfg.width,
            cfg.height,
        )

    cache[key] = asdict(best)
    _save_cache(cache)
    return best
