"""Threaded camera capture with latest-frame semantics.

Reading frames on the main loop couples inference speed to capture speed: if a
frame takes 40 ms to process, ``cap.read()`` hands back a frame from the
driver's queue that is already 40 ms stale, and the lag compounds until the
preview is visibly behind the user's hand.

This reader runs in its own thread and keeps only the newest frame. Consumers
always get the freshest view of the world; frames produced while a consumer was
busy are dropped on purpose and counted, so the drop rate is visible rather than
silently degrading interactivity.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..config import CameraConfig
from ..logsetup import get_logger
from .devices import CameraDevice, DeviceError, backend_flag, resolve_device
from .negotiate import CaptureProfile, apply_format, clear_cache, negotiate

log = get_logger("capture.camera")

# Tried in order when the negotiated backend stops delivering frames.
FALLBACK_BACKENDS = ("dshow", "msmf", "any")


@dataclass(frozen=True)
class Frame:
    """One captured frame plus the metadata downstream stages need."""

    image: np.ndarray  # BGR, HxWx3, uint8
    index: int  # monotonically increasing frame counter
    timestamp: float  # time.perf_counter() at grab time
    monotonic_ms: int  # integer milliseconds, for MediaPipe's timestamp API

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def age_ms(self) -> float:
        """Milliseconds since this frame was grabbed from the driver."""
        return (time.perf_counter() - self.timestamp) * 1000.0


class CameraError(RuntimeError):
    """Raised when a camera cannot be opened or has failed irrecoverably."""


class CameraStream:
    """Background-threaded camera reader.

    Use as a context manager::

        with CameraStream(cfg.capture.camera) as cam:
            frame = cam.read(timeout=2.0)
    """

    def __init__(
        self,
        config: CameraConfig,
        device: CameraDevice | None = None,
        renegotiate: bool = False,
    ) -> None:
        self.config = config
        self._device = device
        self._renegotiate = renegotiate
        self.profile: CaptureProfile | None = None
        self._cap: Any = None

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._frame_ready = threading.Condition()
        self._latest: Frame | None = None
        self._last_delivered_index = -1

        self._frames_grabbed = 0
        self._frames_dropped = 0
        self._reopens = 0
        self._consecutive_failures = 0
        self._start_time = 0.0
        self._error: BaseException | None = None

        # Actual negotiated settings, filled in once the device is open.
        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 0.0
        self.actual_fourcc = ""

    # ------------------------------------------------------------------ #
    # Device setup
    # ------------------------------------------------------------------ #
    @property
    def device(self) -> CameraDevice:
        if self._device is None:
            self._device = resolve_device(
                self.config.device,
                backend=self.config.backend,
                exclude_ir=self.config.exclude_ir,
            )
        return self._device

    def _configure(self, cap: Any, backend: str) -> None:
        """Apply the capture format, then read back what we actually got."""
        import cv2

        from .negotiate import _fourcc_str

        cfg = self.config
        apply_format(cap, cfg, backend)

        self.actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
        self.actual_fourcc = _fourcc_str(cap)

        if (self.actual_width, self.actual_height) != (cfg.width, cfg.height):
            log.warning(
                "camera delivered %dx%d instead of the requested %dx%d",
                self.actual_width,
                self.actual_height,
                cfg.width,
                cfg.height,
            )

    @property
    def backend(self) -> str:
        """The backend actually in use, once negotiation has run."""
        if self.profile is not None:
            return self.profile.backend
        return self.config.backend

    # Reopen attempts when the device opens but yields no frames, which
    # happens when a previous process's capture handle is still being released.
    OPEN_ATTEMPTS = 3
    OPEN_RETRY_DELAY_S = 0.8

    def _open(self) -> Any:
        """Open the camera, falling back to another backend if necessary.

        The negotiated backend is normally right, but a capture backend can go
        bad at runtime independently of our code: on Windows, Media Foundation
        will keep reporting a device as openable while returning no frames at
        all, after a process holding it exited uncleanly. DirectShow continues
        to work in that state.

        Since the negotiated choice is cached, a wedged backend would otherwise
        make every future run fail identically. So a persistent no-frames
        failure is treated as a property of the backend rather than of the
        camera: the alternatives are tried, and the cached verdict is discarded
        so the next run re-measures from scratch.
        """
        dev = self.device

        # Measure which backend actually delivers this format on this camera -
        # the answer is camera- and driver-specific, and getting it wrong is the
        # difference between 30 fps and 5 fps. Cached after the first run.
        if self.profile is None:
            self.profile = negotiate(
                dev.index,
                dev.name,
                self.config,
                use_cache=not self._renegotiate,
            )

        preferred = self.profile.backend
        candidates = [preferred]
        if self.config.backend in ("auto", "any"):
            candidates += [b for b in FALLBACK_BACKENDS if b != preferred]

        first_error: BaseException | None = None
        for i, backend in enumerate(candidates):
            try:
                cap = self._open_with(backend)
            except CameraError as exc:
                first_error = first_error or exc
                if i + 1 < len(candidates):
                    log.warning(
                        "%s backend is not delivering frames; falling back to %s",
                        backend, candidates[i + 1],
                    )
                continue
            if backend != preferred:
                # Remember the working backend for this session and force a
                # fresh measurement next time rather than trusting the old one.
                log.warning(
                    "recovered using the %s backend; clearing the cached "
                    "capture profile so it is re-measured next run",
                    backend,
                )
                self.profile = replace(self.profile, backend=backend)
                clear_cache()
            return cap

        raise first_error or CameraError(f"Could not open camera {dev.label}.")

    def _open_with(self, backend: str, attempt: int = 0) -> Any:
        import cv2

        dev = self.device

        log.info(
            "opening camera %s via %s at %dx%d@%d",
            dev.label,
            backend,
            self.config.width,
            self.config.height,
            self.config.fps,
        )
        cap = cv2.VideoCapture(dev.index, backend_flag(backend))

        deadline = time.perf_counter() + self.config.open_timeout_s
        while not cap.isOpened() and time.perf_counter() < deadline:
            time.sleep(0.05)
        if not cap.isOpened():
            cap.release()
            raise CameraError(
                f"Could not open camera {dev.label} using the {backend} backend.\n"
                "Things worth checking:\n"
                "  - another application (Teams, Zoom, Camera app) may be holding it\n"
                "  - Windows Settings > Privacy & security > Camera must allow desktop apps\n"
                "  - try a different backend: --set capture.camera.backend=msmf"
            )

        self._configure(cap, backend)

        # Some UVC cameras return a few empty frames while exposure settles.
        for _ in range(10):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                break
            time.sleep(0.05)
        else:
            # The device handle opened but produced nothing. The usual cause is
            # that a previous process has exited but Windows has not finished
            # tearing down its capture handle yet, which is common when two
            # ARGUS commands run back to back. Reopening after a short pause
            # recovers it; only a persistent failure is a real error.
            cap.release()
            if attempt + 1 < self.OPEN_ATTEMPTS:
                log.warning(
                    "camera %s opened but delivered no frames; retrying in %.1fs "
                    "(attempt %d of %d)",
                    dev.label, self.OPEN_RETRY_DELAY_S, attempt + 2, self.OPEN_ATTEMPTS,
                )
                time.sleep(self.OPEN_RETRY_DELAY_S)
                return self._open_with(backend, attempt=attempt + 1)
            raise CameraError(
                f"Camera {dev.label} opened but delivered no frames after "
                f"{self.OPEN_ATTEMPTS} attempts on the {backend} backend.\n"
                "  - another application (Teams, Zoom, the Camera app) may be holding it\n"
                "  - a previous ARGUS run may not have exited cleanly"
            )

        log.info(
            "camera ready: %dx%d @ %.1f fps reported, codec %s",
            self.actual_width,
            self.actual_height,
            self.actual_fps,
            self.actual_fourcc or "unknown",
        )
        return cap

    # ------------------------------------------------------------------ #
    # Thread lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> "CameraStream":
        if self._thread is not None:
            return self
        self._cap = self._open()
        self._stop.clear()
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="argus-capture", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        import cv2

        mirror = self.config.mirror
        while not self._stop.is_set():
            try:
                ok, image = self._cap.read()
            except Exception as exc:  # pragma: no cover - driver level failure
                ok, image = False, None
                log.debug("cap.read() raised: %s", exc)

            if not ok or image is None or image.size == 0:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.config.max_consecutive_failures:
                    if self.config.reopen_on_failure and self._try_reopen():
                        continue
                    self._error = CameraError(
                        f"Camera stopped delivering frames after "
                        f"{self._consecutive_failures} consecutive failures."
                    )
                    with self._frame_ready:
                        self._frame_ready.notify_all()
                    return
                time.sleep(0.005)
                continue

            self._consecutive_failures = 0
            now = time.perf_counter()
            if mirror:
                # Horizontal flip gives a mirror ("selfie") view, which is what
                # people expect when steering a UI with their own hands.
                image = cv2.flip(image, 1)

            frame = Frame(
                image=image,
                index=self._frames_grabbed,
                timestamp=now,
                monotonic_ms=int(now * 1000.0),
            )
            self._frames_grabbed += 1

            with self._frame_ready:
                if self._latest is not None and self._latest.index > self._last_delivered_index:
                    # Previous frame was never consumed - count it as dropped.
                    self._frames_dropped += 1
                self._latest = frame
                self._frame_ready.notify_all()

    def _try_reopen(self) -> bool:
        self._reopens += 1
        log.warning("camera read failed repeatedly; reopening (attempt %d)", self._reopens)
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        time.sleep(0.5)
        try:
            self._cap = self._open()
        except Exception as exc:
            log.error("camera reopen failed: %s", exc)
            return False
        self._consecutive_failures = 0
        return True

    # ------------------------------------------------------------------ #
    # Consumption
    # ------------------------------------------------------------------ #
    def read(self, timeout: float = 1.0, allow_repeat: bool = False) -> Frame | None:
        """Return the freshest frame, waiting up to ``timeout`` seconds.

        By default this blocks until a frame *newer* than the last one returned
        is available, so a fast consumer never processes the same image twice.
        """
        deadline = time.perf_counter() + timeout
        with self._frame_ready:
            while True:
                if self._error is not None:
                    raise self._error
                frame = self._latest
                if frame is not None and (allow_repeat or frame.index > self._last_delivered_index):
                    self._last_delivered_index = frame.index
                    return frame
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self._frame_ready.wait(remaining)

    def latest(self) -> Frame | None:
        """Non-blocking peek at the most recent frame, if any."""
        with self._frame_ready:
            return self._latest

    # ------------------------------------------------------------------ #
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and self._error is None

    @property
    def stats(self) -> dict[str, float | int | str]:
        elapsed = max(time.perf_counter() - self._start_time, 1e-9)
        return {
            "device": self.device.name,
            "index": self.device.index,
            "backend": self.backend,
            "resolution": f"{self.actual_width}x{self.actual_height}",
            "fourcc": self.actual_fourcc,
            "grabbed": self._frames_grabbed,
            "dropped": self._frames_dropped,
            "reopens": self._reopens,
            "capture_fps": round(self._frames_grabbed / elapsed, 2),
            "elapsed_s": round(elapsed, 2),
        }

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        with self._frame_ready:
            self._frame_ready.notify_all()

    def __enter__(self) -> "CameraStream":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def open_camera(config: CameraConfig) -> CameraStream:
    """Resolve the configured device and start streaming from it."""
    try:
        device = resolve_device(
            config.device, backend=config.backend, exclude_ir=config.exclude_ir
        )
    except DeviceError as exc:
        raise CameraError(str(exc)) from exc
    return CameraStream(config, device=device).start()
