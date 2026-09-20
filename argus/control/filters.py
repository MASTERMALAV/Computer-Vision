"""Signal conditioning for a hand-driven cursor.

The core problem: a fingertip tracked by a camera jitters by a few pixels even
when the hand is perfectly still, and that jitter is amplified by pointer gain.
A fixed low-pass filter can remove it, but only by adding lag that is instantly
perceptible when the hand moves fast.

The One Euro filter (Casiez, Roussel & Vogel, CHI 2012) resolves the tradeoff by
making the cutoff frequency a function of speed: heavy smoothing while slow
(where jitter is visible and lag is not), light smoothing while fast (where lag
is visible and jitter is not).

Every filter here takes an explicit timestamp rather than assuming a fixed rate.
That is not a detail: this camera's p95 inter-frame gap is 49 ms against a 33 ms
mean, so a filter tuned for "30 Hz" would silently over-smooth on every late
frame and produce speed-dependent lag - exactly the artefact it exists to avoid.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


def _alpha(cutoff_hz: float, dt: float) -> float:
    """Smoothing factor of a first-order low-pass with the given cutoff.

    tau = 1 / (2*pi*fc);  alpha = 1 / (1 + tau/dt)
    """
    if dt <= 0.0:
        return 1.0
    tau = 1.0 / (2.0 * np.pi * max(cutoff_hz, 1e-6))
    return 1.0 / (1.0 + tau / dt)


class LowPass:
    """First-order low-pass with an externally supplied alpha."""

    __slots__ = ("_value", "_initialised")

    def __init__(self) -> None:
        self._value: np.ndarray | float = 0.0
        self._initialised = False

    @property
    def value(self):
        return self._value

    @property
    def initialised(self) -> bool:
        return self._initialised

    def reset(self) -> None:
        self._initialised = False
        self._value = 0.0

    def __call__(self, x, alpha: float):
        if not self._initialised:
            self._value = x
            self._initialised = True
        else:
            self._value = alpha * x + (1.0 - alpha) * self._value
        return self._value


@dataclass
class OneEuroConfig:
    """Tuning for :class:`OneEuroFilter`.

    Tuning procedure, from the paper - do it in this order:

    1. Set ``beta = 0`` and ``min_cutoff = 1.0``.
    2. Hold the hand still and lower ``min_cutoff`` until the cursor stops
       shimmering. Lower = steadier at rest, but laggier everywhere.
    3. Move the hand quickly and raise ``beta`` until the cursor keeps up.
       Higher = less lag when fast, but jitter returns during motion.

    Never tune both at once; they trade against each other.
    """

    min_cutoff: float = 1.0  # Hz. Governs steadiness at rest.
    beta: float = 0.02  # Speed coupling. Governs lag when moving.
    d_cutoff: float = 1.0  # Hz. Cutoff for the speed estimate itself.


class OneEuroFilter:
    """Adaptive low-pass for an N-dimensional signal.

    Works on scalars or numpy vectors; the speed term uses the vector's
    magnitude so both axes of a cursor share one cutoff and cannot shear apart.
    """

    def __init__(self, config: OneEuroConfig | None = None) -> None:
        self.config = config or OneEuroConfig()
        self._x = LowPass()
        self._dx = LowPass()
        self._last_time: float | None = None
        self._last_raw: np.ndarray | float | None = None

    def reset(self) -> None:
        self._x.reset()
        self._dx.reset()
        self._last_time = None
        self._last_raw = None

    @property
    def speed(self) -> float:
        """Current filtered speed estimate, in input units per second."""
        d = self._dx.value
        if isinstance(d, np.ndarray):
            return float(np.linalg.norm(d))
        return abs(float(d))

    def __call__(self, x, timestamp: float):
        """Filter one sample taken at ``timestamp`` (seconds, monotonic)."""
        x = np.asarray(x, dtype=np.float64) if not np.isscalar(x) else float(x)

        if self._last_time is None:
            self._last_time = timestamp
            self._last_raw = x
            self._x(x, 1.0)
            return x

        dt = timestamp - self._last_time
        # Guard against a repeated or backwards timestamp: fall back to a
        # nominal 30 Hz step rather than dividing by zero or going unstable.
        if dt <= 1e-6:
            dt = 1.0 / 30.0
        # A very long gap (hand left the frame, a stall) must not be treated as
        # a real, slow movement - that would produce a huge spurious velocity.
        if dt > 0.25:
            self.reset()
            self._last_time = timestamp
            self._last_raw = x
            self._x(x, 1.0)
            return x
        self._last_time = timestamp

        derivative = (x - self._last_raw) / dt
        self._last_raw = x
        d_hat = self._dx(derivative, _alpha(self.config.d_cutoff, dt))

        speed = float(np.linalg.norm(d_hat)) if isinstance(d_hat, np.ndarray) else abs(float(d_hat))
        cutoff = self.config.min_cutoff + self.config.beta * speed
        return self._x(x, _alpha(cutoff, dt))


class PositionHistory:
    """Short ring buffer of timestamped positions.

    Used to recover where the pointer *was* a moment ago. Gesture recognition
    necessarily lags the physical gesture by its debounce window, so when a
    click is finally confirmed, the hand has already begun to deform. Rolling
    back to the position from before the gesture started removes that error.
    """

    def __init__(self, seconds: float = 0.5, expected_hz: float = 30.0) -> None:
        self.seconds = seconds
        self._buf: deque[tuple[float, np.ndarray]] = deque(
            maxlen=max(8, int(seconds * expected_hz * 2))
        )

    def add(self, position: np.ndarray, timestamp: float) -> None:
        self._buf.append((timestamp, np.array(position, dtype=np.float64)))

    def at(self, timestamp: float) -> np.ndarray | None:
        """Position nearest the given time, or None if the buffer cannot reach it."""
        if not self._buf:
            return None
        best = min(self._buf, key=lambda item: abs(item[0] - timestamp))
        return best[1].copy()

    def before(self, seconds_ago: float, now: float) -> np.ndarray | None:
        return self.at(now - seconds_ago)

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Hermite interpolation, 0 below edge0 and 1 above edge1.

    Used for the gain curve because its zero first-derivative at both ends means
    gain changes have no discontinuity the hand can feel.
    """
    if edge1 <= edge0:
        return 0.0 if x < edge0 else 1.0
    t = min(max((x - edge0) / (edge1 - edge0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class GainConfig:
    """Velocity-dependent pointer gain ("ballistics").

    A single fixed gain cannot work: low enough to hit a 16 px close button is
    far too low to cross a dual-monitor desktop, and high enough to cross the
    desktop makes precision impossible. Every real pointing device solves this
    by making gain rise with speed, so slow movements are precise and fast ones
    cover ground.

    Speeds are in *hand-scale units per second* - one unit is the user's own
    wrist-to-knuckle span - so the curve behaves identically whether the hand is
    30 cm or 60 cm from the camera.
    """

    slow_speed: float = 0.35  # below this, full precision
    fast_speed: float = 3.0  # above this, full gain
    min_gain: float = 0.55
    max_gain: float = 3.4
    # Pixels of cursor travel per hand-scale unit of hand travel, at gain 1.0.
    # One "hand width" of movement should cross a useful fraction of a screen.
    pixels_per_unit: float = 900.0

    def gain(self, speed: float) -> float:
        t = smoothstep(self.slow_speed, self.fast_speed, speed)
        return self.min_gain + (self.max_gain - self.min_gain) * t
