"""Lightweight timing and throughput instrumentation.

Every stage of the pipeline is measured, because on a 4-core laptop the
difference between a usable system and an unusable one is a single stage that
quietly costs 80 ms. These helpers are deliberately allocation-free in the hot
path: a fixed-size ring buffer of floats, no lists growing without bound.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class Stat:
    """Summary of a rolling window of samples, in milliseconds."""

    name: str
    count: int
    mean: float
    median: float
    p95: float
    p99: float
    maximum: float

    def format(self) -> str:
        return (
            f"{self.name:<22} n={self.count:<6d} "
            f"mean={self.mean:7.2f}ms  p50={self.median:7.2f}ms  "
            f"p95={self.p95:7.2f}ms  p99={self.p99:7.2f}ms  max={self.maximum:7.2f}ms"
        )


class RollingWindow:
    """Fixed-capacity ring buffer of float samples with percentile queries."""

    __slots__ = ("_buf", "_capacity", "_index", "_filled", "_total")

    def __init__(self, capacity: int = 120) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        self._buf = np.zeros(capacity, dtype=np.float64)
        self._index = 0
        self._filled = 0
        self._total = 0

    def add(self, value: float) -> None:
        self._buf[self._index] = value
        self._index = (self._index + 1) % self._capacity
        self._filled = min(self._filled + 1, self._capacity)
        self._total += 1

    @property
    def total(self) -> int:
        """Total samples ever added, not just those still in the window."""
        return self._total

    def values(self) -> np.ndarray:
        if self._filled < self._capacity:
            return self._buf[: self._filled]
        return self._buf

    def mean(self) -> float:
        v = self.values()
        return float(v.mean()) if v.size else 0.0

    def percentile(self, q: float) -> float:
        v = self.values()
        return float(np.percentile(v, q)) if v.size else 0.0

    def summary(self, name: str) -> Stat:
        v = self.values()
        if v.size == 0:
            return Stat(name, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return Stat(
            name=name,
            count=self._total,
            mean=float(v.mean()),
            median=float(np.percentile(v, 50)),
            p95=float(np.percentile(v, 95)),
            p99=float(np.percentile(v, 99)),
            maximum=float(v.max()),
        )


class Metrics:
    """Named rolling timers, thread-safe enough for a few producer threads.

    Usage::

        metrics = Metrics(window=120)
        with metrics.timer("face.detect"):
            boxes = detector(frame)
        metrics.mark_frame()
        print(metrics.fps)
    """

    def __init__(self, window: int = 120) -> None:
        self._window = window
        self._timers: OrderedDict[str, RollingWindow] = OrderedDict()
        self._lock = threading.Lock()
        self._frame_times = RollingWindow(window)
        self._last_frame_ts: float | None = None
        self._started = time.perf_counter()
        self._frames = 0

    # ------------------------------------------------------------------ #
    def _window_for(self, name: str) -> RollingWindow:
        w = self._timers.get(name)
        if w is None:
            with self._lock:
                w = self._timers.get(name)
                if w is None:
                    w = RollingWindow(self._window)
                    self._timers[name] = w
        return w

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        """Time a block and record the elapsed milliseconds under ``name``."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self._window_for(name).add((time.perf_counter() - start) * 1000.0)

    def record(self, name: str, milliseconds: float) -> None:
        """Record a pre-measured duration (for work timed elsewhere)."""
        self._window_for(name).add(milliseconds)

    def mark_frame(self) -> None:
        """Call once per delivered frame to drive the FPS estimate."""
        now = time.perf_counter()
        if self._last_frame_ts is not None:
            self._frame_times.add((now - self._last_frame_ts) * 1000.0)
        self._last_frame_ts = now
        self._frames += 1

    # ------------------------------------------------------------------ #
    @property
    def frames(self) -> int:
        return self._frames

    @property
    def fps(self) -> float:
        """Instantaneous FPS over the rolling window of inter-frame gaps."""
        mean_ms = self._frame_times.mean()
        return 1000.0 / mean_ms if mean_ms > 1e-9 else 0.0

    @property
    def fps_overall(self) -> float:
        """Average FPS since this ``Metrics`` object was created."""
        elapsed = time.perf_counter() - self._started
        return self._frames / elapsed if elapsed > 1e-9 else 0.0

    def frame_interval(self) -> Stat:
        return self._frame_times.summary("frame.interval")

    def get(self, name: str) -> Stat | None:
        w = self._timers.get(name)
        return w.summary(name) if w is not None else None

    def mean_ms(self, name: str) -> float:
        w = self._timers.get(name)
        return w.mean() if w is not None else 0.0

    def summaries(self) -> list[Stat]:
        with self._lock:
            names = list(self._timers)
        return [self._timers[n].summary(n) for n in names]

    def report(self, header: str = "") -> str:
        lines: list[str] = []
        if header:
            lines.append(header)
        lines.append(
            f"frames={self._frames}  fps_window={self.fps:.1f}  "
            f"fps_overall={self.fps_overall:.1f}"
        )
        if self._frame_times.total:
            lines.append("  " + self.frame_interval().format())
        for stat in self.summaries():
            lines.append("  " + stat.format())
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot, for writing benchmark results to disk."""
        return {
            "frames": self._frames,
            "fps_window": round(self.fps, 3),
            "fps_overall": round(self.fps_overall, 3),
            "stages": {
                s.name: {
                    "count": s.count,
                    "mean_ms": round(s.mean, 4),
                    "p50_ms": round(s.median, 4),
                    "p95_ms": round(s.p95, 4),
                    "p99_ms": round(s.p99, 4),
                    "max_ms": round(s.maximum, 4),
                }
                for s in [self.frame_interval(), *self.summaries()]
            },
        }
