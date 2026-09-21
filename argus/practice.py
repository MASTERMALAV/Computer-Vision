"""Fitts's law target practice: measure the pointer instead of guessing at it.

Every constant in the cursor was reasoned about; one of them,
``pixels_per_unit``, was never measured, and "does it feel right?" is not a
number anyone can act on. This runs the standard multidirectional tapping task
from ISO 9241-411 and reports **throughput in bits per second**, which is
comparable across input devices and against published figures.

The harness watches the *operating system cursor* and the real left button
rather than hooking into the gesture pipeline. That is deliberate: the same code
then measures a physical mouse, so the hand can be compared against the mouse on
the same task, on this machine, with this person - which is the only comparison
that means anything.

Throughput uses the Shannon formulation with **effective** width::

    We  = 4.133 * SD(endpoint deviation along the movement axis)
    IDe = log2(De / We + 1)
    TP  = IDe / MT

Effective width matters. Raw width measures the task that was set; effective
width measures the task that was actually performed, so someone who is sloppy
and fast is not rewarded over someone accurate and fast.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import ROOT, ArgusConfig
from .control.screens import get_cursor_position, get_virtual_desktop
from .logsetup import get_logger
from .ui.overlay import COLORS, draw_bar, draw_panel, draw_text

log = get_logger("practice")

VK_LBUTTON = 0x01

# Diameter of the ring, and target size, in pixels. Two conditions give two
# indices of difficulty, which is the minimum for a meaningful fit.
CONDITIONS: tuple[tuple[int, int], ...] = (
    (520, 90),   # easy: big targets, short distance
    (760, 48),   # hard: small targets, long distance
)
TARGETS_PER_RING = 9  # odd, so the zig-zag visits every target exactly once

# Published reference points, for context in the report.
REFERENCE_TP = {
    "physical mouse": 4.5,
    "touchpad": 2.9,
    "touchscreen": 4.0,
    "mid-air pointing (published)": 2.0,
}


@dataclass
class Trial:
    condition: int
    distance: float
    width: float
    start: tuple[float, float]
    target: tuple[float, float]
    click: tuple[float, float]
    movement_time: float
    hit: bool

    @property
    def deviation(self) -> float:
        """Signed error along the movement axis, in pixels.

        Only the along-axis component counts: Fitts's law is one-dimensional,
        and sideways error does not affect whether the target was acquired.
        """
        axis = np.array(self.target) - np.array(self.start)
        length = float(np.linalg.norm(axis))
        if length < 1e-6:
            return 0.0
        unit = axis / length
        offset = np.array(self.click) - np.array(self.target)
        return float(np.dot(offset, unit))


@dataclass
class Report:
    trials: list[Trial] = field(default_factory=list)

    def by_condition(self, index: int) -> list[Trial]:
        return [t for t in self.trials if t.condition == index]

    @staticmethod
    def throughput(trials: list[Trial]) -> dict:
        """Shannon throughput with effective width."""
        if len(trials) < 3:
            return {}
        deviations = np.array([t.deviation for t in trials], dtype=np.float64)
        distances = np.array([t.distance for t in trials], dtype=np.float64)
        times = np.array([t.movement_time for t in trials], dtype=np.float64)

        # 4.133 = 2 * 2.066, the constant that makes the effective width span
        # 96% of a normal distribution of endpoints.
        spread = float(deviations.std(ddof=1))
        effective_width = 4.133 * max(spread, 1e-3)
        effective_distance = float(distances.mean())
        ide = math.log2(effective_distance / effective_width + 1.0)
        mean_time = float(times.mean())
        return {
            "n": len(trials),
            "distance_px": round(effective_distance, 1),
            "nominal_width_px": round(float(trials[0].width), 1),
            "effective_width_px": round(effective_width, 1),
            "index_of_difficulty_bits": round(ide, 3),
            "movement_time_s": round(mean_time, 4),
            "throughput_bits_per_s": round(ide / mean_time, 3) if mean_time > 0 else 0.0,
            "error_rate_pct": round(
                100.0 * sum(1 for t in trials if not t.hit) / len(trials), 1
            ),
            "overshoot_pct": round(
                100.0 * sum(1 for t in trials if t.deviation > 0) / len(trials), 1
            ),
        }

    def overall(self) -> dict:
        per = [self.throughput(self.by_condition(i)) for i in range(len(CONDITIONS))]
        per = [p for p in per if p]
        if not per:
            return {}
        return {
            "conditions": per,
            "throughput_bits_per_s": round(
                float(np.mean([p["throughput_bits_per_s"] for p in per])), 3
            ),
            "error_rate_pct": round(float(np.mean([p["error_rate_pct"] for p in per])), 1),
            "overshoot_pct": round(float(np.mean([p["overshoot_pct"] for p in per])), 1),
            "movement_time_s": round(float(np.mean([p["movement_time_s"] for p in per])), 4),
        }


def target_order(count: int) -> list[int]:
    """The ISO zig-zag: each movement crosses the ring rather than hopping along it."""
    order, index = [], 0
    for _ in range(count):
        order.append(index)
        index = (index + count // 2) % count
    return order


def ring_positions(centre: tuple[int, int], diameter: int, count: int) -> list[tuple[float, float]]:
    radius = diameter / 2.0
    return [
        (
            centre[0] + radius * math.cos(2 * math.pi * i / count - math.pi / 2),
            centre[1] + radius * math.sin(2 * math.pi * i / count - math.pi / 2),
        )
        for i in range(count)
    ]


def _button_down() -> bool:
    import ctypes
    import sys

    if sys.platform != "win32":
        return False
    return bool(ctypes.windll.user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)


def advise(result: dict, cfg: ArgusConfig) -> list[str]:
    """Turn the measurements into a concrete suggestion about gain."""
    notes: list[str] = []
    if not result:
        return ["Not enough trials to say anything."]

    tp = result["throughput_bits_per_s"]
    errors = result["error_rate_pct"]
    overshoot = result["overshoot_pct"]
    gain = cfg.control.gain.pixels_per_unit

    if errors > 15:
        notes.append(
            f"Error rate {errors:.0f}% is high. Either the targets are moving away "
            "under you or clicking is disturbing the cursor."
        )
    if overshoot > 70:
        notes.append(
            f"{overshoot:.0f}% of clicks landed past the target - the pointer is "
            f"running ahead of you. Try control.gain.pixels_per_unit={int(gain * 0.8)}."
        )
    elif overshoot < 30:
        notes.append(
            f"Only {overshoot:.0f}% of clicks overshot, so you are consistently "
            f"stopping short and correcting. Try "
            f"control.gain.pixels_per_unit={int(gain * 1.2)}."
        )
    else:
        notes.append(
            f"Overshoot {overshoot:.0f}% is well balanced - the gain is about right."
        )

    if tp < 1.5:
        notes.append("Throughput is low; check the frame rate and the smoothing settings.")
    elif tp > 3.0:
        notes.append("Throughput is good for free-air pointing.")
    return notes


# --------------------------------------------------------------------------- #
def run_practice(
    cfg: ArgusConfig,
    rounds: int = 1,
    out: str | None = None,
) -> int:
    """Run the tapping task and report throughput."""
    import cv2

    desktop = get_virtual_desktop()
    monitor = desktop.primary
    width, height = monitor.width, monitor.height
    centre = (width // 2, height // 2)

    window = "ARGUS practice"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, width, height)
    cv2.moveWindow(window, monitor.left, monitor.top)

    report = Report()
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    sequence: list[tuple[int, int, tuple[float, float]]] = []
    for _ in range(max(1, rounds)):
        for condition, (diameter, size) in enumerate(CONDITIONS):
            points = ring_positions(centre, diameter, TARGETS_PER_RING)
            for i in target_order(TARGETS_PER_RING):
                sequence.append((condition, size, points[i]))

    print()
    print("  Fitts's law target practice")
    print(f"  {len(sequence)} targets over {len(CONDITIONS)} difficulty levels.")
    print("  Click each highlighted circle as quickly and accurately as you can.")
    print("  Works with the hand (arm it first with F9) or with a normal mouse -")
    print("  it measures the cursor, so the two are directly comparable.")
    print("  Press Esc to abort.")
    print()

    was_down = _button_down()
    index = 0
    previous_point: tuple[float, float] = centre
    started_at = time.perf_counter()
    aborted = False

    try:
        while index < len(sequence):
            condition, size, point = sequence[index]

            cx, cy = get_cursor_position()
            local = (float(cx - monitor.left), float(cy - monitor.top))

            canvas[:] = COLORS["bg"]
            # Every target in the ring, so the pattern is visible and the eye
            # can plan ahead - the standard task shows them all.
            for other in ring_positions(centre, CONDITIONS[condition][0], TARGETS_PER_RING):
                cv2.circle(canvas, (int(other[0]), int(other[1])), size // 2,
                           (60, 55, 50), 1, cv2.LINE_AA)
            cv2.circle(canvas, (int(point[0]), int(point[1])), size // 2,
                       COLORS["accent"], -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(local[0]), int(local[1])), 7, COLORS["ok"], 2, cv2.LINE_AA)

            done = report.trials
            draw_panel(
                canvas,
                [
                    f"target     {index + 1} / {len(sequence)}",
                    f"condition  {condition + 1}: {size}px targets, "
                    f"{CONDITIONS[condition][0]}px apart",
                    f"hits       {sum(1 for t in done if t.hit)} / {len(done)}",
                    "Esc to stop",
                ],
                origin=(24, 24),
                title="ARGUS  -  practice",
                min_width=420,
            )
            draw_bar(canvas, (34, 132), 380, index / max(len(sequence), 1), COLORS["ok"])

            cv2.imshow(window, canvas)
            if (cv2.waitKey(1) & 0xFF) == 27:
                aborted = True
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                aborted = True
                break

            down = _button_down()
            if down and not was_down:
                now = time.perf_counter()
                distance = float(np.linalg.norm(np.array(point) - np.array(previous_point)))
                hit = float(np.linalg.norm(np.array(local) - np.array(point))) <= size / 2.0
                report.trials.append(
                    Trial(
                        condition=condition,
                        distance=distance,
                        width=float(size),
                        start=previous_point,
                        target=point,
                        click=local,
                        movement_time=now - started_at,
                        hit=hit,
                    )
                )
                previous_point = point
                started_at = now
                index += 1
            was_down = down
    finally:
        cv2.destroyWindow(window)
        for _ in range(4):
            cv2.waitKey(1)

    if len(report.trials) < 6:
        print("\n  Too few trials to measure anything. Nothing recorded.\n")
        return 1

    result = report.overall()
    print()
    print("  Results")
    for i, per in enumerate(result["conditions"]):
        print(
            f"    condition {i + 1}: ID {per['index_of_difficulty_bits']:.2f} bits   "
            f"MT {per['movement_time_s'] * 1000:6.0f} ms   "
            f"TP {per['throughput_bits_per_s']:.2f} bits/s   "
            f"errors {per['error_rate_pct']:.0f}%"
        )
    print()
    print(f"    throughput   {result['throughput_bits_per_s']:.2f} bits/s")
    print(f"    mean time    {result['movement_time_s'] * 1000:.0f} ms per target")
    print(f"    error rate   {result['error_rate_pct']:.0f}%")
    print(f"    overshoot    {result['overshoot_pct']:.0f}% of clicks landed past the target")
    print()
    print("  For comparison")
    for name, value in REFERENCE_TP.items():
        print(f"    {name:<30} {value:.1f} bits/s")
    print()
    print("  What this suggests")
    for note in advise(result, cfg):
        print(f"    - {note}")
    print()

    if out:
        path = Path(out)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "aborted": aborted,
            "gain_pixels_per_unit": cfg.control.gain.pixels_per_unit,
            "result": result,
            "trials": [
                {
                    "condition": t.condition,
                    "distance_px": round(t.distance, 1),
                    "width_px": t.width,
                    "movement_time_s": round(t.movement_time, 4),
                    "deviation_px": round(t.deviation, 2),
                    "hit": t.hit,
                }
                for t in report.trials
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  Written to {path}")
        print()
    return 0
