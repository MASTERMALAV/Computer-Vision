"""Fit gesture thresholds to the operator's own hand.

Default thresholds are a compromise. Thumb length relative to palm span varies
enough between people that a pinch which reads as 0.28 on one hand reads as 0.40
on another - and a threshold set between those either fires constantly or never
fires at all.

This walks through the three poses that define the thresholds, measures each,
and derives values with proper separation between them. It refuses to emit
thresholds when the measured poses overlap, because thresholds fitted to
overlapping distributions cannot work no matter where they are placed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..capture.camera import CameraStream
from ..capture.devices import resolve_device
from ..config import ROOT, ArgusConfig
from ..hands.engine import HandEngine
from ..logsetup import get_logger
from ..ui.hands import draw_hand
from ..ui.overlay import COLORS, draw_bar, draw_panel, draw_text

log = get_logger("gestures.calibrate")

SAMPLES_PER_POSE = 45  # ~1.5 s at 30 fps
COUNTDOWN_S = 3.0


@dataclass
class Pose:
    key: str
    title: str
    instruction: str
    measure: str  # which quantity this pose defines
    samples: list[float] = field(default_factory=list)


POSES = [
    Pose(
        "open",
        "OPEN HAND",
        "Index finger extended, thumb comfortably apart. Hold still.",
        "pinch_index",
    ),
    Pose(
        "pinch",
        "PINCH",
        "Touch thumb and index fingertips together. Hold the pinch.",
        "pinch_index",
    ),
    Pose(
        "pinch_middle",
        "MIDDLE PINCH",
        "Touch thumb and MIDDLE fingertips together. Hold.",
        "pinch_middle",
    ),
    Pose(
        "extended",
        "POINTING",
        "Index finger fully extended, other fingers relaxed.",
        "index_extension",
    ),
    Pose(
        "curled",
        "RELAXED",
        "Curl your fingers into a loose fist - the cursor-off pose.",
        "index_extension",
    ),
]


def _measure(hand, quantity: str) -> float:
    if quantity == "pinch_index":
        return hand.pinch("index")
    if quantity == "pinch_middle":
        return hand.pinch("middle")
    return hand.finger_extension("index")


def _stats(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "p05": float(np.percentile(a, 5)),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
    }


def derive_thresholds(results: dict[str, dict]) -> tuple[dict, list[str]]:
    """Turn measured pose distributions into thresholds, with warnings."""
    warnings: list[str] = []
    out: dict[str, float] = {}

    open_s, pinch_s = results.get("open"), results.get("pinch")
    if open_s and pinch_s:
        # Separation check: the widest pinch must sit clearly below the
        # narrowest open hand, or no threshold can separate them.
        if pinch_s["p95"] >= open_s["p05"]:
            warnings.append(
                f"pinch (p95 {pinch_s['p95']:.2f}) overlaps open hand "
                f"(p05 {open_s['p05']:.2f}). Move your hand closer to the camera, "
                "or make the pinch and the open pose more distinct, and retry."
            )
        # Close just above the pinch spread; open just below the open-hand
        # spread. The gap between them is the hysteresis band.
        close = pinch_s["p95"] + 0.04
        open_at = max(close + 0.10, min(open_s["p05"] - 0.05, close + 0.30))
        out["pinch_close"] = round(float(close), 3)
        out["pinch_open"] = round(float(open_at), 3)
        # Approach fires early enough to freeze the cursor before the click.
        out["pinch_approach"] = round(float(open_at + 0.10), 3)

    mid = results.get("pinch_middle")
    if mid and open_s and mid["p95"] >= open_s["p05"]:
        warnings.append(
            "middle pinch overlaps the open hand; right-click may be unreliable"
        )

    ext, curl = results.get("extended"), results.get("curled")
    if ext and curl:
        if curl["p95"] >= ext["p05"]:
            warnings.append(
                f"pointing (p05 {ext['p05']:.2f}) and relaxed (p95 {curl['p95']:.2f}) "
                "overlap; the clutch will be unreliable. Curl your fingers further."
            )
        extended = max(curl["p95"] + 0.10, ext["p05"] - 0.05)
        curled = min(extended - 0.15, curl["p95"] + 0.03)
        out["finger_extended"] = round(float(extended), 3)
        out["finger_curled"] = round(float(curled), 3)

    return out, warnings


def run_calibration(cfg: ArgusConfig, write: str | None = None) -> int:
    """Interactive calibration with a live preview."""
    import cv2

    device = resolve_device(
        cfg.capture.camera.device,
        backend=cfg.capture.camera.backend,
        exclude_ir=cfg.capture.camera.exclude_ir,
    )
    cam = CameraStream(cfg.capture.camera, device=device).start()
    engine = HandEngine(cfg.hands, mirrored_input=cfg.capture.camera.mirror).start()

    window = cfg.ui.window_name + " - calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, cfg.ui.preview_width, int(cfg.ui.preview_width * 9 / 16))

    scale_factor = (
        cfg.capture.detect_width / float(cam.actual_width)
        if 0 < cfg.capture.detect_width < cam.actual_width
        else 1.0
    )

    print()
    print("  Gesture calibration")
    print("  Five short poses. Hold each one still while the bar fills.")
    print("  Press q at any time to abort.")
    print()

    aborted = False
    try:
        for pose in POSES:
            print(f"  -> {pose.title}: {pose.instruction}")
            phase_start = time.perf_counter()
            counting_down = True

            while len(pose.samples) < SAMPLES_PER_POSE:
                frame = cam.read(timeout=2.0)
                if frame is None:
                    if not cam.is_running:
                        aborted = True
                        break
                    continue

                small = (
                    cv2.resize(frame.image, None, fx=scale_factor, fy=scale_factor,
                               interpolation=cv2.INTER_AREA)
                    if scale_factor < 1.0
                    else frame.image
                )
                result = engine.process(small, frame.monotonic_ms, frame.index)
                if scale_factor < 1.0:
                    for h in result.hands:
                        h.pixels /= scale_factor
                        h.frame_size = (frame.width, frame.height)
                hand = result.primary

                elapsed = time.perf_counter() - phase_start
                if counting_down and elapsed >= COUNTDOWN_S:
                    counting_down = False
                if not counting_down and hand is not None and hand.is_reliable:
                    pose.samples.append(_measure(hand, pose.measure))

                canvas = frame.image.copy()
                if hand is not None:
                    draw_hand(canvas, hand, show_pinch=True)

                if counting_down:
                    remaining = COUNTDOWN_S - elapsed
                    status = f"get ready... {remaining:0.1f}s"
                    progress = 1.0 - remaining / COUNTDOWN_S
                    colour = COLORS["warn"]
                else:
                    status = f"hold still  {len(pose.samples)}/{SAMPLES_PER_POSE}"
                    progress = len(pose.samples) / SAMPLES_PER_POSE
                    colour = COLORS["ok"]

                draw_panel(
                    canvas,
                    [
                        pose.instruction,
                        status,
                        "" if hand is not None else "NO HAND DETECTED",
                    ],
                    origin=(12, 12),
                    title=f"{pose.title}   ({POSES.index(pose) + 1}/{len(POSES)})",
                    min_width=430,
                )
                draw_bar(canvas, (22, 118), 400, progress, colour, height=10)
                if hand is not None:
                    draw_text(
                        canvas,
                        f"{pose.measure} = {_measure(hand, pose.measure):.3f}",
                        (22, 152), 0.6, COLORS["accent"], 2,
                    )

                cv2.imshow(window, canvas)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    aborted = True
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    aborted = True
                    break
            if aborted:
                break
            print(f"     captured {len(pose.samples)} samples")
    finally:
        engine.close()
        cam.stop()
        cv2.destroyAllWindows()
        for _ in range(4):
            cv2.waitKey(1)

    measured = {p.key: _stats(p.samples) for p in POSES if len(p.samples) >= 10}
    if aborted and len(measured) < len(POSES):
        print("\n  Calibration aborted - nothing was changed.\n")
        return 1

    print()
    print("  Measured (in hand-scale units)")
    for key, s in measured.items():
        print(
            f"    {key:<14} p05 {s['p05']:.3f}   median {s['p50']:.3f}   "
            f"p95 {s['p95']:.3f}   (n={s['n']})"
        )

    thresholds, warnings = derive_thresholds(measured)
    print()
    if warnings:
        print("  Warnings")
        for w in warnings:
            print(f"    ! {w}")
        print()
    if not thresholds:
        print("  Not enough clean data to derive thresholds.\n")
        return 1

    print("  Derived thresholds")
    for key, value in thresholds.items():
        current = getattr(cfg.gestures, key, None)
        arrow = f"  (was {current})" if current is not None else ""
        print(f"    {key:<18} {value}{arrow}")

    lines = ["# Gesture thresholds fitted to this operator by `argus calibrate`.",
             "# Load with:  python -m argus mouse -c configs/calibrated.yaml",
             "gestures:"]
    lines += [f"  {k}: {v}" for k, v in thresholds.items()]
    text = "\n".join(lines) + "\n"

    if write:
        path = Path(write)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"\n  Written to {path}")
        print(f"  Use it with:  python -m argus mouse -c {path.name}")
    else:
        print("\n  Add to your config (or re-run with --write configs/calibrated.yaml):\n")
        print("    " + text.replace("\n", "\n    "))
    print()
    return 0
