"""Fit gesture thresholds to the operator's own hand.

Default thresholds are a compromise. Thumb length relative to palm span varies
enough between people that a pinch which reads as 0.28 on one hand reads as 0.40
on another - and a threshold set between those either fires constantly or never
fires at all.

This walks through five poses, measures each, and derives values with proper
separation between them. Two independent checks guard the result:

* **Separation** - the poses must be distinguishable from each other, or no
  threshold placed between them can work.
* **Plausibility** - each pose must land in the range a human hand can actually
  produce. This matters because a run where the operator never quite performed
  a pose can still separate cleanly while being nonsense: a "pinch" measuring
  1.29 hand-widths separates perfectly from an open hand, and would make the
  system click continuously.

The live view shows the measurement against its expected range *before* anything
is recorded, so a pose that is not registering can be corrected rather than
discovered afterwards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..capture.camera import CameraStream
from ..capture.picker import select_camera
from ..config import ROOT, ArgusConfig
from ..logsetup import get_logger
from ..ui.hands import draw_hand
from ..ui.overlay import COLORS, draw_bar, draw_panel, draw_text

log = get_logger("gestures.calibrate")

SAMPLES_PER_POSE = 45  # ~1.5 s of holding still at 30 fps

# Time to read the instruction and settle into the pose before anything is
# recorded. Deliberately unhurried - the procedure runs once, and a rushed pose
# produces thresholds that are wrong in ways which are hard to diagnose later.
# SPACE skips the wait once the pose already reads READY.
COUNTDOWN_S = 8.0


@dataclass
class Pose:
    key: str
    title: str
    instruction: str
    measure: str  # which quantity this pose defines
    detail: str = ""  # the specific thing people get wrong
    samples: list[float] = field(default_factory=list)


POSES: list[Pose] = [
    Pose(
        "open", "1 / 5   OPEN HAND",
        "Index finger straight, thumb held well away from it.",
        "pinch_index",
        "Like holding an invisible cup. The thumb must be clearly apart.",
    ),
    Pose(
        "pinch", "2 / 5   PINCH",
        "Thumb and INDEX fingertips touching.",
        "pinch_index",
        "They must actually touch, pad to pad - not hover near each other.",
    ),
    Pose(
        "pinch_middle", "3 / 5   MIDDLE PINCH",
        "Thumb and MIDDLE fingertips touching.",
        "pinch_middle",
        "Same as before, but the middle finger. The index may stay out.",
    ),
    Pose(
        "extended", "4 / 5   POINTING",
        "Index finger fully straight, as if pointing at the screen.",
        "index_extension",
        "Straighten it completely. A half-bent finger fails this check.",
    ),
    Pose(
        "curled", "5 / 5   RELAXED",
        "Curl your fingers into a loose fist.",
        "index_extension",
        "This is the cursor-off pose. Let the hand go slack.",
    ),
]


# Physically plausible ranges for each pose, in hand-scale units. These are
# properties of human hands, not tuning knobs: the thumb tip cannot sit 1.3
# wrist-widths from the index tip while the two are touching.
PLAUSIBLE: dict[str, tuple[float, float, str, str]] = {
    "pinch": (0.05, 0.60, "PINCH", "your thumb and index fingertips must actually touch"),
    "pinch_middle": (0.05, 0.60, "MIDDLE PINCH", "your thumb and middle fingertips must touch"),
    "open": (0.55, 2.20, "OPEN HAND", "hold your thumb clearly away from your index finger"),
    "extended": (1.00, 1.90, "POINTING", "straighten your index finger fully"),
    "curled": (0.10, 0.95, "RELAXED", "curl your fingers into a loose fist"),
}


def pose_verdict(pose_key: str, value: float | None) -> tuple[bool, str]:
    """Whether a live measurement is inside the range this pose must land in."""
    if value is None:
        return False, "no hand detected"
    limits = PLAUSIBLE.get(pose_key)
    if limits is None:
        return True, f"{value:.2f}"
    low, high, _title, advice = limits
    if value < low:
        return False, f"{value:.2f} - too small, need {low:.2f}-{high:.2f}"
    if value > high:
        return False, f"{value:.2f} - too large, need {low:.2f}-{high:.2f}"
    return True, f"{value:.2f}, inside {low:.2f}-{high:.2f}"


def implausible(results: dict[str, dict]) -> list[str]:
    """Poses whose measurements are outside what a hand can physically do."""
    problems: list[str] = []
    for key, (low, high, title, advice) in PLAUSIBLE.items():
        stats = results.get(key)
        if not stats:
            continue
        median = stats["p50"]
        if not (low <= median <= high):
            problems.append(
                f"the {title} pose measured {median:.2f}, outside the plausible "
                f"range {low:.2f}-{high:.2f}. That pose was probably not performed "
                f"as intended - {advice}."
            )
    return problems


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
    warnings: list[str] = implausible(results)
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


# --------------------------------------------------------------------------- #
def run_calibration(
    cfg: ArgusConfig,
    write: str | None = None,
    countdown: float = COUNTDOWN_S,
) -> int:
    """Interactive calibration with a live preview and per-pose feedback."""
    import cv2

    from ..hands.engine import HandEngine

    device = select_camera(cfg)
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
    print(f"  Five poses. You get {countdown:.0f} seconds to read each one and get your")
    print("  hand into position, then about 1.5 seconds of holding still.")
    print()
    print("  The panel shows READY or ADJUST live, so you can fix your hand")
    print("  before anything is recorded.")
    print()
    print("    SPACE   start recording now (once it says READY)")
    print("    R       redo the current pose")
    print("    Q       quit without saving")
    print()

    aborted = False
    index = 0
    try:
        while index < len(POSES):
            pose = POSES[index]
            pose.samples.clear()
            print(f"  -> {pose.title.strip()}: {pose.instruction}")
            phase_start = time.perf_counter()
            counting_down = True
            redo = False

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

                value = _measure(hand, pose.measure) if hand is not None else None
                in_range, verdict = pose_verdict(pose.key, value)

                elapsed = time.perf_counter() - phase_start
                if counting_down and elapsed >= countdown:
                    counting_down = False
                if not counting_down and hand is not None and hand.is_reliable:
                    pose.samples.append(value)

                canvas = frame.image.copy()
                if hand is not None:
                    draw_hand(canvas, hand, show_pinch=True)

                if counting_down:
                    remaining = countdown - elapsed
                    status = f"get into position...  {remaining:0.1f}s"
                    progress = 1.0 - remaining / max(countdown, 1e-6)
                    colour = COLORS["ok"] if in_range else COLORS["warn"]
                else:
                    status = f"HOLD STILL   {len(pose.samples)} / {SAMPLES_PER_POSE}"
                    progress = len(pose.samples) / SAMPLES_PER_POSE
                    colour = COLORS["ok"] if in_range else COLORS["error"]

                draw_panel(
                    canvas,
                    [
                        pose.instruction,
                        pose.detail,
                        "",
                        status,
                        ("READY    " if in_range else "ADJUST   ") + verdict,
                    ],
                    origin=(12, 12),
                    title=pose.title,
                    min_width=600,
                )
                draw_bar(canvas, (22, 168), 560, progress, colour, height=12)
                draw_text(canvas, "SPACE start now     R redo     Q quit",
                          (22, 206), 0.5, COLORS["muted"])

                cv2.imshow(window, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    aborted = True
                    break
                if key == ord(" ") and counting_down:
                    counting_down = False
                elif key == ord("r"):
                    redo = True
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    aborted = True
                    break

            if aborted:
                break
            if redo:
                print("     redoing this pose")
                continue

            median = float(np.median(pose.samples)) if pose.samples else 0.0
            ok, _verdict = pose_verdict(pose.key, median)
            print(f"     captured {len(pose.samples)}   median {median:.2f}   "
                  f"{'OK' if ok else 'OUT OF RANGE'}")
            index += 1
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
    for key, st in measured.items():
        ok, _ = pose_verdict(key, st["p50"])
        print(
            f"    {key:<14} p05 {st['p05']:.3f}   median {st['p50']:.3f}   "
            f"p95 {st['p95']:.3f}   {'OK' if ok else '<- OUT OF RANGE'}"
        )

    thresholds, warnings = derive_thresholds(measured)
    print()
    if warnings:
        print("  Problems")
        for w in warnings:
            print(f"    ! {w}")
        print()
    if not thresholds:
        print("  Not enough clean data to derive thresholds.\n")
        return 1

    if implausible(measured):
        # Writing these would be worse than writing nothing: the defaults at
        # least work, whereas thresholds fitted to a pose that never happened
        # make the system fire continuously or never at all.
        print("  Refusing to save thresholds derived from implausible measurements.")
        print("  Re-run and hold each pose exactly as described:")
        print("    - PINCH means the thumb and index fingertips touching, pad to pad")
        print("    - POINTING means the index finger completely straight")
        print("  Keep your whole hand in frame, and reasonably close to the camera.")
        print()
        return 1

    print("  Derived thresholds")
    for key, value in thresholds.items():
        current = getattr(cfg.gestures, key, None)
        arrow = f"   (was {current})" if current is not None else ""
        print(f"    {key:<24} {value}{arrow}")

    lines = [
        "# Gesture thresholds fitted to this operator by `argus calibrate`.",
        "# Load with:  argus mouse -c configs/calibrated.yaml",
        "gestures:",
    ]
    lines += [f"  {k}: {v}" for k, v in thresholds.items()]
    text = "\n".join(lines) + "\n"

    if write:
        path = Path(write)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"\n  Written to {path}")
        print(f"  Use it with:  argus mouse -c configs/{path.name}")
    else:
        print("\n  Add to your config (or re-run with --write):\n")
        print("    " + text.replace("\n", "\n    "))
    print()
    return 0
