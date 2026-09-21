"""Phase 2 verification: live hand preview and a headless landmark benchmark.

The benchmark answers the question the whole design depends on: can this CPU
landmark a hand fast enough that the cursor feels attached to it? Everything
downstream - filtering, gain, clicking - is built on the assumption that a fresh
hand position arrives roughly every 33 ms. If that is false, the design has to
change rather than the constants.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..capture.camera import CameraStream
from ..capture.picker import select_camera
from ..config import ROOT, ArgusConfig
from ..logsetup import get_logger
from ..metrics import Metrics
from ..ui.hands import draw_hands
from ..ui.overlay import COLORS, draw_panel, draw_text, fps_color
from .engine import HandEngine

log = get_logger("hands.preview")

WARMUP_FRAMES = 20

HELP_LINES = [
    "q / ESC  quit",
    "p        pinch distances on/off",
    "h        hide this panel",
]


def _detect_size(cfg: ArgusConfig, width: int) -> float:
    """Scale factor applied before inference, from capture.detect_width."""
    target = cfg.capture.detect_width
    if target <= 0 or target >= width:
        return 1.0
    return target / float(width)


def run_hands_preview(cfg: ArgusConfig, seconds: float = 0.0, renegotiate: bool = False) -> int:
    """Live hand landmarks with per-stage timing on screen."""
    import cv2

    device = select_camera(cfg)
    metrics = Metrics(window=cfg.runtime.metrics_window)
    show_help = True
    show_pinch = True
    window = cfg.ui.window_name + " - hands"
    deadline = time.perf_counter() + seconds if seconds > 0 else None

    cam = CameraStream(cfg.capture.camera, device=device, renegotiate=renegotiate).start()
    engine = HandEngine(cfg.hands, mirrored_input=cfg.capture.camera.mirror).start()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, cfg.ui.preview_width, int(cfg.ui.preview_width * 9 / 16))

    try:
        while True:
            frame = cam.read(timeout=2.0)
            if frame is None:
                if not cam.is_running:
                    break
                continue
            metrics.mark_frame()

            scale = _detect_size(cfg, frame.width)
            with metrics.timer("hands.preprocess"):
                if scale < 1.0:
                    small = cv2.resize(
                        frame.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                    )
                else:
                    small = frame.image

            result = engine.process(small, frame.monotonic_ms, frame.index)
            metrics.record("hands.inference", result.inference_ms)

            with metrics.timer("hands.render"):
                canvas = frame.image.copy()
                # Landmarks were computed on the downscaled image; scale the
                # pixel coordinates back up so they land on the full-res frame.
                if scale < 1.0:
                    for hand in result.hands:
                        hand.pixels /= scale
                        hand.frame_size = (frame.width, frame.height)
                draw_hands(canvas, result.hands, show_pinch=show_pinch)

                fps = metrics.fps
                lines = [
                    f"hands     {result.count}",
                    f"inference {metrics.mean_ms('hands.inference'):5.1f} ms"
                    f"  (p95 {metrics.get('hands.inference').p95:5.1f})",
                    f"pipeline  {fps:5.1f} fps   capture {cam.stats['capture_fps']} fps",
                    f"latency   {frame.age_ms:5.1f} ms at draw",
                    f"detect at {small.shape[1]}x{small.shape[0]}",
                ]
                for hand in result.hands:
                    lines.append(
                        f"{hand.handedness[:1]}: pinch {hand.pinch('index'):.2f}"
                        f" / mid {hand.pinch('middle'):.2f}"
                        f"  ext {''.join('1' if hand.finger_extension(f) > 1.05 else '0' for f in ('index','middle','ring','pinky'))}"
                    )
                draw_panel(canvas, lines, origin=(12, 12), title="ARGUS  -  hands", min_width=350)
                draw_text(canvas, f"{fps:4.1f} FPS", (canvas.shape[1] - 150, 40),
                          0.9, fps_color(fps), 2)
                if show_help:
                    draw_panel(canvas, HELP_LINES,
                               origin=(12, canvas.shape[0] - 24 - 22 * len(HELP_LINES)),
                               scale=0.45, alpha=0.5, color=COLORS["muted"])

            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("h"):
                show_help = not show_help
            elif key == ord("p"):
                show_pinch = not show_pinch

            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if deadline is not None and time.perf_counter() > deadline:
                break
    finally:
        engine.close()
        cam.stop()
        cv2.destroyAllWindows()
        for _ in range(4):
            cv2.waitKey(1)

    print()
    print(metrics.report(header="Hand landmarks"))
    return 0


def bench_hands(
    cfg: ArgusConfig,
    seconds: float = 20.0,
    out: str | None = None,
    renegotiate: bool = False,
) -> int:
    """Headless landmark benchmark - the Phase 2 acceptance gate."""
    import cv2

    device = select_camera(cfg)
    cam = CameraStream(cfg.capture.camera, device=device, renegotiate=renegotiate).start()
    engine = HandEngine(cfg.hands, mirrored_input=cfg.capture.camera.mirror).start()

    scale = _detect_size(cfg, cam.actual_width)
    print(
        f"Benchmarking hand landmarks for {seconds:.0f} s\n"
        f"  camera   {device.name} via {cam.backend} "
        f"{cam.actual_width}x{cam.actual_height}\n"
        f"  inference at {int(cam.actual_width * scale)}x{int(cam.actual_height * scale)}"
        f"  (detect_width={cfg.capture.detect_width}, max_hands={cfg.hands.max_hands},"
        f" complexity={cfg.hands.model_complexity})\n"
        "  Hold a hand in view for a representative measurement."
    )

    for _ in range(WARMUP_FRAMES):
        frame = cam.read(timeout=2.0)
        if frame is not None:
            small = (
                cv2.resize(frame.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                if scale < 1.0
                else frame.image
            )
            engine.process(small, frame.monotonic_ms, frame.index)

    metrics = Metrics(window=int(max(120, seconds * 60)))
    frames_with_hand = 0
    total = 0
    end = time.perf_counter() + seconds
    try:
        while time.perf_counter() < end:
            frame = cam.read(timeout=2.0)
            if frame is None:
                continue
            with metrics.timer("pipeline.total"):
                with metrics.timer("hands.preprocess"):
                    small = (
                        cv2.resize(
                            frame.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                        )
                        if scale < 1.0
                        else frame.image
                    )
                result = engine.process(small, frame.monotonic_ms, frame.index)
            metrics.record("hands.inference", result.inference_ms)
            metrics.record("capture.frame_age", frame.age_ms)
            metrics.mark_frame()
            total += 1
            if result.count:
                frames_with_hand += 1
    finally:
        engine.close()
        cam.stop()

    detection_rate = 100.0 * frames_with_hand / total if total else 0.0
    inference = metrics.get("hands.inference")
    pipeline = metrics.get("pipeline.total")

    print()
    print(metrics.report(header="Hand landmark benchmark"))
    print(f"  frames with a hand detected: {frames_with_hand}/{total} ({detection_rate:.0f}%)")

    # The budget that matters is end-to-end: a fresh hand position must be
    # available within one camera frame period, or the cursor falls behind the
    # hand no matter how good the filtering is.
    frame_budget_ms = 1000.0 / max(cfg.capture.camera.fps, 1)
    checks = [
        ("pipeline fps", metrics.fps_overall >= 25.0,
         f"{metrics.fps_overall:5.1f} fps  (need >= 25.0)"),
        ("inference mean", inference.mean <= frame_budget_ms,
         f"{inference.mean:5.1f} ms   (need <= {frame_budget_ms:.1f})"),
        ("inference p95", inference.p95 <= frame_budget_ms * 1.5,
         f"{inference.p95:5.1f} ms   (need <= {frame_budget_ms * 1.5:.1f})"),
        ("total per frame", pipeline.p95 <= frame_budget_ms * 1.5,
         f"{pipeline.p95:5.1f} ms   (need <= {frame_budget_ms * 1.5:.1f})"),
    ]
    print()
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}]  {name:<16} {detail}")
    ok = all(p for _, p, _ in checks)
    print(f"\n  Phase 2 landmark gate: {'PASS' if ok else 'FAIL'}\n")

    if out:
        path = Path(out)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "device": device.name,
                    "backend": cam.backend,
                    "capture": f"{cam.actual_width}x{cam.actual_height}",
                    "inference_size": [
                        int(cam.actual_width * scale),
                        int(cam.actual_height * scale),
                    ],
                    "max_hands": cfg.hands.max_hands,
                    "detection_rate_pct": round(detection_rate, 1),
                    "frames": total,
                    "passed": ok,
                    "metrics": metrics.to_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  results written to {path}")

    return 0 if ok else 1
