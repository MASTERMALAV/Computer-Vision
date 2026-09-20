"""Phase 1 verification tools: live preview and a headless capture benchmark.

The preview is not a toy. It is the instrument used to confirm that the capture
layer really delivers what it promises: the negotiated resolution and codec, the
true frame rate (not the one the driver claims), the drop rate, and the age of
the frame at the moment it is drawn - which is the latency the user actually
feels when waving a hand at the camera.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import ROOT, ArgusConfig
from ..logsetup import get_logger
from ..metrics import Metrics
from ..ui.overlay import COLORS, draw_panel, draw_text, fps_color
from .camera import CameraStream
from .devices import enumerate_devices, resolve_device

log = get_logger("capture.preview")

# Frames discarded before timing starts, so that exposure/gain settling is not
# charged against the steady-state frame rate.
WARMUP_FRAMES = 15

HELP_LINES = [
    "q / ESC  quit",
    "n        next camera",
    "s        save snapshot",
    "m        mirror on/off",
    "h        hide this panel",
]


def _snapshot_path() -> Path:
    path = ROOT / "captures"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"


def run_preview(
    cfg: ArgusConfig,
    seconds: float = 0.0,
    save_to: str | None = None,
    renegotiate: bool = False,
) -> int:
    """Open the configured camera and show it, with live capture statistics."""
    import cv2

    devices = enumerate_devices(backend=cfg.capture.camera.backend)
    device = resolve_device(
        cfg.capture.camera.device,
        devices,
        exclude_ir=cfg.capture.camera.exclude_ir,
    )
    # Rotation order for the 'n' key: skip IR cameras, they are unusable here.
    cycle = [d for d in devices if not (cfg.capture.camera.exclude_ir and d.is_ir)] or devices
    cycle_pos = next((i for i, d in enumerate(cycle) if d.index == device.index), 0)

    metrics = Metrics(window=cfg.runtime.metrics_window)
    show_help = True
    mirror = cfg.capture.camera.mirror
    window = cfg.ui.window_name
    deadline = time.perf_counter() + seconds if seconds > 0 else None

    cam = CameraStream(cfg.capture.camera, device=device, renegotiate=renegotiate).start()

    # Headless one-shot capture, for scripted verification.
    if save_to:
        frame = cam.read(timeout=5.0)
        cam.stop()
        if frame is None:
            log.error("no frame received within 5 s")
            return 1
        out = Path(save_to)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), frame.image)
        print(f"Saved {frame.width}x{frame.height} frame from '{device.name}' to {out}")
        return 0

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, cfg.ui.preview_width, int(cfg.ui.preview_width * 9 / 16))

    try:
        while True:
            frame = cam.read(timeout=2.0)
            if frame is None:
                log.warning("no frame for 2 s; is the camera still connected?")
                if not cam.is_running:
                    break
                continue

            metrics.mark_frame()
            with metrics.timer("preview.render"):
                canvas = frame.image.copy()
                if mirror != cfg.capture.camera.mirror:
                    # Live toggle without reopening the device.
                    canvas = cv2.flip(canvas, 1)

                stats = cam.stats
                grabbed = int(stats["grabbed"])
                dropped = int(stats["dropped"])
                drop_pct = 100.0 * dropped / grabbed if grabbed else 0.0

                fps = metrics.fps
                draw_panel(
                    canvas,
                    [
                        f"device    {device.name[:34]}",
                        f"index     {device.index}   backend {cam.backend}",
                        f"stream    {cam.actual_width}x{cam.actual_height}"
                        f"  {cam.actual_fourcc or '?'}",
                        f"capture   {stats['capture_fps']} fps",
                        f"delivered {fps:5.1f} fps  (p95 gap "
                        f"{metrics.frame_interval().p95:.1f} ms)",
                        f"latency   {frame.age_ms:5.1f} ms at draw time",
                        f"dropped   {dropped} of {grabbed}  ({drop_pct:.1f}%)",
                    ],
                    origin=(12, 12),
                    title="ARGUS  -  capture",
                    min_width=330,
                )

                # Big FPS readout, colour-coded against a 25 fps interactive target.
                label = f"{fps:4.1f} FPS"
                draw_text(
                    canvas,
                    label,
                    (canvas.shape[1] - 150, 40),
                    scale=0.9,
                    color=fps_color(fps),
                    thickness=2,
                )

                if show_help:
                    draw_panel(
                        canvas,
                        HELP_LINES,
                        origin=(12, canvas.shape[0] - 24 - 22 * len(HELP_LINES)),
                        scale=0.45,
                        alpha=0.5,
                        color=COLORS["muted"],
                    )

            cv2.imshow(window, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("h"):
                show_help = not show_help
            elif key == ord("m"):
                mirror = not mirror
            elif key == ord("s"):
                path = _snapshot_path()
                cv2.imwrite(str(path), canvas)
                log.info("snapshot saved to %s", path)
            elif key == ord("n") and len(cycle) > 1:
                cycle_pos = (cycle_pos + 1) % len(cycle)
                device = cycle[cycle_pos]
                log.info("switching to camera %s", device.label)
                cam.stop()
                metrics = Metrics(window=cfg.runtime.metrics_window)
                try:
                    cam = CameraStream(cfg.capture.camera, device=device, renegotiate=renegotiate).start()
                except Exception as exc:
                    log.error("could not switch to %s: %s", device.label, exc)
                    cycle_pos = (cycle_pos - 1) % len(cycle)
                    device = cycle[cycle_pos]
                    cam = CameraStream(cfg.capture.camera, device=device, renegotiate=renegotiate).start()

            # cv2 windows report 0 visibility once the user clicks the X button.
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if deadline is not None and time.perf_counter() > deadline:
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        # Windows needs a few event-loop turns to actually tear the window down.
        for _ in range(4):
            cv2.waitKey(1)

    print()
    print(metrics.report(header=f"Capture preview - {device.name}"))
    print(f"  camera: {cam.stats}")
    return 0


def bench_capture(
    cfg: ArgusConfig,
    seconds: float = 15.0,
    out: str | None = None,
    renegotiate: bool = False,
) -> int:
    """Headless capture benchmark - the Phase 1 acceptance gate."""
    device = resolve_device(
        cfg.capture.camera.device,
        backend=cfg.capture.camera.backend,
        exclude_ir=cfg.capture.camera.exclude_ir,
    )
    open_start = time.perf_counter()
    cam = CameraStream(cfg.capture.camera, device=device, renegotiate=renegotiate).start()
    startup_s = time.perf_counter() - open_start

    # Warm up before timing anything. Exposure and gain are still settling for
    # the first frames, and a cold first frame would otherwise be charged to the
    # steady-state frame rate the gate is measuring.
    for _ in range(WARMUP_FRAMES):
        cam.read(timeout=2.0)

    # Metrics are created only now, so device negotiation and camera open do
    # not count against the measured throughput.
    metrics = Metrics(window=int(max(60, seconds * 60)))
    baseline = cam.stats
    print(
        f"Benchmarking capture from '{device.name}' via {cam.backend} "
        f"for {seconds:.0f} s (startup took {startup_s:.2f} s) ..."
    )

    end = time.perf_counter() + seconds
    try:
        while time.perf_counter() < end:
            frame = cam.read(timeout=2.0)
            if frame is None:
                log.warning("frame timeout during benchmark")
                continue
            metrics.mark_frame()
            metrics.record("capture.frame_age", frame.age_ms)
    finally:
        cam.stop()

    stats = cam.stats
    report = {
        "device": device.name,
        "index": device.index,
        "backend": cam.backend,
        "requested": {
            "width": cfg.capture.camera.width,
            "height": cfg.capture.camera.height,
            "fps": cfg.capture.camera.fps,
            "fourcc": cfg.capture.camera.fourcc,
        },
        "negotiated": {
            "width": cam.actual_width,
            "height": cam.actual_height,
            "fourcc": cam.actual_fourcc,
            "driver_reported_fps": cam.actual_fps,
        },
        "camera": stats,
        "startup_s": round(startup_s, 3),
        "grabbed_during_measurement": int(stats["grabbed"]) - int(baseline["grabbed"]),
        "metrics": metrics.to_dict(),
    }

    print()
    print(metrics.report(header=f"Capture benchmark - {device.name}"))
    print(
        f"  negotiated {cam.actual_width}x{cam.actual_height} "
        f"{cam.actual_fourcc}  |  grabbed {stats['grabbed']}  dropped {stats['dropped']}"
    )

    if out:
        path = Path(out)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  results written to {path}")

    # Acceptance gate. Average frame rate alone is not enough: a pipeline that
    # averages 30 fps but stalls for 200 ms once a second feels broken, so the
    # p95 inter-frame gap is checked too. Startup time is checked because a
    # camera that takes 25 s to open is unusable even at a perfect frame rate.
    fps_target = cfg.capture.camera.fps * 0.8
    achieved = metrics.fps_overall
    p95_gap = metrics.frame_interval().p95
    # Two frame periods: no more than 5% of frames may arrive a full frame late.
    # Tighter than this fails ordinary laptop webcams whose auto-exposure makes
    # their timing irregular without making them unusable.
    gap_budget = 2000.0 / cfg.capture.camera.fps if cfg.capture.camera.fps else 1e9
    drop_pct = 100.0 * int(stats["dropped"]) / max(int(stats["grabbed"]), 1)

    checks = [
        ("sustained fps", achieved >= fps_target, f"{achieved:5.1f} fps  (need >= {fps_target:.1f})"),
        ("p95 frame gap", p95_gap <= gap_budget, f"{p95_gap:5.1f} ms   (need <= {gap_budget:.1f})"),
        ("startup time", startup_s <= 5.0, f"{startup_s:5.2f} s    (need <= 5.00)"),
        ("frame latency", metrics.mean_ms("capture.frame_age") <= 20.0,
         f"{metrics.mean_ms('capture.frame_age'):5.2f} ms   (need <= 20.00)"),
        ("drop rate", drop_pct <= 5.0, f"{drop_pct:5.1f} %    (need <= 5.0)"),
    ]

    print()
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}]  {name:<14} {detail}")
    ok = all(passed for _, passed, _ in checks)
    print(f"\n  Phase 1 capture gate: {'PASS' if ok else 'FAIL'}\n")
    return 0 if ok else 1
