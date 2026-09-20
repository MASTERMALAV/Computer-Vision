"""The integrated virtual mouse: capture -> landmarks -> gestures -> cursor.

Safety model, in order of precedence:

1. **Disarmed by default.** Everything runs and the HUD shows exactly what would
   happen, but no input reaches the OS until the operator presses F9.
2. **Global panic key.** Holding Esc disarms from anywhere, read straight from
   the keyboard rather than through the preview window - because once the system
   clicks something, the preview no longer has focus.
3. **Buttons are always released** on every exit path, so quitting mid-drag can
   never leave the desktop with a stuck mouse button.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .capture.camera import CameraStream
from .capture.devices import resolve_device
from .config import ROOT, ArgusConfig
from .control.dispatcher import ActionDispatcher
from .control.filters import GainConfig, OneEuroConfig
from .control.hotkeys import VK_F9, VK_F10, EdgeDetector, PanicSwitch
from .control.injector import MouseInjector
from .control.pointer import PointerConfig, PointerEngine
from .control.screens import get_virtual_desktop
from .gestures.fsm import GestureEngine, GestureThresholds, GestureType
from .hands.engine import HandEngine
from .logsetup import get_logger
from .metrics import Metrics
from .ui.hands import draw_hand
from .ui.overlay import COLORS, draw_bar, draw_panel, draw_text, fps_color

log = get_logger("app")


def _pointer_config(cfg: ArgusConfig) -> PointerConfig:
    c = cfg.control
    return PointerConfig(
        gain=GainConfig(
            slow_speed=c.gain.slow_speed,
            fast_speed=c.gain.fast_speed,
            min_gain=c.gain.min_gain,
            max_gain=c.gain.max_gain,
            pixels_per_unit=c.gain.pixels_per_unit,
        ),
        euro=OneEuroConfig(
            min_cutoff=c.smoothing.min_cutoff,
            beta=c.smoothing.beta,
            d_cutoff=c.smoothing.d_cutoff,
        ),
        source=c.source,
        dead_zone=c.dead_zone,
        freeze_timeout_s=c.freeze_timeout_s,
        max_delta_units=c.max_delta_units,
        max_jump_px=c.max_jump_px,
    )


def _gesture_thresholds(cfg: ArgusConfig) -> GestureThresholds:
    g = cfg.gestures
    return GestureThresholds(
        pinch_close=g.pinch_close,
        pinch_open=g.pinch_open,
        pinch_approach=g.pinch_approach,
        finger_extended=g.finger_extended,
        finger_curled=g.finger_curled,
        debounce_frames=g.debounce_frames,
        clutch_debounce_frames=g.clutch_debounce_frames,
        drag_dwell_s=g.drag_dwell_s,
        click_cooldown_s=g.click_cooldown_s,
        double_click_s=g.double_click_s,
        motion_gate_speed=g.motion_gate_speed,
    )


@dataclass
class SessionSummary:
    frames: int = 0
    events: dict[str, int] = field(default_factory=dict)
    actions_executed: int = 0
    actions_blocked: int = 0
    cursor_px: float = 0.0
    armed_seconds: float = 0.0


HELP = [
    "F9       arm / disarm cursor control",
    "Esc      HOLD to disarm (works anywhere)",
    "F10      re-centre cursor on primary",
    "q        quit",
    "h        hide this panel",
]


def _select_hand(result, wanted: str):
    """Pick the hand allowed to drive the cursor."""
    if not result.hands:
        return None
    if wanted in ("Left", "Right"):
        return result.by_handedness(wanted)
    return result.primary


def run_mouse(
    cfg: ArgusConfig,
    seconds: float = 0.0,
    start_armed: bool = False,
    renegotiate: bool = False,
    out: str | None = None,
) -> int:
    """Run the virtual mouse with an on-screen HUD."""
    import cv2

    desktop = get_virtual_desktop()
    log.info("display layout:\n%s", desktop.describe())

    device = resolve_device(
        cfg.capture.camera.device,
        backend=cfg.capture.camera.backend,
        exclude_ir=cfg.capture.camera.exclude_ir,
    )
    cam = CameraStream(cfg.capture.camera, device=device, renegotiate=renegotiate).start()
    engine = HandEngine(cfg.hands, mirrored_input=cfg.capture.camera.mirror).start()
    gestures = GestureEngine(_gesture_thresholds(cfg))
    injector = MouseInjector(desktop, armed=False)
    pointer = PointerEngine(injector, _pointer_config(cfg))
    dispatcher = ActionDispatcher(injector, cfg.security)

    if start_armed:
        injector.arm()

    panic = PanicSwitch()
    arm_key = EdgeDetector(VK_F9)
    center_key = EdgeDetector(VK_F10)

    metrics = Metrics(window=cfg.runtime.metrics_window)
    summary = SessionSummary()
    recent_events: list[str] = []
    show_help = True
    armed_since: float | None = time.perf_counter() if start_armed else None

    window = cfg.ui.window_name + " - virtual mouse"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, cfg.ui.preview_width, int(cfg.ui.preview_width * 9 / 16))

    scale_factor = (
        cfg.capture.detect_width / float(cam.actual_width)
        if 0 < cfg.capture.detect_width < cam.actual_width
        else 1.0
    )
    deadline = time.perf_counter() + seconds if seconds > 0 else None

    print()
    print("  ARGUS virtual mouse")
    print(f"  {desktop.describe()}".replace("\n", "\n  "))
    print()
    print("  DISARMED. Press F9 to arm. Hold Esc anywhere to disarm.")
    print("  Point with an extended index finger to move; pinch thumb+index to click.")
    print()

    try:
        while True:
            frame = cam.read(timeout=2.0)
            if frame is None:
                if not cam.is_running:
                    break
                continue
            now = time.perf_counter()
            metrics.mark_frame()
            summary.frames += 1

            # ---- global hotkeys, read straight from the keyboard ---------- #
            if panic.triggered(now):
                if injector.armed:
                    log.warning("PANIC: disarming")
                    dispatcher.shutdown()
                    injector.disarm()
                    if armed_since is not None:
                        summary.armed_seconds += now - armed_since
                        armed_since = None
                    recent_events.append("PANIC DISARM")
                panic.reset()
            if arm_key.pressed():
                if injector.armed:
                    dispatcher.shutdown()
                    injector.disarm()
                    if armed_since is not None:
                        summary.armed_seconds += now - armed_since
                        armed_since = None
                else:
                    injector.arm()
                    pointer.sync_from_system()
                    armed_since = now
            if center_key.pressed():
                cx, cy = desktop.primary.center
                pointer._cursor[:] = (cx, cy)
                injector.move_to(cx, cy)

            # ---- perception ------------------------------------------------ #
            with metrics.timer("stage.preprocess"):
                small = (
                    cv2.resize(
                        frame.image, None, fx=scale_factor, fy=scale_factor,
                        interpolation=cv2.INTER_AREA,
                    )
                    if scale_factor < 1.0
                    else frame.image
                )

            result = engine.process(small, frame.monotonic_ms, frame.index)
            metrics.record("stage.landmarks", result.inference_ms)

            if scale_factor < 1.0:
                for h in result.hands:
                    h.pixels /= scale_factor
                    h.frame_size = (frame.width, frame.height)

            hand = _select_hand(result, cfg.control.hand)

            with metrics.timer("stage.gestures"):
                events = gestures.update(hand, now)
            with metrics.timer("stage.pointer"):
                pstate = pointer.update(hand, gestures.state, events, now)
            with metrics.timer("stage.dispatch"):
                records = dispatcher.dispatch(events, now)

            for ev in events:
                if ev.type not in (GestureType.PINCH_APPROACH, GestureType.PINCH_ABORT):
                    summary.events[ev.type.value] = summary.events.get(ev.type.value, 0) + 1
                    recent_events.append(str(ev))
            for rec in records:
                if not rec.executed:
                    summary.actions_blocked += 1
                else:
                    summary.actions_executed += 1
            if len(recent_events) > 6:
                del recent_events[: len(recent_events) - 6]

            # ---- HUD -------------------------------------------------------- #
            with metrics.timer("stage.render"):
                canvas = frame.image.copy()
                if hand is not None:
                    draw_hand(canvas, hand, show_pinch=True)

                armed = injector.armed
                gs = gestures.state
                fps = metrics.fps

                banner = "ARMED - CURSOR LIVE" if armed else "DISARMED - dry run"
                bcolor = COLORS["error"] if armed else COLORS["ok"]
                bw = canvas.shape[1]
                cv2.rectangle(canvas, (0, 0), (bw, 34), bcolor, -1)
                draw_text(canvas, banner, (14, 24), 0.7, (20, 20, 20), 2, shadow=False)
                draw_text(
                    canvas,
                    f"{fps:4.1f} FPS   {metrics.mean_ms('stage.landmarks'):.0f}ms lm",
                    (bw - 260, 24), 0.55, (20, 20, 20), 1, shadow=False,
                )

                lines = [
                    f"clutch    {'ENGAGED' if gs.clutch_engaged else 'released'}"
                    f"{'  (FROZEN)' if pstate.frozen else ''}",
                    f"cursor    {pstate.position[0]:7.0f}, {pstate.position[1]:6.0f}"
                    f"   monitor {pstate.monitor}",
                    f"gain      {pstate.gain:4.2f}x   speed {gs.hand_speed:4.2f} u/s",
                    f"pinch  i  {gs.pinch_index:5.2f}   m {gs.pinch_middle:5.2f}"
                    f"   (close<{cfg.gestures.pinch_close})",
                    f"drag      {'YES' if gs.dragging else 'no'}"
                    f"   actions {dispatcher.stats.executed}"
                    f" / blocked {summary.actions_blocked}",
                ]
                if gs.suppressed_by_motion:
                    lines.append("moving too fast - gestures gated")
                draw_panel(canvas, lines, origin=(12, 46),
                           title="ARGUS  -  virtual mouse", min_width=390)

                # Pinch bar: the fastest way to see where the thresholds sit.
                by = 46 + 22 * (len(lines) + 1) + 18
                draw_text(canvas, "pinch", (14, by - 4), 0.42, COLORS["muted"])
                draw_bar(canvas, (62, by - 12), 200,
                         1.0 - min(gs.pinch_index / max(cfg.gestures.pinch_open, 1e-6), 1.0),
                         COLORS["accent"] if gs.pinch_index > cfg.gestures.pinch_close
                         else COLORS["ok"])

                if recent_events:
                    draw_panel(canvas, recent_events[-6:], origin=(canvas.shape[1] - 330, 46),
                               scale=0.45, alpha=0.5, title="events", min_width=300)
                if show_help:
                    draw_panel(canvas, HELP,
                               origin=(12, canvas.shape[0] - 24 - 22 * len(HELP)),
                               scale=0.45, alpha=0.55, color=COLORS["muted"])

            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("h"):
                show_help = not show_help

            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if deadline is not None and now > deadline:
                break
    finally:
        # Order matters: release buttons before tearing anything else down.
        dispatcher.shutdown()
        injector.disarm()
        engine.close()
        cam.stop()
        cv2.destroyAllWindows()
        for _ in range(4):
            cv2.waitKey(1)

    if armed_since is not None:
        summary.armed_seconds += time.perf_counter() - armed_since
    summary.actions_executed = dispatcher.stats.executed
    summary.cursor_px = pointer.total_moved_px

    print()
    print(metrics.report(header="Virtual mouse session"))
    print(f"  gesture events : {summary.events or 'none'}")
    print(f"  actions        : {dispatcher.stats.as_dict()}")
    print(f"  injector       : {injector.stats.as_dict()}")
    print(f"  cursor travel  : {summary.cursor_px:.0f} px")
    print(f"  armed for      : {summary.armed_seconds:.1f} s")

    if out:
        path = Path(out)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "frames": summary.frames,
                    "events": summary.events,
                    "actions": dispatcher.stats.as_dict(),
                    "injector": injector.stats.as_dict(),
                    "cursor_travel_px": round(summary.cursor_px, 1),
                    "armed_seconds": round(summary.armed_seconds, 2),
                    "metrics": metrics.to_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  results written to {path}")
    return 0
