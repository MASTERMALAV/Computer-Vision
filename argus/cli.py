"""ARGUS command-line interface.

Every phase of the system is reachable from one entry point::

    argus cameras            # discover and choose a camera
    argus preview            # live capture preview + FPS
    argus models pull        # fetch model weights
    argus hands              # live hand landmark preview
    argus mouse              # the virtual mouse
    argus calibrate          # fit gesture thresholds to your hand
    argus enroll <name>      # enrol a face for identity gating
    argus face               # live face recognition preview
    argus screens            # display layout + DPI
    argus bench capture      # headless throughput benchmark
    argus config             # show the effective configuration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ROOT, ArgusConfig, ConfigError
from .logsetup import configure_native_backends, setup_logging

# Native backends read their log level from the environment at import time, so
# this has to run before cv2 / mediapipe are imported anywhere.
configure_native_backends()


# --------------------------------------------------------------------------- #
# Shared argument wiring
# --------------------------------------------------------------------------- #
def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="YAML config file to load (default: configs/default.yaml if present)",
    )
    parser.add_argument(
        "-s",
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set capture.camera.width=1920 (repeatable)",
    )
    parser.add_argument(
        "-p",
        "--profile",
        choices=["fast", "balanced", "accurate"],
        default=None,
        help="Speed/accuracy preset (overrides the config file)",
    )
    parser.add_argument("--camera", default=None, help="Camera index, name substring, or 'auto'")
    parser.add_argument(
        "--renegotiate",
        action="store_true",
        help="re-measure the best capture backend instead of using the cached result",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )


def _load_config(args: argparse.Namespace) -> ArgusConfig:
    """Resolve config file + CLI conveniences into one validated config."""
    path = args.config
    if path is None:
        default = ROOT / "configs" / "default.yaml"
        path = default if default.exists() else None

    overrides = list(getattr(args, "overrides", []) or [])
    # Dedicated flags are just sugar over --set, applied first so an explicit
    # --set on the same key still wins.
    if getattr(args, "profile", None):
        overrides.insert(0, f"runtime.profile={args.profile}")
    if getattr(args, "camera", None):
        overrides.append(f"capture.camera.device={args.camera}")
    if getattr(args, "log_level", None):
        overrides.append(f"runtime.log_level={args.log_level}")

    cfg = ArgusConfig.load(path, overrides)
    setup_logging(cfg.runtime.log_level)
    return cfg


# --------------------------------------------------------------------------- #
# cameras
# --------------------------------------------------------------------------- #
def cmd_cameras(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    from .capture.devices import enumerate_devices, resolve_device

    devices = enumerate_devices(
        backend=cfg.capture.camera.backend,
        probe=args.probe,
        verify=args.probe or args.verify,
    )

    if args.json:
        print(json.dumps([d.to_dict() for d in devices], indent=2))
        return 0

    if not devices:
        print("No cameras found.")
        print("  - check the camera is plugged in")
        print("  - Windows Settings > Privacy & security > Camera > allow desktop apps")
        return 1

    try:
        selected = resolve_device(
            cfg.capture.camera.device,
            devices,
            exclude_ir=cfg.capture.camera.exclude_ir,
        )
        selected_index = selected.index
    except Exception:
        selected_index = -1

    print(f"\n  Cameras available ({len(devices)} found)\n")
    for dev in devices:
        marker = "->" if dev.index == selected_index else "  "
        tags = []
        if dev.is_ir:
            tags.append("infrared - not usable for face recognition")
        if dev.is_virtual:
            tags.append("virtual")
        if dev.working is False:
            tags.append("NOT OPENABLE")
        elif dev.working is True:
            tags.append("verified")
        note = f"   ({'; '.join(tags)})" if tags else ""
        print(f"  {marker} [{dev.index}] {dev.name}{note}")
        if dev.modes:
            modes = ", ".join(f"{w}x{h}" for w, h in dev.modes)
            print(f"        modes: {modes}")

    if selected_index >= 0:
        print(f"\n  Current selection: [{selected_index}] (config: capture.camera.device)")
    print("\n  Choose one with:   argus preview --camera brio")
    print("  Make it permanent: --set capture.camera.device=brio in configs/default.yaml\n")
    return 0


# --------------------------------------------------------------------------- #
# preview
# --------------------------------------------------------------------------- #
def cmd_preview(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    from .capture.preview import run_preview

    return run_preview(cfg, seconds=args.seconds, save_to=args.save,
                       renegotiate=args.renegotiate)


# --------------------------------------------------------------------------- #
# hands
# --------------------------------------------------------------------------- #
def cmd_hands(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    from .hands.preview import run_hands_preview

    return run_hands_preview(cfg, seconds=args.seconds, renegotiate=args.renegotiate)


# --------------------------------------------------------------------------- #
# mouse
# --------------------------------------------------------------------------- #
def cmd_mouse(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    from .app import run_mouse

    return run_mouse(
        cfg,
        seconds=args.seconds,
        start_armed=args.armed,
        renegotiate=args.renegotiate,
        out=args.out,
    )


# --------------------------------------------------------------------------- #
# face
# --------------------------------------------------------------------------- #
def cmd_enroll(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    from .face.enroll import run_enrolment

    return run_enrolment(cfg, args.name, per_pose=args.samples, reset=args.reset)


def cmd_faces(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    from .face.enroll import forget, show_gallery

    if args.forget:
        return forget(cfg, args.forget)
    return show_gallery(cfg, analyse=args.analyse)


def cmd_face_preview(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    from .face.preview import run_face_preview

    return run_face_preview(cfg, seconds=args.seconds)


# --------------------------------------------------------------------------- #
# calibrate
# --------------------------------------------------------------------------- #
def cmd_calibrate(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    from .gestures.calibrate import run_calibration

    return run_calibration(cfg, write=args.write)


# --------------------------------------------------------------------------- #
# screens
# --------------------------------------------------------------------------- #
def cmd_screens(args: argparse.Namespace) -> int:
    _load_config(args)
    from .control.screens import ensure_dpi_aware, get_cursor_position, get_virtual_desktop

    awareness = ensure_dpi_aware()
    desktop = get_virtual_desktop()
    if args.json:
        print(json.dumps({
            "dpi_awareness": awareness,
            "virtual_desktop": {
                "left": desktop.left, "top": desktop.top,
                "width": desktop.width, "height": desktop.height,
            },
            "monitors": [
                {"index": m.index, "left": m.left, "top": m.top,
                 "width": m.width, "height": m.height,
                 "primary": m.is_primary, "dpi": m.dpi}
                for m in desktop.monitors
            ],
            "cursor": list(get_cursor_position()),
        }, indent=2))
        return 0
    print()
    print(f"  DPI awareness: {awareness}")
    for line in desktop.describe().splitlines():
        print("  " + line)
    print(f"  cursor currently at {get_cursor_position()}")
    print()
    return 0


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def cmd_models(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    from .models import registry

    if args.action == "pull":
        names = args.names or registry.models_for_profile(cfg.runtime.profile)
        for name in names:
            path = registry.ensure(name)
            print(f"  {name:<16} -> {path}")
        return 0

    if args.action == "verify":
        failed = False
        for name, ok, note in registry.verify_installed():
            print(f"  [{'OK  ' if ok else 'FAIL'}]  {name:<16} {note}")
            failed |= not ok
        return 1 if failed else 0

    rows = registry.status()
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print()
    print(f"  Models ({sum(1 for r in rows if r['installed'])}/{len(rows)} installed)")
    print()
    for row in rows:
        mark = "installed" if row["installed"] else "not fetched"
        size = f"{row['size_mb']} MB" if row["size_mb"] else ""
        print(f"  [{mark:>11}] {row['name']:<16} {size:>9}  {row['description']}")
        print(f"                {' ' * 16} {row['credit']}")
    print()
    print("  Fetch what the current profile needs:  argus models pull")
    print()
    return 0


# --------------------------------------------------------------------------- #
# bench
# --------------------------------------------------------------------------- #
def cmd_bench(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    if args.target == "capture":
        from .capture.preview import bench_capture

        return bench_capture(cfg, seconds=args.seconds, out=args.out,
                             renegotiate=args.renegotiate)
    if args.target == "hands":
        from .hands.preview import bench_hands

        return bench_hands(cfg, seconds=args.seconds, out=args.out,
                           renegotiate=args.renegotiate)
    if args.target == "face":
        from .face.preview import bench_face

        return bench_face(cfg, seconds=args.seconds, out=args.out)
    print(f"Unknown benchmark target: {args.target}", file=sys.stderr)
    return 2


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def cmd_config(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    text = cfg.dump_yaml()
    if args.out:
        path = Path(args.out)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"Wrote effective config to {path}")
    else:
        print(text)
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="argus",
        description="ARGUS - real-time face, gesture and voice perception.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    sub = parser.add_subparsers(dest="command")

    p_cam = sub.add_parser("cameras", help="list cameras and show which one is selected")
    _add_common(p_cam)
    p_cam.add_argument(
        "--probe", action="store_true", help="test each camera for supported resolutions (slow)"
    )
    p_cam.add_argument("--verify", action="store_true", help="open each camera and grab one frame")
    p_cam.add_argument("--json", action="store_true", help="machine-readable output")
    p_cam.set_defaults(func=cmd_cameras)

    p_prev = sub.add_parser("preview", help="live camera preview with capture statistics")
    _add_common(p_prev)
    p_prev.add_argument("--seconds", type=float, default=0.0, help="auto-exit after N seconds")
    p_prev.add_argument("--save", default=None, help="write one frame to this path and exit")
    p_prev.set_defaults(func=cmd_preview)

    p_bench = sub.add_parser("bench", help="run a headless benchmark")
    _add_common(p_bench)
    p_bench.add_argument(
        "target", choices=["capture", "hands", "face"], help="what to benchmark"
    )
    p_bench.add_argument("--seconds", type=float, default=15.0, help="benchmark duration")
    p_bench.add_argument("--out", default=None, help="write JSON results to this path")
    p_bench.set_defaults(func=cmd_bench)

    p_hands = sub.add_parser("hands", help="live hand landmark preview")
    _add_common(p_hands)
    p_hands.add_argument("--seconds", type=float, default=0.0, help="auto-exit after N seconds")
    p_hands.set_defaults(func=cmd_hands)

    p_mouse = sub.add_parser("mouse", help="run the virtual mouse (disarmed until you press F9)")
    _add_common(p_mouse)
    p_mouse.add_argument("--seconds", type=float, default=0.0, help="auto-exit after N seconds")
    p_mouse.add_argument(
        "--armed", action="store_true",
        help="start with cursor control live (default: start disarmed)",
    )
    p_mouse.add_argument("--out", default=None, help="write a session report to this path")
    p_mouse.set_defaults(func=cmd_mouse)

    p_enroll = sub.add_parser("enroll", help="enrol a face so actions can be identity-gated")
    _add_common(p_enroll)
    p_enroll.add_argument("name", help="who is being enrolled")
    p_enroll.add_argument("--samples", type=int, default=6, help="samples per pose (5 poses)")
    p_enroll.add_argument(
        "--reset", action="store_true", help="discard the existing gallery first"
    )
    p_enroll.set_defaults(func=cmd_enroll)

    p_faces = sub.add_parser("faces", help="list enrolled identities")
    _add_common(p_faces)
    p_faces.add_argument(
        "--analyse", action="store_true", help="report similarity separation and a threshold"
    )
    p_faces.add_argument("--forget", metavar="NAME", default=None, help="remove an identity")
    p_faces.set_defaults(func=cmd_faces)

    p_face = sub.add_parser("face", help="live face detection and recognition preview")
    _add_common(p_face)
    p_face.add_argument("--seconds", type=float, default=0.0, help="auto-exit after N seconds")
    p_face.set_defaults(func=cmd_face_preview)

    p_cal = sub.add_parser("calibrate", help="fit gesture thresholds to your hand")
    _add_common(p_cal)
    p_cal.add_argument(
        "--write",
        nargs="?",
        const="configs/calibrated.yaml",
        default=None,
        help="write the fitted thresholds to a config file (default: configs/calibrated.yaml)",
    )
    p_cal.set_defaults(func=cmd_calibrate)

    p_screens = sub.add_parser("screens", help="show display layout and DPI awareness")
    _add_common(p_screens)
    p_screens.add_argument("--json", action="store_true", help="machine-readable output")
    p_screens.set_defaults(func=cmd_screens)

    p_models = sub.add_parser("models", help="download and verify model weights")
    _add_common(p_models)
    p_models.add_argument(
        "action", nargs="?", default="status", choices=["status", "pull", "verify"]
    )
    p_models.add_argument("names", nargs="*", help="specific models (default: what the profile needs)")
    p_models.add_argument("--json", action="store_true", help="machine-readable output")
    p_models.set_defaults(func=cmd_models)

    p_cfg = sub.add_parser("config", help="print the effective configuration")
    _add_common(p_cfg)
    p_cfg.add_argument("--out", default=None, help="write the config to this path instead")
    p_cfg.set_defaults(func=cmd_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from . import __version__

        print(f"argus {__version__}")
        return 0
    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"\nConfiguration error: {exc}\n", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
