"""ARGUS command-line interface.

Every phase of the system is reachable from one entry point::

    python -m argus cameras            # discover and choose a camera
    python -m argus preview            # live capture preview + FPS
    python -m argus bench capture      # headless throughput benchmark
    python -m argus config             # show the effective configuration
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
    print("\n  Choose one with:   python -m argus preview --camera brio")
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
# bench
# --------------------------------------------------------------------------- #
def cmd_bench(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    if args.target == "capture":
        from .capture.preview import bench_capture

        return bench_capture(cfg, seconds=args.seconds, out=args.out,
                             renegotiate=args.renegotiate)
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
    p_bench.add_argument("target", choices=["capture"], help="what to benchmark")
    p_bench.add_argument("--seconds", type=float, default=15.0, help="benchmark duration")
    p_bench.add_argument("--out", default=None, help="write JSON results to this path")
    p_bench.set_defaults(func=cmd_bench)

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
