"""One command that checks everything and says what to do about it.

Most of these checks answer "is it plugged in": the models are present, the
camera delivers frames, DPI awareness was granted. Useful, but shallow.

The check that earns its place is the threshold margin. A calibration can be
perfectly *valid* - every threshold ordered correctly, the config loading
without complaint - and still be fragile, because a threshold sits only a
hundredth away from the value the hand actually produces. That reads as "works,
mostly", which is the hardest kind of fault to chase. It is measured here and
reported as a number.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum

from .config import ArgusConfig
from .logsetup import get_logger

log = get_logger("doctor")


class Status(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


SYMBOL = {
    Status.PASS: "  ok  ",
    Status.WARN: " warn ",
    Status.FAIL: " FAIL ",
    Status.SKIP: " skip ",
}


@dataclass
class Check:
    name: str
    status: Status
    detail: str = ""
    fix: str = ""


@dataclass
class Group:
    title: str
    checks: list[Check] = field(default_factory=list)

    def add(self, name, status, detail="", fix="") -> None:
        self.checks.append(Check(name, status, detail, fix))

    @property
    def worst(self) -> Status:
        for level in (Status.FAIL, Status.WARN):
            if any(c.status is level for c in self.checks):
                return level
        return Status.PASS


# --------------------------------------------------------------------------- #
def check_environment() -> Group:
    g = Group("Environment")

    version = sys.version_info
    g.add(
        "python",
        Status.PASS if version >= (3, 10) else Status.FAIL,
        f"{version.major}.{version.minor}.{version.micro}",
        "" if version >= (3, 10) else "Python 3.10 or newer is required.",
    )

    packages = {}
    for module, label in (
        ("cv2", "opencv"),
        ("mediapipe", "mediapipe"),
        ("onnxruntime", "onnxruntime"),
        ("numpy", "numpy"),
    ):
        try:
            mod = __import__(module)
            packages[label] = getattr(mod, "__version__", "?")
            g.add(label, Status.PASS, packages[label])
        except Exception as exc:
            g.add(label, Status.FAIL, str(exc),
                  "Reinstall: .venv\\Scripts\\python.exe -m pip install -r requirements-core.txt")

    # Two OpenCV distributions install into the same cv2 directory, so whichever
    # pip wrote last silently wins while both claim to own the files. This cost
    # real debugging time once already.
    try:
        from importlib.metadata import distributions

        installed = {
            d.metadata["Name"].lower()
            for d in distributions()
            if d.metadata.get("Name")
            and d.metadata["Name"].lower().startswith("opencv")
        }
        if len(installed) > 1:
            g.add(
                "one opencv only", Status.FAIL, ", ".join(sorted(installed)),
                "Two OpenCV packages share the same cv2 directory. Keep one:\n"
                "        pip uninstall -y opencv-python opencv-contrib-python\n"
                "        pip install opencv-contrib-python==4.12.0.88",
            )
        else:
            g.add("one opencv only", Status.PASS, ", ".join(installed) or "none")
    except Exception as exc:
        g.add("one opencv only", Status.SKIP, str(exc))
    return g


def check_models(cfg: ArgusConfig) -> Group:
    g = Group("Models")
    from .models import registry

    needed = registry.models_for_profile(cfg.runtime.profile)
    for name in needed:
        if registry.is_installed(name):
            g.add(name, Status.PASS, "installed")
        else:
            g.add(name, Status.FAIL, "missing", "Fetch them:  argus models pull")

    if all(registry.is_installed(n) for n in needed):
        bad = [n for n, ok, _ in registry.verify_installed() if not ok and n in needed]
        if bad:
            g.add("checksums", Status.WARN, ", ".join(bad),
                  "The cached download no longer matches its pinned hash.")
        else:
            g.add("checksums", Status.PASS, "verified")
    return g


def check_display() -> Group:
    g = Group("Displays")
    from .control.screens import ensure_dpi_aware, get_virtual_desktop

    awareness = ensure_dpi_aware()
    if awareness == "per-monitor-v2":
        g.add("dpi awareness", Status.PASS, awareness)
    elif awareness in ("per-monitor", "system"):
        g.add("dpi awareness", Status.WARN, awareness,
              "Only per-monitor v2 keeps coordinates correct across mixed scaling.")
    else:
        g.add("dpi awareness", Status.FAIL, awareness,
              "Cursor coordinates will be wrong on any display not at 100% scaling.")

    desktop = get_virtual_desktop()
    g.add("monitors", Status.PASS,
          f"{len(desktop.monitors)}, desktop {desktop.width}x{desktop.height}")

    scaled = [m for m in desktop.monitors if m.dpi != 96]
    if scaled:
        g.add("mixed scaling", Status.PASS,
              ", ".join(f"monitor {m.index} at {int(m.scale * 100)}%" for m in scaled))

    if desktop.left < 0 or desktop.top < 0:
        g.add("negative coordinates", Status.PASS,
              f"origin ({desktop.left}, {desktop.top}) - handled")

    # The corners must reach the ends of the absolute range, or the far edges of
    # the desktop are unclickable.
    worst = 0.0
    for px in range(desktop.left, desktop.right, max(1, desktop.width // 50)):
        for py in range(desktop.top, desktop.bottom, max(1, desktop.height // 30)):
            nx, ny = desktop.to_absolute(px, py)
            rx = desktop.left + nx * (desktop.width - 1) / 65535.0
            ry = desktop.top + ny * (desktop.height - 1) / 65535.0
            worst = max(worst, abs(rx - px), abs(ry - py))
    ok = worst < 1.0 and desktop.to_absolute(desktop.right - 1, desktop.bottom - 1) == (65535, 65535)
    g.add("coordinate mapping", Status.PASS if ok else Status.FAIL,
          f"max round-trip error {worst:.3f}px",
          "" if ok else "The far edges of the desktop cannot be reached.")
    return g


def check_injection() -> Group:
    g = Group("Input injection")
    import ctypes

    try:
        from .control.injector import INPUT, MouseInjector

        expected = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
        size = ctypes.sizeof(INPUT)
        g.add("INPUT struct", Status.PASS if size == expected else Status.FAIL,
              f"{size} bytes",
              "" if size == expected else "SendInput will silently reject every event.")
        injector = MouseInjector()
        g.add("injector", Status.PASS, f"disarmed, backend ready ({len(injector.desktop.monitors)} displays)")
    except Exception as exc:
        g.add("injector", Status.FAIL, str(exc))
    return g


def check_camera(cfg: ArgusConfig, quick: bool = True) -> Group:
    g = Group("Camera")
    from .capture.picker import load_scan, select_camera

    scan = load_scan()
    if not scan:
        g.add("scan cache", Status.WARN, "not built yet",
              "Build it once:  argus cameras --scan")
    else:
        working = [c for c in scan if c.works and not c.blank]
        blank = [c for c in scan if c.works and c.blank]
        g.add("cameras found", Status.PASS if working else Status.FAIL,
              f"{len(working)} usable of {len(scan)} tested",
              "" if working else "No camera delivers a usable image.")
        if blank:
            g.add("black frames", Status.WARN,
                  ", ".join(c.spec for c in blank),
                  "A camera is delivering only black - is a privacy shutter closed?")

    try:
        device = select_camera(cfg)
        g.add("selected", Status.PASS, f"{device.name} ({device.spec})")
    except Exception as exc:
        g.add("selected", Status.FAIL, str(exc).splitlines()[0],
              "Choose one by eye:  argus cameras --pick")
        return g

    if quick:
        g.add("live capture", Status.SKIP, "use --live to open the camera")
        return g

    try:
        from .capture.camera import CameraStream

        cam = CameraStream(cfg.capture.camera, device=device).start()
        try:
            good = sum(1 for _ in range(20) if cam.read(timeout=2.0) is not None)
            rate = cam.stats["capture_fps"]
        finally:
            cam.stop()
        ok = good >= 18
        g.add("live capture", Status.PASS if ok else Status.WARN,
              f"{good}/20 frames, {rate} fps, {cam.actual_width}x{cam.actual_height}",
              "" if ok else "The camera is dropping frames.")
    except Exception as exc:
        g.add("live capture", Status.FAIL, str(exc).splitlines()[0])
    return g


def check_thresholds(cfg: ArgusConfig) -> Group:
    """Margins, not just validity.

    A threshold can be correctly ordered and still sit so close to the value a
    hand produces that the gesture works only sometimes - which presents as
    "it is a bit unreliable" rather than as a fault.
    """
    g = Group("Gesture thresholds")
    t = cfg.gestures

    pinch_gap = t.pinch_open - t.pinch_close
    g.add(
        "pinch hysteresis",
        Status.PASS if pinch_gap >= 0.10 else Status.WARN,
        f"{pinch_gap:.3f} between close ({t.pinch_close}) and open ({t.pinch_open})",
        "" if pinch_gap >= 0.10
        else "A narrow band lets the pinch chatter, firing repeated clicks. "
             "Re-run: argus calibrate --write",
    )

    approach_gap = t.pinch_approach - t.pinch_open
    g.add(
        "click freeze margin",
        Status.PASS if approach_gap >= 0.05 else Status.WARN,
        f"{approach_gap:.3f} above the open threshold",
        "" if approach_gap >= 0.05
        else "The cursor may not freeze before a click registers, so clicks drift.",
    )

    clutch_gap = t.finger_extended - t.finger_curled
    g.add(
        "clutch hysteresis",
        Status.PASS if clutch_gap >= 0.15 else Status.WARN,
        f"{clutch_gap:.3f} between curled ({t.finger_curled}) and extended "
        f"({t.finger_extended})",
        "" if clutch_gap >= 0.15 else "The cursor will flicker on and off.",
    )

    # A pointing index usually measures 1.2-1.4. An engage threshold close to
    # or above that leaves almost no room, and the cursor becomes reluctant.
    typical_pointing = 1.25
    headroom = typical_pointing - t.finger_extended
    if headroom >= 0.15:
        status, fix = Status.PASS, ""
    elif headroom >= 0.05:
        status, fix = Status.WARN, (
            "Engaging the cursor may feel hesitant. Straighten your index fully "
            "during the POINTING pose and re-run: argus calibrate --write"
        )
    else:
        status, fix = Status.FAIL, (
            "The cursor will often refuse to engage. Re-run calibration and "
            "straighten your index finger completely: argus calibrate --write"
        )
    g.add("clutch headroom", status,
          f"{headroom:+.3f} below a typical pointing finger ({typical_pointing})", fix)

    g.add("scroll dwell", Status.PASS if 0.15 <= t.scroll_dwell_s <= 0.6 else Status.WARN,
          f"{t.scroll_dwell_s}s before a middle pinch becomes a scroll")
    return g


def check_actions(cfg: ArgusConfig, quick: bool = True) -> Group:
    g = Group("Actions")
    if not cfg.actions.enabled:
        g.add("action layer", Status.SKIP, "disabled in config")
        return g
    g.add("action layer", Status.PASS, "open palm = volume, V = brightness, thumbs up = launch")

    target = cfg.actions.launch_target
    if not target:
        g.add("launch target", Status.WARN, "not set",
              "Find your app:  argus actions --find whatsapp --write")
    else:
        g.add("launch target", Status.PASS, target[:60])

    if quick:
        g.add("brightness", Status.SKIP, "use --live to query the display")
    else:
        from .control.system import BrightnessControl

        b = BrightnessControl().start()
        if b.available:
            g.add("brightness", Status.PASS, f"available, currently {b.level}%")
        else:
            g.add("brightness", Status.WARN, "unavailable",
                  "Only the internal panel is controllable; external monitors "
                  "need DDC/CI.")
        b.stop()
    return g


def check_identity(cfg: ArgusConfig) -> Group:
    g = Group("Identity")
    if not cfg.security.require_identity:
        g.add("gating", Status.SKIP, "off - actions are not identity-gated")
    from .face.gallery import Gallery

    try:
        gallery = Gallery.load(cfg.face.gallery_path, model=cfg.face.recognizer.model)
    except Exception as exc:
        g.add("gallery", Status.FAIL, str(exc).splitlines()[0],
              "Re-enrol:  argus enroll <name> --reset")
        return g

    if gallery.is_empty:
        status = Status.FAIL if cfg.security.require_identity else Status.SKIP
        g.add("gallery", status, "nobody enrolled",
              "Enrol yourself:  argus enroll \"Your Name\"" if cfg.security.require_identity else "")
    else:
        g.add("gallery", Status.PASS,
              f"{len(gallery.names)} identities, {gallery.total_samples()} samples")
        report = gallery.analyse()
        if report.get("overlap"):
            g.add("separation", Status.WARN, "identities overlap",
                  "Recognition will be unreliable. Re-enrol with better lighting.")
        elif report.get("genuine"):
            g.add("separation", Status.PASS,
                  f"suggested threshold {report.get('suggested_threshold')}")

    if cfg.security.require_identity and cfg.security.operator:
        known = cfg.security.operator in gallery.names
        g.add("operator", Status.PASS if known else Status.FAIL,
              cfg.security.operator,
              "" if known else "That operator is not enrolled; every action will be blocked.")
    return g


# --------------------------------------------------------------------------- #
def run_doctor(cfg: ArgusConfig, live: bool = False) -> int:
    groups = [
        check_environment(),
        check_models(cfg),
        check_display(),
        check_injection(),
        check_camera(cfg, quick=not live),
        check_thresholds(cfg),
        check_actions(cfg, quick=not live),
        check_identity(cfg),
    ]

    print()
    print("  ARGUS health check")
    if not live:
        print("  (add --live to open the camera and query the display)")
    print()

    fixes: list[tuple[str, str]] = []
    for group in groups:
        print(f"  {group.title}")
        for check in group.checks:
            line = f"    [{SYMBOL[check.status]}] {check.name:<22} {check.detail}"
            print(line)
            if check.fix:
                fixes.append((check.name, check.fix))
        print()

    failures = sum(1 for g in groups for c in g.checks if c.status is Status.FAIL)
    warnings = sum(1 for g in groups for c in g.checks if c.status is Status.WARN)

    if fixes:
        print("  What to do")
        for name, fix in fixes:
            print(f"    {name}:")
            for line in fix.splitlines():
                print(f"      {line}")
        print()

    if failures:
        print(f"  {failures} failure(s), {warnings} warning(s).")
    elif warnings:
        print(f"  No failures, {warnings} warning(s) worth a look.")
    else:
        print("  Everything checks out.")
    print()
    return 1 if failures else 0
