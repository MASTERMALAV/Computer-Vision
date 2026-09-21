"""Find every camera that actually works, and let the operator choose one.

An index is not an identity. Each capture backend enumerates devices
independently and they can disagree - on the development machine DirectShow
lists ``[Integrated Camera, Brio 100]`` while Media Foundation's index 0 *is*
the Brio. Selecting by name under one backend and capturing under the other
therefore opens the wrong camera, confidently and silently.

So this module does not assume anything. It opens every (backend, index) pair,
keeps the ones that deliver real frames, and identifies them by what they show:
a thumbnail from an MSMF device is compared against the thumbnails of the
DirectShow devices, whose names *are* authoritative. Two cameras in the same
room differ enormously - different position, field of view and colour response -
so the match is unambiguous when it exists, and is reported as uncertain when it
is not.

The picker then shows each working camera live, so the final choice is made by
looking at the picture rather than trusting any of this.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..config import ROOT, ArgusConfig, CameraConfig
from ..logsetup import get_logger
from ..ui.overlay import COLORS, draw_panel, draw_text
from .devices import CameraDevice, _matches_any, _IR_PATTERNS, _VIRTUAL_PATTERNS, backend_flag
from .devices import _names_via_pygrabber

log = get_logger("capture.picker")

MAX_INDEX = 3  # how far to scan per backend
THUMB = (160, 90)
# Enough for auto-exposure to settle without making the one-time scan tedious.
# A backend that negotiates an uncompressed format runs at 5 fps, so every
# frame here costs 200 ms on that path.
SETTLE_FRAMES = 6
MEASURE_FRAMES = 8

# Below this mean absolute difference two thumbnails are the same camera; the
# best match must also beat the runner-up by MATCH_MARGIN to be trusted.
# Measured on this machine: the same camera seen through two backends differs
# by about 5, two different cameras by about 57. The threshold sits well clear
# of both, because auto-exposure drifts between the two grabs.
SAME_CAMERA_DIFF = 26.0
MATCH_MARGIN = 10.0


@dataclass
class Candidate:
    """One (backend, index) pair that was actually opened and tested."""

    backend: str
    index: int
    works: bool
    width: int = 0
    height: int = 0
    fps: float = 0.0
    name: str = ""
    name_source: str = "unknown"  # dshow | inferred | unknown
    thumbnail: np.ndarray | None = field(default=None, repr=False)
    blank: bool = False  # delivers frames, but they are black

    @property
    def spec(self) -> str:
        return f"{self.backend}:{self.index}"

    @property
    def display_name(self) -> str:
        if self.name:
            return self.name
        return f"Camera {self.index}"

    def to_device(self) -> CameraDevice:
        return CameraDevice(
            index=self.index,
            name=self.display_name,
            backend=self.backend,
            backend_locked=True,
            name_is_exact=self.name_source == "dshow",
            is_ir=_matches_any(self.display_name, _IR_PATTERNS),
            is_virtual=_matches_any(self.display_name, _VIRTUAL_PATTERNS),
            working=self.works,
        )

    def describe(self) -> str:
        if not self.works:
            return f"{self.spec:<10} {self.display_name:<26} unusable"
        note = "  (black frames)" if self.blank else ""
        tag = {"dshow": "", "inferred": "  ~name", "unknown": "  ?name"}[self.name_source]
        return (
            f"{self.spec:<10} {self.display_name:<26} "
            f"{self.width}x{self.height} @ {self.fps:4.1f} fps{note}{tag}"
        )


def _probe(index: int, backend: str, cfg: CameraConfig) -> Candidate:
    """Open one pair, grab frames, and measure what it delivers."""
    import cv2

    cand = Candidate(backend=backend, index=index, works=False)
    cap = cv2.VideoCapture(index, backend_flag(backend))
    if not cap.isOpened():
        cap.release()
        return cand
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)

        frame = None
        for _ in range(SETTLE_FRAMES):
            ok, f = cap.read()
            if ok and f is not None and f.size:
                frame = f
        if frame is None:
            return cand

        start = time.perf_counter()
        good = 0
        for _ in range(MEASURE_FRAMES):
            ok, f = cap.read()
            if ok and f is not None and f.size:
                good += 1
                frame = f
        elapsed = time.perf_counter() - start

        cand.works = good > 0
        cand.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cand.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cand.fps = good / elapsed if elapsed > 0 else 0.0
        cand.thumbnail = cv2.resize(frame, THUMB).astype(np.float32)
        # A camera behind a closed privacy shutter opens and delivers frames
        # happily - they are just black. Worth saying out loud rather than
        # letting someone wonder why tracking never works.
        cand.blank = bool(cand.thumbnail.std() < 3.0)
        return cand
    finally:
        cap.release()


def _infer_names(candidates: list[Candidate]) -> None:
    """Name non-DirectShow cameras by matching what they see."""
    named = [c for c in candidates
             if c.backend == "dshow" and c.works and c.thumbnail is not None and not c.blank]
    if not named:
        return

    for cand in candidates:
        if cand.backend == "dshow" or not cand.works or cand.thumbnail is None:
            continue
        if cand.blank:
            continue
        scores = sorted(
            ((float(np.abs(cand.thumbnail - ref.thumbnail).mean()), ref) for ref in named),
            key=lambda item: item[0],
        )
        best_diff, best_ref = scores[0]
        runner_diff = scores[1][0] if len(scores) > 1 else float("inf")
        if best_diff <= SAME_CAMERA_DIFF and (runner_diff - best_diff) >= MATCH_MARGIN:
            cand.name = best_ref.name
            cand.name_source = "inferred"
            log.debug("%s identified as %r (diff %.1f)", cand.spec, cand.name, best_diff)


def scan_cameras(cfg: ArgusConfig, backends: tuple[str, ...] = ("dshow", "msmf")) -> list[Candidate]:
    """Test every (backend, index) pair and return the ones that work."""
    names = _names_via_pygrabber() or []
    candidates: list[Candidate] = []

    for backend in backends:
        misses = 0
        limit = max(len(names), MAX_INDEX) if backend == "dshow" else MAX_INDEX
        for index in range(limit):
            log.debug("probing %s:%d", backend, index)
            cand = _probe(index, backend, cfg.capture.camera)
            if backend == "dshow" and index < len(names):
                cand.name = names[index]
                cand.name_source = "dshow"
            if not cand.works:
                misses += 1
                if misses >= 2 and any(c.works for c in candidates if c.backend == backend):
                    break
                continue
            misses = 0
            candidates.append(cand)

    _infer_names(candidates)
    return candidates


def best_candidate(candidates: list[Candidate], exclude_ir: bool = True) -> Candidate | None:
    """Pick the best working camera: real image, fastest, external preferred."""
    usable = [c for c in candidates if c.works and not c.blank]
    if exclude_ir:
        usable = [c for c in usable if not _matches_any(c.display_name, _IR_PATTERNS)] or usable
    if not usable:
        return None

    def rank(c: Candidate) -> tuple:
        low = c.display_name.lower()
        external = 0 if ("integrated" in low or "built-in" in low or "internal" in low) else 1
        known = 1 if any(k in low for k in ("brio", "logi", "razer", "elgato")) else 0
        # Frame rate dominates, in coarse buckets. A recognisable brand name is
        # no use at 5 fps: the same physical camera often appears under two
        # backends, one of which negotiates an uncompressed format and stalls,
        # and the fast one is the right answer even when it is the one whose
        # name could not be recovered.
        speed = round(min(c.fps, 60.0) / 5.0)
        return (speed, external, known, c.width * c.height)

    return max(usable, key=rank)


# --------------------------------------------------------------------------- #
def run_picker(cfg: ArgusConfig, write: str | None = None) -> int:
    """Show each working camera live and let the operator choose one."""
    import cv2

    print()
    print("  Scanning for cameras (opening each one to see what it really is) ...")
    candidates = scan_cameras(cfg)
    working = [c for c in candidates if c.works]

    if not working:
        print("\n  No working cameras found.")
        print("  - check the camera is plugged in")
        print("  - Windows Settings > Privacy & security > Camera > allow desktop apps")
        print("  - close any app that may be holding it (Teams, Zoom, Camera)\n")
        return 1

    print()
    for cand in candidates:
        print("   " + cand.describe())
    print()
    print("  Use the arrow of your eyes, not the labels: the picker shows each")
    print("  camera live. N = next, ENTER = choose this one, Q = cancel.")
    print()

    window = cfg.ui.window_name + " - choose a camera"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, cfg.ui.preview_width, int(cfg.ui.preview_width * 9 / 16))

    chosen: Candidate | None = None
    position = 0
    try:
        while True:
            cand = working[position]
            cap = cv2.VideoCapture(cand.index, backend_flag(cand.backend))
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.capture.camera.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.capture.camera.height)

            advance = 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok or frame is None or not frame.size:
                        frame = np.zeros((360, 640, 3), dtype=np.uint8)
                    if cfg.capture.camera.mirror:
                        frame = cv2.flip(frame, 1)

                    draw_panel(
                        frame,
                        [
                            f"{cand.display_name}",
                            f"{cand.spec}    {cand.width}x{cand.height}  ~{cand.fps:.0f} fps",
                            "black frames - is a privacy shutter closed?" if cand.blank else "",
                            "",
                            f"camera {position + 1} of {len(working)}",
                        ],
                        origin=(12, 12),
                        title="Is this the camera you want?",
                        min_width=520,
                    )
                    draw_text(frame, "N next     ENTER choose this     Q cancel",
                              (22, frame.shape[0] - 24), 0.6, COLORS["accent"], 2)

                    cv2.imshow(window, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("n"), 83, 9):  # n, right arrow, tab
                        advance = 1
                        break
                    if key in (13, 10):  # enter
                        chosen = cand
                        advance = 0
                        break
                    if key in (ord("q"), 27):
                        advance = -1
                        break
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        advance = -1
                        break
            finally:
                cap.release()

            if chosen is not None or advance == -1:
                break
            position = (position + 1) % len(working)
    finally:
        cv2.destroyAllWindows()
        for _ in range(4):
            cv2.waitKey(1)

    if chosen is None:
        print("  Cancelled - nothing was changed.\n")
        return 1

    print(f"  Selected: {chosen.display_name}  ({chosen.spec})")

    target = ROOT / (write or "configs/default.yaml")
    try:
        _write_camera_choice(target, chosen.spec)
    except Exception as exc:
        print(f"\n  Could not update {target}: {exc}")
        print(f"  Set it by hand:  capture.camera.device: {chosen.spec}\n")
        return 1

    print(f"  Saved to {target}")
    print(f"  Every command will now use this camera. Override once with:")
    print(f"    argus mouse --camera {chosen.spec}")
    print()
    return 0


def _write_camera_choice(path, spec: str) -> None:
    """Rewrite the ``device:`` line under capture.camera, preserving comments.

    Done textually rather than by re-serialising the YAML so that the file's
    comments - which explain every option - survive. A round trip through a
    YAML dumper would silently delete all of them.
    """
    import re

    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if not text.strip():
        path.write_text(
            f"capture:\n  camera:\n    device: {spec}\n", encoding="utf-8"
        )
        return

    pattern = re.compile(r"^(\s*device:\s*)(\S.*?)(\s*)$", re.MULTILINE)
    replaced = False

    def _sub(match):
        nonlocal replaced
        if replaced:
            return match.group(0)
        replaced = True
        return f"{match.group(1)}{spec}"

    new_text = pattern.sub(_sub, text, count=1)
    if not replaced:
        new_text = text.rstrip("\n") + f"\n\ncapture:\n  camera:\n    device: {spec}\n"
    path.write_text(new_text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Cached selection
# --------------------------------------------------------------------------- #
SCAN_CACHE = "data/camera_scan.json"
SCAN_VERSION = 1


def _cache_path():
    from ..config import DATA_DIR

    return DATA_DIR / "camera_scan.json"


def save_scan(candidates: list[Candidate]) -> None:
    """Remember what the scan found, so startup does not repeat it."""
    import json

    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "version": SCAN_VERSION,
                    "cameras": [
                        {
                            "backend": c.backend,
                            "index": c.index,
                            "works": c.works,
                            "blank": c.blank,
                            "fps": round(c.fps, 2),
                            "width": c.width,
                            "height": c.height,
                            "name": c.name,
                            "name_source": c.name_source,
                        }
                        for c in candidates
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover - read-only disk etc.
        log.debug("could not cache camera scan: %s", exc)


def load_scan() -> list[Candidate]:
    """Previously scanned cameras, or an empty list."""
    import json

    path = _cache_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != SCAN_VERSION:
            return []
        return [
            Candidate(
                backend=row["backend"], index=row["index"], works=row["works"],
                width=row.get("width", 0), height=row.get("height", 0),
                fps=row.get("fps", 0.0), name=row.get("name", ""),
                name_source=row.get("name_source", "unknown"),
                blank=row.get("blank", False),
            )
            for row in data.get("cameras", [])
        ]
    except Exception as exc:  # pragma: no cover - corrupt cache is not fatal
        log.debug("ignoring unreadable camera scan cache: %s", exc)
        return []


def clear_scan() -> None:
    path = _cache_path()
    if path.exists():
        path.unlink()


# --------------------------------------------------------------------------- #
# The single selection entry point
# --------------------------------------------------------------------------- #
def select_camera(cfg, rescan: bool = False):
    """Resolve the configured camera into a concrete, backend-locked device.

    An explicit spec is obeyed as given. ``auto`` uses a cached scan in which
    every (backend, index) pair was opened and measured; if no cache exists one
    is built, which takes a few seconds and then never happens again unless the
    hardware changes.
    """
    from .devices import resolve_device

    spec = str(cfg.capture.camera.device).strip()

    if spec.lower() != "auto":
        # Named or pinned: resolve against the scan when we have one, so a name
        # like "brio" can map to the pair that actually is the Brio.
        candidates = load_scan()
        if candidates:
            devices = [c.to_device() for c in candidates if c.works]
            try:
                return resolve_device(
                    spec, devices, exclude_ir=cfg.capture.camera.exclude_ir
                )
            except Exception:
                pass
        return resolve_device(
            spec,
            backend=cfg.capture.camera.backend,
            exclude_ir=cfg.capture.camera.exclude_ir,
        )

    candidates = [] if rescan else load_scan()
    if not candidates:
        log.info(
            "no cached camera scan yet - opening each camera once to see what it "
            "really is. This takes a few seconds and is remembered afterwards."
        )
        candidates = scan_cameras(cfg)
        save_scan(candidates)

    best = best_candidate(candidates, exclude_ir=cfg.capture.camera.exclude_ir)
    if best is None:
        # Nothing in the cache works any more - the camera may have been
        # unplugged since. Re-scan before giving up.
        candidates = scan_cameras(cfg)
        save_scan(candidates)
        best = best_candidate(candidates, exclude_ir=cfg.capture.camera.exclude_ir)
    if best is None:
        from .devices import DeviceError

        raise DeviceError(
            "No working camera found.\n"
            "  - check the camera is plugged in\n"
            "  - Windows Settings > Privacy & security > Camera > allow desktop apps\n"
            "  - close any app holding it (Teams, Zoom, Camera)\n"
            "  - choose one manually:  argus cameras --pick"
        )

    log.info("auto-selected %s (%s, %.0f fps measured)",
             best.display_name, best.spec, best.fps)
    return best.to_device()
