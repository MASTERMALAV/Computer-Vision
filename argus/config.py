"""Typed, layered configuration for ARGUS.

Configuration is a tree of dataclasses. It is loaded in layers, each overriding
the previous:

    1. dataclass defaults (this file - always valid, always complete)
    2. a YAML file (``configs/default.yaml`` unless overridden)
    3. explicit dot-path overrides (``--set camera.width=1920``)

Unknown keys are a hard error rather than a silent no-op: a typo in a config key
is one of the easiest ways to spend an hour debugging a pipeline that is quietly
running with defaults.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

from .logsetup import get_logger

log = get_logger("config")

# Repository root: <root>/argus/config.py -> <root>
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
CONFIGS_DIR = ROOT / "configs"


class ConfigError(ValueError):
    """Raised when a configuration file or override is malformed."""


# Keys that were valid in an earlier version. They are accepted and ignored
# rather than rejected: a config file written against an older release was
# correct when it was written, and failing to load it punishes the user for a
# change they did not make. Each entry says what replaced it.
DEPRECATED_KEYS: dict[str, str] = {
    "gestures.scroll_middle_extended": (
        "scrolling is now a held thumb-to-middle pinch rather than a two-finger "
        "pose, so this threshold is no longer read. Remove it, or re-run "
        "`argus calibrate --write`. See gestures.scroll_dwell_s."
    ),
    "gestures.scroll_debounce_frames": (
        "the two-finger scroll pose it debounced no longer exists. Remove it, "
        "or re-run `argus calibrate --write`."
    ),
}


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #
@dataclass
class CameraConfig:
    """How to open a single camera.

    ``device`` accepts an integer index, a case-insensitive substring of the
    device's friendly name (e.g. ``"brio"``), or ``"auto"`` to pick the highest
    scoring device found by :mod:`argus.capture.devices`.
    """

    device: str = "auto"
    backend: str = "auto"  # auto | dshow | msmf | any
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str = "MJPG"  # MJPG keeps USB bandwidth sane at 720p30+
    mirror: bool = True  # selfie view - flip horizontally for natural interaction
    autofocus: bool | None = None  # None = leave the driver default alone
    autoexposure: bool | None = None
    exclude_ir: bool = True  # skip Windows Hello IR cameras during auto-select

    # Threaded reader tuning
    queue_size: int = 1  # 1 == always process the freshest frame, drop the rest
    reopen_on_failure: bool = True
    max_consecutive_failures: int = 30
    open_timeout_s: float = 10.0


@dataclass
class CaptureConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    # Downscale sent to detectors; capture stays at full res for crops/preview.
    detect_width: int = 640


# --------------------------------------------------------------------------- #
# Face
# --------------------------------------------------------------------------- #
@dataclass
class FaceDetectorConfig:
    model: str = "scrfd_500m"  # scrfd_500m | scrfd_2_5g | scrfd_10g
    conf_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_faces: int = 8
    min_face_px: int = 40  # reject faces too small to embed reliably
    input_size: int = 640  # SCRFD square input; must be a multiple of 32


@dataclass
class FaceRecognizerConfig:
    model: str = "arcface_mbf"  # arcface_mbf (fast) | arcface_r50 (accurate)
    # Cosine similarity above which a probe is accepted as a gallery identity.
    match_threshold: float = 0.42
    # Margin the best match must beat the runner-up by, to avoid confident
    # confusion between two enrolled people who look alike.
    margin_threshold: float = 0.05
    embed_every_n_frames: int = 5  # re-embed a tracked face only occasionally
    # Number of consistent observations before an identity is "locked".
    lock_votes: int = 3
    # Consecutive contradicting votes before an identity is released.
    unlock_votes: int = 5


@dataclass
class FaceTrackerConfig:
    iou_threshold: float = 0.3
    max_age_frames: int = 15  # keep a lost track alive this long before dropping
    min_hits: int = 2  # observations before a track is reported as confirmed


@dataclass
class FaceConfig:
    enabled: bool = True
    detector: FaceDetectorConfig = field(default_factory=FaceDetectorConfig)
    recognizer: FaceRecognizerConfig = field(default_factory=FaceRecognizerConfig)
    tracker: FaceTrackerConfig = field(default_factory=FaceTrackerConfig)
    gallery_path: str = "data/gallery.npz"
    detect_every_n_frames: int = 2  # track in between full detections


# --------------------------------------------------------------------------- #
# Hands
# --------------------------------------------------------------------------- #
@dataclass
class HandsConfig:
    enabled: bool = True
    # One hand by default: the virtual mouse only needs one, and each extra hand
    # costs a full landmark-model pass. Raise to 2 for two-handed gestures.
    max_hands: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1  # 0 = lite/faster, 1 = full/accurate
    # Static gesture classification
    gesture_threshold: float = 0.6
    smoothing_window: int = 5  # majority vote over this many frames
    # Landmark trajectory history kept per hand, for dynamic gestures
    history_frames: int = 45


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
@dataclass
class AudioConfig:
    enabled: bool = False  # opt-in: Phase 4
    device: str = "auto"
    sample_rate: int = 16000
    block_ms: int = 32
    vad_threshold: float = 0.5
    asr_model: str = "base.en"
    asr_compute_type: str = "int8"
    speaker_verification: bool = False


# --------------------------------------------------------------------------- #
# Gestures
# --------------------------------------------------------------------------- #
@dataclass
class GestureConfig:
    """Gesture geometry, in hand-scale units (see argus/hands/landmarks.py).

    Every distance is divided by the user's own wrist-to-knuckle span, so these
    thresholds hold at any distance from the camera. Measure values fitted to
    your own hand with: argus calibrate
    """

    pinch_close: float = 0.34
    pinch_open: float = 0.50
    pinch_approach: float = 0.62
    finger_extended: float = 1.05
    finger_curled: float = 0.85
    debounce_frames: int = 2
    clutch_debounce_frames: int = 3
    drag_dwell_s: float = 0.35
    click_cooldown_s: float = 0.25
    double_click_s: float = 0.40
    motion_gate_speed: float = 2.6
    # Hold the thumb-middle pinch this long and it becomes a scroll, exactly as
    # holding thumb-index becomes a drag.
    scroll_dwell_s: float = 0.30


# --------------------------------------------------------------------------- #
# Cursor control
# --------------------------------------------------------------------------- #
@dataclass
class GainCurveConfig:
    slow_speed: float = 0.35
    fast_speed: float = 3.0
    min_gain: float = 0.55
    max_gain: float = 3.4
    pixels_per_unit: float = 900.0


@dataclass
class ScrollSettings:
    units_per_notch: float = 0.16
    dead_zone: float = 0.008
    max_notches_per_frame: int = 4
    invert: bool = False
    # Slow movement is precise; fast movement covers ground.
    boost_start: float = 0.45
    boost_full: float = 3.20
    max_boost: float = 3.50
    # A flick keeps scrolling after release and decays, like a phone. A slow,
    # deliberate release stops dead, which is what you want when positioning.
    momentum: bool = True
    min_flick_speed: float = 1.10
    momentum_tau_s: float = 0.38
    max_coast_s: float = 1.60
    coast_stop_speed: float = 0.30


@dataclass
class SmoothingConfig:
    min_cutoff: float = 1.0
    beta: float = 0.02
    d_cutoff: float = 1.0


@dataclass
class ControlConfig:
    """Cursor control.

    ``armed`` is the safety interlock. While false the whole pipeline runs and
    the HUD shows exactly what it would do, but no input reaches the operating
    system. Nothing arms itself automatically.
    """

    enabled: bool = True
    armed: bool = False
    source: str = "palm"  # palm | index_mcp | index_tip
    dead_zone: float = 0.0016
    freeze_timeout_s: float = 0.45
    max_delta_units: float = 0.30
    max_jump_px: float = 1400.0
    gain: GainCurveConfig = field(default_factory=GainCurveConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    scroll: ScrollSettings = field(default_factory=ScrollSettings)
    # Which hand may drive the cursor: Left | Right | any
    hand: str = "any"


# --------------------------------------------------------------------------- #
# Security
# --------------------------------------------------------------------------- #
@dataclass
class SecurityConfig:
    """Identity gating.

    Cursor motion rides an authenticated *session* rather than a per-frame face
    match: recognition at 30 fps is unaffordable on this CPU, and gating motion
    frame-by-frame would freeze the cursor every time the operator glanced down
    at the keyboard. Destructive actions still demand a fresh match.
    """

    require_identity: bool = False
    operator: str = ""
    session_timeout_s: float = 120.0
    reverify_interval_s: float = 15.0
    fresh_match_within_s: float = 5.0
    confirm_countdown_s: float = 5.0


# --------------------------------------------------------------------------- #
# Runtime / UI
# --------------------------------------------------------------------------- #
@dataclass
class UIConfig:
    show_window: bool = True
    window_name: str = "ARGUS"
    draw_face: bool = True
    draw_hands: bool = True
    draw_hud: bool = True
    draw_fps: bool = True
    preview_width: int = 1280


@dataclass
class RuntimeConfig:
    # Worker threads handed to ONNX Runtime. The box has 4 physical cores and
    # also needs headroom for capture + MediaPipe, so we do not take all of them.
    onnx_intra_threads: int = 2
    onnx_inter_threads: int = 1
    log_level: str = "INFO"
    metrics_window: int = 120  # frames used for rolling FPS/latency stats
    profile: str = "balanced"  # fast | balanced | accurate


@dataclass
class ArgusConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    hands: HandsConfig = field(default_factory=HandsConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # ---------------- loading ---------------- #
    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        overrides: list[str] | None = None,
        apply_profile: bool = True,
    ) -> "ArgusConfig":
        """Build a config from defaults + optional YAML + optional dot overrides.

        Profile expansion happens *before* the YAML/CLI layers are applied, so an
        explicit ``face.detector.input_size`` in a config file always wins over
        the value the profile would have chosen.
        """
        cfg = cls()

        raw: dict[str, Any] = {}
        if path is not None:
            p = Path(path)
            if not p.is_absolute():
                p = ROOT / p
            if not p.exists():
                raise ConfigError(f"Config file not found: {p}")
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if not isinstance(raw, dict):
                raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}")

        # Resolve the requested profile first, then let explicit settings override it.
        if apply_profile:
            requested = str(
                _peek(raw, "runtime", "profile")
                or _peek_overrides(overrides, "runtime.profile")
                or cfg.runtime.profile
            )
            cfg.runtime.profile = requested
            cfg.apply_profile()

        if raw:
            cfg = _merge_into(cfg, raw, prefix="")
        for item in overrides or []:
            cfg = _apply_override(cfg, item)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def dump_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, indent=2)

    def validate(self) -> None:
        cam = self.capture.camera
        if cam.width <= 0 or cam.height <= 0:
            raise ConfigError("camera.width/height must be positive")
        if cam.backend not in {"auto", "dshow", "msmf", "any"}:
            raise ConfigError(f"camera.backend must be auto|dshow|msmf|any, got {cam.backend!r}")
        if len(cam.fourcc) != 4:
            raise ConfigError(f"camera.fourcc must be exactly 4 characters, got {cam.fourcc!r}")
        if self.face.detector.input_size % 32 != 0:
            raise ConfigError("face.detector.input_size must be a multiple of 32 (SCRFD stride)")
        if not 0.0 < self.face.recognizer.match_threshold < 1.0:
            raise ConfigError("face.recognizer.match_threshold must be in (0, 1)")
        if self.runtime.profile not in {"fast", "balanced", "accurate"}:
            raise ConfigError("runtime.profile must be fast|balanced|accurate")
        if self.hands.max_hands < 1:
            raise ConfigError("hands.max_hands must be >= 1")
        if self.hands.model_complexity not in (0, 1):
            raise ConfigError("hands.model_complexity must be 0 or 1")
        if self.face.detect_every_n_frames < 1:
            raise ConfigError("face.detect_every_n_frames must be >= 1")
        g = self.gestures
        if not g.pinch_close < g.pinch_open:
            raise ConfigError(
                "gestures.pinch_close must be smaller than gestures.pinch_open; "
                "without that gap the pinch has no hysteresis and will chatter"
            )
        if g.pinch_approach < g.pinch_open:
            raise ConfigError(
                "gestures.pinch_approach must be >= gestures.pinch_open so the "
                "cursor freezes before the click registers"
            )
        if not g.finger_curled < g.finger_extended:
            raise ConfigError("gestures.finger_curled must be < gestures.finger_extended")
        if self.control.source not in {"palm", "index_mcp", "index_tip"}:
            raise ConfigError("control.source must be palm|index_mcp|index_tip")
        if self.control.hand not in {"Left", "Right", "any"}:
            raise ConfigError("control.hand must be Left|Right|any")
        if self.control.scroll.units_per_notch <= 0:
            raise ConfigError("control.scroll.units_per_notch must be positive")
        if self.gestures.scroll_dwell_s <= 0:
            raise ConfigError("gestures.scroll_dwell_s must be positive")
        gc = self.control.gain
        if gc.slow_speed >= gc.fast_speed:
            raise ConfigError("control.gain.slow_speed must be < fast_speed")
        if gc.min_gain > gc.max_gain:
            raise ConfigError("control.gain.min_gain must be <= max_gain")

    def apply_profile(self) -> "ArgusConfig":
        """Re-tune speed/accuracy knobs from ``runtime.profile``.

        A one-word profile switch should move every knob that matters, rather
        than making the user hand-edit six fields that have to stay consistent.
        """
        p = self.runtime.profile
        if p == "fast":
            self.face.detector.model = "scrfd_500m"
            self.face.detector.input_size = 448
            self.face.recognizer.model = "arcface_mbf"
            self.face.detect_every_n_frames = 3
            self.hands.model_complexity = 0
            self.capture.detect_width = 512
        elif p == "balanced":
            self.face.detector.model = "scrfd_500m"
            self.face.detector.input_size = 640
            self.face.recognizer.model = "arcface_mbf"
            self.face.detect_every_n_frames = 2
            self.hands.model_complexity = 1
            self.capture.detect_width = 640
        elif p == "accurate":
            self.face.detector.model = "scrfd_10g"
            self.face.detector.input_size = 640
            self.face.recognizer.model = "arcface_r50"
            self.face.detect_every_n_frames = 1
            self.hands.model_complexity = 1
            self.capture.detect_width = 800
        return self


# --------------------------------------------------------------------------- #
# Merge / override machinery
# --------------------------------------------------------------------------- #
def _peek(raw: dict[str, Any], section: str, key: str) -> Any:
    """Read ``raw[section][key]`` without failing on missing/odd structure."""
    sect = raw.get(section)
    if isinstance(sect, dict):
        return sect.get(key)
    return None


def _peek_overrides(overrides: list[str] | None, dotted: str) -> Any:
    """Last value assigned to ``dotted`` in a list of CLI overrides, if any."""
    found = None
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if sep and key.strip() == dotted:
            found = value.strip()
    return found


def _coerce(value: Any, target_type: Any, dotted: str) -> Any:
    """Coerce a YAML/CLI scalar into the field's declared type."""
    # Unwrap Optional[X] / X | None
    origin = get_origin(target_type)
    if origin is not None:
        args = [a for a in get_args(target_type) if a is not type(None)]
        if value is None:
            return None
        if len(args) == 1:
            return _coerce(value, args[0], dotted)
        return value  # unions we do not need to be clever about

    if target_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            low = value.strip().lower()
            if low in {"true", "yes", "on", "1"}:
                return True
            if low in {"false", "no", "off", "0"}:
                return False
            if low in {"none", "null"}:
                return None
        raise ConfigError(f"{dotted}: cannot read {value!r} as a boolean")
    if target_type is int:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{dotted}: expected an integer, got {value!r}") from exc
    if target_type is float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{dotted}: expected a number, got {value!r}") from exc
    if target_type is str:
        return str(value)
    return value


def _merge_into(node: Any, raw: dict[str, Any], prefix: str) -> Any:
    """Recursively overlay ``raw`` onto a dataclass instance, strictly."""
    hints = get_type_hints(type(node))
    known = {f.name for f in fields(node)}
    for key, value in raw.items():
        dotted = f"{prefix}{key}"
        if key not in known:
            if dotted in DEPRECATED_KEYS:
                log.warning("ignoring retired setting %r: %s", dotted, DEPRECATED_KEYS[dotted])
                continue
            valid = ", ".join(sorted(known))
            raise ConfigError(f"Unknown config key {dotted!r}. Valid keys here: {valid}")
        current = getattr(node, key)
        if is_dataclass(current) and not isinstance(current, type):
            if not isinstance(value, dict):
                raise ConfigError(f"{dotted}: expected a mapping, got {type(value).__name__}")
            setattr(node, key, _merge_into(current, value, prefix=f"{dotted}."))
        else:
            setattr(node, key, _coerce(value, hints[key], dotted))
    return node


def _apply_override(cfg: ArgusConfig, item: str) -> ArgusConfig:
    """Apply a single ``a.b.c=value`` override string."""
    if "=" not in item:
        raise ConfigError(f"Override {item!r} must look like 'section.key=value'")
    dotted, _, value = item.partition("=")
    parts = [p for p in dotted.strip().split(".") if p]
    if not parts:
        raise ConfigError(f"Override {item!r} has an empty key")

    node: Any = cfg
    for i, part in enumerate(parts[:-1]):
        if not hasattr(node, part):
            raise ConfigError(f"Unknown config section {'.'.join(parts[: i + 1])!r}")
        node = getattr(node, part)
        if not is_dataclass(node):
            raise ConfigError(f"{'.'.join(parts[: i + 1])!r} is a value, not a section")
    leaf = parts[-1]
    if not hasattr(node, leaf):
        if dotted.strip() in DEPRECATED_KEYS:
            log.warning(
                "ignoring retired setting %r: %s",
                dotted.strip(), DEPRECATED_KEYS[dotted.strip()],
            )
            return cfg
        raise ConfigError(f"Unknown config key {dotted!r}")
    hints = get_type_hints(type(node))
    setattr(node, leaf, _coerce(yaml.safe_load(value), hints[leaf], dotted))
    return cfg
