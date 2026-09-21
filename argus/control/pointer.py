"""The pointer engine: hand motion in, cursor motion out.

Design decisions that matter, and why:

**Relative, not absolute.** The cursor moves by the *displacement* of the hand,
like a mouse, not to a position the hand points at. A single 2D webcam cannot
recover the ray from a fingertip to a screen accurately - the depth estimate is
weak and the camera is not at the user's eye - and absolute mapping would tie
screen resolution to the small patch of air the hand can comfortably reach.

**The clutch makes the reachable area unbounded.** Because the mapping is
relative, the hand can be re-centred at any time exactly as a mouse is lifted
and repositioned. This is what lets the hand stay low near the desk instead of
held up in the air.

**Palm, not fingertip, drives motion.** The palm centroid barely moves when
fingers flex, so pinching to click does not drag the cursor off the target. A
fingertip would - that click-induced drift is the single largest source of
error in mid-air pointing, and choosing a stable landmark removes it at the
source rather than filtering it afterwards.

**Displacement is measured in hand-scale units.** Dividing by the user's own
wrist-to-knuckle span makes the mapping depth-invariant: one hand-width of
motion moves the cursor the same distance whether the hand is 30 cm or 60 cm
from the camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..gestures.fsm import GestureEvent, GestureState, GestureType
from ..hands.landmarks import INDEX_MCP, INDEX_TIP, Hand
from ..logsetup import get_logger
from .filters import GainConfig, OneEuroConfig, OneEuroFilter, PositionHistory, smoothstep
from .injector import MouseInjector

log = get_logger("control.pointer")


@dataclass
class ScrollConfig:
    """Two-finger... no: held-middle-pinch scroll.

    Two things make wheel scrolling from a hand feel wrong if you ignore them.

    **Travel runs out.** A hand has perhaps 15 cm of comfortable vertical range,
    which is a few hundred pixels of document. A mouse wheel has no such limit.
    So movement is boosted with speed, and a quick flick *coasts* after release
    the way a phone does, decoupling distance scrolled from distance moved.

    **Notches are discrete.** Emitting only whole notches per frame throws away
    the remainder, so slow movement scrolls nothing at all. The remainder is
    accumulated across frames instead, and a slow drag produces a steady trickle.
    """

    # Hand travel, in hand-scale units, for one wheel notch at unity boost.
    units_per_notch: float = 0.16
    # Ignore movement below this, so a resting hand never creeps.
    dead_zone: float = 0.008
    # Cap per frame, so one landmark glitch cannot scroll to the end of a page.
    max_notches_per_frame: int = 4
    # Natural scrolling: move the hand up and the content follows the hand.
    invert: bool = False

    # --- speed boost: slow is precise, fast covers ground ---------------- #
    boost_start: float = 0.45  # hand-scale units/sec where boost begins
    boost_full: float = 3.20  # ... and where it saturates
    max_boost: float = 3.50

    # --- momentum ------------------------------------------------------- #
    momentum: bool = True
    # Release slower than this and the scroll stops dead, which is what you
    # want when positioning carefully. Faster, and it coasts.
    min_flick_speed: float = 1.10
    momentum_tau_s: float = 0.38  # exponential decay constant
    max_coast_s: float = 1.60  # hard stop, so it can never run away
    coast_stop_speed: float = 0.30


class ScrollEngine:
    """Turns vertical hand movement into mouse wheel notches."""

    def __init__(self, config: ScrollConfig | None = None) -> None:
        self.config = config or ScrollConfig()
        self._accumulator = 0.0
        self._prev_y: float | None = None
        self._prev_time: float | None = None
        self._velocity = 0.0  # smoothed, hand-scale units per second
        self._coasting = False
        self._coast_until = 0.0
        self._coast_time = 0.0

    def reset(self) -> None:
        """Stop immediately, cancelling any momentum."""
        self._accumulator = 0.0
        self._prev_y = None
        self._prev_time = None
        self._velocity = 0.0
        self._coasting = False

    @property
    def coasting(self) -> bool:
        return self._coasting

    @property
    def velocity(self) -> float:
        return self._velocity

    # ------------------------------------------------------------------ #
    def _boost(self, speed: float) -> float:
        cfg = self.config
        t = smoothstep(cfg.boost_start, cfg.boost_full, abs(speed))
        return 1.0 + (cfg.max_boost - 1.0) * t

    def _emit(self, units: float) -> int:
        """Accumulate fractional notches and release whole ones."""
        cfg = self.config
        # Screen y grows downward; a wheel notch is positive when scrolling up.
        direction = 1.0 if cfg.invert else -1.0
        self._accumulator += direction * units / max(cfg.units_per_notch, 1e-6)

        notches = int(self._accumulator)
        if not notches:
            return 0
        self._accumulator -= notches
        limit = cfg.max_notches_per_frame
        return max(-limit, min(limit, notches))

    def update(self, palm_y: float, hand_scale: float, active: bool, now: float) -> int:
        """Return whole wheel notches to emit this frame.

        ``active`` is true while the scroll gesture is held. When it goes false
        the engine may keep emitting for a moment, coasting on the momentum of
        the release.
        """
        cfg = self.config
        if hand_scale <= 1e-6:
            active = False

        if active:
            self._coasting = False
            if self._prev_y is None or self._prev_time is None:
                self._prev_y, self._prev_time = palm_y, now
                return 0

            dt = now - self._prev_time
            if dt <= 1e-6 or dt > 0.25:
                # A stall or a repeated timestamp: re-anchor rather than
                # inventing a huge velocity from a long gap.
                self._prev_y, self._prev_time = palm_y, now
                return 0

            delta_units = (palm_y - self._prev_y) / hand_scale
            self._prev_y, self._prev_time = palm_y, now

            instant = delta_units / dt
            self._velocity = 0.6 * self._velocity + 0.4 * instant
            if abs(delta_units) < cfg.dead_zone:
                return 0
            return self._emit(delta_units * self._boost(instant))

        # ---- released ------------------------------------------------- #
        if self._prev_y is not None:
            self._prev_y = None
            self._prev_time = None
            if cfg.momentum and abs(self._velocity) >= cfg.min_flick_speed:
                self._coasting = True
                self._coast_until = now + cfg.max_coast_s
                self._coast_time = now
            else:
                self.reset()

        if not self._coasting:
            return 0

        dt = now - self._coast_time
        self._coast_time = now
        if dt <= 0.0 or dt > 0.25:
            dt = 1.0 / 30.0

        self._velocity *= float(np.exp(-dt / max(cfg.momentum_tau_s, 1e-6)))
        if abs(self._velocity) < cfg.coast_stop_speed or now >= self._coast_until:
            self.reset()
            return 0
        return self._emit(self._velocity * dt * self._boost(self._velocity))


@dataclass
class PointerConfig:
    gain: GainConfig = field(default_factory=GainConfig)
    euro: OneEuroConfig = field(default_factory=OneEuroConfig)
    scroll: "ScrollConfig" = field(default_factory=lambda: ScrollConfig())

    # palm | index_mcp | index_tip
    # 'palm' is the default because it is immune to click-induced drift.
    source: str = "palm"

    # Hand movement below this (hand-scale units/frame) is treated as tremor and
    # ignored, so a perfectly still hand produces a perfectly still cursor.
    dead_zone: float = 0.0016

    # How long the cursor stays locked after the fingers start closing.
    freeze_timeout_s: float = 0.45
    # A drag needs to move, so the freeze is dropped as soon as one starts.
    freeze_during_drag: bool = False

    # Physical plausibility limit on the INPUT: a hand cannot move more than
    # this fraction of its own width between two frames at 30 fps. Clamping here
    # rather than on the output preserves the gain curve - an output clamp low
    # enough to catch glitches also flattens every fast deliberate movement.
    max_delta_units: float = 0.30

    # Backstop on the OUTPUT for genuine landmark glitches. Deliberately set
    # above what the gain curve can legitimately produce
    # (max_delta_units * pixels_per_unit * max_gain), so it never fires in
    # normal use.
    max_jump_px: float = 1400.0


@dataclass
class PointerState:
    engaged: bool = False
    frozen: bool = False
    position: tuple[float, float] = (0.0, 0.0)
    hand_speed: float = 0.0
    gain: float = 1.0
    moved_px: float = 0.0
    monitor: int = 0
    scroll_notches: int = 0
    coasting: bool = False
    suppressed: bool = False


class PointerEngine:
    """Maps hand motion onto the system cursor."""

    def __init__(
        self,
        injector: MouseInjector,
        config: PointerConfig | None = None,
    ) -> None:
        self.injector = injector
        self.config = config or PointerConfig()

        self._filter = OneEuroFilter(self.config.euro)
        self.scroll = ScrollEngine(self.config.scroll)
        self._history = PositionHistory(seconds=0.5)
        self._prev_source: np.ndarray | None = None
        self._prev_time: float | None = None

        self._cursor = np.array(injector.desktop.primary.center, dtype=np.float64)
        self._frozen_until: float = 0.0
        self._frozen = False

        self.state = PointerState(position=(float(self._cursor[0]), float(self._cursor[1])))
        self.total_moved_px = 0.0

    # ------------------------------------------------------------------ #
    def _source_point(self, hand: Hand) -> np.ndarray:
        src = self.config.source
        if src == "index_tip":
            return hand.pixels[INDEX_TIP].astype(np.float64)
        if src == "index_mcp":
            return hand.pixels[INDEX_MCP].astype(np.float64)
        return np.asarray(hand.palm_center, dtype=np.float64)

    def sync_from_system(self) -> None:
        """Adopt the OS cursor position as our own.

        Called whenever the clutch engages, so that if the user moved the real
        mouse while disengaged, the cursor continues from where it actually is
        instead of teleporting back to where this engine last left it.
        """
        x, y = self.injector.position()
        self._cursor = np.array([float(x), float(y)], dtype=np.float64)
        self.state.position = (float(x), float(y))

    # ------------------------------------------------------------------ #
    def _handle_events(self, events: list[GestureEvent], now: float) -> None:
        for event in events:
            if event.type is GestureType.CLUTCH_ENGAGE:
                # Re-anchor: the hand's current position becomes the new origin,
                # so engaging never jumps the cursor.
                self._prev_source = None
                self._filter.reset()
                self.sync_from_system()
            elif event.type is GestureType.CLUTCH_RELEASE:
                self._prev_source = None
                self._unfreeze()
            elif event.type is GestureType.PINCH_APPROACH:
                self._freeze(now)
            elif event.type in (
                GestureType.PINCH_ABORT,
                GestureType.CLICK,
                GestureType.DOUBLE_CLICK,
                GestureType.RIGHT_CLICK,
            ):
                self._unfreeze()
            elif event.type is GestureType.DRAG_START:
                if not self.config.freeze_during_drag:
                    self._unfreeze()
            elif event.type is GestureType.DRAG_END:
                self._unfreeze()

    def _freeze(self, now: float) -> None:
        self._frozen = True
        self._frozen_until = now + self.config.freeze_timeout_s

    def _unfreeze(self) -> None:
        self._frozen = False
        self._frozen_until = 0.0

    # ------------------------------------------------------------------ #
    def update(
        self,
        hand: Hand | None,
        gesture_state: GestureState,
        events: list[GestureEvent],
        now: float,
        suppressed: bool = False,
    ) -> PointerState:
        """Advance the cursor by one frame and return the resulting state.

        ``suppressed`` holds the cursor still without losing track of where the
        hand is - the same mechanism as the click freeze. Skipping the update
        entirely would leave a stale anchor, and the first frame after
        suppression lifted would compute its displacement against a position
        from seconds ago and fling the cursor.
        """
        self._handle_events(events, now)

        # A freeze must expire on its own: if a pinch is begun and simply held
        # halfway, the cursor has to come back to life rather than stay stuck.
        if self._frozen and now >= self._frozen_until:
            self._unfreeze()

        self.state.engaged = gesture_state.clutch_engaged
        self.state.frozen = self._frozen or suppressed
        self.state.suppressed = suppressed
        self.state.hand_speed = gesture_state.hand_speed
        self.state.moved_px = 0.0

        self.state.scroll_notches = 0
        scrolling = gesture_state.mode == "scroll"
        # volume / brightness knobs also take the hand out of cursor duty.
        acting = gesture_state.mode not in ("point", "scroll")

        # The scroll engine is driven every frame, not only while the gesture is
        # held, because a flick keeps emitting for a moment after release.
        notches = self.scroll.update(
            float(hand.palm_center[1]) if hand is not None else 0.0,
            hand.scale if hand is not None else 0.0,
            active=scrolling and hand is not None,
            now=now,
        )
        if notches:
            self.injector.scroll(notches)
            self.state.scroll_notches = notches
        self.state.coasting = self.scroll.coasting

        if hand is None or not gesture_state.clutch_engaged:
            self._prev_source = None
            self._prev_time = None
            return self.state

        # Scrolling drives the wheel instead of the cursor. The pointer is
        # deliberately left where it is: scrolling a window should not also move
        # the pointer out of it.
        if scrolling or acting:
            self._prev_source = None
            self._prev_time = None
            return self.state

        raw = self._source_point(hand)
        smoothed = np.asarray(self._filter(raw, now), dtype=np.float64)
        self._history.add(smoothed, now)

        if self._prev_source is None or self._prev_time is None:
            self._prev_source = smoothed
            self._prev_time = now
            return self.state

        dt = now - self._prev_time
        if dt <= 1e-6 or dt > 0.25:
            self._prev_source = smoothed
            self._prev_time = now
            return self.state

        # Displacement in hand-scale units: this is what makes the mapping
        # independent of how far the hand is from the camera.
        scale = max(hand.scale, 1e-6)
        delta_units = (smoothed - self._prev_source) / scale
        self._prev_source = smoothed
        self._prev_time = now

        magnitude = float(np.linalg.norm(delta_units))
        if magnitude < self.config.dead_zone:
            return self.state

        # Reject physically impossible jumps at the input, before they are
        # amplified by gain.
        if magnitude > self.config.max_delta_units:
            delta_units *= self.config.max_delta_units / magnitude
            magnitude = self.config.max_delta_units
            log.debug("clamped an implausible hand displacement")

        speed = magnitude / dt
        gain = self.config.gain.gain(speed)
        self.state.gain = gain

        if self._frozen or suppressed:
            # Position is still tracked above so that motion during the freeze
            # is discarded rather than accumulated and released in a lump.
            return self.state

        move = delta_units * self.config.gain.pixels_per_unit * gain

        jump = float(np.linalg.norm(move))
        if jump > self.config.max_jump_px:
            move *= self.config.max_jump_px / jump
            log.debug("clamped a %.0f px cursor jump", jump)

        self._cursor = np.asarray(
            self.injector.desktop.clamp(self._cursor[0] + move[0], self._cursor[1] + move[1]),
            dtype=np.float64,
        )
        self.injector.move_to(self._cursor[0], self._cursor[1])

        moved = float(np.linalg.norm(move))
        self.state.moved_px = moved
        self.total_moved_px += moved
        self.state.position = (float(self._cursor[0]), float(self._cursor[1]))
        self.state.monitor = self.injector.desktop.monitor_at(*self._cursor).index
        return self.state

    def reset(self) -> None:
        self._filter.reset()
        self._history.clear()
        self._prev_source = None
        self._prev_time = None
        self._unfreeze()
