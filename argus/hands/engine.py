"""MediaPipe HandLandmarker wrapper.

Running mode is VIDEO, deliberately:

* ``IMAGE`` re-runs the expensive BlazePalm detector on every single frame and
  throws away the tracking optimisation - roughly 3x the cost for no benefit.
* ``LIVE_STREAM`` is asynchronous: you push a frame and a callback fires later.
  It drops frames internally and gives no back-pressure, so the result you get
  belongs to an unknown earlier frame. We already own a capture thread with
  latest-frame semantics, so adding a second, opaque queue would make latency
  unmeasurable - exactly what this system must not do.
* ``VIDEO`` is synchronous and deterministic (the result belongs to the frame
  you just passed, so it can be timed honestly) while still using the
  detector-then-tracker path: palm detection only re-runs when tracking is lost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..config import HandsConfig
from ..logsetup import get_logger
from ..models.registry import ensure
from .landmarks import Hand, hand_from_mediapipe

log = get_logger("hands.engine")


@dataclass
class HandsResult:
    """Everything the landmarker produced for one frame."""

    hands: list[Hand]
    frame_index: int
    timestamp: float
    inference_ms: float
    detector_ran: bool = False

    @property
    def count(self) -> int:
        return len(self.hands)

    def by_handedness(self, label: str) -> Hand | None:
        for hand in self.hands:
            if hand.handedness == label:
                return hand
        return None

    @property
    def primary(self) -> Hand | None:
        """The most usable hand: largest in frame, which is the one nearest the camera."""
        if not self.hands:
            return None
        return max(self.hands, key=lambda h: h.scale)


class HandEngine:
    """Detects hand landmarks in a stream of frames."""

    def __init__(self, config: HandsConfig, mirrored_input: bool = True) -> None:
        self.config = config
        self.mirrored_input = mirrored_input
        self._landmarker = None
        self._last_timestamp_ms = -1
        self._frames = 0
        self._empty_streak = 0

    # ------------------------------------------------------------------ #
    def start(self) -> "HandEngine":
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        model_path = ensure("hand_landmarker")
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=self.config.max_hands,
            min_hand_detection_confidence=self.config.min_detection_confidence,
            # Presence and tracking confidence govern how readily the tracker
            # gives up and pays for palm detection again. Keeping them equal to
            # the tracking threshold avoids a detector re-run on every small
            # dip in confidence, which is the main source of frame-time spikes.
            min_hand_presence_confidence=self.config.min_tracking_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        t0 = time.perf_counter()
        self._landmarker = HandLandmarker.create_from_options(options)
        log.info(
            "hand landmarker ready in %.2f s (max_hands=%d, VIDEO mode)",
            time.perf_counter() - t0,
            self.config.max_hands,
        )
        return self

    # ------------------------------------------------------------------ #
    def _next_timestamp_ms(self, monotonic_ms: int) -> int:
        """MediaPipe requires strictly increasing integer millisecond stamps.

        Frames can arrive within the same millisecond (and after a camera
        reopen the clock can even appear to step back), either of which makes
        MediaPipe raise. Forcing monotonicity here keeps a real timing source
        while guaranteeing the contract.
        """
        ts = int(monotonic_ms)
        if ts <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = ts
        return ts

    def process(
        self,
        image_bgr: np.ndarray,
        monotonic_ms: int,
        frame_index: int = 0,
    ) -> HandsResult:
        """Run hand landmarking on one BGR frame."""
        if self._landmarker is None:
            raise RuntimeError("HandEngine.start() must be called before process()")

        import mediapipe as mp

        height, width = image_bgr.shape[:2]

        start = time.perf_counter()
        # MediaPipe wants RGB. cvtColor is a cheap native call; slicing with
        # [..., ::-1] would hand it a non-contiguous view it must copy anyway.
        import cv2

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        raw = self._landmarker.detect_for_video(mp_image, self._next_timestamp_ms(monotonic_ms))
        inference_ms = (time.perf_counter() - start) * 1000.0

        hands: list[Hand] = []
        if raw is not None and raw.hand_landmarks:
            world_sets = raw.hand_world_landmarks or [None] * len(raw.hand_landmarks)
            for i, landmarks in enumerate(raw.hand_landmarks):
                category = raw.handedness[i][0] if raw.handedness and i < len(raw.handedness) else None
                hands.append(
                    hand_from_mediapipe(
                        landmarks=landmarks,
                        world_landmarks=world_sets[i] if i < len(world_sets) else None,
                        handedness_label=category.category_name if category else "Unknown",
                        handedness_score=category.score if category else 0.0,
                        frame_size=(width, height),
                        input_was_mirrored=self.mirrored_input,
                    )
                )

        self._frames += 1
        self._empty_streak = 0 if hands else self._empty_streak + 1

        return HandsResult(
            hands=hands,
            frame_index=frame_index,
            timestamp=time.perf_counter(),
            inference_ms=inference_ms,
        )

    # ------------------------------------------------------------------ #
    @property
    def frames_processed(self) -> int:
        return self._frames

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:  # pragma: no cover - teardown races in native code
                pass
            self._landmarker = None

    def __enter__(self) -> "HandEngine":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()
