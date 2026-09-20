"""Face recognition, scheduled so it fits in the leftover CPU budget.

Hand tracking already consumes about 12 ms of every 33 ms frame. Detection plus
embedding costs far more than the remainder, so running face recognition every
frame is simply not available - and it is also unnecessary. Identity changes on
a timescale of minutes, not milliseconds.

The scheduler therefore does the least work that keeps the answer correct:

* **No session** - the operator needs to be recognised, so run every
  ``detect_every_n_frames`` frames until one is.
* **Session open and recently confirmed** - skip entirely. This is the steady
  state, and it costs nothing.
* **Session open but due for re-confirmation** - run once, then go quiet again.

In practice that means face work happens at roughly 15 Hz while logging in and
once every fifteen seconds thereafter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..config import ArgusConfig
from ..logsetup import get_logger
from .align import align_face, face_quality
from .detector import FaceBox, FaceDetector
from .embedder import FaceEmbedder
from .gallery import Gallery, MatchResult
from .session import IdentitySession, SessionEvent

log = get_logger("face.pipeline")


@dataclass
class FaceObservation:
    """The most recent thing the face stage saw."""

    box: FaceBox | None = None
    match: MatchResult | None = None
    quality: dict = field(default_factory=dict)
    timestamp: float = 0.0
    inference_ms: float = 0.0

    @property
    def has_face(self) -> bool:
        return self.box is not None


class FacePipeline:
    """Detection, embedding, gallery matching and session upkeep."""

    def __init__(self, cfg: ArgusConfig) -> None:
        self.cfg = cfg
        self.detector = FaceDetector(cfg.face.detector, cfg.runtime)
        self.embedder = FaceEmbedder(cfg.face.recognizer, cfg.runtime)
        self.gallery = Gallery(model=cfg.face.recognizer.model)
        self.session = IdentitySession(security=cfg.security)

        self.last = FaceObservation()
        self.events: list[SessionEvent] = []
        self.runs = 0
        self.skipped = 0
        self._last_run_at = -1e9
        self._started = False

    # ------------------------------------------------------------------ #
    def start(self) -> "FacePipeline":
        self.detector.start()
        self.embedder.start()
        self.gallery = Gallery.load(
            self.cfg.face.gallery_path,
            model=self.cfg.face.recognizer.model,
            strict_model=True,
        )
        if self.gallery.is_empty and self.cfg.security.require_identity:
            log.warning(
                "identity gating is on but no one is enrolled; every action will "
                "be blocked. Enrol with:  argus enroll <your name>"
            )
        self._started = True
        return self

    def close(self) -> None:
        self.detector.close()
        self.embedder.close()
        self._started = False

    # ------------------------------------------------------------------ #
    def should_run(self, frame_index: int, now: float) -> bool:
        """Decide whether this frame is worth spending face inference on.

        The governing question is whether the answer could change anything. Work
        that cannot affect a decision is not cheap work - measured on this
        machine, hunting for a face every fifth frame costs about 10 fps and
        also slows hand tracking by competing for the same cores.
        """
        if not self._started or not self.cfg.face.enabled:
            return False

        gating = self.cfg.security.require_identity

        # Nothing to decide: no gate to enforce and nobody to recognise.
        if not gating and self.gallery.is_empty:
            return False

        # Authenticated and recently confirmed - the steady state, and free.
        if self.session.is_open and not self.session.needs_reverify(now):
            return False

        if not gating:
            # Identity is being shown, not enforced. A name on the HUD does not
            # justify a per-frame cost, so it refreshes at the re-verify cadence.
            if (now - self._last_run_at) < self.cfg.security.reverify_interval_s:
                return False
            return True

        return frame_index % max(self.cfg.face.detect_every_n_frames, 1) == 0

    def update(
        self,
        image_bgr: np.ndarray,
        frame_index: int,
        now: float | None = None,
    ) -> FaceObservation:
        """Run the face stage if it is due; otherwise return the previous result."""
        now = time.perf_counter() if now is None else now

        event = self.session.tick(now)
        if event:
            self.events.append(event)

        if not self.should_run(frame_index, now):
            self.skipped += 1
            return self.last

        self._last_run_at = now
        start = time.perf_counter()
        faces = self.detector.detect(image_bgr)
        observation = FaceObservation(timestamp=now)

        if faces:
            # The largest face is the one closest to the camera, i.e. the person
            # actually at the desk rather than someone walking past behind them.
            box = max(faces, key=lambda f: f.area)
            observation.box = box
            observation.quality = face_quality(box, image_bgr.shape)
            try:
                aligned = align_face(image_bgr, box.keypoints)
                embedding = self.embedder.embed(aligned)
                match = self.gallery.match(
                    embedding,
                    threshold=self.cfg.face.recognizer.match_threshold,
                    margin=self.cfg.face.recognizer.margin_threshold,
                )
                observation.match = match
                event = (
                    self.session.observe_match(match.name, match.score, now)
                    if match.is_known
                    else self.session.observe_no_match(now)
                )
                if event:
                    self.events.append(event)
            except ValueError as exc:
                # Degenerate landmarks: treat as "saw a face, could not identify".
                log.debug("alignment failed: %s", exc)
                observation.match = None
        else:
            # No face at all is not evidence of an impostor, so an open session
            # is left alone to expire on its own timer. Punishing absence would
            # log the operator out every time they leaned out of frame.
            pass

        observation.inference_ms = (time.perf_counter() - start) * 1000.0
        self.runs += 1
        self.last = observation
        return observation

    # ------------------------------------------------------------------ #
    def status(self, now: float | None = None):
        return self.session.status(now)

    def stats(self) -> dict:
        total = self.runs + self.skipped
        return {
            "runs": self.runs,
            "skipped": self.skipped,
            "duty_cycle_pct": round(100.0 * self.runs / total, 1) if total else 0.0,
            "identities": len(self.gallery.names),
            "session": self.session.describe(),
        }

    def drain_events(self) -> list[SessionEvent]:
        events, self.events = self.events, []
        return events
