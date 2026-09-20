"""Hand landmark geometry: indices, scale normalisation, derived measurements.

Two things in here are easy to get wrong and quietly ruin every downstream
threshold:

**Aspect distortion.** MediaPipe returns landmarks normalised to 0..1 on each
axis *independently*. On a 16:9 frame one x-unit is 1280 px while one y-unit is
720 px, so a raw Euclidean distance in normalised space is stretched horizontally
by 1.78x. Every distance in this module is therefore computed in *pixels* first.

**Depth invariance.** A pinch 30 cm from the camera spans far fewer pixels than
the same pinch at 60 cm. Every distance that feeds a threshold is divided by the
hand's own scale - the wrist to middle-finger MCP span - which makes the ratio
independent of how far away the hand is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# The 21 MediaPipe hand landmarks
# --------------------------------------------------------------------------- #
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

NUM_LANDMARKS = 21

# Finger name -> (mcp, pip, dip, tip)
FINGERS: dict[str, tuple[int, int, int, int]] = {
    "thumb": (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

TIPS = {"thumb": THUMB_TIP, "index": INDEX_TIP, "middle": MIDDLE_TIP,
        "ring": RING_TIP, "pinky": PINKY_TIP}

# Bones, for drawing.
CONNECTIONS: tuple[tuple[int, int], ...] = (
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (INDEX_MCP, MIDDLE_MCP),
    (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (MIDDLE_MCP, RING_MCP),
    (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (RING_MCP, PINKY_MCP),
    (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
)

# A hand smaller than this many pixels across (wrist -> middle MCP) is too far
# away or too poorly resolved for reliable pinch thresholds.
MIN_HAND_SCALE_PX = 22.0


@dataclass
class Hand:
    """One detected hand, in a form every downstream stage can use directly.

    Attributes:
        pixels: (21, 2) float32 landmark positions in image pixels.
        normalized: (21, 3) float32 as MediaPipe returned them (x, y in 0..1, z relative).
        world: (21, 3) float32 metric landmarks, origin at the hand's centre, in metres.
        handedness: "Left" or "Right" - the user's physical hand.
        score: handedness classification confidence.
        frame_size: (width, height) of the image the landmarks came from.
    """

    pixels: np.ndarray
    normalized: np.ndarray
    world: np.ndarray
    handedness: str
    score: float
    frame_size: tuple[int, int]

    # ------------------------------------------------------------------ #
    # Scale
    # ------------------------------------------------------------------ #
    @property
    def scale(self) -> float:
        """Hand size in pixels: wrist to middle-finger MCP.

        This span is used rather than, say, the bounding box because it is a
        rigid bone length. It does not change when fingers curl, so thresholds
        divided by it stay stable through a pinch - which a bounding box would
        not.
        """
        return float(np.linalg.norm(self.pixels[MIDDLE_MCP] - self.pixels[WRIST]))

    @property
    def is_reliable(self) -> bool:
        """False when the hand is too small in frame for thresholds to mean much."""
        return self.scale >= MIN_HAND_SCALE_PX

    def norm_dist(self, a: int, b: int) -> float:
        """Distance between two landmarks, in units of hand scale.

        Aspect-correct (computed in pixels) and depth-invariant (divided by the
        hand's own size), so a threshold expressed in these units holds whether
        the hand is near or far, centred or at the edge of the frame.
        """
        scale = self.scale
        if scale < 1e-6:
            return 0.0
        return float(np.linalg.norm(self.pixels[a] - self.pixels[b]) / scale)

    def norm_dist_to_point(self, a: int, point: np.ndarray) -> float:
        scale = self.scale
        if scale < 1e-6:
            return 0.0
        return float(np.linalg.norm(self.pixels[a] - point) / scale)

    # ------------------------------------------------------------------ #
    # Derived geometry
    # ------------------------------------------------------------------ #
    @property
    def palm_center(self) -> np.ndarray:
        """Centroid of the palm quad (wrist + the four MCP knuckles).

        Much steadier than any fingertip: it barely moves when fingers flex, so
        it is the right anchor for a cursor that must not drift while pinching.
        """
        return self.pixels[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]].mean(axis=0)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        x1, y1 = self.pixels.min(axis=0)
        x2, y2 = self.pixels.max(axis=0)
        return float(x1), float(y1), float(x2), float(y2)

    def pinch(self, finger: str = "index") -> float:
        """Thumb-tip to fingertip distance, in hand-scale units."""
        tip = TIPS.get(finger)
        if tip is None:
            raise KeyError(f"unknown finger {finger!r}; known: {', '.join(TIPS)}")
        return self.norm_dist(THUMB_TIP, tip)

    def finger_extension(self, finger: str) -> float:
        """How extended a finger is, as tip distance from its MCP over hand scale.

        Chosen over joint angles because it degrades gracefully: joint angles
        become unstable when a finger points toward the camera and the PIP/DIP
        landmarks collapse onto each other, whereas this ratio just shrinks
        smoothly. Roughly: > 1.1 extended, < 0.7 curled, for non-thumb fingers.
        """
        if finger == "thumb":
            # The thumb's MCP barely moves relative to the wrist, so measuring
            # from the wrist gives a usable range where MCP-relative does not.
            return self.norm_dist(WRIST, THUMB_TIP)
        mcp, _pip, _dip, tip = FINGERS[finger]
        return self.norm_dist(mcp, tip)

    def extensions(self) -> dict[str, float]:
        return {name: self.finger_extension(name) for name in FINGERS}

    @property
    def span(self) -> float:
        """Index-tip to pinky-tip distance in hand-scale units - hand openness."""
        return self.norm_dist(INDEX_TIP, PINKY_TIP)

    def to_dict(self) -> dict:
        return {
            "handedness": self.handedness,
            "score": round(self.score, 4),
            "scale_px": round(self.scale, 2),
            "palm_center": [round(float(v), 2) for v in self.palm_center],
            "pinch_index": round(self.pinch("index"), 4),
            "pinch_middle": round(self.pinch("middle"), 4),
            "extensions": {k: round(v, 4) for k, v in self.extensions().items()},
        }


def hand_from_mediapipe(
    landmarks,
    world_landmarks,
    handedness_label: str,
    handedness_score: float,
    frame_size: tuple[int, int],
    input_was_mirrored: bool,
) -> Hand:
    """Build a :class:`Hand` from one MediaPipe result entry.

    Args:
        input_was_mirrored: whether the frame fed to MediaPipe had already been
            flipped horizontally. MediaPipe assigns handedness *assuming a
            mirrored selfie image*; if the frame was not flipped, the label
            refers to the opposite physical hand and must be swapped.
    """
    width, height = frame_size

    norm = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    # Normalised -> pixels. Doing this before any distance is what removes the
    # 16:9 aspect distortion from every downstream ratio.
    pixels = np.empty((len(norm), 2), dtype=np.float32)
    pixels[:, 0] = norm[:, 0] * width
    pixels[:, 1] = norm[:, 1] * height

    if world_landmarks is not None:
        world = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks], dtype=np.float32)
    else:
        world = np.zeros((len(norm), 3), dtype=np.float32)

    label = handedness_label
    if not input_was_mirrored:
        label = {"Left": "Right", "Right": "Left"}.get(label, label)

    return Hand(
        pixels=pixels,
        normalized=norm,
        world=world,
        handedness=label,
        score=float(handedness_score),
        frame_size=(int(width), int(height)),
    )
