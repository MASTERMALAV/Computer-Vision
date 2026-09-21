"""Shared test fixtures: synthetic hands with exactly controlled geometry.

Building hands analytically rather than replaying recordings means a test can
state precisely what it is testing - "a pinch of 0.30 hand-scale units at twice
the distance from the camera" - and the assertion has no ambiguity.

One thing this deliberately does *not* do is let a caller specify every distance
independently. A hand has fewer degrees of freedom than that: with the index
extended and the middle curled, the thumb physically cannot be 0.15 hand-widths
from the middle fingertip *and* 0.95 from the index fingertip, because those two
tips are two hand-widths apart. So the thumb is placed relative to one named
target and the other distances follow from the geometry, exactly as on a real
hand. An earlier version of this fixture allowed the contradiction, which
silently produced hands whose middle finger was implausibly extended.
"""

from __future__ import annotations

import numpy as np
import pytest

from argus.hands.landmarks import (
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    NUM_LANDMARKS,
    PINKY_MCP,
    RING_MCP,
    THUMB_TIP,
    WRIST,
    Hand,
)


def make_hand(
    palm: tuple[float, float] = (640.0, 360.0),
    scale: float = 100.0,
    pinch_index: float = 0.9,
    pinch_middle: float = 0.9,
    index_extension: float = 1.3,
    middle_extension: float = 0.75,
    thumb: str = "index",
    handedness: str = "Right",
    frame_size: tuple[int, int] = (1280, 720),
) -> Hand:
    """A synthetic hand with the requested normalised measurements.

    Args:
        palm: wrist position in pixels.
        scale: wrist-to-middle-MCP span in pixels - the unit every normalised
            distance divides by.
        index_extension: index tip distance from its MCP, over hand scale.
        middle_extension: same for the middle finger. Below ~1.0 the finger is
            curled toward the palm, which is the pointing pose; above ~1.05 it
            is extended, which is the two-finger scroll pose.
        thumb: which fingertip the thumb is placed relative to - ``"index"``
            uses ``pinch_index``, ``"middle"`` uses ``pinch_middle``.

    Guarantees, to floating-point precision::

        hand.scale                       == scale
        hand.finger_extension("index")   == index_extension
        hand.finger_extension("middle")  == middle_extension
        hand.pinch(thumb)                == pinch_index / pinch_middle
    """
    s = float(scale)
    wx, wy = float(palm[0]), float(palm[1])
    pts = np.zeros((NUM_LANDMARKS, 2), dtype=np.float32)

    pts[WRIST] = (wx, wy)
    # Defines the scale: |middle_mcp - wrist| == s exactly.
    pts[MIDDLE_MCP] = (wx, wy - s)
    pts[INDEX_MCP] = (wx + 0.25 * s, wy - 0.90 * s)
    pts[RING_MCP] = (wx - 0.25 * s, wy - 0.95 * s)
    pts[PINKY_MCP] = (wx - 0.50 * s, wy - 0.85 * s)

    # An extended finger points away from the palm; a curled one folds back
    # toward it. Either way the tip sits exactly `extension * scale` from its
    # knuckle, which is what finger_extension() measures.
    def fingertip(mcp: np.ndarray, extension: float) -> np.ndarray:
        direction = np.array([0.0, -1.0]) if extension >= 1.0 else np.array([0.0, 1.0])
        return (mcp + direction * extension * s).astype(np.float32)

    pts[INDEX_TIP] = fingertip(pts[INDEX_MCP], index_extension)
    pts[MIDDLE_TIP] = fingertip(pts[MIDDLE_MCP], middle_extension)

    if thumb == "middle":
        pts[THUMB_TIP] = pts[MIDDLE_TIP] + np.array([pinch_middle * s, 0.0], dtype=np.float32)
    elif thumb == "index":
        pts[THUMB_TIP] = pts[INDEX_TIP] + np.array([pinch_index * s, 0.0], dtype=np.float32)
    else:
        raise ValueError(f"thumb must be 'index' or 'middle', got {thumb!r}")

    # Remaining joints get plausible positions; no test depends on them.
    for idx in range(NUM_LANDMARKS):
        if not pts[idx].any():
            pts[idx] = (wx, wy - 0.5 * s)

    return Hand(
        pixels=pts,
        normalized=np.zeros((NUM_LANDMARKS, 3), dtype=np.float32),
        world=np.zeros((NUM_LANDMARKS, 3), dtype=np.float32),
        handedness=handedness,
        score=0.99,
        frame_size=frame_size,
    )


def pointing_hand(**kw) -> Hand:
    """Clutch engaged, one finger out - the cursor pose."""
    kw.setdefault("index_extension", 1.3)
    kw.setdefault("middle_extension", 0.75)
    return make_hand(**kw)


def scrolling_hand(**kw) -> Hand:
    """Clutch engaged, thumb holding the middle fingertip - the scroll grip."""
    kw.setdefault("index_extension", 1.3)
    kw.setdefault("middle_extension", 0.75)
    kw.setdefault("thumb", "middle")
    kw.setdefault("pinch_middle", 0.15)
    return make_hand(**kw)


@pytest.fixture
def hand():
    return make_hand()


@pytest.fixture
def open_hand():
    """Clutch engaged, pinch open."""
    return pointing_hand(pinch_index=0.95)


@pytest.fixture
def pinched_hand():
    """Clutch engaged, index pinch closed."""
    return pointing_hand(pinch_index=0.20)
