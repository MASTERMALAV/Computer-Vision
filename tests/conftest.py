"""Shared test fixtures: synthetic hands with exactly controlled geometry.

Building hands analytically rather than replaying recordings means a test can
state precisely what it is testing - "a pinch of 0.30 hand-scale units at twice
the distance from the camera" - and the assertion has no ambiguity.
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
    handedness: str = "Right",
    frame_size: tuple[int, int] = (1280, 720),
) -> Hand:
    """A synthetic hand with the requested normalised measurements.

    ``palm`` positions the wrist; ``scale`` is the wrist-to-middle-MCP span in
    pixels, which is what every normalised distance divides by. The resulting
    hand satisfies, to floating-point precision:

        hand.scale                    == scale
        hand.pinch("index")           == pinch_index
        hand.pinch("middle")          == pinch_middle
        hand.finger_extension("index")== index_extension
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

    # |index_tip - index_mcp| == index_extension * s
    pts[INDEX_TIP] = pts[INDEX_MCP] + np.array([0.0, -index_extension * s], dtype=np.float32)
    # |thumb_tip - index_tip| == pinch_index * s
    pts[THUMB_TIP] = pts[INDEX_TIP] + np.array([pinch_index * s, 0.0], dtype=np.float32)
    # |middle_tip - thumb_tip| == pinch_middle * s
    pts[MIDDLE_TIP] = pts[THUMB_TIP] + np.array([0.0, -pinch_middle * s], dtype=np.float32)

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


@pytest.fixture
def hand():
    return make_hand()


@pytest.fixture
def open_hand():
    """Clutch engaged, both pinches well open."""
    return make_hand(pinch_index=0.95, pinch_middle=0.95, index_extension=1.3)


@pytest.fixture
def pinched_hand():
    """Clutch engaged, index pinch closed."""
    return make_hand(pinch_index=0.20, pinch_middle=0.95, index_extension=1.3)
