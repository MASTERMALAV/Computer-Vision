"""Hand landmark rendering."""

from __future__ import annotations

import numpy as np

from ..hands.landmarks import (
    CONNECTIONS,
    INDEX_TIP,
    MIDDLE_TIP,
    THUMB_TIP,
    WRIST,
    Hand,
)
from .overlay import COLORS, draw_text

# Fingertips are drawn larger; they are what the interaction actually uses.
_TIP_IDS = frozenset({4, 8, 12, 16, 20})


def hand_color(hand: Hand) -> tuple[int, int, int]:
    return COLORS["left_hand"] if hand.handedness == "Left" else COLORS["right_hand"]


def draw_hand(
    image: np.ndarray,
    hand: Hand,
    color: tuple[int, int, int] | None = None,
    show_label: bool = True,
    show_pinch: bool = False,
) -> None:
    """Draw one hand's skeleton onto ``image`` in place."""
    import cv2

    color = color or hand_color(hand)
    pts = hand.pixels.astype(np.int32)

    for a, b in CONNECTIONS:
        cv2.line(image, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)

    for i, (x, y) in enumerate(pts):
        radius = 5 if i in _TIP_IDS else 3
        cv2.circle(image, (int(x), int(y)), radius, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(image, (int(x), int(y)), radius - 1, color, -1, cv2.LINE_AA)

    if show_pinch:
        # A live line between thumb and index makes the pinch threshold visible,
        # which is the fastest way to tune it by eye.
        cv2.line(image, tuple(pts[THUMB_TIP]), tuple(pts[INDEX_TIP]),
                 COLORS["accent"], 1, cv2.LINE_AA)
        mid = (pts[THUMB_TIP] + pts[INDEX_TIP]) // 2
        draw_text(image, f"{hand.pinch('index'):.2f}", (int(mid[0]) + 6, int(mid[1])),
                  0.45, COLORS["accent"])
        cv2.line(image, tuple(pts[THUMB_TIP]), tuple(pts[MIDDLE_TIP]),
                 COLORS["muted"], 1, cv2.LINE_AA)

    if show_label:
        wx, wy = pts[WRIST]
        label = f"{hand.handedness} {hand.score:.2f}  scale {hand.scale:.0f}px"
        if not hand.is_reliable:
            label += "  TOO FAR"
        draw_text(image, label, (int(wx) - 30, int(wy) + 26), 0.45, color)


def draw_hands(
    image: np.ndarray,
    hands: list[Hand],
    show_pinch: bool = False,
) -> None:
    for hand in hands:
        draw_hand(image, hand, show_pinch=show_pinch)
