"""Drawing primitives shared by every ARGUS preview window.

Kept free of pipeline knowledge on purpose: these take geometry and strings, so
the same helpers serve the capture preview, the face view and the full HUD.
"""

from __future__ import annotations

import numpy as np

# BGR, because that is what OpenCV expects.
COLORS: dict[str, tuple[int, int, int]] = {
    "bg": (24, 20, 18),
    "panel": (38, 32, 28),
    "text": (240, 240, 240),
    "muted": (160, 160, 160),
    "accent": (255, 176, 46),  # amber
    "ok": (110, 220, 120),
    "warn": (60, 200, 255),  # amber-yellow
    "error": (70, 80, 255),
    "known": (120, 230, 140),
    "unknown": (90, 120, 255),
    "left_hand": (255, 190, 80),
    "right_hand": (140, 210, 255),
}

_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX, avoids importing cv2 at module scope


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.5,
    color: tuple[int, int, int] = COLORS["text"],
    thickness: int = 1,
    shadow: bool = True,
) -> None:
    """Draw a single line of text with a subtle shadow for legibility."""
    import cv2

    x, y = origin
    if shadow:
        # LINE_8, not LINE_AA: the shadow only provides contrast behind the
        # glyph, so antialiasing it is pure cost. Measured across a full HUD
        # this roughly halves text rendering time.
        cv2.putText(image, text, (x + 1, y + 1), _FONT, scale, (0, 0, 0), thickness + 1, cv2.LINE_8)
    cv2.putText(image, text, (x, y), _FONT, scale, color, thickness, cv2.LINE_AA)


def text_size(text: str, scale: float = 0.5, thickness: int = 1) -> tuple[int, int]:
    import cv2

    (w, h), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    return w, h


def draw_panel(
    image: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (12, 12),
    scale: float = 0.5,
    alpha: float = 0.55,
    color: tuple[int, int, int] = COLORS["text"],
    title: str | None = None,
    min_width: int = 0,
) -> tuple[int, int]:
    """Draw a translucent panel of text lines. Returns its (width, height)."""
    import cv2

    if not lines and not title:
        return (0, 0)

    pad = 10
    line_h = int(22 * (scale / 0.5))
    all_lines = ([title] if title else []) + lines
    width = max([text_size(s, scale)[0] for s in all_lines] + [min_width]) + pad * 2
    height = line_h * len(all_lines) + pad * 2 - 4

    x, y = origin
    x = max(0, min(x, image.shape[1] - 1))
    y = max(0, min(y, image.shape[0] - 1))
    x2 = min(x + width, image.shape[1])
    y2 = min(y + height, image.shape[0])
    if x2 <= x or y2 <= y:
        return (0, 0)

    roi = image[y:y2, x:x2]
    panel = np.full(roi.shape, COLORS["panel"], dtype=np.uint8)
    cv2.addWeighted(panel, alpha, roi, 1.0 - alpha, 0.0, dst=roi)
    cv2.rectangle(image, (x, y), (x2 - 1, y2 - 1), COLORS["muted"], 1, cv2.LINE_AA)

    ty = y + pad + line_h - 8
    if title:
        draw_text(image, title, (x + pad, ty), scale * 1.05, COLORS["accent"], 1)
        ty += line_h
    for line in lines:
        draw_text(image, line, (x + pad, ty), scale, color, 1)
        ty += line_h
    return (width, height)


def draw_box(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    color: tuple[int, int, int] = COLORS["accent"],
    thickness: int = 2,
    label: str | None = None,
    sublabel: str | None = None,
    corner_style: bool = True,
) -> None:
    """Draw a bounding box, optionally with corner brackets and a label chip."""
    import cv2

    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return

    if corner_style:
        # Corner brackets read more cleanly than a full rectangle when several
        # boxes overlap, and they leave the subject's face unobscured.
        arm = max(12, int(min(x2 - x1, y2 - y1) * 0.22))
        for px, py, dx, dy in (
            (x1, y1, 1, 1),
            (x2, y1, -1, 1),
            (x1, y2, 1, -1),
            (x2, y2, -1, -1),
        ):
            cv2.line(image, (px, py), (px + dx * arm, py), color, thickness, cv2.LINE_AA)
            cv2.line(image, (px, py), (px, py + dy * arm), color, thickness, cv2.LINE_AA)
    else:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    if label:
        scale = 0.52
        tw, th = text_size(label, scale)
        chip_h = th + 12
        cy1 = max(0, y1 - chip_h)
        cv2.rectangle(image, (x1, cy1), (min(x1 + tw + 14, w - 1), cy1 + chip_h), color, -1)
        draw_text(image, label, (x1 + 7, cy1 + chip_h - 8), scale, (18, 18, 18), 1, shadow=False)
    if sublabel:
        draw_text(image, sublabel, (x1 + 2, min(y2 + 18, h - 4)), 0.45, color, 1)


def draw_bar(
    image: np.ndarray,
    origin: tuple[int, int],
    width: int,
    value: float,
    color: tuple[int, int, int] = COLORS["accent"],
    height: int = 6,
) -> None:
    """Horizontal progress/confidence bar with ``value`` clamped to [0, 1]."""
    import cv2

    x, y = origin
    value = float(np.clip(value, 0.0, 1.0))
    cv2.rectangle(image, (x, y), (x + width, y + height), COLORS["panel"], -1)
    if value > 0:
        cv2.rectangle(image, (x, y), (x + int(width * value), y + height), color, -1)
    cv2.rectangle(image, (x, y), (x + width, y + height), COLORS["muted"], 1)


def fps_color(fps: float, target: float = 25.0) -> tuple[int, int, int]:
    """Green when comfortably real-time, amber when marginal, red when not."""
    if fps >= target:
        return COLORS["ok"]
    if fps >= target * 0.6:
        return COLORS["warn"]
    return COLORS["error"]
