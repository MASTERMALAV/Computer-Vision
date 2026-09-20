"""Face alignment: warp a detected face onto the ArcFace reference frame.

ArcFace was trained on faces normalised so that the eyes, nose and mouth corners
land on fixed pixel positions in a 112x112 crop. Feeding it a raw bounding-box
crop instead costs a large amount of accuracy, because the network never learned
to be invariant to in-plane rotation or scale - the alignment step was supposed
to remove those.

The transform is a *similarity* transform (rotation, uniform scale, translation
- four degrees of freedom), fitted by the Umeyama least-squares method. An
affine fit would have six and would shear the face to match the template, which
distorts exactly the geometry the embedding is meant to measure.
"""

from __future__ import annotations

import numpy as np

# The canonical ArcFace 5-point template for a 112x112 crop, in pixels:
# left eye, right eye, nose tip, left mouth corner, right mouth corner.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

REFERENCE_SIZE = 112


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform mapping ``src`` onto ``dst``.

    Returns a 2x3 matrix suitable for ``cv2.warpAffine``.

    Implemented here rather than pulled from scikit-image (one function versus a
    large dependency) and preferred over ``cv2.estimateAffinePartial2D``, whose
    RANSAC/LMEDS estimators are randomised and can return slightly different
    results run to run - which would make embeddings non-reproducible.

    Reference: Umeyama, "Least-squares estimation of transformation parameters
    between two point patterns", IEEE PAMI 1991.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(f"expected matching (N, 2) arrays, got {src.shape} and {dst.shape}")

    num = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    covariance = dst_demean.T @ src_demean / num
    d = np.ones(2, dtype=np.float64)
    if np.linalg.det(covariance) < 0:
        d[1] = -1.0

    U, S, Vt = np.linalg.svd(covariance)
    rank = np.linalg.matrix_rank(covariance)
    if rank == 0:
        return np.full((2, 3), np.nan)
    if rank == 1:
        # Degenerate: the points are collinear. Still recoverable, but the
        # reflection has to be resolved from the determinants.
        if np.linalg.det(U) * np.linalg.det(Vt) > 0:
            rotation = U @ Vt
        else:
            s = d[1]
            d[1] = -1.0
            rotation = U @ np.diag(d) @ Vt
            d[1] = s
    else:
        rotation = U @ np.diag(d) @ Vt

    variance = src_demean.var(axis=0).sum()
    scale = 1.0 if variance < 1e-12 else (S @ d) / variance

    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = scale * rotation
    matrix[:2, 2] = dst_mean - scale * rotation @ src_mean
    return matrix[:2, :].astype(np.float32)


def align_face(
    image_bgr: np.ndarray,
    keypoints: np.ndarray,
    size: int = REFERENCE_SIZE,
) -> np.ndarray:
    """Warp a face into the ArcFace reference frame.

    Args:
        image_bgr: the full frame.
        keypoints: (5, 2) landmarks from the detector, in frame pixels.
        size: output side length; the template scales with it.
    """
    import cv2

    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.shape != (5, 2):
        raise ValueError(f"expected 5 keypoints, got {keypoints.shape}")

    template = ARCFACE_TEMPLATE * (size / REFERENCE_SIZE)
    matrix = umeyama(keypoints, template)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("degenerate keypoints; cannot align this face")
    return cv2.warpAffine(image_bgr, matrix, (size, size), flags=cv2.INTER_LINEAR,
                          borderValue=0.0)


def face_quality(box, image_shape: tuple[int, int]) -> dict[str, float]:
    """Cheap quality signals, used to reject poor enrolment samples.

    None of these need a model. They catch the cases that actually spoil a
    gallery: a face too small to resolve, one cut off by the frame edge, and one
    turned so far that the two eyes are no longer symmetric about the nose.
    """
    h, w = image_shape[:2]
    x1, y1, x2, y2 = box.bbox
    kps = box.keypoints

    # How far the eye-line is from horizontal.
    eye_vector = kps[1] - kps[0]
    roll = float(np.degrees(np.arctan2(eye_vector[1], eye_vector[0])))

    # Yaw proxy: the nose sits midway between the eyes when facing the camera,
    # and slides toward one of them as the head turns.
    eye_mid = (kps[0] + kps[1]) * 0.5
    eye_span = float(np.linalg.norm(eye_vector)) or 1.0
    yaw_ratio = float((kps[2][0] - eye_mid[0]) / eye_span)

    margin = min(x1, y1, w - 1 - x2, h - 1 - y2)
    return {
        "size_px": float(min(x2 - x1, y2 - y1)),
        "roll_deg": abs(roll),
        "yaw_ratio": abs(yaw_ratio),
        "edge_margin_px": float(margin),
        "score": float(box.score),
    }


def quality_problems(
    quality: dict[str, float],
    min_size: int = 80,
    max_roll: float = 25.0,
    max_yaw: float = 0.35,
    min_margin: float = 4.0,
) -> list[str]:
    """Human-readable reasons a face is unsuitable for enrolment."""
    problems: list[str] = []
    if quality["size_px"] < min_size:
        problems.append(f"too small ({quality['size_px']:.0f}px, need {min_size})")
    if quality["roll_deg"] > max_roll:
        problems.append(f"head tilted ({quality['roll_deg']:.0f} deg)")
    if quality["yaw_ratio"] > max_yaw:
        problems.append("turned too far from the camera")
    if quality["edge_margin_px"] < min_margin:
        problems.append("cut off by the frame edge")
    return problems
