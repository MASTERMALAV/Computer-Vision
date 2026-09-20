"""SCRFD face detection on ONNX Runtime.

Implemented directly against the ONNX graph rather than through the
``insightface`` package, for three reasons: that package needs a C++ toolchain
to build on Windows, it pulls in a large dependency tree for code we use a
fraction of, and decoding the outputs ourselves means the anchor maths is
visible and testable rather than opaque.

Graph signature, verified against ``scrfd_500m.onnx``:

    input   (1, 3, H, W)  float32, RGB, (pixel - 127.5) / 128.0
    outputs 9 tensors, grouped by stride 8 / 16 / 32:
              [0:3] scores  (N, 1)
              [3:6] bboxes  (N, 4)   distances to l/t/r/b, in stride units
              [6:9] kps     (N, 10)  five keypoint offsets, in stride units

At 640x640 the row counts are 12800 / 3200 / 800, which is
(640/stride)^2 x 2 - confirming two anchors per spatial location.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import FaceDetectorConfig, RuntimeConfig
from ..logsetup import get_logger
from ..models.registry import ensure

log = get_logger("face.detector")

STRIDES = (8, 16, 32)
NUM_ANCHORS = 2
FEATURE_MAP_COUNT = 3


@dataclass
class FaceBox:
    """One detected face in original-image pixel coordinates."""

    bbox: np.ndarray  # (4,) x1, y1, x2, y2
    keypoints: np.ndarray  # (5, 2) eye_l, eye_r, nose, mouth_l, mouth_r
    score: float

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)

    @property
    def center(self) -> np.ndarray:
        return np.array(
            [(self.bbox[0] + self.bbox[2]) * 0.5, (self.bbox[1] + self.bbox[3]) * 0.5],
            dtype=np.float32,
        )


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Anchor centres + per-side distances -> x1, y1, x2, y2."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Anchor centres + keypoint offsets -> (N, 5, 2) keypoints."""
    coords = []
    for i in range(0, distance.shape[1], 2):
        coords.append(points[:, 0] + distance[:, i])
        coords.append(points[:, 1] + distance[:, i + 1])
    return np.stack(coords, axis=-1).reshape(distance.shape[0], -1, 2)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Greedy non-maximum suppression. Returns kept indices, best first."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(xx2 - xx1, 0.0) * np.maximum(yy2 - yy1, 0.0)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = order[1:][iou <= threshold]
    return keep


class FaceDetector:
    """SCRFD detector. Call :meth:`detect` with a BGR image."""

    def __init__(
        self,
        config: FaceDetectorConfig,
        runtime: RuntimeConfig | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime or RuntimeConfig()
        self._session = None
        self._input_name = ""
        self._anchor_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def start(self) -> "FaceDetector":
        import onnxruntime as ort

        path = ensure(self.config.model)
        options = ort.SessionOptions()
        # Leave cores for capture and MediaPipe; taking all of them makes the
        # whole pipeline slower, not faster.
        options.intra_op_num_threads = self.runtime.onnx_intra_threads
        options.inter_op_num_threads = self.runtime.onnx_inter_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3

        self._session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        outputs = len(self._session.get_outputs())
        if outputs != 9:
            raise RuntimeError(
                f"{self.config.model} has {outputs} outputs; this decoder expects 9 "
                "(scores/bboxes/keypoints for three strides)"
            )
        log.info("face detector ready: %s", path.name)
        return self

    # ------------------------------------------------------------------ #
    def _anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        """Anchor centres for one feature map, cached per shape."""
        key = (height, width, stride)
        cached = self._anchor_cache.get(key)
        if cached is not None:
            return cached
        # mgrid gives (row, col); reversed to (x, y) before scaling by stride.
        centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        centers = (centers * stride).reshape(-1, 2)
        if NUM_ANCHORS > 1:
            # Both anchors at a location share the same centre.
            centers = np.stack([centers] * NUM_ANCHORS, axis=1).reshape(-1, 2)
        self._anchor_cache[key] = centers
        return centers

    def _preprocess(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Letterbox to a square input, preserving aspect ratio."""
        import cv2

        size = self.config.input_size
        h, w = image_bgr.shape[:2]
        scale = min(size / max(h, 1), size / max(w, 1))
        new_w, new_h = int(round(w * scale)), int(round(h * scale))

        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        # Padding sits at the right/bottom only, so the mapping back to original
        # coordinates is a single division with no offset to track.
        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / 128.0, (size, size), (127.5, 127.5, 127.5), swapRB=True
        )
        return blob, scale, (new_w, new_h)

    def detect(self, image_bgr: np.ndarray) -> list[FaceBox]:
        """Detect faces. Returns boxes in the coordinates of ``image_bgr``."""
        if self._session is None:
            raise RuntimeError("FaceDetector.start() must be called before detect()")

        blob, scale, _ = self._preprocess(image_bgr)
        outputs = self._session.run(None, {self._input_name: blob})

        size = self.config.input_size
        all_boxes: list[np.ndarray] = []
        all_kps: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []

        for idx, stride in enumerate(STRIDES):
            scores = outputs[idx].reshape(-1)
            keep = np.where(scores >= self.config.conf_threshold)[0]
            if keep.size == 0:
                continue

            bbox_preds = outputs[idx + FEATURE_MAP_COUNT].reshape(-1, 4) * stride
            kps_preds = outputs[idx + FEATURE_MAP_COUNT * 2].reshape(-1, 10) * stride
            centers = self._anchor_centers(size // stride, size // stride, stride)

            all_boxes.append(distance2bbox(centers[keep], bbox_preds[keep]))
            all_kps.append(distance2kps(centers[keep], kps_preds[keep]))
            all_scores.append(scores[keep])

        if not all_boxes:
            return []

        boxes = np.vstack(all_boxes) / scale
        kps = np.vstack(all_kps) / scale
        scores = np.concatenate(all_scores)

        keep = nms(boxes, scores, self.config.nms_threshold)

        h, w = image_bgr.shape[:2]
        results: list[FaceBox] = []
        for i in keep[: self.config.max_faces]:
            box = boxes[i]
            box[0::2] = np.clip(box[0::2], 0, w - 1)
            box[1::2] = np.clip(box[1::2], 0, h - 1)
            face = FaceBox(bbox=box.astype(np.float32),
                           keypoints=kps[i].astype(np.float32),
                           score=float(scores[i]))
            # A face smaller than this cannot produce a trustworthy embedding;
            # letting it through would poison the gallery during enrolment.
            if min(face.width, face.height) < self.config.min_face_px:
                continue
            results.append(face)
        return results

    def close(self) -> None:
        self._session = None
