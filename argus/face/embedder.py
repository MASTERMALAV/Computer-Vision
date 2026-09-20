"""ArcFace embeddings.

Produces a 512-dimensional vector per face, L2-normalised so that the cosine
similarity between two embeddings is simply their dot product - which keeps the
matching code in :mod:`argus.face.gallery` trivial and fast.

Graph signature, verified against ``arcface_mbf.onnx``:

    input   (None, 3, 112, 112)  float32, RGB, (pixel - 127.5) / 127.5
    output  (1, 512)             float32

The output batch dimension is fixed at 1, so faces are embedded one at a time
regardless of how many are in frame. That is not a real cost here: after the
first authentication the pipeline embeds roughly one face every fifteen seconds,
not one per frame.
"""

from __future__ import annotations

import numpy as np

from ..config import FaceRecognizerConfig, RuntimeConfig
from ..logsetup import get_logger
from ..models.registry import ensure

log = get_logger("face.embedder")

EMBEDDING_DIM = 512
INPUT_SIZE = 112


class FaceEmbedder:
    """Turns an aligned 112x112 face crop into a normalised 512-d vector."""

    def __init__(
        self,
        config: FaceRecognizerConfig,
        runtime: RuntimeConfig | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime or RuntimeConfig()
        self._session = None
        self._input_name = ""

    def start(self) -> "FaceEmbedder":
        import onnxruntime as ort

        path = ensure(self.config.model)
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.runtime.onnx_intra_threads
        options.inter_op_num_threads = self.runtime.onnx_inter_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3

        self._session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        dim = self._session.get_outputs()[0].shape[-1]
        if dim != EMBEDDING_DIM:
            raise RuntimeError(f"{path.name} outputs {dim} dimensions, expected {EMBEDDING_DIM}")
        log.info("face embedder ready: %s", path.name)
        return self

    # ------------------------------------------------------------------ #
    @staticmethod
    def _preprocess(aligned_bgr: np.ndarray) -> np.ndarray:
        import cv2

        if aligned_bgr.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
            aligned_bgr = cv2.resize(aligned_bgr, (INPUT_SIZE, INPUT_SIZE))
        return cv2.dnn.blobFromImage(
            aligned_bgr, 1.0 / 127.5, (INPUT_SIZE, INPUT_SIZE),
            (127.5, 127.5, 127.5), swapRB=True,
        )

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Embed one aligned face crop. Returns a unit-length (512,) vector."""
        if self._session is None:
            raise RuntimeError("FaceEmbedder.start() must be called before embed()")

        blob = self._preprocess(aligned_bgr)
        raw = self._session.run(None, {self._input_name: blob})[0].reshape(-1)
        return normalize(raw)

    def embed_many(self, aligned: list[np.ndarray]) -> np.ndarray:
        """Embed several aligned crops. Returns (N, 512)."""
        if not aligned:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        return np.stack([self.embed(a) for a in aligned])

    def close(self) -> None:
        self._session = None


def normalize(vector: np.ndarray) -> np.ndarray:
    """L2-normalise, so cosine similarity becomes a dot product."""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return vector
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors, safe for non-normalised input."""
    a = normalize(a)
    b = normalize(b)
    return float(np.dot(a, b))


def similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, D) against (M, D) -> (N, M) cosine similarities."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    return (an @ bn.T).astype(np.float32)
