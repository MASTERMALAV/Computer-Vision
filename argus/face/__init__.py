"""Face detection, alignment, embedding, gallery and identity sessions."""

from __future__ import annotations

from .align import align_face, face_quality, umeyama
from .detector import FaceBox, FaceDetector
from .embedder import FaceEmbedder, cosine_similarity, normalize
from .gallery import Gallery, GalleryError, MatchResult
from .pipeline import FaceObservation, FacePipeline
from .session import IdentitySession, SessionEvent

__all__ = [
    "FaceBox",
    "FaceDetector",
    "FaceEmbedder",
    "FaceObservation",
    "FacePipeline",
    "Gallery",
    "GalleryError",
    "IdentitySession",
    "MatchResult",
    "SessionEvent",
    "align_face",
    "cosine_similarity",
    "face_quality",
    "normalize",
    "umeyama",
]
