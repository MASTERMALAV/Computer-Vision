"""ARGUS - Adaptive Recognition & Gesture Understanding System.

A CPU-first, real-time multimodal perception stack: who is in front of the
camera (face detection + recognition), what they are doing with their hands
(landmarks + static and dynamic gestures), and what they said (VAD + ASR).

The perception core is deliberately separate from anything that acts on the
world. :mod:`argus.fusion` decides *what happened*; acting on it is opt-in and
gated behind operator identity.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
