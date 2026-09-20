"""Model weight registry: download, verify, locate."""

from __future__ import annotations

from .registry import (
    REGISTRY,
    ModelError,
    ModelSpec,
    ensure,
    ensure_many,
    is_installed,
    models_for_profile,
    status,
    verify_installed,
)

__all__ = [
    "REGISTRY",
    "ModelError",
    "ModelSpec",
    "ensure",
    "ensure_many",
    "is_installed",
    "models_for_profile",
    "status",
    "verify_installed",
]
