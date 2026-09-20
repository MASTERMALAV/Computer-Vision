"""Cursor control: filtering, screen geometry, input injection, pointer engine."""

from __future__ import annotations

from .dispatcher import ActionDispatcher, IdentityStatus
from .filters import GainConfig, OneEuroConfig, OneEuroFilter, PositionHistory
from .injector import MouseInjector
from .pointer import PointerConfig, PointerEngine, PointerState
from .screens import Monitor, VirtualDesktop, get_virtual_desktop

__all__ = [
    "ActionDispatcher",
    "GainConfig",
    "IdentityStatus",
    "Monitor",
    "MouseInjector",
    "OneEuroConfig",
    "OneEuroFilter",
    "PointerConfig",
    "PointerEngine",
    "PointerState",
    "PositionHistory",
    "VirtualDesktop",
    "get_virtual_desktop",
]
