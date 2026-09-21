"""Camera discovery and threaded frame capture."""

from __future__ import annotations

from .camera import CameraError, CameraStream, Frame, open_camera
from .devices import CameraDevice, DeviceError, enumerate_devices, resolve_device
from .picker import best_candidate, scan_cameras, select_camera

__all__ = [
    "CameraDevice",
    "CameraError",
    "CameraStream",
    "DeviceError",
    "Frame",
    "best_candidate",
    "enumerate_devices",
    "open_camera",
    "resolve_device",
    "scan_cameras",
    "select_camera",
]
