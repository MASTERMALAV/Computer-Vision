"""Logging setup for ARGUS.

One entry point, :func:`setup_logging`, so every command-line tool produces the
same output format. Uses ``rich`` when available (colour, aligned columns) and
degrades to the stdlib formatter when it is not installed.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_CONFIGURED = False


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    """Configure the root logger once and return the ``argus`` logger."""
    global _CONFIGURED

    numeric = getattr(logging, str(level).upper(), logging.INFO)
    root = logging.getLogger()

    if not _CONFIGURED:
        root.handlers.clear()
        handler: logging.Handler
        try:
            from rich.logging import RichHandler

            handler = RichHandler(
                rich_tracebacks=True,
                show_path=False,
                markup=False,
                log_time_format="%H:%M:%S.%f",
            )
            handler.setFormatter(logging.Formatter("%(name)-22s %(message)s"))
        except ImportError:  # pragma: no cover - rich is a declared dependency
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)-7s %(name)-22s %(message)s")
            )
        root.addHandler(handler)

        if log_file is not None:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)-7s %(name)-22s %(message)s")
            )
            root.addHandler(file_handler)

        # These libraries are extremely chatty at DEBUG and drown out our logs.
        for noisy in ("matplotlib", "PIL", "urllib3", "numba", "absl", "comtypes"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        _CONFIGURED = True

    root.setLevel(numeric)
    return logging.getLogger("argus")


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``argus`` namespace."""
    return logging.getLogger(f"argus.{name}" if not name.startswith("argus") else name)


def configure_native_backends() -> None:
    """Tune and quieten OpenCV / MediaPipe / TensorFlow Lite native code.

    These libraries read their configuration from environment variables at
    import time and log straight to stderr from C++, where Python's logging
    module cannot reach them. So this must run *before* they are imported.
    """
    # --- logging: these are read at import time by the native layers --------
    os.environ.setdefault("GLOG_minloglevel", "2")  # MediaPipe / glog: errors only
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    os.environ.setdefault("ABSL_LOGGING_MIN_SEVERITY", "2")

    # --- Media Foundation startup latency ----------------------------------
    # Measured on the development machine opening a Logitech Brio at 1280x720:
    #
    #     hardware transforms ON (default):  open 7.31 s + property set 17.26 s
    #     hardware transforms OFF:           open 0.57 s + property set  0.03 s
    #
    # Same 30 fps either way. MSMF's hardware transform pipeline negotiates
    # with the GPU/driver on open, and on machines with an old or unusual
    # display driver that negotiation stalls for tens of seconds. Turning it
    # off makes MSMF open essentially instantly and costs nothing measurable,
    # since frames are converted on the CPU regardless.
    os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

    # Deliberately NOT setting OPENCV_VIDEOIO_PRIORITY_MSMF=0. MSMF is the
    # backend that reaches full frame rate on USB webcams where DirectShow
    # negotiates an uncompressed format and stalls - see capture/negotiate.py.


# Backwards-compatible alias for the original, narrower name.
quiet_native_backends = configure_native_backends
