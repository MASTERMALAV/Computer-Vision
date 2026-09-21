"""Camera identity: (backend, index) pairs, naming and ranking.

Regression cover for a real bug. DirectShow enumerated ``[Integrated, Brio]``
while Media Foundation's index 0 *was* the Brio - the orders were reversed.
Selecting "brio" resolved to DirectShow index 1 and then captured from MSMF
index 1, which was the built-in camera. Everything looked correct: the right
name was reported, frames arrived, nothing errored.
"""

from __future__ import annotations

import numpy as np
import pytest

from argus.capture.devices import CameraDevice, resolve_device
from argus.capture.picker import Candidate, _infer_names, best_candidate


def candidate(backend, index, name="", fps=30.0, blank=False, works=True,
              thumb_value=None, source="unknown", width=1280, height=720):
    thumb = None
    if thumb_value is not None:
        thumb = np.full((90, 160, 3), float(thumb_value), dtype=np.float32)
        # A little structure so std() is non-zero and `blank` is not inferred.
        thumb[::2] += 20.0
    return Candidate(
        backend=backend, index=index, works=works, width=width, height=height,
        fps=fps, name=name, name_source=source, thumbnail=thumb, blank=blank,
    )


# --------------------------------------------------------------------------- #
# A camera is a (backend, index) pair
# --------------------------------------------------------------------------- #
def test_spec_is_backend_and_index():
    assert CameraDevice(index=0, name="x", backend="msmf").spec == "msmf:0"


def test_explicit_spec_locks_the_backend():
    dev = resolve_device("msmf:0", [])
    assert dev.backend == "msmf"
    assert dev.index == 0
    assert dev.backend_locked is True


def test_spec_parsing_is_case_insensitive_and_tolerates_spaces():
    assert resolve_device("MSMF:1", []).spec == "msmf:1"
    assert resolve_device("dshow : 2", []).spec == "dshow:2"


def test_spec_matches_an_enumerated_device_of_that_backend():
    devices = [
        CameraDevice(index=1, name="Brio 100", backend="dshow"),
        CameraDevice(index=1, name="Integrated Camera", backend="msmf"),
    ]
    # The same index under two backends is two different cameras.
    assert resolve_device("dshow:1", devices).name == "Brio 100"
    assert resolve_device("msmf:1", devices).name == "Integrated Camera"


def test_reversed_backend_orders_do_not_confuse_explicit_specs():
    """The exact situation that caused the bug."""
    devices = [
        CameraDevice(index=0, name="Integrated Camera", backend="dshow"),
        CameraDevice(index=1, name="Brio 100", backend="dshow"),
        CameraDevice(index=0, name="Brio 100", backend="msmf"),
        CameraDevice(index=1, name="Integrated Camera", backend="msmf"),
    ]
    assert resolve_device("msmf:0", devices).name == "Brio 100"
    assert resolve_device("dshow:1", devices).name == "Brio 100"
    # Both are the Brio, under different backends - and neither is index-only.
    assert resolve_device("msmf:0", devices).spec != resolve_device("dshow:1", devices).spec


def test_plain_index_is_not_backend_locked():
    """An index alone cannot identify a camera, so it must not claim to."""
    assert resolve_device("1", []).backend_locked is False


# --------------------------------------------------------------------------- #
# Naming by what the camera sees
# --------------------------------------------------------------------------- #
def test_msmf_camera_is_named_from_the_matching_dshow_view():
    cands = [
        candidate("dshow", 0, "Integrated Camera", fps=5.0, thumb_value=30, source="dshow"),
        candidate("dshow", 1, "Brio 100", fps=5.0, thumb_value=140, source="dshow"),
        candidate("msmf", 0, fps=30.0, thumb_value=141),  # looks like the Brio
    ]
    _infer_names(cands)
    assert cands[2].name == "Brio 100"
    assert cands[2].name_source == "inferred"


def test_an_ambiguous_view_is_left_unnamed():
    """Two references that look alike must not produce a confident guess."""
    cands = [
        candidate("dshow", 0, "Camera A", thumb_value=100, source="dshow"),
        candidate("dshow", 1, "Camera B", thumb_value=101, source="dshow"),
        candidate("msmf", 0, thumb_value=100),
    ]
    _infer_names(cands)
    assert cands[2].name == ""


def test_a_wildly_different_view_is_left_unnamed():
    cands = [
        candidate("dshow", 1, "Brio 100", thumb_value=20, source="dshow"),
        candidate("msmf", 0, thumb_value=200),
    ]
    _infer_names(cands)
    assert cands[1].name == ""


def test_blank_cameras_are_not_used_as_naming_references():
    """A lens-capped camera is all black and would match anything else black."""
    cands = [
        candidate("dshow", 0, "Integrated Camera", thumb_value=0, blank=True, source="dshow"),
        candidate("dshow", 1, "Brio 100", thumb_value=140, source="dshow"),
        candidate("msmf", 0, thumb_value=0, blank=True),
    ]
    _infer_names(cands)
    assert cands[2].name == ""


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def test_frame_rate_beats_a_recognisable_name():
    """The same camera often appears twice, one path stalling at 5 fps.

    Picking the branded-but-slow entry is what an earlier version did, and a
    5 fps cursor is unusable regardless of which camera produced it.
    """
    cands = [
        candidate("dshow", 1, "Brio 100", fps=5.0, thumb_value=140),
        candidate("msmf", 0, "Brio 100", fps=30.0, thumb_value=140),
    ]
    best = best_candidate(cands)
    assert best.spec == "msmf:0"


def test_black_cameras_are_never_chosen():
    cands = [
        candidate("msmf", 1, "Integrated Camera", fps=30.0, blank=True, thumb_value=0),
        candidate("msmf", 0, "Brio 100", fps=28.0, thumb_value=140),
    ]
    assert best_candidate(cands).spec == "msmf:0"


def test_external_preferred_when_speed_is_equal():
    cands = [
        candidate("msmf", 1, "Integrated Camera", fps=30.0, thumb_value=100),
        candidate("msmf", 0, "Brio 100", fps=30.0, thumb_value=140),
    ]
    assert best_candidate(cands).spec == "msmf:0"


def test_non_working_cameras_are_excluded():
    cands = [
        candidate("msmf", 0, "Brio 100", fps=0.0, works=False),
        candidate("msmf", 1, "Integrated Camera", fps=25.0, thumb_value=100),
    ]
    assert best_candidate(cands).spec == "msmf:1"


def test_nothing_usable_returns_none():
    assert best_candidate([]) is None
    assert best_candidate([candidate("msmf", 0, works=False)]) is None


def test_all_blank_still_returns_none_rather_than_a_black_camera():
    cands = [candidate("msmf", 0, "Integrated", fps=30.0, blank=True, thumb_value=0)]
    assert best_candidate(cands) is None


def test_candidate_converts_to_a_locked_device():
    dev = candidate("msmf", 0, "Brio 100", fps=30.0, thumb_value=140).to_device()
    assert dev.backend_locked is True
    assert dev.spec == "msmf:0"
    assert dev.name == "Brio 100"
