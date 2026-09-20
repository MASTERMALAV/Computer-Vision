"""Face stack: detector decoding, alignment, gallery matching, sessions."""

from __future__ import annotations

import numpy as np
import pytest

from argus.config import SecurityConfig
from argus.face.align import ARCFACE_TEMPLATE, quality_problems, umeyama
from argus.face.detector import distance2bbox, distance2kps, nms
from argus.face.embedder import EMBEDDING_DIM, cosine_similarity, normalize, similarity_matrix
from argus.face.gallery import Gallery, GalleryError
from argus.face.session import IdentitySession


# --------------------------------------------------------------------------- #
# Detector decoding
# --------------------------------------------------------------------------- #
def test_distance2bbox_decodes_around_the_anchor():
    points = np.array([[10.0, 20.0]])
    # distances to left, top, right, bottom
    distance = np.array([[1.0, 2.0, 3.0, 4.0]])
    box = distance2bbox(points, distance)[0]
    assert box.tolist() == [9.0, 18.0, 13.0, 24.0]


def test_distance2kps_produces_five_points():
    points = np.array([[10.0, 20.0]])
    distance = np.arange(10, dtype=np.float64).reshape(1, 10)
    kps = distance2kps(points, distance)
    assert kps.shape == (1, 5, 2)
    assert kps[0][0].tolist() == [10.0, 21.0]
    assert kps[0][4].tolist() == [18.0, 29.0]


def test_nms_suppresses_overlapping_boxes():
    boxes = np.array(
        [
            [0, 0, 10, 10],
            [1, 1, 11, 11],  # almost identical - should be suppressed
            [100, 100, 110, 110],  # far away - should survive
        ],
        dtype=np.float64,
    )
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, 0.4)
    assert keep[0] == 0
    assert 1 not in keep
    assert 2 in keep


def test_nms_keeps_the_highest_scoring_box():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float64)
    keep = nms(boxes, np.array([0.3, 0.95]), 0.4)
    assert keep == [1]


def test_nms_on_empty_input():
    assert nms(np.zeros((0, 4)), np.zeros(0), 0.4) == []


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def test_umeyama_recovers_an_exact_similarity():
    theta = np.radians(23.0)
    scale = 1.6
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    src = ARCFACE_TEMPLATE.astype(np.float64)
    dst = scale * src @ rot.T + np.array([11.0, -6.0])

    matrix = umeyama(src, dst)
    mapped = np.c_[src, np.ones(len(src))] @ matrix.T
    assert np.abs(mapped - dst).max() < 1e-3


def test_umeyama_identity():
    matrix = umeyama(ARCFACE_TEMPLATE, ARCFACE_TEMPLATE)
    assert np.allclose(matrix, np.array([[1, 0, 0], [0, 1, 0]]), atol=1e-6)


def test_umeyama_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        umeyama(np.zeros((5, 2)), np.zeros((4, 2)))


def test_quality_problems_flag_bad_enrolment_samples():
    good = {"size_px": 160.0, "roll_deg": 5.0, "yaw_ratio": 0.05,
            "edge_margin_px": 40.0, "score": 0.9}
    assert quality_problems(good) == []

    assert any("small" in p for p in quality_problems({**good, "size_px": 30.0}))
    assert any("tilted" in p for p in quality_problems({**good, "roll_deg": 40.0}))
    assert any("turned" in p for p in quality_problems({**good, "yaw_ratio": 0.9}))
    assert any("edge" in p for p in quality_problems({**good, "edge_margin_px": 0.0}))


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def test_normalize_produces_unit_vectors():
    v = normalize(np.array([3.0, 4.0]))
    assert np.linalg.norm(v) == pytest.approx(1.0)


def test_normalize_survives_a_zero_vector():
    v = normalize(np.zeros(8))
    assert np.all(np.isfinite(v))


def test_cosine_similarity_bounds():
    a = np.array([1.0, 0.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0)
    assert cosine_similarity(a, np.array([-1.0, 0.0])) == pytest.approx(-1.0)
    assert cosine_similarity(a, np.array([0.0, 1.0])) == pytest.approx(0.0, abs=1e-6)


def test_similarity_matrix_shape_and_values():
    a = np.eye(3, dtype=np.float32)
    sims = similarity_matrix(a, a)
    assert sims.shape == (3, 3)
    assert np.allclose(np.diag(sims), 1.0)


# --------------------------------------------------------------------------- #
# Gallery
# --------------------------------------------------------------------------- #
def person(seed: int, n: int = 5, spread: float = 0.05) -> np.ndarray:
    """N embeddings clustered around one random direction."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=EMBEDDING_DIM)
    base /= np.linalg.norm(base)
    out = base[None, :] + rng.normal(scale=spread, size=(n, EMBEDDING_DIM))
    return (out / np.linalg.norm(out, axis=1, keepdims=True)).astype(np.float32)


def test_empty_gallery_rejects_everything():
    g = Gallery(model="m")
    result = g.match(person(0, 1)[0], threshold=0.4)
    assert not result.accepted
    assert "empty" in result.reason


def test_matches_the_enrolled_person():
    g = Gallery(model="m")
    samples = person(1)
    g.add("malav", samples)
    probe = person(1, n=1)[0]  # same cluster
    result = g.match(probe, threshold=0.4)
    assert result.is_known
    assert result.name == "malav"


def test_rejects_a_stranger():
    g = Gallery(model="m")
    g.add("malav", person(1))
    stranger = person(999, n=1)[0]  # an unrelated direction
    result = g.match(stranger, threshold=0.4)
    assert not result.accepted
    assert "threshold" in result.reason


def test_margin_rejects_an_ambiguous_match():
    """Two people who score almost identically must not be guessed between."""
    g = Gallery(model="m")
    shared = person(2, n=1)
    g.add("a", shared)
    g.add("b", shared)  # deliberately identical
    result = g.match(shared[0], threshold=0.3, margin=0.05)
    assert not result.accepted
    assert "close to" in result.reason


def test_match_uses_the_best_sample_not_the_average():
    """A person enrolled across several poses must match any one of them."""
    g = Gallery(model="m")
    a, b = person(3, n=1), person(4, n=1)
    g.add("multi", np.vstack([a, b]))
    # The mean of two near-orthogonal vectors resembles neither; max does.
    assert g.match(a[0], threshold=0.5).is_known
    assert g.match(b[0], threshold=0.5).is_known


def test_add_rejects_empty_names_and_wrong_dimensions():
    g = Gallery(model="m")
    with pytest.raises(GalleryError):
        g.add("  ", person(1))
    with pytest.raises(GalleryError, match="512"):
        g.add("x", np.zeros((2, 128), dtype=np.float32))


def test_remove_identity():
    g = Gallery(model="m")
    g.add("a", person(1))
    assert g.remove("a") is True
    assert g.remove("a") is False
    assert g.is_empty


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "gallery.npz"
    g = Gallery(model="arcface_mbf")
    g.add("malav", person(1))
    g.add("other", person(2))
    g.save(path)

    loaded = Gallery.load(path, model="arcface_mbf")
    assert loaded.names == ["malav", "other"]
    assert loaded.count("malav") == 5
    assert loaded.model == "arcface_mbf"
    probe = person(1, n=1)[0]
    assert loaded.match(probe, 0.4).name == "malav"


def test_loading_with_a_different_model_is_refused(tmp_path):
    """Embeddings from different models are not comparable."""
    path = tmp_path / "gallery.npz"
    g = Gallery(model="arcface_mbf")
    g.add("malav", person(1))
    g.save(path)
    with pytest.raises(GalleryError, match="not comparable"):
        Gallery.load(path, model="arcface_r50")


def test_loading_a_missing_file_gives_an_empty_gallery(tmp_path):
    g = Gallery.load(tmp_path / "nope.npz", model="m")
    assert g.is_empty


def test_analyse_separates_genuine_from_impostor():
    g = Gallery(model="m")
    g.add("a", person(10, n=8, spread=0.05))
    g.add("b", person(20, n=8, spread=0.05))
    report = g.analyse()
    assert report["genuine"]["mean"] > report["impostor"]["mean"]
    assert report["separation"] > 0
    assert not report["overlap"]
    assert 0.0 < report["suggested_threshold"] < 1.0


def test_analyse_flags_overlap_when_people_look_alike():
    g = Gallery(model="m")
    base = person(30, n=8, spread=0.4)  # very noisy -> poor within-person match
    g.add("a", base[:4])
    g.add("b", base[4:])
    report = g.analyse()
    assert report["overlap"] is True


def test_analyse_with_one_person_says_so():
    g = Gallery(model="m")
    g.add("solo", person(40, n=6))
    report = g.analyse()
    assert report["impostor"] is None
    assert "note" in report


# --------------------------------------------------------------------------- #
# Identity session
# --------------------------------------------------------------------------- #
def make_session(**kw) -> IdentitySession:
    security = SecurityConfig(
        require_identity=True,
        session_timeout_s=kw.pop("timeout", 120.0),
        reverify_interval_s=kw.pop("reverify", 15.0),
        fresh_match_within_s=kw.pop("fresh", 5.0),
        **kw,
    )
    return IdentitySession(security=security)


def test_session_opens_on_first_match():
    s = make_session()
    event = s.observe_match("malav", 0.8, now=100.0)
    assert event.kind == "opened"
    assert s.is_valid(100.0)
    assert s.status(100.0).authenticated


def test_session_survives_the_face_being_hidden():
    """The whole point: an occluded face must not log the operator out."""
    s = make_session(timeout=120.0)
    s.observe_match("malav", 0.8, now=100.0)
    # 60 s later with no observations at all.
    assert s.is_valid(160.0)
    assert s.status(160.0).authenticated


def test_session_expires_after_the_timeout():
    s = make_session(timeout=30.0)
    s.observe_match("malav", 0.8, now=100.0)
    assert not s.is_valid(131.0)
    event = s.tick(131.0)
    assert event.kind == "expired"
    assert not s.status(131.0).authenticated


def test_freshness_is_stricter_than_validity():
    s = make_session(timeout=120.0, fresh=5.0)
    s.observe_match("malav", 0.8, now=100.0)
    assert s.is_valid(110.0)
    assert not s.is_fresh(110.0), "a destructive action needs a recent match"
    assert s.is_fresh(103.0)


def test_reverify_becomes_due():
    s = make_session(reverify=15.0)
    s.observe_match("malav", 0.8, now=100.0)
    assert not s.needs_reverify(105.0)
    assert s.needs_reverify(116.0)


def test_one_bad_observation_does_not_close_the_session():
    s = make_session()
    s.observe_match("malav", 0.8, now=100.0)
    assert s.observe_no_match(101.0) is None
    assert s.is_valid(101.0)


def test_repeated_rejection_closes_the_session():
    s = make_session()
    s.observe_match("malav", 0.8, now=100.0)
    event = None
    for i in range(s.reject_streak_to_close):
        event = s.observe_no_match(101.0 + i)
    assert event is not None and event.kind == "rejected"
    assert not s.is_open


def test_a_good_match_clears_the_rejection_streak():
    s = make_session()
    s.observe_match("malav", 0.8, now=100.0)
    s.observe_no_match(101.0)
    s.observe_no_match(102.0)
    s.observe_match("malav", 0.9, now=103.0)
    for i in range(3):
        assert s.observe_no_match(104.0 + i) is None
    assert s.is_open


def test_a_different_person_starts_a_new_session():
    s = make_session()
    s.observe_match("malav", 0.8, now=100.0)
    event = s.observe_match("someone_else", 0.8, now=101.0)
    assert event.kind == "switched"
    assert s.name == "someone_else"
    assert s.opened_at == 101.0


def test_operator_restriction_rejects_other_enrolled_people():
    s = make_session(operator="malav")
    s.observe_match("colleague", 0.9, now=100.0)
    status = s.status(100.0)
    assert not status.authenticated
    assert status.name == "colleague"


def test_gating_disabled_is_always_authenticated():
    s = IdentitySession(security=SecurityConfig(require_identity=False))
    assert s.status(100.0).authenticated
