"""Config layering, coercion and validation."""

from __future__ import annotations

import pytest

from argus.config import ArgusConfig, ConfigError


def test_defaults_are_valid():
    cfg = ArgusConfig()
    cfg.validate()
    assert cfg.capture.camera.device == "auto"
    assert cfg.capture.camera.fourcc == "MJPG"


def test_override_scalar():
    cfg = ArgusConfig.load(None, ["capture.camera.width=1920"])
    assert cfg.capture.camera.width == 1920
    assert isinstance(cfg.capture.camera.width, int)


def test_override_coerces_types():
    cfg = ArgusConfig.load(None, ["capture.camera.mirror=false", "face.detector.conf_threshold=.7"])
    assert cfg.capture.camera.mirror is False
    assert cfg.face.detector.conf_threshold == pytest.approx(0.7)


def test_override_optional_bool_accepts_null():
    cfg = ArgusConfig.load(None, ["capture.camera.autofocus=null"])
    assert cfg.capture.camera.autofocus is None
    cfg = ArgusConfig.load(None, ["capture.camera.autofocus=true"])
    assert cfg.capture.camera.autofocus is True


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match="Unknown config key"):
        ArgusConfig.load(None, ["capture.camera.widht=1920"])


def test_unknown_section_is_rejected():
    with pytest.raises(ConfigError, match="Unknown config section"):
        ArgusConfig.load(None, ["camrea.width=1920"])


def test_malformed_override_is_rejected():
    with pytest.raises(ConfigError, match="section.key=value"):
        ArgusConfig.load(None, ["capture.camera.width"])


def test_yaml_layer_and_strictness(tmp_path):
    good = tmp_path / "good.yaml"
    good.write_text("capture:\n  camera:\n    device: brio\n    width: 1920\n", encoding="utf-8")
    cfg = ArgusConfig.load(good)
    assert cfg.capture.camera.device == "brio"
    assert cfg.capture.camera.width == 1920

    bad = tmp_path / "bad.yaml"
    bad.write_text("capture:\n  camera:\n    widht: 1920\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown config key"):
        ArgusConfig.load(bad)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        ArgusConfig.load(tmp_path / "nope.yaml")


def test_profile_moves_multiple_knobs():
    fast = ArgusConfig.load(None, ["runtime.profile=fast"])
    accurate = ArgusConfig.load(None, ["runtime.profile=accurate"])
    assert fast.face.detector.input_size < accurate.face.detector.input_size
    assert fast.hands.model_complexity == 0
    assert accurate.face.recognizer.model == "arcface_r50"
    assert fast.face.detect_every_n_frames > accurate.face.detect_every_n_frames


def test_explicit_setting_beats_profile():
    # The profile would pick 448 for 'fast'; an explicit value must survive.
    cfg = ArgusConfig.load(None, ["runtime.profile=fast", "face.detector.input_size=640"])
    assert cfg.face.detector.input_size == 640


def test_explicit_yaml_beats_profile(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "runtime:\n  profile: fast\nhands:\n  model_complexity: 1\n",
        encoding="utf-8",
    )
    cfg = ArgusConfig.load(path)
    assert cfg.runtime.profile == "fast"
    assert cfg.hands.model_complexity == 1


def test_validation_rejects_bad_values():
    with pytest.raises(ConfigError, match="multiple of 32"):
        ArgusConfig.load(None, ["face.detector.input_size=500"])
    with pytest.raises(ConfigError, match="backend"):
        ArgusConfig.load(None, ["capture.camera.backend=v4l2"])
    with pytest.raises(ConfigError, match="fourcc"):
        ArgusConfig.load(None, ["capture.camera.fourcc=MJPEG"])
    with pytest.raises(ConfigError, match="match_threshold"):
        ArgusConfig.load(None, ["face.recognizer.match_threshold=1.5"])


def test_roundtrip_through_yaml():
    cfg = ArgusConfig.load(None, ["capture.camera.device=brio"])
    text = cfg.dump_yaml()
    assert "brio" in text
    # The dumped config must itself be loadable - it is what `config --out` writes.
    import yaml

    reparsed = ArgusConfig.load(None, [])
    from argus.config import _merge_into

    _merge_into(reparsed, yaml.safe_load(text), prefix="")
    reparsed.validate()
    assert reparsed.capture.camera.device == "brio"


def test_shipped_default_config_is_valid():
    """configs/default.yaml must always load - it is what users start from."""
    from argus.config import ROOT

    path = ROOT / "configs" / "default.yaml"
    if path.exists():
        ArgusConfig.load(path).validate()


# --------------------------------------------------------------------------- #
# Retired keys
#
# Unknown keys are a hard error so a typo fails loudly. A key that used to be
# valid is different: the user's file was correct when written, and the software
# changed underneath it. Scroll moved from a two-finger pose to a held middle
# pinch, which retired gestures.scroll_middle_extended - and every previously
# written calibration file contains it.
# --------------------------------------------------------------------------- #
def test_retired_key_is_ignored_not_rejected(tmp_path):
    path = tmp_path / "old.yaml"
    path.write_text(
        "gestures:\n  pinch_close: 0.2\n  scroll_middle_extended: 0.958\n",
        encoding="utf-8",
    )
    cfg = ArgusConfig.load(path)
    cfg.validate()
    assert cfg.gestures.pinch_close == 0.2


def test_retired_key_as_an_override_is_ignored():
    cfg = ArgusConfig.load(None, ["gestures.scroll_middle_extended=1.0"])
    cfg.validate()


def test_a_genuine_typo_is_still_rejected():
    with pytest.raises(ConfigError, match="Unknown config key"):
        ArgusConfig.load(None, ["gestures.pinch_clsoe=0.3"])


def test_retired_keys_all_name_their_replacement():
    from argus.config import DEPRECATED_KEYS

    assert DEPRECATED_KEYS
    for key, message in DEPRECATED_KEYS.items():
        assert "." in key, "a retired key must be a dotted path"
        assert len(message) > 30, f"{key} needs an explanation, not a stub"
