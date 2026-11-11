"""Tracker settings for the person detector (``passenger.detector.tracker``): ByteTrack memory and thresholds from config.yaml."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from passenger.detector import PersonDetector, tracker_yaml


def test_overrides_are_merged_over_the_stock_bytetrack_settings():
    path = tracker_yaml({"track_buffer": 60, "new_track_thresh": 0.35})
    cfg = yaml.safe_load(Path(path).read_text())
    assert cfg["tracker_type"] == "bytetrack" and cfg["track_buffer"] == 60 and cfg["new_track_thresh"] == 0.35
    assert cfg["track_low_thresh"] == 0.1 and cfg["match_thresh"] == 0.8                   # untouched stock values


def test_the_same_settings_give_the_same_file_and_different_settings_a_different_one():
    a, b, c = tracker_yaml({"track_buffer": 60}), tracker_yaml({"track_buffer": 60}), tracker_yaml({"track_buffer": 61})
    assert a == b and a != c


def test_an_unknown_or_misspelt_setting_is_refused():
    with pytest.raises(ValueError, match="track_bufer"):
        tracker_yaml({"track_bufer": 60})
    with pytest.raises(ValueError):
        tracker_yaml({"track_buffer": "long"})


class SpyModel:
    def __init__(self):
        self.kwargs = None

    def track(self, frame, **kw):
        self.kwargs = kw

        class R:
            boxes = None
        return [R()]


def detector(tracker=None):
    d = PersonDetector.__new__(PersonDetector)                  # no weights needed to check which tracker config is used
    d._model, d.conf, d.imgsz, d.classes = SpyModel(), 0.35, 416, [0]
    if tracker is not None:
        d.tracker = tracker_yaml(tracker)
    return d


def test_the_stock_tracker_is_used_unless_the_config_names_settings():
    d = detector()
    d.detect(np.zeros((9, 16, 3), np.uint8))
    assert d._model.kwargs["tracker"] == "bytetrack.yaml"


def test_configured_tracker_settings_reach_the_model():
    d = detector({"track_buffer": 60})
    d.detect(np.zeros((9, 16, 3), np.uint8))
    assert d._model.kwargs["tracker"] == tracker_yaml({"track_buffer": 60}) and d._model.kwargs["persist"] is True


# ---- duplicate boxes: how much two boxes may overlap before they count as one person ------------------
def test_the_overlap_limit_reaches_the_model_only_when_configured():
    d = detector()
    d.detect(np.zeros((9, 16, 3), np.uint8))
    assert "iou" not in d._model.kwargs                                   # stock behaviour: the model's own default
    d = detector()
    d.iou = 0.5
    d.detect(np.zeros((9, 16, 3), np.uint8))
    assert d._model.kwargs["iou"] == 0.5


def test_the_overlap_limit_is_validated_by_the_constructor(monkeypatch):
    import types
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=lambda *a, **k: object()))
    assert PersonDetector("m.pt", iou=0.5).iou == 0.5
    assert PersonDetector("m.pt").iou is None
    for bad in (0, -0.1, 1.5, "half"):
        with pytest.raises(ValueError, match="iou"):
            PersonDetector("m.pt", iou=bad)
