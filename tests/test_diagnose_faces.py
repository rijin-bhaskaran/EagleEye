import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("diagnose_faces", Path(__file__).resolve().parents[1] / "scripts" / "diagnose_faces.py")
df = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(df)

KW = dict(min_person_px=60, min_face_px=24, min_conf=0.5)


def test_stage_order():
    assert df.stage_of(40, (0, 0, 50, 50), 0.9, **KW) == "person_too_short"
    assert df.stage_of(100, None, None, **KW) == "no_face"
    assert df.stage_of(100, (0, 0, 20, 40), 0.9, **KW) == "face_too_small"
    assert df.stage_of(100, (0, 0, 40, 40), 0.4, **KW) == "low_confidence"
    assert df.stage_of(100, (0, 0, 40, 40), 0.8, **KW) == "labelled"


def test_child_is_labelled_regardless_of_gender_confidence():
    assert df.stage_of(100, (0, 0, 40, 40), 0.3, child=True, **KW) == "labelled"


def test_summarise_shares_sum_to_one_and_handle_empty():
    s = df.summarise(["no_face", "no_face", "labelled", "person_too_short"])
    assert s["no_face"] == (2, 0.5)
    assert abs(sum(v[1] for v in s.values()) - 1.0) < 1e-9
    assert all(v == (0, 0.0) for v in df.summarise([]).values())


# --- face finder selection -------------------------------------------------------------------------------------

import numpy as np
import pytest

from eagleeye.config import _wrap, load_config
from passenger import classifier as C

YUNET = Path(__file__).resolve().parents[1] / "models" / "passenger" / "face_detection_yunet_2023mar.onnx"


def test_yunet_missing_weights_falls_back_to_mediapipe(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(C, "MediaPipeFaceFinder", lambda conf: "mp")
    c = _wrap({"face_detector": "yunet", "yunet_weights": str(tmp_path / "none.onnx"), "face_confidence": 0.5})
    assert C.make_face_finder(c) == "mp"
    assert "falling back to MediaPipe" in capsys.readouterr().out


def test_default_detector_is_mediapipe(monkeypatch):
    monkeypatch.setattr(C, "MediaPipeFaceFinder", lambda conf: "mp")
    assert C.make_face_finder(load_config().passenger.classifier) == "mp"


@pytest.mark.skipif(not YUNET.is_file(), reason="run scripts/fetch_passenger_models.py")
def test_yunet_finder_handles_blank_and_tiny_crops():
    f = C.YuNetFaceFinder(YUNET)
    assert f(np.zeros((120, 80, 3), np.uint8)) is None          # no face
    assert f(np.zeros((4, 4, 3), np.uint8)) is None             # too small to search


def test_face_margin_comes_from_config():
    fake = lambda crop: (40, 40, 40, 40)
    tight = C.PassengerClassifier(None, fake, min_person_px=10, face_margin=0.0)
    loose = C.PassengerClassifier(None, fake, min_person_px=10, face_margin=0.5)
    frame = np.zeros((200, 200, 3), np.uint8)
    box = (0.0, 0.0, 1.0, 1.0)
    assert tight.face_crop(frame, box).shape[:2] == (40, 40)
    assert loose.face_crop(frame, box).shape[:2] == (80, 80)
