"""Tests for passenger.classifier, classification metrics and their pipeline/evaluate wiring."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from passenger import metrics
from passenger.classifier import Attribute, AttributeService, PassengerClassifier
from passenger.data_manager import DataManager
from passenger.evaluate import evaluate
from passenger.pipeline import PassengerPipeline
from passenger.seats import Detection, SeatMap

FRAME = np.full((200, 200, 3), 90, np.uint8)
FULL = (0.0, 0.0, 1.0, 1.0)


class FakeModel:
    def __init__(self, probs):
        self.probs, self.last_shape = probs, None

    def predict(self, face):
        self.last_shape = face.shape
        return self.probs


def clf(probs=(0.9, 0.1, 0.0), face=(50, 50, 40, 40), **kw):
    return PassengerClassifier(FakeModel(probs), lambda crop: face, **kw)


# -- PassengerClassifier ---------------------------------------------------
def test_male_female_child_decisions():
    assert clf((0.9, 0.1, 0.0)).classify(FRAME, FULL) == Attribute("male", 0.9)
    assert clf((0.1, 0.9, 0.0)).classify(FRAME, FULL) == Attribute("female", 0.9)
    assert clf((0.9, 0.1, 0.8)).classify(FRAME, FULL) == Attribute("child", 0.8)     # age beats gender


def test_low_confidence_gender_is_unknown_not_a_guess():
    assert clf((0.55, 0.45, 0.0)).classify(FRAME, FULL).label == "unknown"


def test_unknown_when_person_too_small_or_degenerate():
    c = clf(min_person_px=60)
    assert c.classify(FRAME, (0.1, 0.1, 0.3, 0.2)).label == "unknown"        # 20 px tall
    assert c.classify(FRAME, (0.5, 0.5, 0.5, 0.9)).label == "unknown"        # zero width


def test_unknown_when_no_face_or_face_too_small():
    assert clf(face=None).classify(FRAME, FULL).label == "unknown"
    assert clf(face=(10, 10, 12, 30), min_face_px=24).classify(FRAME, FULL).label == "unknown"


def test_box_outside_frame_is_clamped():
    assert clf().classify(FRAME, (-0.5, -0.5, 1.5, 1.5)).label == "male"


def test_face_is_cropped_with_margin_inside_the_person():
    model = FakeModel((0.9, 0.1, 0.0))
    c = PassengerClassifier(model, lambda crop: (50, 50, 40, 40), face_margin=0.5)
    c.classify(FRAME, FULL)
    assert model.last_shape[:2] == (80, 80)                  # 40 face + 20 px margin each side


# -- AttributeService ------------------------------------------------------
class ScriptedClassifier:
    """Returns scripted attributes per call; counts calls."""

    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def classify(self, frame, box):
        a = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return a


def det(tid, box=FULL):
    return Detection(box, 0.9, track_id=tid)


def test_label_needs_min_votes_then_appears():
    svc = AttributeService(ScriptedClassifier([Attribute("female", 0.9)]), min_votes=3)
    out = [svc.update(FRAME, [det(1)])[1] for _ in range(3)]
    assert out == ["unknown", "unknown", "female"]


def test_smoothing_ignores_a_minority_of_bad_frames():
    script = [Attribute("male", 0.9)] * 4 + [Attribute("female", 0.7)] + [Attribute("male", 0.9)] * 3
    svc = AttributeService(ScriptedClassifier(script), min_votes=3)
    labels = [svc.update(FRAME, [det(1)])[1] for _ in range(8)]
    assert labels[-1] == "male" and "female" not in labels[2:]


def test_unknown_results_do_not_vote():
    svc = AttributeService(ScriptedClassifier([Attribute("unknown", 0.0)]), min_votes=1)
    assert svc.update(FRAME, [det(1)]) == {1: "unknown"}


def test_a_failed_face_read_is_not_retried_every_frame():
    scripted = ScriptedClassifier([Attribute("unknown", 0.0)])
    svc = AttributeService(scripted, min_votes=1, retry_after=3)
    for _ in range(7):
        svc.update(FRAME, [det(1)])
    assert scripted.calls == 3                         # ticks 1, 4 and 7, not all seven


def test_budget_caps_calls_and_serves_fewest_votes_first():
    scripted = ScriptedClassifier([Attribute("male", 0.9)])
    svc = AttributeService(scripted, min_votes=1, budget=2)
    svc.update(FRAME, [det(1), det(2), det(3)])
    assert scripted.calls == 2
    out = svc.update(FRAME, [det(1), det(2), det(3)])        # person 3 has no votes, so it goes first
    assert out[3] == "male" and scripted.calls == 4


def test_untracked_detections_are_skipped():
    scripted = ScriptedClassifier([Attribute("male", 0.9)])
    svc = AttributeService(scripted, min_votes=1)
    assert svc.update(FRAME, [Detection(FULL, 0.9, track_id=None)]) == {} and scripted.calls == 0


def test_stale_tracks_are_forgotten():
    svc = AttributeService(ScriptedClassifier([Attribute("male", 0.9)]), min_votes=1, stale_after=3)
    svc.update(FRAME, [det(1)])
    for _ in range(5):
        svc.update(FRAME, [det(2)])
    assert svc.label(1) == "unknown" and 1 not in svc._seen_at


# -- classification metrics ------------------------------------------------
def test_classification_metrics_confusion_and_unknown_as_miss():
    gt = ["male", "male", "male", "female", "child"]
    pred = ["male", "female", "unknown", "female", "child"]
    r = metrics.classification_metrics(pred, gt)
    assert r["confusion"]["male"] == {"male": 1, "female": 1, "child": 0, "unknown": 1}
    assert r["per_class"]["male"]["recall"] == pytest.approx(1 / 3)       # abstention counts against recall
    assert r["per_class"]["male"]["precision"] == 1.0
    assert r["per_class"]["female"]["precision"] == 0.5
    assert r["coverage"] == pytest.approx(4 / 5)
    assert r["accuracy_answered"] == pytest.approx(3 / 4)
    assert r["accuracy_overall"] == pytest.approx(3 / 5)


def test_abstaining_cannot_beat_answering():
    gt = ["male", "female"] * 5
    guesser = metrics.classification_metrics(gt, gt)
    coward = metrics.classification_metrics(["unknown"] * 10, gt)
    assert guesser["macro_f1"] == 1.0 and coward["macro_f1"] == 0.0 and coward["coverage"] == 0.0


def test_macro_f1_skips_classes_absent_from_ground_truth():
    r = metrics.classification_metrics(["male", "male"], ["male", "male"])
    assert r["macro_f1"] == 1.0 and r["per_class"]["child"]["support"] == 0


def test_classification_metrics_empty_and_mismatch():
    assert metrics.classification_metrics([], [])["coverage"] == 0.0
    with pytest.raises(ValueError):
        metrics.classification_metrics(["male"], [])


def test_match_pairs_is_one_to_one():
    a, b = (0.1, 0.1, 0.3, 0.3), (0.11, 0.11, 0.31, 0.31)
    pairs = metrics.match_pairs([(a, 0.9), (b, 0.8)], [a])
    assert pairs == [(0, 0)]                                  # second prediction cannot reuse the same truth box


# -- pipeline wiring ---------------------------------------------------------
class OneDetector:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, frame, track=True):
        return self.dets


def test_pipeline_reports_composition_and_seat_labels():
    box_a, box_b = (0.1, 0.1, 0.3, 0.3), (0.6, 0.6, 0.8, 0.8)
    seats = SeatMap({"A": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]],
                     "B": [[0.6, 0.6], [0.8, 0.6], [0.8, 0.8], [0.6, 0.8]]}, min_overlap=0.5)
    dets = [Detection(box_a, 0.9, 1), Detection(box_b, 0.9, 2), Detection((0.4, 0.4, 0.5, 0.5), 0.9, None)]

    class ByBox:
        def classify(self, frame, box):
            return Attribute("child" if box == box_b else "female", 0.9)

    pipe = PassengerPipeline(OneDetector(dets), seats, attributes=AttributeService(ByBox(), min_votes=1))
    s = pipe.process(FRAME)
    assert s.seat_labels == {"A": "female", "B": "child"}
    assert s.composition == {"male": 0, "female": 1, "child": 1, "unknown": 1}


def test_pipeline_without_classifier_reports_everyone_unknown():
    seats = SeatMap({"A": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]]})
    s = PassengerPipeline(OneDetector([Detection((0.1, 0.1, 0.3, 0.3), 0.9, 1)]), seats).process(FRAME)
    assert s.composition["unknown"] == 1 and s.seat_labels == {"A": "unknown"}


# -- evaluate ---------------------------------------------------------------
def _dataset(tmp_path, class_id=1):
    dm = DataManager(tmp_path / "pd")
    for i in range(3):
        src = tmp_path / f"v{i}.mp4"
        w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
        for k in range(4):
            w.write(np.full((48, 64, 3), 40 * (k + 1) + i, np.uint8))
        w.release()
        dm.extract_frames(dm.register_video(src), target_fps=5, dedupe_threshold=0)
    dm.assign_splits()
    for jpg in dm.frames_dir.glob("*/*.jpg"):
        (dm.labels_dir / f"{jpg.stem}.txt").write_text(f"{class_id} 0.2 0.2 0.2 0.2\n")
    return dm, SeatMap({"A": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]]}, min_overlap=0.5)


class PerfectDetector:
    def detect(self, frame, track=True):
        return [Detection((0.1, 0.1, 0.3, 0.3), 0.9)]


class Always:
    def __init__(self, label):
        self.label = label

    def classify(self, frame, box):
        return Attribute(self.label, 0.9)


def test_evaluate_scores_classification_against_labels(tmp_path):
    dm, seats = _dataset(tmp_path, class_id=1)               # every labelled person is 'female'
    right = evaluate(dm, PerfectDetector(), seats, "test", warmup=0, classifier=Always("female"))
    assert right["classification"]["macro_f1"] == 1.0 and right["classification"]["coverage"] == 1.0
    wrong = evaluate(dm, PerfectDetector(), seats, "test", warmup=0, classifier=Always("male"))
    assert wrong["classification"]["per_class"]["female"]["recall"] == 0.0
    assert "classification" not in evaluate(dm, PerfectDetector(), seats, "test", warmup=0)


# -- real weights smoke test ---------------------------------------------------
_NETS = Path("models/passenger")
_HAVE_NETS = all((_NETS / f).exists() for f in ("gender_net.caffemodel", "age_net.caffemodel",
                                                "gender_deploy.prototxt", "age_deploy.prototxt"))


@pytest.mark.skipif(not _HAVE_NETS, reason="age/gender weights not downloaded")
def test_real_age_gender_net_returns_valid_probabilities():
    from passenger.classifier import AgeGenderNet
    net = AgeGenderNet(_NETS)
    img = np.random.default_rng(0).integers(0, 255, (120, 100, 3), dtype=np.uint8)
    p_male, p_female, p_child = net.predict(img)
    assert p_male + p_female == pytest.approx(1.0, abs=1e-3) and 0.0 <= p_child <= 1.0


# -- BackgroundAttributes --------------------------------------------------
def test_background_attributes_never_block_the_frame_loop():
    import threading
    import time

    from passenger.classifier import BackgroundAttributes

    release = threading.Event()

    class Slow:
        def classify(self, frame, box):
            release.wait(5)
            return Attribute("female", 0.9)

    det = Detection((0.1, 0.1, 0.5, 0.9), 0.9, 7)
    bg = BackgroundAttributes(AttributeService(Slow(), min_votes=1))
    try:
        frame = np.zeros((20, 20, 3), np.uint8)
        t = time.perf_counter()
        first = bg.update(frame, [det])
        assert time.perf_counter() - t < 0.5            # did not wait for the classifier
        assert first == {7: "unknown"}                  # nothing known yet
        release.set()
        for _ in range(100):                            # the label arrives on a later frame
            if bg.update(frame, [det]) == {7: "female"}:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("label never arrived")
    finally:
        release.set()
        bg.close()
