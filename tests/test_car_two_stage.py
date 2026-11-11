"""``vehicle_type: car`` with ``passenger.car.pipeline: two_stage``: person detector + crop classifier, no 6-class cabin model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eagleeye.config import load_config
from passenger import car_pipeline
from passenger.car_pipeline import CarPipeline, TwoStageCarPipeline
from passenger.hub import PassengerHub, build_payload
from passenger.runner import PassengerRunner
from passenger.seats import Detection
from passenger.vehicle import build_pipeline

FRAME = np.zeros((90, 160, 3), np.uint8)
TWO = [Detection((0.1, 0.1, 0.4, 0.9), 0.9, 1), Detection((0.5, 0.2, 0.8, 0.9), 0.8, 2)]


class FakeDetector:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, frame, track=True):
        return list(self.dets)


class FakeAttributes:
    def __init__(self, labels):
        self.labels = labels

    def update(self, frame, dets):
        return {d.track_id: self.labels.get(d.track_id, "unknown") for d in dets if d.track_id is not None}


def test_state_carries_count_labels_and_a_seat_gauge_against_the_configured_seats():
    p = TwoStageCarPipeline(FakeDetector(TWO), FakeAttributes({1: "male", 2: "child"}), seats=5)
    st = p.process(FRAME)
    assert st.count == 2 and st.total_seats == 5 and st.seats_seen == (2, 3)
    assert st.composition == {"male": 1, "female": 0, "child": 1, "unknown": 0}
    assert st.track_labels == {1: "male", 2: "child"} and st.detections == TWO
    assert st.car is None and st.occupied == [] and st.vacant == []      # no belt or child-seat model behind this chain


def test_more_people_than_seats_never_gives_negative_vacant():
    many = [Detection((0.0, 0.0, 0.1, 0.1), 0.9, i) for i in range(1, 8)]
    st = TwoStageCarPipeline(FakeDetector(many), FakeAttributes({}), seats=5).process(FRAME)
    assert st.seats_seen == (5, 0) and st.count == 7


def test_untracked_people_count_but_stay_unknown_and_without_a_classifier_everyone_is_unknown():
    untracked = [Detection((0.1, 0.1, 0.4, 0.9), 0.9, None)]
    st = TwoStageCarPipeline(FakeDetector(untracked), FakeAttributes({}), seats=5).process(FRAME)
    assert st.count == 1 and st.composition["unknown"] == 1
    st = TwoStageCarPipeline(FakeDetector(TWO), None, seats=5).process(FRAME)
    assert st.composition == {"male": 0, "female": 0, "child": 0, "unknown": 2} and st.track_labels == {}


def test_the_count_is_smoothed_over_a_window():
    frames = [TWO, TWO, [], TWO, TWO]

    class Seq:
        i = 0

        def detect(self, frame, track=True):
            out = frames[self.i]
            self.i += 1
            return out

    p = TwoStageCarPipeline(Seq(), None, seats=5, count_window=5)
    counts = [p.process(FRAME).count for _ in frames]
    assert counts[-1] == 2                                              # one empty frame does not drop the count


def test_the_pipeline_says_which_model_is_reading_the_people():
    assert TwoStageCarPipeline(FakeDetector([]), None).classifier_label == "Crop model live"
    assert TwoStageCarPipeline.seatmap is None and TwoStageCarPipeline(FakeDetector([]), None).attributes is None


# ---- the dashboard text -------------------------------------------------------------------------
def test_payload_names_the_crop_model_only_when_a_classifier_is_on():
    st = TwoStageCarPipeline(FakeDetector(TWO), FakeAttributes({})).process(FRAME)
    assert build_payload(st, fps=10, latency_ms=5, classifier_on=True, classifier_label="Crop model live")["text"]["paxClassifier"] == "Crop model live"
    assert build_payload(st, fps=10, latency_ms=5, classifier_on=False, classifier_label="Crop model live")["text"]["paxClassifier"] == "Classifier off"
    assert build_payload(st, fps=10, latency_ms=5, classifier_on=True)["text"]["paxClassifier"] == "Face model live"     # unchanged default
    assert "car" not in build_payload(st, fps=10, latency_ms=5, classifier_on=True)             # no belt / child-seat block


class FakeStream:
    is_file, src_fps, cap = False, 30.0, None

    def read(self):
        return True, np.zeros((90, 160, 3), np.uint8), 0


def test_the_runner_publishes_the_pipelines_label_to_the_dashboard():
    p = TwoStageCarPipeline(FakeDetector(TWO), FakeAttributes({1: "male"}), seats=5)
    hub = PassengerHub()
    PassengerRunner(FakeStream(), p, hub, fps=10).step()
    snap = hub.snapshot()
    assert snap["text"]["paxClassifier"] == "Crop model live" and snap["text"]["paxCount"] == "2" and snap["text"]["paxMale"] == "1"


# ---- choosing the chain from the config ---------------------------------------------------------
def car_cfg(tmp_path, **car):
    from passenger.train_crops import build_net
    import torch

    w = tmp_path / "n.pt"
    torch.save({"state": build_net(3).state_dict(), "meta": {}}, w)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"passenger": {
        "vehicle_type": "car", "car": {"seats": 4, **car},
        "detector": {"model": "fake.pt", "conf": 0.35, "imgsz": 416, "backend": "torch", "classes": [0]},
        "classifier": {"model": "crops", "background": False, "window": 4, "min_votes": 2, "budget_per_frame": 2,
                       "crops": {"weights": [str(w)], "log_offsets": {"male": 0.0, "female": 2.0, "child": 0.5}}}}}))
    return load_config(str(p))


@pytest.fixture
def fake_person_detector(monkeypatch):
    seen = {}

    class Spy:
        def __init__(self, model, conf, imgsz, classes, backend="torch", tracker=None, iou=None):
            seen.update(model=model, conf=conf, imgsz=imgsz, classes=classes, backend=backend, tracker=tracker, iou=iou)

    monkeypatch.setattr("passenger.detector.PersonDetector", Spy)
    return seen


def test_car_two_stage_builds_the_person_detector_and_the_crop_classifier_from_the_car_profile(tmp_path, fake_person_detector):
    pipe = build_pipeline(load_cfg := car_cfg(tmp_path, pipeline="two_stage"))
    assert isinstance(pipe, TwoStageCarPipeline) and pipe.seats == 4
    assert fake_person_detector == {"model": "fake.pt", "conf": 0.35, "imgsz": 416, "classes": [0], "backend": "torch", "tracker": None, "iou": None}
    assert pipe.attributes is not None and load_cfg.passenger.car.pipeline == "two_stage"


def test_no_classifier_flag_leaves_only_the_count(tmp_path, fake_person_detector):
    pipe = build_pipeline(car_cfg(tmp_path, pipeline="two_stage"), with_classifier=False)
    assert isinstance(pipe, TwoStageCarPipeline) and pipe.attributes is None


def test_car_still_defaults_to_the_single_cabin_model(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(CarPipeline, "from_config", classmethod(lambda cls, cfg: seen.setdefault("pipe", "cabin model")))
    assert build_pipeline(car_cfg(tmp_path)) == "cabin model"          # no `pipeline:` key: behaviour from before this change


def test_an_unknown_car_pipeline_name_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="two_stage.*cabin_model"):
        build_pipeline(car_cfg(tmp_path, pipeline="three_stage"))


def test_crop_weights_missing_stops_the_car_start_with_the_fix(tmp_path, fake_person_detector):
    cfg = car_cfg(tmp_path, pipeline="two_stage")
    cfg.passenger.classifier["crops"]["weights"] = [str(tmp_path / "gone.pt")]
    with pytest.raises(FileNotFoundError, match="train_crops"):
        build_pipeline(cfg)


# ---- calmer boxes, IDs and labels (TrackStabiliser in the chain) ---------------------------------------
from passenger.stability import TrackStabiliser                                                       # noqa: E402


class Script:
    """A detector that plays back one list of detections per frame."""

    def __init__(self, frames):
        self.frames, self.i = frames, 0

    def detect(self, frame, track=True):
        out = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return list(out)


class Spy(FakeAttributes):
    def __init__(self, labels):
        super().__init__(labels)
        self.seen = []

    def update(self, frame, dets):
        self.seen.append(list(dets))
        return super().update(frame, dets)


def test_a_lost_box_stays_on_screen_for_a_moment_but_is_neither_counted_nor_classified():
    spy = Spy({1: "child"})
    p = TwoStageCarPipeline(Script([[TWO[0]], [], []]), spy, seats=5, stabiliser=TrackStabiliser(hold_frames=2))
    p.process(FRAME)
    st = p.process(FRAME)                                              # the person was missed this frame
    assert [d.track_id for d in st.detections] == [1]                  # still drawn ...
    assert spy.seen[-1] == []                                          # ... but not sent to the classifier
    assert st.composition["child"] == 0                                # ... and not in the composition
    assert st.track_labels[1] == "child"                               # the held box keeps its label instead of flashing "unknown"


def test_the_held_box_is_gone_after_the_hold():
    p = TwoStageCarPipeline(Script([[TWO[0]], [], [], []]), FakeAttributes({1: "male"}), stabiliser=TrackStabiliser(hold_frames=2))
    states = [p.process(FRAME) for _ in range(4)]
    assert [len(s.detections) for s in states] == [1, 1, 1, 0]


def test_a_person_the_tracker_renames_keeps_the_label_it_had_earned():
    first = [Detection((0.1, 0.2, 0.3, 0.8), 0.9, 5)]
    again = [Detection((0.11, 0.2, 0.31, 0.8), 0.9, 77)]               # same place, new tracker id, after a gap
    spy = Spy({1: "female"})
    p = TwoStageCarPipeline(Script([first, [], [], [], [], [], again]), spy, stabiliser=TrackStabiliser(hold_frames=1))
    for _ in range(7):
        st = p.process(FRAME)
    assert [d.track_id for d in spy.seen[0]] == [1] and [d.track_id for d in spy.seen[-1]] == [1]     # the classifier sees one person
    assert st.track_labels == {1: "female"}


def test_the_ids_the_classifier_and_the_dashboard_see_are_the_stable_ones():
    p = TwoStageCarPipeline(FakeDetector([Detection((0.1, 0.1, 0.4, 0.9), 0.9, 41)]), FakeAttributes({1: "male"}), stabiliser=TrackStabiliser())
    st = p.process(FRAME)
    assert [d.track_id for d in st.detections] == [1] and st.track_labels == {1: "male"} and st.composition["male"] == 1


def test_without_a_stabiliser_nothing_changes():
    p = TwoStageCarPipeline(FakeDetector(TWO), FakeAttributes({1: "male", 2: "child"}))
    st = p.process(FRAME)
    assert st.detections == TWO and st.track_labels == {1: "male", 2: "child"}


def test_the_config_builds_the_stabiliser_and_validates_it(tmp_path, fake_person_detector):
    cfg = car_cfg(tmp_path, pipeline="two_stage", stability={"hold_frames": 6, "id_memory": 50, "id_iou": 0.35, "ema": 0.5})
    pipe = build_pipeline(cfg)
    s = pipe.stabiliser
    assert (s.hold_frames, s.id_memory, s.id_iou, s.ema) == (6, 50, 0.35, 0.5)
    assert build_pipeline(car_cfg(tmp_path, pipeline="two_stage")).stabiliser is None                                  # not configured: as before
    assert build_pipeline(car_cfg(tmp_path, pipeline="two_stage", stability={"enabled": False, "hold_frames": 6})).stabiliser is None
    with pytest.raises(ValueError, match="ema"):
        build_pipeline(car_cfg(tmp_path, pipeline="two_stage", stability={"ema": 0}))


def test_the_tracker_settings_of_the_car_profile_reach_the_detector(tmp_path, fake_person_detector):
    cfg = car_cfg(tmp_path, pipeline="two_stage")
    cfg.passenger.detector["tracker"] = {"track_buffer": 60, "track_low_thresh": 0.1}
    build_pipeline(cfg)
    assert fake_person_detector["tracker"] == {"track_buffer": 60, "track_low_thresh": 0.1}


def test_the_overlap_limit_of_the_car_profile_reaches_the_detector(tmp_path, fake_person_detector):
    cfg = car_cfg(tmp_path, pipeline="two_stage")
    cfg.passenger.detector["iou"] = 0.5
    build_pipeline(cfg)
    assert fake_person_detector["iou"] == 0.5
