"""Tests for seats, occupancy pipeline, metrics and evaluation. Run with:  pytest -q"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eagleeye.config import load_config
from passenger import metrics
from passenger.data_manager import DataManager
from passenger.evaluate import evaluate
from passenger.pipeline import PassengerPipeline, SeatDebouncer
from passenger.seats import Detection, SeatMap, perspective_grid


def square(x, y, s=0.2):
    return [[x, y], [x + s, y], [x + s, y + s], [x, y + s]]


def box_in(poly):
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


# -- seats ---------------------------------------------------------------
def test_grid_is_50_seats_2_plus_3_and_inside_frame():
    g = perspective_grid(10, 2, 3)
    assert len(g) == 50
    assert sum(k.startswith("R1L") for k in g) == 2 and sum(k.startswith("R1R") for k in g) == 3
    assert all(0 <= v <= 1 for poly in g.values() for pt in poly for v in pt)
    width = lambda k: g[k][1][0] - g[k][0][0]
    assert width("R10L1") < width("R1L1")           # perspective: far rows are smaller


def test_seatmap_rejects_bad_polygons():
    with pytest.raises(ValueError):
        SeatMap({})
    with pytest.raises(ValueError):
        SeatMap({"a": [[0, 0], [1, 1]]})
    with pytest.raises(ValueError):
        SeatMap({"a": [[0, 0], [2, 0], [2, 2]]})


def test_assign_person_in_seat_and_overlap_threshold():
    sm = SeatMap({"A": square(0.1, 0.1), "B": square(0.6, 0.6)}, min_overlap=0.5)
    inside = Detection(box_in(square(0.1, 0.1)), 0.9)
    assert list(sm.assign([inside])) == ["A"]
    grazing = Detection((0.25, 0.1, 0.55, 0.3), 0.9)       # only a sliver inside seat A
    assert sm.assign([grazing]) == {}
    assert sm.assign([]) == {}


def test_assign_is_one_to_one_best_first():
    sm = SeatMap({"A": square(0.1, 0.1), "B": square(0.3, 0.1)}, min_overlap=0.2)
    wide = Detection((0.1, 0.1, 0.5, 0.3), 0.9)             # straddles both seats
    exact = Detection(box_in(square(0.1, 0.1)), 0.9)        # fits A perfectly
    out = sm.assign([wide, exact])
    assert out["A"] is exact and out["B"] is wide
    assert len(sm.assign([wide])) == 1                      # one person fills one seat


def test_from_config_uses_polygons_or_placeholder_grid():
    cfg = load_config("config.yaml")
    assert len(SeatMap.from_config(cfg.passenger.seats)) == 50
    custom = {"rows": 1, "left": 1, "right": 1, "polygons": {"X": square(0.2, 0.2)}}
    assert SeatMap.from_config(custom).seat_ids == ["X"]


# -- debounce / pipeline ---------------------------------------------------
def test_debouncer_fast_on_slow_off():
    d = SeatDebouncer(["A"], on_frames=3, off_frames=5)
    assert [d.update({"A"}) for _ in range(3)] == [set(), set(), {"A"}]
    for _ in range(4):
        assert d.update(set()) == {"A"}                      # brief occlusion does not free the seat
    assert d.update(set()) == set()


def test_debouncer_resets_on_flicker():
    d = SeatDebouncer(["A"], on_frames=3, off_frames=5)
    for seen in ({"A"}, {"A"}, set(), {"A"}, {"A"}):
        d.update(seen)
    assert d.occupied == set()                               # streak was broken


class FakeDetector:
    def __init__(self, script):
        self.script, self.i = script, 0

    def detect(self, frame, track=True):
        out = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return out


def test_pipeline_counts_occupied_and_vacant():
    sm = SeatMap({"A": square(0.1, 0.1), "B": square(0.5, 0.5)}, min_overlap=0.5)
    person = Detection(box_in(square(0.1, 0.1)), 0.9, track_id=7)
    pipe = PassengerPipeline(FakeDetector([[person]]), sm, on_frames=2, off_frames=3, count_window=1)
    frame = np.zeros((10, 10, 3), np.uint8)
    pipe.process(frame)
    s = pipe.process(frame)
    assert (s.count, s.occupied, s.vacant, s.total_seats) == (1, ["A"], ["B"], 2)
    assert s.seat_track == {"A": 7} and s.vacant_count == 1


def test_pipeline_count_is_median_smoothed():
    sm = SeatMap({"A": square(0.1, 0.1)})
    p = Detection(box_in(square(0.1, 0.1)), 0.9)
    pipe = PassengerPipeline(FakeDetector([[p], [p], [], [p], [p]]), sm, count_window=5)
    frame = np.zeros((4, 4, 3), np.uint8)
    counts = [pipe.process(frame).count for _ in range(5)]
    assert counts[-1] == 1                                   # the one-frame dropout is ignored


def test_pipeline_from_config_builds():
    cfg = load_config("config.yaml")
    pipe = PassengerPipeline.from_config(cfg, FakeDetector([[]]), with_classifier=False)
    s = pipe.process(np.zeros((4, 4, 3), np.uint8))
    assert s.total_seats == 50 and s.count == 0 and len(s.vacant) == 50


# -- metrics ---------------------------------------------------------------
B = (0.1, 0.1, 0.3, 0.3)
FAR = (0.6, 0.6, 0.8, 0.8)


def test_iou_basics():
    assert metrics.iou(B, B) == pytest.approx(1.0)
    assert metrics.iou(B, FAR) == 0.0
    assert metrics.iou((0, 0, 2, 1), (1, 0, 3, 1)) == pytest.approx(1 / 3)


def test_detection_prf_counts_tp_fp_fn():
    frames = [([(B, 0.9), (FAR, 0.8)], [B, (0.0, 0.7, 0.1, 0.9)])]
    r = metrics.detection_prf(frames)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1)
    assert r["precision"] == r["recall"] == 0.5


def test_detection_prf_empty_is_zero_not_error():
    assert metrics.detection_prf([([], [])])["f1"] == 0.0


def test_ap_perfect_and_ordering_matters():
    assert metrics.average_precision([([(B, 0.9)], [B])]) == pytest.approx(1.0)
    wrong_first = [([(FAR, 0.9), (B, 0.5)], [B])]            # confident FP ranked above the TP
    right_first = [([(B, 0.9), (FAR, 0.5)], [B])]
    assert metrics.average_precision(wrong_first) == pytest.approx(0.5)
    assert metrics.average_precision(right_first) == pytest.approx(1.0)
    assert metrics.average_precision([([], [B])]) == 0.0


def test_map_50_95_drops_with_loose_boxes():
    loose = (0.12, 0.12, 0.32, 0.32)
    frames = [([(loose, 0.9)], [B])]
    assert metrics.average_precision(frames, 0.5) == pytest.approx(1.0)
    assert metrics.map_50_95(frames) < 0.9
    assert metrics.map_50_95([([(B, 0.9)], [B])]) == pytest.approx(1.0)


def test_count_metrics():
    r = metrics.count_metrics([3, 5, 4], [3, 4, 6])
    assert r["mae"] == pytest.approx(1.0)
    assert r["exact"] == pytest.approx(1 / 3)
    assert r["bias"] == pytest.approx(-1 / 3)
    with pytest.raises(ValueError):
        metrics.count_metrics([1], [1, 2])


def test_occupancy_metrics():
    seats = ["A", "B", "C"]
    r = metrics.occupancy_metrics([{"A", "B"}, {"A"}], [{"A", "C"}, {"A"}], seats)
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(2 / 3)
    assert r["frame_exact"] == 0.5
    assert r["per_seat_f1"] == {"A": 1.0, "B": 0.0, "C": 0.0}
    assert metrics.occupancy_metrics([set()], [set()], seats)["macro_f1"] == 0.0     # nothing to score


def test_latency_summary():
    r = metrics.latency_summary([0.1] * 19 + [0.3])
    assert r["mean_ms"] == pytest.approx(110) and r["p95_ms"] > r["mean_ms"]
    assert r["max_fps"] == pytest.approx(1 / 0.11)
    assert metrics.latency_summary([])["max_fps"] == 0.0


# -- evaluate (fake detector, real DataManager) --------------------------
def _labelled_dataset(tmp_path):
    dm = DataManager(tmp_path / "pd")
    for i in range(3):
        src = tmp_path / f"v{i}.mp4"
        w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
        for k in range(4):
            w.write(np.full((48, 64, 3), 40 * (k + 1) + i, np.uint8))
        w.release()
        vid = dm.register_video(src)
        dm.extract_frames(vid, target_fps=5, dedupe_threshold=0)
    dm.assign_splits()
    for jpg in dm.frames_dir.glob("*/*.jpg"):
        (dm.labels_dir / f"{jpg.stem}.txt").write_text("0 0.2 0.2 0.2 0.2\n")
    return dm, SeatMap({"A": square(0.1, 0.1), "B": square(0.6, 0.6)}, min_overlap=0.5)


class Perfect:
    def detect(self, frame, track=True):
        return [Detection(B, 0.9)]


class Blind:
    def detect(self, frame, track=True):
        return []


def test_evaluate_perfect_and_blind_detectors(tmp_path):
    dm, sm = _labelled_dataset(tmp_path)
    good = evaluate(dm, Perfect(), sm, "test", warmup=0)
    assert good["frames"] == 4
    assert good["detection"]["f1"] == 1.0 and good["detection"]["map50_95"] == pytest.approx(1.0)
    assert good["count"]["mae"] == 0.0 and good["occupancy"]["frame_exact"] == 1.0
    assert good["speed"]["max_fps"] > 0
    bad = evaluate(dm, Blind(), sm, "test", warmup=0)
    assert bad["detection"]["recall"] == 0.0 and bad["count"]["bias"] == -1.0
    assert bad["occupancy"]["recall"] == 0.0


def test_evaluate_without_labels_raises(tmp_path):
    dm = DataManager(tmp_path / "pd")
    with pytest.raises(ValueError):
        evaluate(dm, Perfect(), SeatMap({"A": square(0.1, 0.1)}), "test")


# -- real model smoke test -------------------------------------------------
@pytest.mark.skipif(not Path("yolov8n.pt").exists(), reason="needs yolov8n.pt in repo root")
def test_person_detector_runs_tracked_and_untracked():
    from passenger.detector import PersonDetector
    det = PersonDetector("yolov8n.pt", conf=0.3, imgsz=320)
    blank = np.full((240, 320, 3), 127, np.uint8)
    assert det.detect(blank, track=False) == []
    assert det.detect(blank, track=True) == []
