"""Adversarial / regression tests from the Phase 6 QA pass.

Each test targets a defect or risk found by reviewing the passenger code. Run with:  pytest -q
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from passenger import metrics
from passenger.classifier import Attribute, AttributeService
from passenger.data_manager import DataManager
from passenger.evaluate import evaluate
from passenger.hub import PassengerHub, seat_grid_html
from passenger.pipeline import PassengerPipeline, PassengerState
from passenger.runner import PassengerRunner
from passenger.seats import Detection, SeatMap, perspective_grid

FRAME = np.full((200, 200, 3), 90, np.uint8)


def box(i: int, w: float = 0.05):
    x = 0.02 + (i % 15) * 0.06
    y = 0.02 + (i // 15) * 0.1
    return (x, y, x + w, y + 0.08)


def make_video(path: Path, n=6, seed=0) -> Path:
    rng = np.random.default_rng(seed)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
    for k in range(n):
        w.write(np.clip(rng.integers(0, 255, (48, 64, 3)), 0, 255).astype(np.uint8))
    w.release()
    return path


# -- 1. classifier scheduling must not starve people who have a readable face ----------
class FaceOnlyFor:
    """Reports a face (male) only for the box it was built with; everyone else is faceless."""

    def __init__(self, readable_box):
        self.readable_box, self.calls = readable_box, 0

    def classify(self, frame, b):
        self.calls += 1
        return Attribute("male", 0.9) if b == self.readable_box else Attribute("unknown", 0.0)


def test_faceless_passengers_do_not_starve_the_ones_with_faces():
    """20 people in the back rows have unreadable faces; one in the front does. With a budget of 3
    calls per frame the front passenger must still be labelled."""
    faceless = [Detection(box(i), 0.9, track_id=i) for i in range(20)]
    readable = Detection(box(30), 0.9, track_id=99)
    svc = AttributeService(FaceOnlyFor(readable.box), min_votes=3, budget=3)
    dets = faceless + [readable]
    labels = {}
    for _ in range(60):
        labels = svc.update(FRAME, dets)
    assert labels[99] == "male"


def test_every_track_is_attempted_in_rotation():
    scripted = FaceOnlyFor((9, 9, 9, 9))
    seen = []
    orig = scripted.classify

    def spy(frame, b):
        seen.append(b)
        return orig(frame, b)

    scripted.classify = spy
    dets = [Detection(box(i), 0.9, track_id=i) for i in range(9)]
    svc = AttributeService(scripted, budget=3)
    for _ in range(3):
        svc.update(FRAME, dets)
    assert len({tuple(b) for b in seen}) == 9                 # 3 frames x 3 calls cover all 9 people


# -- 2. data-loss guards -----------------------------------------------------------------
@pytest.fixture
def dm(tmp_path):
    return DataManager(tmp_path / "pd")


@pytest.mark.parametrize("name", ["labels", "..", ".", "a/b", "a\\b", "", "../x", "has space"])
def test_export_rejects_dangerous_dataset_names(dm, name):
    (dm.labels_dir / "keep.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    with pytest.raises(ValueError):
        dm.export_yolo(name)
    assert (dm.labels_dir / "keep.txt").exists()               # annotations untouched


def test_export_never_deletes_raw_videos(dm, tmp_path):
    vid = dm.register_video(make_video(tmp_path / "a.mp4"))
    with pytest.raises(ValueError):
        dm.export_yolo("..")
    assert (dm.root / dm.videos[vid]["path"]).is_file()


@pytest.mark.parametrize("vehicle", ["../escape", "a/b", "..", ""])
def test_register_rejects_path_like_vehicle_names(dm, tmp_path, vehicle):
    with pytest.raises(ValueError):
        dm.register_video(make_video(tmp_path / "a.mp4"), vehicle=vehicle)
    assert not (tmp_path / "escape").exists() and not dm.videos


# -- 3. splits must be stable once assigned ---------------------------------------------
def test_adding_videos_does_not_move_existing_ones_between_splits(dm, tmp_path):
    for i in range(6):
        dm.register_video(make_video(tmp_path / f"v{i}.mp4", seed=i))
    dm.assign_splits()
    before = {vid: m["split"] for vid, m in dm.videos.items()}
    for i in range(6, 12):
        dm.register_video(make_video(tmp_path / f"v{i}.mp4", seed=i))
    dm.assign_splits()
    after = {vid: dm.videos[vid]["split"] for vid in before}
    assert after == before                                     # a test video can never become a training video
    assert all(m["split"] in ("train", "val", "test") for m in dm.videos.values())


def test_new_videos_fill_the_split_that_is_furthest_below_target(dm, tmp_path):
    for i in range(10):
        dm.register_video(make_video(tmp_path / f"v{i}.mp4", seed=i))
    dm.assign_splits()
    for i in range(10, 30):
        dm.register_video(make_video(tmp_path / f"v{i}.mp4", seed=i))
    dm.assign_splits()
    counts = {s: sum(m["split"] == s for m in dm.videos.values()) for s in ("train", "val", "test")}
    assert counts["train"] > counts["val"] and counts["train"] > counts["test"]
    assert counts["val"] >= 3 and counts["test"] >= 3          # roughly 70/15/15 of 30


# -- 4. bad labels give a clear error, not an IndexError -----------------------------------
def test_stats_tolerates_out_of_range_class_ids(dm, tmp_path):
    vid = dm.register_video(make_video(tmp_path / "a.mp4"))
    dm.extract_frames(vid, target_fps=5, dedupe_threshold=0)
    dm.assign_splits()
    idx = dm.videos[vid]["frames"][0]
    (dm.labels_dir / f"{vid}_{idx:06d}.txt").write_text("9 0.5 0.5 0.2 0.2\n")
    assert dm.stats()["train"]["boxes"].get("invalid") == 1


def test_evaluate_rejects_invalid_labels_with_a_clear_message(dm, tmp_path):
    for i in range(3):
        vid = dm.register_video(make_video(tmp_path / f"v{i}.mp4", seed=i))
        dm.extract_frames(vid, target_fps=5, dedupe_threshold=0)
    dm.assign_splits()
    for jpg in dm.frames_dir.glob("*/*.jpg"):
        (dm.labels_dir / f"{jpg.stem}.txt").write_text("9 0.5 0.5 0.2 0.2\n")

    class D:
        def detect(self, f, track=True):
            return []

    with pytest.raises(ValueError, match="invalid label"):
        evaluate(dm, D(), SeatMap({"A": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3]]}), "test")


# -- 5. the passenger loop must survive a bad frame ---------------------------------------
class FlakyDetector:
    def __init__(self, fail_on):
        self.fail_on, self.n = set(fail_on), 0

    def detect(self, frame, track=True):
        self.n += 1
        if self.n in self.fail_on:
            raise RuntimeError("model hiccup")
        return []


class Frames:
    is_file, src_fps, frame_index = False, 30.0, 0

    def read(self):
        return True, np.zeros((48, 64, 3), np.uint8), 0.0


def runner(detector, max_errors=30):
    pipe = PassengerPipeline(detector, SeatMap(perspective_grid(2, 2, 3)))
    return PassengerRunner(Frames(), pipe, PassengerHub(), fps=1000.0, max_consecutive_errors=max_errors)


def test_runner_survives_isolated_frame_errors(capsys):
    r = runner(FlakyDetector({2, 5}))
    stop = threading.Event()
    th = threading.Thread(target=r.run, args=(stop,))
    th.start()
    time.sleep(0.4)
    stop.set()
    th.join(2)
    assert r.frames > 5 and r.errors == 2
    assert "model hiccup" in capsys.readouterr().out          # reported, not swallowed silently


def test_runner_gives_up_when_every_frame_fails():
    r = runner(FlakyDetector(range(1, 10_000)), max_errors=5)
    with pytest.raises(RuntimeError, match="model hiccup"):
        r.run(threading.Event())
    assert r.errors == 5


# -- 6. dashboard HTML is escaped -------------------------------------------------------------
def test_seat_grid_escapes_seat_ids_from_config():
    st = PassengerState(count=0, occupied=[], vacant=['x"><script>alert(1)</script>'], total_seats=1)
    html = seat_grid_html(st)
    assert "<script>" not in html and "&lt;script&gt;" in html


# -- 7. SeatMap behaves under hostile input ----------------------------------------------------
def test_assign_handles_degenerate_and_out_of_frame_boxes():
    sm = SeatMap(perspective_grid(3, 2, 3))
    weird = [Detection(b, 0.9) for b in [(0.5, 0.5, 0.5, 0.5), (0.7, 0.7, 0.2, 0.2), (-5, -5, -4, -4),
                                         (2, 2, 3, 3), (-1, -1, 2, 2), (0.0, 0.0, 0.0, 0.0)]]
    out = sm.assign(weird)
    assert len(out) <= len(sm)                                  # no crash, no more people than seats


def test_assign_invariants_hold_on_random_crowds():
    rng = np.random.default_rng(7)
    sm = SeatMap(perspective_grid(10, 2, 3))
    for _ in range(200):
        n = int(rng.integers(0, 60))
        xy = rng.random((n, 2)) * 0.9
        wh = rng.random((n, 2)) * 0.15 + 0.01
        dets = [Detection((x, y, x + w, y + h), 0.9) for (x, y), (w, h) in zip(xy, wh)]
        out = sm.assign(dets)
        assert set(out) <= set(sm.seat_ids)
        assert len({id(d) for d in out.values()}) == len(out)   # one person per seat


def test_assign_is_fast_enough_for_a_full_bus():
    sm = SeatMap(perspective_grid(10, 2, 3))
    rng = np.random.default_rng(1)
    xy = rng.random((60, 2)) * 0.8
    dets = [Detection((x, y, x + 0.1, y + 0.2), 0.9) for x, y in xy]
    t0 = time.perf_counter()
    for _ in range(20):
        sm.assign(dets)
    assert (time.perf_counter() - t0) / 20 < 0.05              # far under the 100 ms frame budget


# -- 8. metric invariants ----------------------------------------------------------------------
def test_metrics_stay_in_range_on_random_inputs():
    rng = np.random.default_rng(3)
    seats = [f"S{i}" for i in range(12)]
    for _ in range(100):
        pred = [set(rng.choice(seats, int(rng.integers(0, 12)), replace=False)) for _ in range(5)]
        gt = [set(rng.choice(seats, int(rng.integers(0, 12)), replace=False)) for _ in range(5)]
        r = metrics.occupancy_metrics(pred, gt, seats)
        assert all(0.0 <= r[k] <= 1.0 for k in ("precision", "recall", "f1", "macro_f1", "frame_exact"))
        assert metrics.occupancy_metrics(gt, gt, seats)["frame_exact"] == 1.0


def test_detection_metrics_stay_in_range_on_random_boxes():
    rng = np.random.default_rng(5)
    for _ in range(50):
        def rand_boxes(n):
            xy = rng.random((n, 2)) * 0.7
            return [(x, y, x + 0.2, y + 0.2) for x, y in xy]
        frames = [([(b, float(rng.random())) for b in rand_boxes(int(rng.integers(0, 6)))],
                   rand_boxes(int(rng.integers(0, 6)))) for _ in range(4)]
        r = metrics.detection_prf(frames)
        assert 0 <= r["precision"] <= 1 and 0 <= r["recall"] <= 1
        assert 0.0 <= metrics.average_precision(frames) <= 1.0 + 1e-9
        assert 0.0 <= metrics.map_50_95(frames) <= 1.0 + 1e-9


# -- 9. full pipeline over a moving crowd -----------------------------------------------------
def test_pipeline_over_a_moving_crowd_keeps_its_invariants():
    sm = SeatMap(perspective_grid(10, 2, 3), min_overlap=0.3)
    ids = sm.seat_ids
    poly_box = lambda sid: (min(p[0] for p in sm.polygons[sid]), min(p[1] for p in sm.polygons[sid]),
                            max(p[0] for p in sm.polygons[sid]), max(p[1] for p in sm.polygons[sid]))
    rng = np.random.default_rng(11)

    class Crowd:
        t = 0

        def detect(self, frame, track=True):
            self.t += 1
            seated = rng.choice(len(ids), 25, replace=False)
            return [Detection(poly_box(ids[i]), 0.9, track_id=int(i)) for i in seated]

    class Labeller:
        def classify(self, frame, b):
            return Attribute(("male", "female", "child")[int(b[0] * 10) % 3], 0.9)

    pipe = PassengerPipeline(Crowd(), sm, attributes=AttributeService(Labeller(), budget=3))
    for _ in range(40):
        s = pipe.process(FRAME)
        assert set(s.occupied) | set(s.vacant) == set(ids) and not set(s.occupied) & set(s.vacant)
        assert sum(s.composition.values()) == len(s.detections)
        assert s.count >= 0 and s.total_seats == 50
