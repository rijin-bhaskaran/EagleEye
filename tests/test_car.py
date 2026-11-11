"""Car vehicle type: cabin rules, config switch, pipeline, payload, overlay and dataset assembly."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eagleeye.config import load_config
from passenger import car_logic as cl
from passenger.car_logic import (ALERT_CHILD_UNSEATED, ALERT_NO_CHILD_SEAT, ALERT_UNBELTED, CabinAnalyzer, CarDetection,
                                 assign_items, inside_fraction)
from passenger.car_pipeline import CarPipeline
from passenger.datasets import car_cabin
from passenger.hub import build_payload, car_status_html
from passenger.overlay import draw
from passenger.vehicle import build_pipeline, vehicle_type

ADULT = (0.1, 0.2, 0.4, 0.9)
KID = (0.6, 0.4, 0.8, 0.9)
BELT_ON_ADULT = (0.15, 0.3, 0.3, 0.7)
SEAT_AROUND_KID = (0.55, 0.35, 0.85, 0.95)


def det(cls, box, tid=None, conf=0.9):
    return CarDetection(cls, box, conf, tid)


def run(an, frames):
    st = None
    for f in frames:
        st = an.update(f)
    return st


# ---- geometry -------------------------------------------------------------------------------------
def test_inside_fraction_and_degenerate_boxes():
    assert inside_fraction((0.2, 0.2, 0.3, 0.3), (0, 0, 1, 1)) == 1.0
    assert inside_fraction((0.0, 0.0, 0.2, 0.2), (0.1, 0.0, 1, 1)) == pytest.approx(0.5)
    assert inside_fraction((0.5, 0.5, 0.5, 0.6), (0, 0, 1, 1)) == 0.0          # zero-width: no area, no divide-by-zero
    assert inside_fraction((0, 0, 0.1, 0.1), (0.5, 0.5, 1, 1)) == 0.0


def test_belt_goes_to_the_smaller_of_two_people_that_both_contain_it():
    big, small = (0.0, 0.0, 1.0, 1.0), (0.2, 0.2, 0.5, 0.8)
    assert assign_items([(0.25, 0.3, 0.4, 0.6)], [big, small], 0.5) == {0: 1}


# ---- belts ----------------------------------------------------------------------------------------
def test_belt_seen_means_belted_and_missing_means_unknown_then_off():
    an = CabinAnalyzer(min_votes=3)
    with_belt = [det("male", ADULT, 1), det("seatbelt", BELT_ON_ADULT, 5)]
    assert run(an, [with_belt] * 4).occupants[0].belt == "on"
    an = CabinAnalyzer(min_votes=3)
    bare = [det("male", ADULT, 1)]
    assert run(an, [bare] * 2).occupants[0].belt == "unknown"                  # too early to accuse anyone
    st = run(an, [bare] * 3)
    assert st.occupants[0].belt == "off" and ALERT_UNBELTED in st.alerts and st.unbelted == 1


def test_one_missed_belt_frame_does_not_flip_a_belted_person():
    an = CabinAnalyzer(min_votes=3)
    with_belt = [det("female", ADULT, 1), det("seatbelt", BELT_ON_ADULT, 5)]
    run(an, [with_belt] * 6)
    assert run(an, [[det("female", ADULT, 1)]]).occupants[0].belt == "on"


def test_belt_outside_every_person_belongs_to_nobody():
    an = CabinAnalyzer(min_votes=1)
    st = run(an, [[det("male", ADULT, 1), det("seatbelt", (0.85, 0.1, 0.95, 0.3), 9)]] * 3)
    assert st.occupants[0].belt == "off"


def test_explicit_no_seatbelt_counts_double_so_it_convinces_faster_than_silence():
    nb = [det("male", ADULT, 1), det("no_seatbelt", (0.12, 0.3, 0.35, 0.7), 7)]
    assert run(CabinAnalyzer(min_votes=3), [nb]).occupants[0].belt == "unknown"      # 2 votes < 3
    assert run(CabinAnalyzer(min_votes=3), [nb] * 2).occupants[0].belt == "off"      # 4 votes
    assert run(CabinAnalyzer(min_votes=3), [[det("male", ADULT, 1)]] * 2).occupants[0].belt == "unknown"   # silence: 2 votes


# ---- child seats ----------------------------------------------------------------------------------
def test_child_without_any_child_seat_alerts():
    st = run(CabinAnalyzer(), [[det("child", KID, 2)]] * 4)
    assert st.child_present and st.child_seats == 0 and ALERT_NO_CHILD_SEAT in st.alerts


def test_child_inside_a_child_seat_is_seated_and_needs_no_belt_verdict():
    st = run(CabinAnalyzer(), [[det("child", KID, 2), det("child_seat", SEAT_AROUND_KID, 8)]] * 4)
    kid = st.occupants[0]
    assert kid.in_child_seat and kid.belt == "child_seat"
    assert st.child_seats == 1 and st.alerts == [] and st.unbelted == 0


def test_empty_child_seat_is_reported_as_installed_with_no_child():
    st = run(CabinAnalyzer(), [[det("female", ADULT, 1), det("child_seat", SEAT_AROUND_KID, 8)]] * 4)
    assert st.child_seats == 1 and not st.child_present and ALERT_NO_CHILD_SEAT not in st.alerts


def test_child_beside_a_child_seat_is_unseated():
    far_seat = (0.0, 0.0, 0.2, 0.2)
    st = run(CabinAnalyzer(), [[det("child", KID, 2), det("child_seat", far_seat, 8)]] * 4)
    assert st.children_unseated == 1 and ALERT_CHILD_UNSEATED in st.alerts and ALERT_NO_CHILD_SEAT not in st.alerts


def test_child_seat_count_ignores_a_single_dropped_frame():
    an = CabinAnalyzer(seat_window=5)
    seat = [det("child", KID, 2), det("child_seat", SEAT_AROUND_KID, 8)]
    run(an, [seat] * 4)
    assert run(an, [[det("child", KID, 2)]]).child_seats == 1


# ---- labels ---------------------------------------------------------------------------------------
def test_class_flicker_is_voted_down_per_track():
    an = CabinAnalyzer(window=9, min_votes=3)
    frames = [[det("male", ADULT, 1)]] * 5 + [[det("female", ADULT, 1)]] + [[det("male", ADULT, 1)]] * 2
    assert run(an, frames).occupants[0].label == "male"


def test_label_follows_a_real_change_once_it_wins_the_vote():
    an = CabinAnalyzer(window=5, min_votes=3)
    st = run(an, [[det("male", ADULT, 1)]] * 2 + [[det("child", ADULT, 1)]] * 5)
    assert st.occupants[0].label == "child"


def test_untracked_people_are_judged_on_the_current_frame_only():
    an = CabinAnalyzer()
    st = an.update([det("male", ADULT, None), det("seatbelt", BELT_ON_ADULT, None)])
    assert st.occupants[0].belt == "on"
    assert an.update([det("male", ADULT, None)]).occupants[0].belt == "unknown"


def test_state_of_vanished_tracks_is_forgotten():
    an = CabinAnalyzer(max_track_age=3)
    an.update([det("male", ADULT, 1)])
    for _ in range(6):
        an.update([])
    assert not an._class and not an._belt


def test_empty_cabin():
    st = CabinAnalyzer().update([])
    assert st.occupants == [] and not st.child_present and st.alerts == []


# ---- pipeline, payload, overlay -------------------------------------------------------------------
class FakeDetector:
    def __init__(self, frames):
        self.frames, self.i = frames, 0

    def detect(self, frame, track=True):
        out = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return out


def _pipeline(frames, seats=5):
    return CarPipeline(FakeDetector(frames), CabinAnalyzer(min_votes=2), seats=seats)


SCENE = [det("male", ADULT, 1), det("seatbelt", BELT_ON_ADULT, 5), det("child", KID, 2), det("child_seat", SEAT_AROUND_KID, 8)]


def test_pipeline_fills_the_state_the_dashboard_reads():
    p = _pipeline([SCENE])
    for _ in range(3):
        st = p.process(np.zeros((90, 160, 3), np.uint8))
    assert st.count == 2 and st.composition == {"male": 1, "female": 0, "child": 1, "unknown": 0}
    assert st.seats_seen == (2, 3) and st.track_labels == {1: "male", 2: "child"} and st.car.child_seats == 1


def test_more_people_than_seats_never_gives_negative_vacant():
    many = [det("male", (0.0, 0.0, 0.1, 0.1), i) for i in range(1, 8)]
    st = _pipeline([many], seats=5).process(np.zeros((90, 160, 3), np.uint8))
    assert st.seats_seen[1] == 0


def test_payload_has_car_fields_and_escapes_nothing_unsafe():
    p = _pipeline([SCENE])
    for _ in range(3):
        st = p.process(np.zeros((90, 160, 3), np.uint8))
    pl = build_payload(st, fps=10, latency_ms=20, classifier_on=False)
    assert pl["text"]["paxClassifier"] == "Cabin model live" and pl["text"]["paxChild"] == "1"
    assert pl["car"]["childSeats"] == 1 and pl["car"]["occupants"][0]["belt"] == "on"
    assert "in child seat" in car_status_html(st) and "belted" in car_status_html(st)


def test_payload_shows_alerts_and_no_belt():
    p = _pipeline([[det("male", ADULT, 1), det("child", KID, 2)]])
    for _ in range(4):
        st = p.process(np.zeros((90, 160, 3), np.uint8))
    html_ = car_status_html(st)
    assert "NO BELT" in html_ and ALERT_NO_CHILD_SEAT in html_ and "no child seat installed" in html_


def test_overlay_draws_for_car_without_a_seat_map():
    p = _pipeline([SCENE])
    for _ in range(3):
        st = p.process(np.zeros((360, 640, 3), np.uint8))
    frame = np.zeros((360, 640, 3), np.uint8)
    assert draw(frame, st, None, draw_seats=True).any()                       # seatmap None must not crash


# ---- config switch --------------------------------------------------------------------------------
def test_vehicle_type_defaults_to_bus_and_rejects_typos():
    assert vehicle_type(load_config(None)) == "bus"
    cfg = load_config(None)
    cfg.passenger.vehicle_type = "Car"
    assert vehicle_type(cfg) == "car"
    cfg.passenger.vehicle_type = "truck"
    with pytest.raises(ValueError, match="bus.*car"):
        vehicle_type(cfg)


def test_repo_config_yaml_is_valid_and_has_the_car_block():
    cfg = load_config("config.yaml")
    assert vehicle_type(cfg) in ("bus", "car")
    assert cfg.passenger.car.rules.belt_on_ratio > 0 and cfg.passenger.car.seats >= 1


def test_car_without_weights_fails_with_a_message_that_says_what_to_do(tmp_path):
    pytest.importorskip("ultralytics")
    cfg = load_config(None)
    cfg.passenger.vehicle_type = "car"
    cfg.passenger.car.weights = str(tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError, match="train_car|vehicle_type: bus"):
        build_pipeline(cfg)


# ---- dataset assembly -----------------------------------------------------------------------------
def make_export(root: Path, names, n_groups=40, per_group=3, classes_per_img=(0,)):
    (root).mkdir(parents=True)
    (root / "data.yaml").write_text(f"names: {names}\nnc: {len(names)}\n")
    for g in range(n_groups):
        for k in range(per_group):
            stem = f"clip{g:03d}_{k:04d}"
            sp = ("train", "valid", "test")[k % 3]
            (root / sp / "images").mkdir(parents=True, exist_ok=True)
            (root / sp / "labels").mkdir(parents=True, exist_ok=True)
            (root / sp / "images" / f"{stem}.jpg").write_bytes(b"x")
            rows = "\n".join(f"{c} 0.5 0.5 0.2 0.2" for c in classes_per_img)
            (root / sp / "labels" / f"{stem}.txt").write_text(rows + "\n")
    return root


def test_group_key_strips_roboflow_suffix_and_frame_counter():
    assert car_cabin.group_of("clip012_0003") == "clip012"
    assert car_cabin.group_of("car_9_jpg.rf.abcdef0123") == "car"
    assert car_cabin.group_of("x_a1", r"^(x)") == "x"
    with pytest.raises(ValueError):
        car_cabin.group_of("abc", r"^zzz")


def test_a_group_never_straddles_splits_and_split_is_stable(tmp_path):
    src = make_export(tmp_path / "src", ["belt", "person"])
    out = tmp_path / "out"
    car_cabin.add(src, out, "ds", {"belt": "seatbelt", "person": "drop"}, license="CC BY 4.0")
    where = {}
    for sp in ("train", "val", "test"):
        for img in (out / "images" / sp).glob("*.jpg"):
            where.setdefault(car_cabin.group_of(img.stem.split("__", 1)[1]), set()).add(sp)
    assert where and all(len(v) == 1 for v in where.values())                    # the leakage guarantee
    before = {p.name for p in (out / "images").rglob("*.jpg")}
    car_cabin.add(src, out, "ds", {"belt": "seatbelt", "person": "drop"}, license="CC BY 4.0")
    assert {p.name for p in (out / "images").rglob("*.jpg")} == before          # rerun is idempotent


def test_classes_are_remapped_and_dropped_ones_leave_no_rows(tmp_path):
    src = make_export(tmp_path / "src", ["Belt", "Man", "Kid seat"], classes_per_img=(0, 1, 2))
    out = tmp_path / "out"
    car_cabin.add(src, out, "ds", car_cabin.parse_map(["belt=seatbelt", "man=drop", "kid seat=child_seat"]), license="x")
    ids = {int(l.split()[0]) for f in (out / "labels").rglob("*.txt") for l in f.read_text().splitlines()}
    assert ids == {cl.CAR_CLASSES.index("seatbelt"), cl.CAR_CLASSES.index("child_seat")}
    assert "names:" in (out / "data.yaml").read_text() and "child_seat" in (out / "data.yaml").read_text()


def test_unmapped_class_is_an_error_not_a_silent_drop(tmp_path):
    src = make_export(tmp_path / "src", ["belt", "mystery"])
    with pytest.raises(ValueError, match="mystery"):
        car_cabin.add(src, tmp_path / "out", "ds", {"belt": "seatbelt"}, license="x")
    car_cabin.add(src, tmp_path / "out", "ds", {"belt": "seatbelt"}, license="x", drop_unmapped=True)


def test_licence_is_required_and_bad_targets_and_names_are_refused(tmp_path):
    src = make_export(tmp_path / "src", ["belt"])
    with pytest.raises(ValueError, match="license"):
        car_cabin.add(src, tmp_path / "out", "ds", {"belt": "seatbelt"}, license=" ")
    with pytest.raises(ValueError, match="target"):
        car_cabin.parse_map(["belt=helmet"])
    with pytest.raises(ValueError, match="source=target"):
        car_cabin.parse_map(["belt"])
    for bad in ("../x", "a/b", "", "a b"):
        with pytest.raises(ValueError, match="name"):
            car_cabin.add(src, tmp_path / "out", bad, {"belt": "seatbelt"}, license="x")


def test_rereading_one_dataset_leaves_other_datasets_alone(tmp_path):
    a = make_export(tmp_path / "a", ["belt"])
    b = make_export(tmp_path / "b", ["seat"])
    out = tmp_path / "out"
    car_cabin.add(a, out, "aa", {"belt": "seatbelt"}, license="x")
    car_cabin.add(b, out, "bb", {"seat": "child_seat"}, license="y")
    n_b = len(list((out / "images").rglob("bb__*")))
    car_cabin.add(a, out, "aa", {"belt": "seatbelt"}, license="x")
    assert n_b > 0 and len(list((out / "images").rglob("bb__*"))) == n_b
    import json
    assert set(json.loads((out / "sources.json").read_text())) == {"aa", "bb"}


def test_label_with_unknown_class_id_is_reported(tmp_path):
    src = make_export(tmp_path / "src", ["belt"], n_groups=1, per_group=1, classes_per_img=(3,))
    with pytest.raises(ValueError, match="class id"):
        car_cabin.add(src, tmp_path / "out", "ds", {"belt": "seatbelt"}, license="x")


def test_stats_counts_every_class_even_when_absent(tmp_path):
    src = make_export(tmp_path / "src", ["belt"])
    car_cabin.add(src, tmp_path / "out", "ds", {"belt": "seatbelt"}, license="x")
    s = car_cabin.stats(tmp_path / "out")
    assert set(s["boxes"]["train"]) == set(cl.CAR_CLASSES) and s["boxes"]["train"]["child_seat"] == 0


# ---- scoring --------------------------------------------------------------------------------------
def test_best_f1_handles_classes_with_no_boxes_in_the_split():
    """ultralytics' curves only have rows for classes present in the split: absent ones must read as None, not crash."""
    from types import SimpleNamespace

    from passenger.train_car import best_f1

    curve = np.zeros((2, 1000))
    curve[0, 300] = 0.8                       # class index 2 ('child')
    curve[1, 300] = 0.6                       # class index 5 ('child_seat')
    res = SimpleNamespace(box=SimpleNamespace(f1_curve=curve, ap_class_index=np.array([2, 5])))
    op = best_f1(res)
    assert op["f1"]["child"] == 0.8 and op["f1"]["child_seat"] == 0.6 and op["f1"]["male"] is None
    assert op["f1_mean"] == pytest.approx(0.7) and op["conf"] == pytest.approx(0.3, abs=0.01)


# ---- startup failure ------------------------------------------------------------------------------
def test_shutting_down_a_server_that_never_started_returns_instead_of_hanging():
    """run_passenger shuts the server down when the model or camera fails to load, before start() was ever called.
    socketserver's shutdown() waits forever in that state, which left the process alive holding the port."""
    import threading

    from passenger.hub import PassengerHub
    from passenger.server import PassengerServer

    srv = PassengerServer(PassengerHub(), port=0)
    t = threading.Thread(target=srv.shutdown, daemon=True)
    t.start()
    t.join(5)
    assert not t.is_alive(), "shutdown() of a never-started server hung"
    PassengerServer(PassengerHub(), port=srv.port).shutdown()               # and the port is free again
