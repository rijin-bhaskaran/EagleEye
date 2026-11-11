"""Seat-state model wired into the pipeline, payload and video overlay."""
from __future__ import annotations

import numpy as np

from eagleeye.config import load_config
from passenger.hub import build_payload, seat_grid_html
from passenger.overlay import draw
from passenger.pipeline import PassengerPipeline, PassengerState
from passenger.seat_state import SeatBox, StableCount
from passenger.seats import Detection, SeatMap, perspective_grid


class FakeDetector:
    def __init__(self, n):
        self.n = n

    def detect(self, frame, track=True):
        return [Detection((0.1 + 0.1 * i, 0.2, 0.18 + 0.1 * i, 0.7), 0.9, i + 1) for i in range(self.n)]


class FakeSeatState:
    def __init__(self, occ, vac):
        self.occ, self.vac, self.reads = occ, vac, 0

    def update(self, frame):
        self.reads += 1
        boxes = [SeatBox((0.0, 0.0, 0.1, 0.1), True, 0.9)] * self.occ + [SeatBox((0.2, 0.0, 0.3, 0.1), False, 0.9)] * self.vac
        return boxes, self.occ, self.vac


def frame():
    return np.full((240, 320, 3), 60, np.uint8)


def test_pipeline_reports_seat_counts_from_the_seat_state_model():
    p = PassengerPipeline(FakeDetector(3), SeatMap(perspective_grid(2, 2, 3)), seat_state=FakeSeatState(7, 4))
    st = p.process(frame())
    assert st.seats_seen == (7, 4) and len(st.seat_boxes) == 11 and st.count == 3


def test_pipeline_without_seat_state_keeps_the_grid_numbers():
    st = PassengerPipeline(FakeDetector(2), SeatMap(perspective_grid(2, 2, 3))).process(frame())
    assert st.seats_seen is None and st.seat_boxes == []


def test_payload_uses_seat_state_counts_over_the_placeholder_grid():
    st = PassengerState(count=5, occupied=[], vacant=["a", "b"], total_seats=50, seats_seen=(6, 4))
    t = build_payload(st, fps=10, latency_ms=5, classifier_on=False)["text"]
    assert (t["paxOccupied"], t["paxVacant"], t["paxTotal"], t["paxOccPct"]) == ("6", "4", "10", "60%")


def test_payload_with_seat_state_seeing_no_seats_does_not_divide_by_zero():
    st = PassengerState(count=0, occupied=[], vacant=[], total_seats=0, seats_seen=(0, 0))
    assert build_payload(st, fps=1, latency_ms=1, classifier_on=False)["text"]["paxOccPct"] == "0%"


def test_overlay_has_no_hud_and_no_seat_annotations():
    seats = SeatMap(perspective_grid(1, 1, 1))
    st = PassengerState(count=3, occupied=[], vacant=[], total_seats=2, seats_seen=(1, 1),
                        composition={"male": 2, "female": 1, "child": 0, "unknown": 0},
                        seat_boxes=[SeatBox((0.5, 0.5, 0.7, 0.9), True, 0.9), SeatBox((0.1, 0.5, 0.3, 0.9), False, 0.9)])
    img = frame()
    draw(img, st, seats, draw_seats=False)
    assert (img == 60).all()                                   # nothing drawn: count and seats belong to the dashboard


def _fixed_state(**kw):
    ids = SeatMap(perspective_grid(3, 2, 2)).seat_ids                  # 12 slots, 3 rows of 2 + 2
    base = dict(count=1, occupied=[], vacant=ids, total_seats=len(ids))
    base.update(kw)
    return PassengerState(**base)


def _marked(html):
    return sorted(t.split(" · ")[0] for t in __import__("re").findall(r'title="([^"]+)"', html) if "vacant" not in t)


def test_seat_map_is_idle_when_the_model_sees_no_occupied_seat():
    html = seat_grid_html(_fixed_state(seats_seen=(0, 0)))
    assert html.count("<span title=") == 12 and _marked(html) == [] and html.index("R1L1") < html.index("R3L1")


def test_seat_map_marks_the_first_n_slots_in_fixed_order_coloured_by_label():
    st = _fixed_state(seats_seen=(3, 2), composition={"male": 1, "female": 1, "child": 0, "unknown": 1})
    html = seat_grid_html(st)
    assert html.count("<span title=") == 12                               # the grid never grows or shrinks
    assert _marked(html) == ["R1L1", "R1L2", "R1R1"]                      # front row first, left to right
    assert "R1L1 · male" in html and "R1L2 · female" in html and "R1R1 · unknown" in html


def test_seat_map_changes_only_when_the_count_changes_not_with_the_boxes():
    boxes_a = [SeatBox((0.1, 0.5, 0.3, 0.9), True, 0.9)]
    boxes_b = [SeatBox((0.6, 0.2, 0.8, 0.4), True, 0.9), SeatBox((0.1, 0.1, 0.2, 0.2), True, 0.9)]   # other boxes, same count
    one = seat_grid_html(_fixed_state(seats_seen=(2, 0), seat_boxes=boxes_a * 2))
    two = seat_grid_html(_fixed_state(seats_seen=(2, 0), seat_boxes=boxes_b))
    assert one == two


def test_seat_map_never_marks_more_slots_than_the_grid_has():
    html = seat_grid_html(_fixed_state(seats_seen=(40, 0)))
    assert html.count("<span title=") == 12 and len(_marked(html)) == 12


def test_stable_count_ignores_a_stray_read_and_follows_a_real_change():
    c = StableCount(window=5, hold=3)
    shown = [c.update(x) for x in (8, 8, 8, 8, 12, 8, 8, 8, 2, 8, 8)]
    assert set(shown) == {8}                                              # outliers never move it
    shown = [c.update(x) for x in (5,) * 12]
    assert shown[-1] == 5 and shown[0] == 8                               # a sustained change gets through, after a delay
    assert [c.update(5) for _ in range(3)] == [5, 5, 5]


def test_pipeline_fallback_counts_are_stable_while_the_raw_model_flickers():
    from passenger.pipeline import PassengerPipeline

    class Raw:
        reads = 0

        def update(self, frame):
            self.reads += 1
            n = [7, 8, 6, 8, 7, 9, 8, 7, 8, 10, 8, 7][self.reads % 12]       # raw occupied seats per read
            return [SeatBox((0.1, 0.1, 0.2, 0.2), True, 0.9)] * n, n, 0
    p = PassengerPipeline(FakeDetector(1), SeatMap(perspective_grid(2, 2, 3)), seat_state=Raw())
    seen = [p.process(frame()).seats_seen[0] for _ in range(60)]
    raw_changes = len({7, 8, 6, 9, 10})
    assert len(set(seen[12:])) == 1 and raw_changes > 1                  # settles on one value and stays


def test_seat_map_without_the_model_still_uses_polygon_seats():
    plain = PassengerState(count=0, occupied=["R1L1"], vacant=["R1L2"], total_seats=2)
    assert "R1L1" in seat_grid_html(plain)


def test_overlay_labels_unknown_people_in_text():
    seats = SeatMap(perspective_grid(1, 1, 1))
    st = PassengerState(count=1, occupied=[], vacant=[], total_seats=2,
                        detections=[Detection((0.5, 0.4, 0.7, 0.9), 0.9, 3)])
    img = frame()
    draw(img, st, seats, draw_seats=False)
    assert (img[85:100, 160:215] != 60).any()                   # 'unknown' text sits above the box


def test_config_has_a_seat_state_block():
    cfg = load_config("config.yaml")
    assert cfg.passenger.seat_state.weights.endswith(".pt") and "every" in cfg.passenger.seat_state
