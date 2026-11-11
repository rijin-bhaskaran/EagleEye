"""Fixed seat zones driven by the seat-state model: no shifting, no blinking."""
from __future__ import annotations

import numpy as np
import pytest

from passenger.hub import seat_grid_html
from passenger.pipeline import PassengerPipeline
from passenger.seat_state import SeatBox, SeatZones
from passenger.seats import Detection, SeatMap, load_zones
from passenger.zone_editor import ZoneEditor


def square(cx, cy, r=0.08):
    return [[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r]]


ZONES = {"R1L1": square(0.2, 0.8), "R1R1": square(0.8, 0.8), "R2L1": square(0.2, 0.4), "R2R1": square(0.8, 0.4)}


def seat_box(cx, cy, occupied=True):
    return SeatBox((cx - 0.1, cy - 0.15, cx + 0.1, cy + 0.15), occupied, 0.9)


def calibrated(zones=ZONES):
    seatmap = SeatMap(zones)
    seatmap.calibrated = True
    return seatmap


# -- zones ----------------------------------------------------------------------
def test_a_box_only_lights_the_zone_that_holds_its_centre():
    z = SeatZones(SeatMap(ZONES), on_reads=1, off_reads=1)
    assert z.update([seat_box(0.8, 0.4)], 1) == {"R2R1"}


def test_boxes_outside_every_zone_and_vacant_boxes_light_nothing():
    z = SeatZones(SeatMap(ZONES), on_reads=1, off_reads=1)
    assert z.update([seat_box(0.5, 0.6), seat_box(0.2, 0.8, occupied=False)], 1) == set()


def test_a_centre_inside_two_overlapping_zones_is_ambiguous_and_ignored():
    zones = {"A": square(0.5, 0.5, 0.2), "B": square(0.55, 0.5, 0.2)}
    assert SeatZones(SeatMap(zones), on_reads=1, off_reads=1).update([seat_box(0.52, 0.5)], 1) == set()


def test_a_slot_needs_consecutive_reads_to_light_and_to_clear():
    z = SeatZones(SeatMap(ZONES), on_reads=2, off_reads=3)
    here, gone = [seat_box(0.2, 0.8)], []
    assert z.update(here, 1) == set()                       # one read is not enough
    assert z.update(here, 2) == {"R1L1"}
    assert z.update(gone, 3) == {"R1L1"} and z.update(gone, 4) == {"R1L1"}   # a missed read or two changes nothing
    assert z.update(gone, 5) == set()                       # three misses in a row


def test_a_short_dropout_never_blinks_the_slot():
    z = SeatZones(SeatMap(ZONES), on_reads=2, off_reads=5)
    seq = [True, True, True, False, True, True, False, False, True, True, True]   # noisy model, seat really occupied
    out = [z.update([seat_box(0.2, 0.8)] if on else [], i + 1) == {"R1L1"} for i, on in enumerate(seq)]
    assert out == [False] + [True] * 10                     # lights once, never goes out


def test_only_a_fresh_model_read_moves_the_counters():
    z = SeatZones(SeatMap(ZONES), on_reads=2, off_reads=2)
    z.update([seat_box(0.2, 0.8)], 1)
    for _ in range(10):
        assert z.update([seat_box(0.2, 0.8)], 1) == set()   # same read replayed on later frames: nothing changes
    assert z.update([seat_box(0.2, 0.8)], 2) == {"R1L1"}


def test_other_slots_do_not_move_when_one_box_is_missed():
    z = SeatZones(SeatMap(ZONES), on_reads=1, off_reads=1)
    both = [seat_box(0.2, 0.8), seat_box(0.8, 0.4)]
    assert z.update(both, 1) == {"R1L1", "R2R1"}
    assert z.update(both[1:], 2) == {"R2R1"}                # R2R1 stays R2R1, it does not slide to another slot


# -- pipeline + dashboard -------------------------------------------------------
class FakeDetector:
    def detect(self, frame, track=True):
        return [Detection((0.14, 0.6, 0.26, 0.95), 0.9, 7)]


class FakeSeatState:
    def __init__(self, script):
        self.script, self.reads = script, 0

    def update(self, frame):
        boxes = self.script[self.reads % len(self.script)]
        self.reads += 1
        return boxes, sum(b.occupied for b in boxes), sum(not b.occupied for b in boxes)


def test_pipeline_with_measured_seats_reports_slots_not_seats_seen():
    seatmap = calibrated()
    script = [[seat_box(0.2, 0.8)], [], [seat_box(0.2, 0.8)], [seat_box(0.2, 0.8), seat_box(0.8, 0.4)], [seat_box(0.2, 0.8)]]
    p = PassengerPipeline(FakeDetector(), seatmap, seat_state=FakeSeatState(script),
                          seat_zones=SeatZones(seatmap, on_reads=1, off_reads=3))
    states = [p.process(np.zeros((120, 160, 3), np.uint8)) for _ in range(5)]
    assert all(s.seats_seen is None and s.total_seats == 4 for s in states)
    assert [s.occupied for s in states][1:] == [["R1L1"], ["R1L1"], ["R1L1", "R2R1"], ["R1L1", "R2R1"]]   # one missed read does not blink
    html = seat_grid_html(states[-1])
    assert html.count("<span title=") == 4 and "R1L1 ·" in html and "R2R1 ·" in html


def test_dashboard_grid_marks_the_same_slots_every_time_for_the_same_occupancy():
    seatmap = calibrated()
    p = PassengerPipeline(FakeDetector(), seatmap, seat_state=FakeSeatState([[seat_box(0.2, 0.8)]]),
                          seat_zones=SeatZones(seatmap, on_reads=1, off_reads=1))
    htmls = {seat_grid_html(p.process(np.zeros((120, 160, 3), np.uint8))) for _ in range(6)}
    assert len(htmls) == 1


# -- loading zones / editor -----------------------------------------------------
def test_zone_editor_numbers_seats_by_row_side_and_position_and_roundtrips(tmp_path):
    ed = ZoneEditor(200, 100)
    for _ in range(2):                                      # two seats on the left of row 1
        for x, y in ((10, 10), (30, 10), (30, 30), (10, 30)):
            ed.click(x, y)
        ed.commit()
    ed.set_side("R")
    for x, y in ((120, 10), (140, 10), (140, 30)):
        ed.click(x, y)
    assert ed.commit() == "R1R1"
    ed.next_row()
    for x, y in ((10, 60), (30, 60), (30, 80)):
        ed.click(x, y)
    assert ed.commit() == "R2L1"
    assert list(ed.seats) == ["R1L1", "R1L2", "R1R1", "R2L1"]
    path = tmp_path / "zones.yaml"
    ed.save(path)
    zones = load_zones(path)
    assert list(zones) == list(ed.seats) and SeatMap(zones).seat_ids == list(zones)


def test_zone_editor_undo_and_too_few_corners(tmp_path):
    ed = ZoneEditor(100, 100)
    ed.click(1, 1)
    ed.click(5, 5)
    assert ed.commit() is None                              # needs 3 corners
    ed.click(9, 1)
    assert ed.commit() == "R1L1"
    ed.click(50, 50)
    ed.undo()                                               # drops the pending corner
    assert ed.seats and not ed.points
    ed.undo()                                               # then the last finished seat
    assert not ed.seats and ed.next_id == "R1L1"
    ed.undo()                                               # nothing left: harmless
    with pytest.raises(ValueError):
        ed.save(tmp_path / "never.yaml")
    assert not (tmp_path / "never.yaml").exists()
    ed.click(-50, 500)                                      # clicks are clamped to the image
    assert ed.points == [(0.0, 1.0)]
    with pytest.raises(ValueError):
        ed.set_side("X")
    with pytest.raises(ValueError):
        ZoneEditor(0, 10)


def test_load_zones_rejects_missing_and_malformed_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_zones(tmp_path / "nope.yaml")
    (tmp_path / "bad.yaml").write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_zones(tmp_path / "bad.yaml")
    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_zones(tmp_path / "empty.yaml")
    (tmp_path / "out.yaml").write_text("seats:\n  A: [[0.1, 0.1], [2.0, 0.1], [0.5, 0.5]]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        SeatMap(load_zones(tmp_path / "out.yaml"))          # coordinates outside 0-1 are refused


def test_seatmap_from_config_marks_measured_seats_calibrated(tmp_path):
    f = tmp_path / "z.yaml"
    f.write_text("seats:\n  R1L1: [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3]]\n", encoding="utf-8")
    cfg = {"rows": 2, "left": 1, "right": 1, "polygons": {}, "polygons_file": str(f)}
    assert SeatMap.from_config(cfg).calibrated and SeatMap.from_config(cfg).seat_ids == ["R1L1"]
    assert not SeatMap.from_config({"rows": 2, "left": 1, "right": 1, "polygons": {}}).calibrated
