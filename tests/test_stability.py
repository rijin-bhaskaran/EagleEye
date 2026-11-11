"""TrackStabiliser: calmer boxes and IDs for the car chain (smoothing, a short hold for a missed frame, a returning person keeps their ID)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from passenger.seats import Detection
from passenger.stability import TrackStabiliser

A = (0.10, 0.20, 0.30, 0.80)
B = (0.60, 0.20, 0.80, 0.80)


def d(box, tid, conf=0.9):
    return Detection(tuple(box), conf, tid)


def shift(box, dx=0.0, dy=0.0):
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def test_a_steady_person_keeps_the_id_and_the_box():
    s = TrackStabiliser()
    for _ in range(5):
        shown, real = s.update([d(A, 3)])
    assert [x.track_id for x in real] == [1] and shown == real
    assert real[0].box == pytest.approx(A, abs=1e-9)


def test_the_stable_ids_are_small_numbers_in_order_of_first_appearance():
    s = TrackStabiliser()
    _, real = s.update([d(A, 41), d(B, 17)])
    assert sorted(x.track_id for x in real) == [1, 2]


def test_smoothing_reduces_frame_to_frame_jitter():
    rng = np.random.default_rng(0)
    raw = [shift(A, rng.normal(0, 0.01), rng.normal(0, 0.01)) for _ in range(60)]
    s = TrackStabiliser(ema=0.4)
    out = [s.update([d(b, 1)])[1][0].box for b in raw]

    def jitter(boxes):
        return float(np.mean([abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(boxes, boxes[1:])]))

    assert jitter(out) < 0.7 * jitter(raw)


def test_smoothing_still_follows_a_person_who_really_moves():
    s = TrackStabiliser(ema=0.6)
    box = A
    for _ in range(40):
        box = shift(box, 0.005)
        _, real = s.update([d(box, 1)])
    assert abs(real[0].box[0] - box[0]) < 0.01                       # lags a little, does not stay behind


def test_a_missed_frame_keeps_the_box_on_screen_but_not_in_the_real_detections():
    s = TrackStabiliser(hold_frames=3)
    s.update([d(A, 1)])
    shown, real = s.update([])
    assert real == [] and [x.track_id for x in shown] == [1] and shown[0].box == pytest.approx(A)


def test_the_hold_ends_after_hold_frames():
    s = TrackStabiliser(hold_frames=3)
    s.update([d(A, 1)])
    for k in range(3):
        shown, _ = s.update([])
        assert len(shown) == 1, k
    shown, _ = s.update([])
    assert shown == []


def test_a_returning_person_with_a_new_tracker_id_gets_the_old_stable_id():
    s = TrackStabiliser(hold_frames=2, id_memory=30)
    _, first = s.update([d(A, 5)])
    for _ in range(6):                                                # gone for longer than the hold
        s.update([])
    _, back = s.update([d(shift(A, 0.01), 99)])
    assert back[0].track_id == first[0].track_id


def test_the_returning_person_takes_the_id_of_the_track_it_overlaps_not_another():
    s = TrackStabiliser(hold_frames=1, id_memory=30)
    _, both = s.update([d(A, 1), d(B, 2)])
    ida, idb = both[0].track_id, both[1].track_id
    for _ in range(4):
        s.update([])
    _, back = s.update([d(shift(B, 0.02), 50)])
    assert back[0].track_id == idb and back[0].track_id != ida


def test_a_new_person_far_from_every_lost_track_is_a_new_id():
    s = TrackStabiliser(hold_frames=1, id_memory=30)
    _, first = s.update([d(A, 1)])
    for _ in range(4):
        s.update([])
    _, other = s.update([d(B, 2)])
    assert other[0].track_id != first[0].track_id


def test_after_the_memory_expires_the_same_place_is_a_new_person():
    s = TrackStabiliser(hold_frames=1, id_memory=5)
    _, first = s.update([d(A, 1)])
    for _ in range(8):
        s.update([])
    _, again = s.update([d(A, 2)])
    assert again[0].track_id != first[0].track_id


def test_two_new_tracker_ids_cannot_both_inherit_one_lost_track():
    s = TrackStabiliser(hold_frames=1, id_memory=30)
    _, first = s.update([d(A, 1)])
    for _ in range(4):
        s.update([])
    _, back = s.update([d(shift(A, 0.005), 8), d(shift(A, 0.03), 9)])
    ids = [x.track_id for x in back]
    assert len(set(ids)) == 2 and first[0].track_id in ids


def test_a_tracker_id_that_is_still_alive_keeps_its_stable_id_however_far_it_moves():
    s = TrackStabiliser()
    _, a = s.update([d(A, 4)])
    _, b = s.update([d(B, 4)])                                         # same tracker id, far away (fast motion, camera cut)
    assert a[0].track_id == b[0].track_id


def test_untracked_detections_pass_through_without_an_id_or_a_hold():
    s = TrackStabiliser(hold_frames=3)
    shown, real = s.update([d(A, None)])
    assert real[0].track_id is None and real[0].box == pytest.approx(A)
    shown, real = s.update([])
    assert shown == [] and real == []


def test_the_confidence_of_a_held_box_is_kept_so_it_can_be_drawn_dimmer():
    s = TrackStabiliser(hold_frames=2)
    s.update([d(A, 1, conf=0.8)])
    shown, _ = s.update([])
    assert shown[0].conf == pytest.approx(0.8)


def test_settings_are_validated():
    for bad in (dict(hold_frames=-1), dict(id_memory=-1), dict(ema=0.0), dict(ema=1.5), dict(id_iou=1.5)):
        with pytest.raises(ValueError):
            TrackStabiliser(**bad)
