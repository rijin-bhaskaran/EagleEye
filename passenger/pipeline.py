"""Per-frame passenger pipeline: detect -> seat assignment -> debounced occupancy -> count."""
from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Protocol, Sequence, Set, Union

import numpy as np

from .classifier import LABELS, UNKNOWN, AttributeService, BackgroundAttributes, make_classifier
from .seats import Detection, SeatMap


class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray, track: bool = True) -> List[Detection]: ...


@dataclass
class PassengerState:
    count: int                                   # smoothed number of people in view
    occupied: List[str]                          # seat ids, debounced
    vacant: List[str]
    total_seats: int
    seat_track: Dict[str, int] = field(default_factory=dict)   # seat id -> track id (when tracked)
    seat_labels: Dict[str, str] = field(default_factory=dict)  # seat id -> male / female / child / unknown
    composition: Dict[str, int] = field(default_factory=dict)  # label -> people in view (incl. 'unknown')
    track_labels: Dict[int, str] = field(default_factory=dict) # track id -> label, seated or standing
    detections: List[Detection] = field(default_factory=list)  # this frame, for drawing
    seat_boxes: List = field(default_factory=list)             # SeatBox list from the seat-state model (empty if off)
    seats_seen: Optional[tuple] = None                         # (occupied, vacant) from the seat-state model, else None
    car: Optional[object] = None                               # CabinStatus when vehicle_type is car, else None

    @property
    def vacant_count(self) -> int:
        return len(self.vacant)


class SeatDebouncer:
    """A seat flips state only after ``on_frames`` hits or ``off_frames`` misses in a row.

    Seated passengers are often occluded by the seat in front for a few frames, so
    going vacant must be much slower than becoming occupied.
    """

    def __init__(self, seat_ids: Sequence[str], on_frames: int, off_frames: int) -> None:
        self._on, self._off = on_frames, off_frames
        self.occupied: Set[str] = set()
        self._run = {sid: 0 for sid in seat_ids}        # consecutive frames disagreeing with current state

    def update(self, seen: Set[str]) -> Set[str]:
        for sid in self._run:
            if (sid in seen) == (sid in self.occupied):
                self._run[sid] = 0
                continue
            self._run[sid] += 1
            if self._run[sid] >= (self._off if sid in self.occupied else self._on):
                self.occupied.symmetric_difference_update({sid})
                self._run[sid] = 0
        return set(self.occupied)


class PassengerPipeline:
    def __init__(self, detector: Detector, seatmap: SeatMap, *, on_frames: int = 3,
                 off_frames: int = 15, count_window: int = 5,
                 attributes: Optional[Union[AttributeService, BackgroundAttributes]] = None,
                 seat_state=None, seat_zones=None, count_reads: int = 9, hold_reads: int = 3) -> None:
        self.count_reads, self.hold_reads = count_reads, hold_reads
        self.detector, self.seatmap, self.attributes = detector, seatmap, attributes
        self.seat_state = seat_state
        self.seat_zones = seat_zones           # SeatZones: fixed measured seats driven by the seat-state model
        self._seat_counts = None               # (StableCount occupied, StableCount vacant) when seats are not measured
        self._counted_read = -1
        self._debounce = SeatDebouncer(seatmap.seat_ids, on_frames, off_frames)
        self._counts: Deque[int] = deque(maxlen=max(count_window, 1))

    def process(self, frame_bgr: np.ndarray) -> PassengerState:
        dets = self.detector.detect(frame_bgr, track=True)
        by_seat = self.seatmap.assign(dets)
        occupied = self._debounce.update(set(by_seat))
        self._counts.append(len(dets))
        ordered = self.seatmap.seat_ids

        labels = self.attributes.update(frame_bgr, dets) if self.attributes else {}
        label_of = lambda d: labels.get(d.track_id, UNKNOWN)       # noqa: E731 - untracked people stay unknown
        composition = {name: 0 for name in (*LABELS, UNKNOWN)}
        for d in dets:
            composition[label_of(d)] += 1
        seat_boxes, seats_seen = [], None
        if self.seat_state is not None:
            seat_boxes, n_occ, n_vac = self.seat_state.update(frame_bgr)
            if self.seat_zones is not None:            # measured seats: each slot is debounced, the layout cannot move
                occupied = self.seat_zones.update(seat_boxes, self.seat_state.reads)
            else:                                      # uncalibrated: stable counts of the seats the model sees
                if self._seat_counts is None:
                    from .seat_state import StableCount
                    self._seat_counts = (StableCount(self.count_reads, self.hold_reads), StableCount(self.count_reads, self.hold_reads))
                if self.seat_state.reads != self._counted_read:          # move only on a fresh model read
                    self._counted_read = self.seat_state.reads
                    for c, x in zip(self._seat_counts, (sum(b.occupied for b in seat_boxes), sum(not b.occupied for b in seat_boxes))):
                        c.update(x)
                seats_seen = (self._seat_counts[0].value, self._seat_counts[1].value)
        return PassengerState(
            seat_boxes=seat_boxes, seats_seen=seats_seen,
            count=int(statistics.median(self._counts)),
            occupied=[s for s in ordered if s in occupied],
            vacant=[s for s in ordered if s not in occupied],
            total_seats=len(ordered),
            seat_track={s: d.track_id for s, d in by_seat.items() if d.track_id is not None},
            seat_labels={s: label_of(by_seat[s]) if s in by_seat else UNKNOWN for s in occupied} if self.seat_zones is not None
                        else {s: label_of(d) for s, d in by_seat.items()},
            composition=composition,
            track_labels=labels,
            detections=dets,
        )

    @classmethod
    def from_config(cls, cfg, detector: Detector, with_classifier: bool = True) -> "PassengerPipeline":
        occ = cfg.passenger.occupancy
        seatmap = SeatMap.from_config(cfg.passenger.seats, occ.min_overlap)
        c = cfg.passenger.classifier
        attributes = AttributeService(make_classifier(cfg), window=c.window,
                                      min_votes=c.min_votes, budget=c.budget_per_frame) if with_classifier else None
        if attributes is not None and c.get("background", True):
            attributes = BackgroundAttributes(attributes)      # face reads off the frame loop
        seat_state = None
        ss = cfg.passenger.get("seat_state")
        if ss and ss.get("enabled", False):
            from pathlib import Path
            from .seat_state import SeatStateDetector
            if not Path(ss["weights"]).is_file():
                print(f"[passenger] seat-state weights {ss['weights']} not found; falling back to the seat grid", flush=True)
            else:
                seat_state = SeatStateDetector(ss["weights"], ss.get("conf", 0.32), ss.get("imgsz", 640),
                                               ss.get("every", 3), ss.get("window", 5))
        seat_zones = None
        if seat_state is not None and seatmap.calibrated:       # measured seats exist: drive them with the model
            from .seat_state import SeatZones
            seat_zones = SeatZones(seatmap, ss.get("on_reads", 2), ss.get("off_reads", 5))
        return cls(detector, seatmap, on_frames=occ.on_frames, off_frames=occ.off_frames,
                   count_window=occ.count_window, attributes=attributes, seat_state=seat_state, seat_zones=seat_zones,
                   count_reads=int(ss.get("count_reads", 9)) if ss else 9, hold_reads=int(ss.get("hold_reads", 3)) if ss else 3)
