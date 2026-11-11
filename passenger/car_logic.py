"""Car-cabin decisions from a multi-class detector: who is there, are they belted, is a child in a child seat.

The car model is one YOLO with these classes (``CAR_CLASSES``):

    male, female, child        the occupant (whole visible body/torso)
    seatbelt                   a belt worn across an occupant
    no_seatbelt                an occupant clearly without a belt (only if the training data labels it)
    child_seat                 an installed child seat, occupied or empty

Everything here is a pure function or a small state holder, so the rules are testable without weights.
Boxes are normalised ``(x1, y1, x2, y2)`` like the rest of the passenger package.

Rules:
* A belt belongs to the occupant that contains most of the belt box (``belt_overlap``).
* An occupant is *belted* when belt evidence shows in at least ``belt_on_ratio`` of the recent frames, *unbelted*
  when enough frames were seen without it, otherwise *unknown*. The ratio is low on purpose: belts are thin and
  often hidden, so a single sighting should outweigh many misses, or the system would cry wolf.
* A child is *seated* when most of its box lies inside a child-seat box (``child_seat_overlap``). Seated children
  are harnessed by the seat, so no belt verdict is given for them.
* ``child_present`` with no child seat anywhere in view, or with a child outside every seat, raises an alert.
"""
from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]

CAR_CLASSES: Tuple[str, ...] = ("male", "female", "child", "seatbelt", "no_seatbelt", "child_seat")
PERSON_CLASSES = ("male", "female", "child")

BELT_ON, BELT_OFF, BELT_UNKNOWN, BELT_SEATED = "on", "off", "unknown", "child_seat"

ALERT_NO_CHILD_SEAT = "child on board but no child seat installed"
ALERT_CHILD_UNSEATED = "child is not in a child seat"
ALERT_UNBELTED = "occupant without seatbelt"


@dataclass(frozen=True)
class CarDetection:
    cls: str                       # one of CAR_CLASSES
    box: Box
    conf: float
    track_id: Optional[int] = None


@dataclass
class OccupantStatus:
    track_id: Optional[int]
    label: str                     # male / female / child
    box: Box
    belt: str                      # on / off / unknown / child_seat
    in_child_seat: bool = False


@dataclass
class CabinStatus:
    occupants: List[OccupantStatus] = field(default_factory=list)
    child_present: bool = False
    child_seats: int = 0                       # installed child seats in view, empty ones included
    children_unseated: int = 0
    unbelted: int = 0
    alerts: List[str] = field(default_factory=list)
    child_seat_boxes: List[Box] = field(default_factory=list)
    belt_boxes: List[Box] = field(default_factory=list)


def area(b: Box) -> float:
    return max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)


def inside_fraction(inner: Box, outer: Box) -> float:
    """Share of ``inner``'s area that lies within ``outer`` (0 when ``inner`` has no area)."""
    w = min(inner[2], outer[2]) - max(inner[0], outer[0])
    h = min(inner[3], outer[3]) - max(inner[1], outer[1])
    a = area(inner)
    return (w * h) / a if (w > 0 and h > 0 and a > 0) else 0.0


def assign_items(items: Sequence[Box], people: Sequence[Box], min_inside: float) -> Dict[int, int]:
    """item index -> person index: each item goes to the person holding most of it (needs ``min_inside``)."""
    out: Dict[int, int] = {}
    for i, item in enumerate(items):
        best, best_f = -1, min_inside
        for j, person in enumerate(people):
            f = inside_fraction(item, person)
            if f >= best_f and (f > best_f or best < 0 or area(person) < area(people[best])):
                best, best_f = j, f
        if best >= 0:
            out[i] = best
    return out


class _Votes:
    """Recent yes/no sightings for one thing (a person's belt, a child's seat)."""

    def __init__(self, window: int) -> None:
        self.v: Deque[bool] = deque(maxlen=max(window, 1))

    def add(self, x: bool) -> None:
        self.v.append(x)

    def share(self) -> float:
        return sum(self.v) / len(self.v) if self.v else 0.0

    def __len__(self) -> int:
        return len(self.v)


class CabinAnalyzer:
    """Per-frame detections -> smoothed ``CabinStatus``. Keeps the per-person votes that stop flicker."""

    def __init__(self, *, window: int = 15, min_votes: int = 3, belt_overlap: float = 0.5, belt_on_ratio: float = 0.3,
                 child_seat_overlap: float = 0.5, seat_window: int = 15, max_track_age: int = 90) -> None:
        self.window, self.min_votes = window, min_votes
        self.belt_overlap, self.belt_on_ratio = belt_overlap, belt_on_ratio
        self.child_seat_overlap = child_seat_overlap
        self.max_track_age = max_track_age
        self._class: Dict[int, Deque[str]] = {}
        self._belt: Dict[int, _Votes] = {}
        self._seated: Dict[int, _Votes] = {}
        self._seen_at: Dict[int, int] = {}
        self._seat_counts: Deque[int] = deque(maxlen=max(seat_window, 1))
        self._frame = 0

    # -- per person -------------------------------------------------------------------------------
    def _label(self, tid: Optional[int], cls: str) -> str:
        if tid is None:
            return cls
        votes = self._class.setdefault(tid, deque(maxlen=self.window))
        votes.append(cls)
        if len(votes) < self.min_votes:
            return cls
        return max(set(votes), key=lambda c: (votes.count(c), c == cls))     # majority; the newest read breaks ties

    def _belt_status(self, tid: Optional[int], seen_on: bool, seen_off: bool) -> str:
        if tid is None:                                  # untracked: this frame only
            return BELT_ON if seen_on else (BELT_OFF if seen_off else BELT_UNKNOWN)
        v = self._belt.setdefault(tid, _Votes(self.window))
        v.add(seen_on)
        if seen_off and not seen_on:
            v.add(False)                                 # an explicit 'no belt' read counts twice
        if v.share() >= self.belt_on_ratio and any(v.v):
            return BELT_ON
        return BELT_OFF if len(v) >= self.min_votes else BELT_UNKNOWN

    def _seated_status(self, tid: Optional[int], now: bool) -> bool:
        if tid is None:
            return now
        v = self._seated.setdefault(tid, _Votes(self.window))
        v.add(now)
        return now if len(v) < self.min_votes else v.share() >= 0.5

    def _forget_old(self) -> None:
        stale = [t for t, at in self._seen_at.items() if self._frame - at > self.max_track_age]
        for t in stale:
            for d in (self._class, self._belt, self._seated, self._seen_at):
                d.pop(t, None)

    # -- per frame --------------------------------------------------------------------------------
    def update(self, dets: Sequence[CarDetection]) -> CabinStatus:
        self._frame += 1
        people = [d for d in dets if d.cls in PERSON_CLASSES]
        belts = [d for d in dets if d.cls == "seatbelt"]
        no_belts = [d for d in dets if d.cls == "no_seatbelt"]
        seats = [d for d in dets if d.cls == "child_seat"]

        pboxes = [p.box for p in people]
        belt_of = assign_items([b.box for b in belts], pboxes, self.belt_overlap)
        nobelt_of = assign_items([b.box for b in no_belts], pboxes, self.belt_overlap)
        on_set, off_set = set(belt_of.values()), set(nobelt_of.values())

        self._seat_counts.append(len(seats))
        n_seats = int(statistics.median(self._seat_counts))

        status = CabinStatus(child_seats=n_seats, child_seat_boxes=[s.box for s in seats], belt_boxes=[b.box for b in belts])
        for j, p in enumerate(people):
            if p.track_id is not None:
                self._seen_at[p.track_id] = self._frame
            label = self._label(p.track_id, p.cls)
            in_seat = False
            if label == "child":
                in_seat = self._seated_status(p.track_id, any(inside_fraction(p.box, s.box) >= self.child_seat_overlap for s in seats))
            belt = BELT_SEATED if in_seat else self._belt_status(p.track_id, j in on_set, j in off_set)
            status.occupants.append(OccupantStatus(p.track_id, label, p.box, belt, in_seat))

        children = [o for o in status.occupants if o.label == "child"]
        status.child_present = bool(children)
        status.children_unseated = sum(not o.in_child_seat for o in children)
        status.unbelted = sum(o.belt == BELT_OFF for o in status.occupants)
        if status.child_present and n_seats == 0:
            status.alerts.append(ALERT_NO_CHILD_SEAT)
        elif status.children_unseated:
            status.alerts.append(ALERT_CHILD_UNSEATED)
        if status.unbelted:
            status.alerts.append(ALERT_UNBELTED)
        self._forget_old()
        return status
