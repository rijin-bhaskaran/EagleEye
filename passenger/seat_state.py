"""Seat-state detector: finds every seat in view and says whether it is Occupied or Unoccupied.

Trained on the Roboflow "busCabin" dataset (``passenger.train_seats``). Unlike the polygon seat map it needs
no calibration, so ``occupied`` / ``vacant`` counts work on any camera angle the model was trained for.
The model is slower than the person detector's budget allows every frame, so it runs every ``every`` frames
and the last result is reused in between (seats do not move).
"""
from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np

from .seat_data import NAMES
from .seats import Box

OCCUPIED, UNOCCUPIED = NAMES


@dataclass(frozen=True)
class SeatBox:
    box: Box
    occupied: bool
    conf: float


class SeatStateDetector:
    def __init__(self, weights: str, conf: float = 0.32, imgsz: int = 640, every: int = 3, window: int = 5) -> None:
        from ultralytics import YOLO                      # lazy: tests do not need the weights

        self._model = YOLO(weights, task="detect")
        self.conf, self.imgsz, self.every = conf, imgsz, max(int(every), 1)
        self._tick = 0
        self.reads = 0                                    # model runs so far; lets callers act once per fresh read
        self._last: List[SeatBox] = []
        self._occ: Deque[int] = deque(maxlen=max(window, 1))
        self._vac: Deque[int] = deque(maxlen=max(window, 1))

    def update(self, frame_bgr: np.ndarray) -> Tuple[List[SeatBox], int, int]:
        """(boxes, occupied seats, vacant seats), counts smoothed by a median over the last few reads."""
        if self._tick % self.every == 0:
            res = self._model.predict(frame_bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
            boxes: List[SeatBox] = []
            if res.boxes is not None and len(res.boxes):
                for b, c, k in zip(res.boxes.xyxyn.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                   res.boxes.cls.cpu().numpy().astype(int)):
                    boxes.append(SeatBox(tuple(map(float, b)), NAMES[k] == OCCUPIED, float(c)))
            self._last = boxes
            self.reads += 1
            self._occ.append(sum(s.occupied for s in boxes))
            self._vac.append(sum(not s.occupied for s in boxes))
        self._tick += 1
        return self._last, int(statistics.median(self._occ)), int(statistics.median(self._vac))


class SeatZones:
    """Fixed seats, filled and emptied by the seat-state model, one slot at a time.

    A seat is a polygon measured once on the (fixed) cabin camera. A model box marks the seat whose polygon
    holds the box centre, so a seat can only ever light its own slot: a missed or extra detection elsewhere
    cannot shift anything. Each slot is then debounced in *model reads* (the model does not run every frame):
    occupied after ``on_reads`` consecutive reads that show it, vacant after ``off_reads`` that do not.
    """

    def __init__(self, seatmap, on_reads: int = 2, off_reads: int = 5) -> None:
        from .pipeline import SeatDebouncer             # local: pipeline imports this module's callers

        self.seatmap = seatmap
        self._debounce = SeatDebouncer(seatmap.seat_ids, on_reads, off_reads)
        self._masks = seatmap._masks
        self._reads = -1
        self.occupied: set = set()

    def _zone_of(self, x: float, y: float) -> int:
        from .seats import _MASK_H, _MASK_W

        px, py = min(max(int(x * _MASK_W), 0), _MASK_W - 1), min(max(int(y * _MASK_H), 0), _MASK_H - 1)
        hits = [i for i, m in enumerate(self._masks) if m[py, px]]
        return hits[0] if len(hits) == 1 else -1       # centre in no zone, or in an overlap of two: ambiguous, ignore it

    def update(self, seat_boxes: List[SeatBox], reads: int) -> set:
        """Occupied seat ids. Only a fresh model read (``reads`` changed) moves the debounce counters."""
        if reads == self._reads:
            return set(self.occupied)
        self._reads = reads
        seen = set()
        for sb in seat_boxes:
            if not sb.occupied:
                continue
            zi = self._zone_of((sb.box[0] + sb.box[2]) / 2, (sb.box[1] + sb.box[3]) / 2)
            if zi >= 0:
                seen.add(self.seatmap.seat_ids[zi])
        self.occupied = self._debounce.update(seen)
        return set(self.occupied)


class StableCount:
    """A count that moves only when the evidence says so: the median of the last ``window`` reads must differ from
    the shown value, and agree with itself, for ``hold`` reads in a row. One odd read cannot nudge it."""

    def __init__(self, window: int = 9, hold: int = 3) -> None:
        self._seen: Deque[int] = deque(maxlen=max(window, 1))
        self._hold = max(hold, 1)
        self.value = 0
        self._cand, self._run, self._started = 0, 0, False

    def update(self, x: int) -> int:
        self._seen.append(int(x))
        m = int(statistics.median(self._seen))
        if not self._started:
            self.value, self._started = m, True
        elif m == self.value:
            self._cand, self._run = m, 0
        else:
            self._run = self._run + 1 if m == self._cand else 1
            self._cand = m
            if self._run >= self._hold:
                self.value, self._run = m, 0
        return self.value
