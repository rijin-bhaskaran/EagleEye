"""Calmer boxes and IDs for the car chain.

The detector's tracker drops a box for a frame or two, jitters it, and hands a person a new ID when it loses them for a moment.
On the dashboard that is boxes that blink, and labels that restart: a new ID has no votes yet, so the person is ``unknown`` again.
``TrackStabiliser`` sits between the detector and the classifier and does three small things:

* **smooths** each box (exponential moving average) so it stops jittering;
* **holds** a box on screen for ``hold_frames`` frames after the tracker lost it (shown, but not counted and not classified);
* **keeps the ID**: a tracker ID it has never seen that lands on top of a recently lost person takes that person's stable ID,
  so their votes (and label) carry over.

Stable IDs are small numbers in order of first appearance. A tracker ID that is still alive always keeps its stable ID.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .seats import Box, Detection


def _iou(a: Box, b: Box) -> float:
    ix, iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    box: Box
    conf: float
    last_seen: int


class TrackStabiliser:
    def __init__(self, hold_frames: int = 4, id_memory: int = 45, id_iou: float = 0.3, ema: float = 0.6) -> None:
        if hold_frames < 0 or id_memory < 0:
            raise ValueError("hold_frames and id_memory must be 0 or more")
        if not 0.0 < ema <= 1.0:
            raise ValueError("ema must be in (0, 1]: 1 means no smoothing")
        if not 0.0 <= id_iou <= 1.0:
            raise ValueError("id_iou must be between 0 and 1")
        self.hold_frames, self.id_memory, self.id_iou, self.ema = hold_frames, id_memory, id_iou, ema
        self._frame = 0
        self._next_id = 1
        self._tracks: Dict[int, _Track] = {}              # stable id -> last known state
        self._stable_of: Dict[int, int] = {}              # tracker id -> stable id

    def update(self, dets: Sequence[Detection]) -> Tuple[List[Detection], List[Detection]]:
        """``(shown, real)``: ``real`` is this frame's detections with stable IDs and smoothed boxes (count and classify these);
        ``shown`` is ``real`` plus the boxes of people lost within ``hold_frames`` frames (draw these)."""
        self._frame += 1
        real: List[Detection] = []
        matched: set = set()
        fresh: List[Detection] = []                                        # tracker IDs not yet mapped to a live stable track
        for d in dets:
            if d.track_id is None:
                real.append(d)                                             # no identity: nothing to smooth, hold or vote on
                continue
            sid = self._stable_of.get(d.track_id)
            if sid is not None and sid in self._tracks and sid not in matched:
                real.append(self._follow(sid, d))
                matched.add(sid)
            else:
                self._stable_of.pop(d.track_id, None)
                fresh.append(d)

        # a new tracker ID on top of a recently lost person is that person: best overlaps first, one-to-one
        lost = {sid: t for sid, t in self._tracks.items() if sid not in matched and self._frame - t.last_seen <= self.id_memory}
        pairs = sorted(((_iou(d.box, t.box), i, sid) for i, d in enumerate(fresh) for sid, t in lost.items()), reverse=True)
        taken_d, taken_s = set(), set()
        assigned: Dict[int, int] = {}
        for score, i, sid in pairs:
            if score < self.id_iou or i in taken_d or sid in taken_s:
                continue
            taken_d.add(i)
            taken_s.add(sid)
            assigned[i] = sid
        for i, d in enumerate(fresh):
            if i in assigned:
                sid = assigned[i]
                real.append(self._follow(sid, d))
            else:
                sid = self._next_id
                self._next_id += 1
                self._tracks[sid] = _Track(d.box, d.conf, self._frame)
                real.append(Detection(d.box, d.conf, sid))
            self._stable_of[d.track_id] = sid
            matched.add(sid)

        held = [Detection(t.box, t.conf, sid) for sid, t in self._tracks.items()
                if sid not in matched and 0 < self._frame - t.last_seen <= self.hold_frames]
        for sid in [s for s, t in self._tracks.items() if self._frame - t.last_seen > max(self.id_memory, self.hold_frames)]:
            del self._tracks[sid]
        for raw in [r for r, s in self._stable_of.items() if s not in self._tracks]:
            del self._stable_of[raw]
        return real + held, real

    def _follow(self, sid: int, d: Detection) -> Detection:
        """Update the track ``sid`` with detection ``d``: smoothed while it is continuous, reset after a longer gap."""
        t = self._tracks[sid]
        if self._frame - t.last_seen <= self.hold_frames + 1:
            box = tuple(self.ema * n + (1.0 - self.ema) * o for n, o in zip(d.box, t.box))
        else:
            box = tuple(d.box)
        t.box, t.conf, t.last_seen = box, d.conf, self._frame
        return Detection(box, d.conf, sid)
