"""Draw detections and seat outlines onto the passenger frame."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict

import cv2
import numpy as np

from .hub import COLORS, hex_to_bgr
from .pipeline import PassengerState
from .seats import SeatMap

_BGR = {label: hex_to_bgr(color) for label, color in COLORS.items()}
_VACANT = (200, 200, 200)
_CHILD_SEAT = (0, 165, 255)        # orange (BGR)
_NO_BELT = (77, 72, 229)           # red (BGR)


@lru_cache(maxsize=4)
def _pixel_polygons(seatmap: SeatMap, w: int, h: int) -> Dict[str, np.ndarray]:
    """Seat polygons in pixels. They only change with the frame size, so convert once, not per frame."""
    return {sid: (np.array(poly) * [w, h]).astype(np.int32) for sid, poly in seatmap.polygons.items()}


def draw(frame_bgr: np.ndarray, state: PassengerState, seatmap: SeatMap, draw_seats: bool = True) -> np.ndarray:
    """Annotate ``frame_bgr`` in place and return it."""
    h, w = frame_bgr.shape[:2]
    scale = max(w / 1280.0, 0.5)
    thin, thick = 1, max(int(round(2 * scale)), 1)

    if draw_seats and seatmap is not None:
        occupied = set(state.occupied)
        by_color: Dict[tuple, list] = {}               # one polylines call per colour, not per seat
        for sid, pts in _pixel_polygons(seatmap, w, h).items():
            color = _BGR.get(state.seat_labels.get(sid, "unknown"), _BGR["unknown"]) if sid in occupied else _VACANT
            by_color.setdefault(color, []).append(pts)
        for color, polys in by_color.items():
            cv2.polylines(frame_bgr, polys, True, color, thin)     # aliased: far cheaper, fine for 1 px outlines

    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.5 * max(scale, 0.8)
    car = state.car
    if car is not None:
        px = lambda b: tuple(int(v) for v in (b[0] * w, b[1] * h, b[2] * w, b[3] * h))      # noqa: E731
        for b in car.child_seat_boxes:
            x1, y1, x2, y2 = px(b)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), _CHILD_SEAT, thick)
            cv2.putText(frame_bgr, "child seat", (x1, max(y1 - 6, 12)), font, fs, _CHILD_SEAT, thick)
        belt_by_track = {o.track_id: o.belt for o in car.occupants}
    for d in state.detections:
        x1, y1, x2, y2 = (int(v) for v in (d.box[0] * w, d.box[1] * h, d.box[2] * w, d.box[3] * h))
        label = state.track_labels.get(d.track_id, "unknown")
        color = _BGR.get(label, _BGR["unknown"])
        if car is not None and d.track_id in belt_by_track:
            belt = belt_by_track[d.track_id]
            label += {"off": " NO BELT", "on": " +belt", "child_seat": " +seat"}.get(belt, "")
            color = _NO_BELT if belt == "off" else color
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thick if label != "unknown" else thin)
        cv2.putText(frame_bgr, label, (x1, max(y1 - 6, 12)), font, fs, color, thick)   # no anti-aliasing: 3x faster, needed on a Pi

    return frame_bgr

