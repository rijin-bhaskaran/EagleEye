"""Seat layout and person-to-seat assignment.

A seat is a polygon in normalised image coordinates (0-1). Seats are rasterised
once to small masks, so assigning a person is a cheap array slice instead of
polygon clipping, and no geometry library is needed.

A person box usually overlaps several seats in perspective (heads of the row in
front cover the row behind), so each person is scored against each seat and
matched one-to-one, best score first.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

Box = Tuple[float, float, float, float]          # x1, y1, x2, y2, normalised
Polygon = Sequence[Sequence[float]]

_MASK_W, _MASK_H = 320, 240


@dataclass
class Detection:
    box: Box
    conf: float
    track_id: Optional[int] = None


class SeatMap:
    calibrated = False          # True when the polygons were measured (config / zones file), False for the placeholder grid

    def __init__(self, polygons: Mapping[str, Polygon], min_overlap: float = 0.30) -> None:
        if not polygons:
            raise ValueError("a seat map needs at least one seat")
        self.min_overlap = min_overlap
        self.seat_ids: List[str] = list(polygons)
        self.polygons: Dict[str, List[List[float]]] = {sid: [list(map(float, pt)) for pt in poly]
                                                       for sid, poly in polygons.items()}   # for drawing
        self._masks: List[np.ndarray] = []
        self._bounds: List[Tuple[int, int, int, int]] = []     # x0, y0, x1, y1 in mask pixels
        for sid, poly in polygons.items():
            pts = np.array(poly, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
                raise ValueError(f"seat {sid!r} needs at least 3 [x, y] points")
            if pts.min() < 0.0 or pts.max() > 1.0:
                raise ValueError(f"seat {sid!r} has coordinates outside 0-1")
            px = np.round(pts * [_MASK_W - 1, _MASK_H - 1]).astype(np.int32)
            mask = np.zeros((_MASK_H, _MASK_W), np.uint8)
            cv2.fillPoly(mask, [px], 1)
            self._masks.append(mask)
            self._bounds.append((int(px[:, 0].min()), int(px[:, 1].min()),
                                 int(px[:, 0].max()) + 1, int(px[:, 1].max()) + 1))

    def __len__(self) -> int:
        return len(self.seat_ids)

    def assign(self, detections: Sequence[Detection]) -> Dict[str, Detection]:
        """Map seat id -> the person sitting in it (one person per seat, one seat per person)."""
        scores: List[Tuple[float, int, int]] = []
        for di, det in enumerate(detections):
            x1, y1, x2, y2 = det.box
            bx1, by1 = max(int(x1 * _MASK_W), 0), max(int(y1 * _MASK_H), 0)
            bx2, by2 = min(int(np.ceil(x2 * _MASK_W)), _MASK_W), min(int(np.ceil(y2 * _MASK_H)), _MASK_H)
            area = (bx2 - bx1) * (by2 - by1)
            if area <= 0:
                continue
            for si, (sx0, sy0, sx1, sy1) in enumerate(self._bounds):
                if sx1 <= bx1 or sx0 >= bx2 or sy1 <= by1 or sy0 >= by2:
                    continue
                inter = int(self._masks[si][by1:by2, bx1:bx2].sum())
                score = inter / area
                if score >= self.min_overlap:
                    scores.append((score, di, si))

        taken_d: Set[int] = set()
        taken_s: Set[int] = set()
        out: Dict[str, Detection] = {}
        for score, di, si in sorted(scores, key=lambda t: -t[0]):
            if di in taken_d or si in taken_s:
                continue
            taken_d.add(di)
            taken_s.add(si)
            out[self.seat_ids[si]] = detections[di]
        return out

    @classmethod
    def from_config(cls, seats_cfg, min_overlap: float = 0.30) -> "SeatMap":
        """Use explicit polygons if configured, otherwise the placeholder grid."""
        polys = dict(seats_cfg.get("polygons") or {})
        if not polys and seats_cfg.get("polygons_file"):
            polys = load_zones(seats_cfg["polygons_file"])
        calibrated = bool(polys)
        if not polys:
            polys = perspective_grid(int(seats_cfg["rows"]), int(seats_cfg["left"]), int(seats_cfg["right"]))
        seatmap = cls(polys, min_overlap)
        seatmap.calibrated = calibrated
        return seatmap


def load_zones(path) -> Dict[str, List[List[float]]]:
    """Seat polygons (0-1 image coordinates) from a YAML file written by ``scripts/draw_seat_zones.py``."""
    import yaml

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"seat zones file {p} not found (draw it with: python scripts/draw_seat_zones.py)")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    zones = data.get("seats", data) if isinstance(data, dict) else None
    if not isinstance(zones, dict) or not zones:
        raise ValueError(f"{p}: expected a 'seats:' mapping of seat id -> [[x, y], ...]")
    return {str(k): v for k, v in zones.items()}


def perspective_grid(rows: int, left: int, right: int, *, seat_w: float = 0.45, aisle_w: float = 0.40,
                     row_pitch: float = 0.80, first_row_z: float = 1.5, cam_height: float = 1.8,
                     head_height: float = 1.25, horizon_y: float = 0.30,
                     fx: float = 0.33, fy: float = 0.36) -> Dict[str, List[List[float]]]:
    """PLACEHOLDER seat polygons for a camera at the front looking down the aisle.

    Each seat is the projection of a 3D column (floor to head height) through a
    pinhole camera centred on the aisle. Parameters are guesses in metres; real
    footage must replace these with measured polygons. Left/right are image
    sides: ``left`` seats per row on the image-left block, ``right`` on the other.
    Ids are ``R<row>L<n>`` / ``R<row>R<n>`` with row 1 nearest the camera.
    """
    block_l, block_r = left * seat_w, right * seat_w
    # lateral position of the cabin's left edge, with the whole cabin centred on the camera
    total = block_l + aisle_w + block_r
    x0 = -total / 2
    polys: Dict[str, List[List[float]]] = {}

    def project(x: float, y_up: float, z: float) -> List[float]:
        return [min(max(0.5 + x * fx / z, 0.0), 1.0),
                min(max(horizon_y + (cam_height - y_up) * fy / z, 0.0), 1.0)]

    for r in range(rows):
        z_mid = first_row_z + (r + 0.5) * row_pitch     # seat occupant is centred in the row
        for side, count, start in (("L", left, x0), ("R", right, x0 + block_l + aisle_w)):
            for n in range(count):
                xa, xb = start + n * seat_w, start + (n + 1) * seat_w
                polys[f"R{r + 1}{side}{n + 1}"] = [
                    project(xa, 0.0, z_mid), project(xb, 0.0, z_mid),
                    project(xb, head_height, z_mid), project(xa, head_height, z_mid),
                ]
    return polys
