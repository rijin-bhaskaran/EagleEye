"""State machine behind ``scripts/draw_seat_zones.py``: build seat polygons by clicking their corners.

Seats are numbered the way the dashboard map expects: ``R<row><L|R><n>``, row 1 nearest the camera,
``L`` / ``R`` the image side of the aisle, ``n`` counted left to right. The editor keeps no window or
image, so it can be tested without a display. Coordinates are stored normalised (0-1).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import yaml


class ZoneEditor:
    def __init__(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("image size must be positive")
        self.w, self.h = width, height
        self.row, self.side = 1, "L"
        self.seats: Dict[str, List[Tuple[float, float]]] = {}
        self.points: List[Tuple[float, float]] = []

    @property
    def next_id(self) -> str:
        n = 1 + sum(1 for s in self.seats if s.startswith(f"R{self.row}{self.side}") and s[len(f"R{self.row}{self.side}"):].isdigit())
        return f"R{self.row}{self.side}{n}"

    def click(self, x: float, y: float) -> None:
        self.points.append((min(max(x / self.w, 0.0), 1.0), min(max(y / self.h, 0.0), 1.0)))

    def commit(self) -> str | None:
        """Finish the seat being drawn. Needs at least 3 corners; returns its id (None if there are too few)."""
        if len(self.points) < 3:
            return None
        sid = self.next_id
        self.seats[sid] = list(self.points)
        self.points = []
        return sid

    def undo(self) -> None:
        """Remove the last corner, or, with none pending, the last finished seat."""
        if self.points:
            self.points.pop()
        elif self.seats:
            self.seats.pop(next(reversed(self.seats)))

    def set_side(self, side: str) -> None:
        if side not in ("L", "R"):
            raise ValueError("side is L or R")
        self.side = side

    def next_row(self) -> None:
        self.row += 1
        self.side = "L"

    def to_yaml(self) -> str:
        data = {"seats": {sid: [[round(x, 4), round(y, 4)] for x, y in pts] for sid, pts in self.seats.items()}}
        return yaml.safe_dump(data, sort_keys=False)

    def save(self, path: str | Path) -> None:
        if not self.seats:
            raise ValueError("no seats drawn yet")
        Path(path).write_text(self.to_yaml(), encoding="utf-8")
