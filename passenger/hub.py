"""Shared state between the passenger loop and its HTTP server.

Mirrors the driver ``TelemetryHub`` contract so the dashboard handles both the
same way: the page is a thin renderer, and the payload is a flat map of
element id -> value (``text`` / ``width`` / ``html``). All formatting lives
here, in Python.
"""
from __future__ import annotations

import html
import re
import threading
import time
from typing import Dict, Iterator, Optional

import cv2

from .pipeline import PassengerState

#: One colour per label, shared by the dashboard seat map and the video overlay.
COLORS = {"male": "#2F80ED", "female": "#FF4FA3", "child": "#22B14C", "unknown": "#9A9AA5"}
_VACANT_FILL, _VACANT_EDGE = "#FDEBE7", "#F6D5CE"
_SEAT_ID = re.compile(r"^R(\d+)([LR])(\d+)$")


def hex_to_bgr(color: str) -> tuple:
    return int(color[5:7], 16), int(color[3:5], 16), int(color[1:3], 16)


class PassengerHub:
    """Latest annotated frame (condition-variable pub/sub) and latest telemetry payload."""

    def __init__(self, jpeg_quality: int = 80) -> None:
        self._jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        self._cond = threading.Condition()
        self._frame: Optional[bytes] = None
        self._seq = 0
        self.closed = False
        self._lock = threading.Lock()
        self._payload: Dict[str, object] = {}

    def publish_frame(self, frame_bgr) -> None:
        ok, buf = cv2.imencode(".jpg", frame_bgr, self._jpeg_params)
        if not ok:
            return
        with self._cond:
            self._frame = buf.tobytes()
            self._seq += 1
            self._cond.notify_all()

    def frames(self) -> Iterator[bytes]:
        """Yield each new JPEG once, for the MJPEG response."""
        last = -1
        while not self.closed:
            with self._cond:
                if self._seq == last:
                    self._cond.wait(1.0)
                if self._seq == last or self._frame is None:
                    continue
                last = self._seq
                data = self._frame
            yield data

    def publish(self, state: PassengerState, *, fps: float, latency_ms: float, classifier_on: bool,
                classifier_label: Optional[str] = None) -> None:
        payload = build_payload(state, fps=fps, latency_ms=latency_ms, classifier_on=classifier_on,
                                classifier_label=classifier_label)
        with self._lock:
            self._payload = payload

    def snapshot(self) -> Dict[str, object]:
        """Copy of the latest payload plus its age, so a stalled loop is visible to the dashboard."""
        with self._lock:
            payload = dict(self._payload)
        if payload:
            payload["age_s"] = round(time.time() - float(payload["ts"]), 2)
        return payload

    def close(self) -> None:
        self.closed = True
        with self._cond:
            self._cond.notify_all()


_BELT_TEXT = {"on": ("belted", "#22B14C"), "off": ("NO BELT", "#E5484D"), "unknown": ("belt ?", "#9A9AA5"),
              "child_seat": ("in child seat", "#22B14C")}


def car_status_html(state: PassengerState) -> str:
    """Car panel in place of the bus seat map: one line per occupant, the child-seat line, then alerts."""
    car = state.car
    rows = []
    for o in sorted(car.occupants, key=lambda o: (o.track_id is None, o.track_id or 0)):
        text, color = _BELT_TEXT.get(o.belt, _BELT_TEXT["unknown"])
        who = html.escape(o.label) + (f" #{o.track_id}" if o.track_id is not None else "")
        rows.append(f'<div class="flex items-center justify-between gap-3 text-xs"><span class="font-semibold" '
                    f'style="color:{COLORS.get(o.label, COLORS["unknown"])}">{who}</span>'
                    f'<span class="font-mono" style="color:{color}">{text}</span></div>')
    seats = (f"{car.child_seats} child seat{'s' if car.child_seats != 1 else ''} installed" if car.child_seats
             else "no child seat installed")
    rows.append(f'<div class="mt-1 text-xs text-slate-500">{seats}</div>')
    for a in car.alerts:
        rows.append(f'<div class="mt-1 rounded-md px-2 py-1 text-xs font-semibold" style="background:#FDEBE7;color:#B42318">'
                    f'{html.escape(a)}</div>')
    return '<div class="flex w-full flex-col gap-1">' + "".join(rows) + "</div>"


def build_payload(state: PassengerState, *, fps: float, latency_ms: float, classifier_on: bool,
                  classifier_label: Optional[str] = None) -> Dict[str, object]:
    if state.seats_seen is not None:               # seat-state model: counts the seats it can see, no calibration needed
        occupied, vacant = state.seats_seen
        seats = occupied + vacant
    else:
        occupied, vacant, seats = len(state.occupied), state.vacant_count, state.total_seats
    total = max(seats, 1)
    comp = state.composition or {}
    car = state.car
    payload = {
        "ts": time.time(),
        "online": True,
        "text": {
            "paxCount": str(state.count),
            "paxOccupied": str(occupied),
            "paxVacant": str(vacant),
            "paxTotal": str(seats),
            "paxOccPct": f"{100 * occupied / total:.0f}%",
            "paxMale": str(comp.get("male", 0)),
            "paxFemale": str(comp.get("female", 0)),
            "paxChild": str(comp.get("child", 0)),
            "paxUnknown": str(comp.get("unknown", 0)),
            "paxFps": f"{fps:.1f} FPS",
            "paxLatency": f"{latency_ms:.0f} ms",
            "paxMode": "CAM_02 [ACTIVE]",
            "paxClassifier": "Face model live" if classifier_on else "Classifier off",
        },
        "width": {"paxOccBar": f"{100 * occupied / total:.0f}%"},
        "html": {},                                   # seat map removed from the dashboard
    }
    if classifier_on and classifier_label:         # the pipeline names its own model ("Crop model live")
        payload["text"]["paxClassifier"] = classifier_label
    if car is not None:
        payload["text"]["paxClassifier"] = "Cabin model live"
        payload["car"] = {"childPresent": car.child_present, "childSeats": car.child_seats,
                          "childrenUnseated": car.children_unseated, "unbelted": car.unbelted, "alerts": list(car.alerts),
                          "occupants": [{"id": o.track_id, "label": o.label, "belt": o.belt, "inChildSeat": o.in_child_seat}
                                        for o in car.occupants]}
    return payload


def _cell(seat_id: str, state: PassengerState, occupied: set) -> str:
    if seat_id in occupied:
        label = state.seat_labels.get(seat_id, "unknown")
        style = f"background:{COLORS.get(label, COLORS['unknown'])}"
        title = f"{seat_id} · {label}"
    else:
        style = f"background:{_VACANT_FILL};border:1px solid {_VACANT_EDGE}"
        title = f"{seat_id} · vacant"
    title = html.escape(title)                      # seat ids come from config.yaml
    return f'<span title="{title}" class="inline-block h-3.5 w-3.5 rounded-[3px]" style="{style}"></span>'


def model_seat_grid_html(state: PassengerState, ids) -> str:
    """The fixed seat grid (``ids``), idle until the seat model reports occupancy.

    Without measured seats the model cannot say *which* seat is taken, so the map is a stable gauge: the first
    ``occupied`` slots (front row first, left to right) are marked, coloured by the people's labels in turn
    (male, female, child, then unknown). The layout never moves, and the marks change only when the smoothed
    seat count changes, so a single odd model read cannot make it blink. Draw real seat zones
    (``scripts/draw_seat_zones.py``) to mark the actual seats instead.
    """
    occupied_n = state.seats_seen[0]
    comp = state.composition or {}
    colours = [lab for lab in ("male", "female", "child") for _ in range(comp.get(lab, 0))]
    colours += ["unknown"] * max(occupied_n - len(colours), 0)
    parsed = {sid: _SEAT_ID.match(sid) for sid in ids}
    grid: Dict[int, Dict[str, list]] = {}
    for sid, m in parsed.items():
        grid.setdefault(int(m[1]), {"L": [], "R": []})[m[2]].append((int(m[3]), sid))
    order = [sid for r in sorted(grid) for side in ("L", "R") for _, sid in sorted(grid[r][side])]
    fill = {sid: colours[k] for k, sid in enumerate(order[:occupied_n])}
    out = []
    for r in sorted(grid):
        cells = {}
        for side in ("L", "R"):
            html_cells = []
            for _, sid in sorted(grid[r][side]):
                if sid in fill:
                    style, title = f"background:{COLORS.get(fill[sid], COLORS['unknown'])}", f"{sid} · {fill[sid]}"
                else:
                    style, title = f"background:{_VACANT_FILL};border:1px solid {_VACANT_EDGE}", f"{sid} · vacant"
                html_cells.append(f'<span title="{html.escape(title)}" class="inline-block h-3.5 w-3.5 rounded-[3px]" style="{style}"></span>')
            cells[side] = "".join(html_cells)
        out.append('<div class="flex items-center justify-center gap-1">'
                   f'<span class="flex gap-1">{cells["L"]}</span><span class="w-4"></span>'
                   f'<span class="flex gap-1">{cells["R"]}</span></div>')
    return '<div class="flex flex-col gap-1">' + "".join(out) + "</div>"


def seat_grid_html(state: PassengerState) -> str:
    """Seat map, front row first. Falls back to a flat list for custom seat ids."""
    ids = [*state.occupied, *state.vacant]
    if state.seats_seen is not None and ids and all(_SEAT_ID.match(s) for s in ids):
        return model_seat_grid_html(state, ids)         # same fixed grid, slots marked from the seat-state model
    occupied = set(state.occupied)
    ids = [*state.occupied, *state.vacant]
    parsed = {sid: _SEAT_ID.match(sid) for sid in ids}
    if not ids or not all(parsed.values()):
        return '<div class="flex flex-wrap gap-1">' + "".join(_cell(s, state, occupied) for s in sorted(ids)) + "</div>"

    rows: Dict[int, Dict[str, list]] = {}
    for sid, m in parsed.items():
        rows.setdefault(int(m[1]), {"L": [], "R": []})[m[2]].append((int(m[3]), sid))
    out = []
    for r in sorted(rows):
        sides = {k: "".join(_cell(s, state, occupied) for _, s in sorted(v)) for k, v in rows[r].items()}
        out.append('<div class="flex items-center justify-center gap-1">'
                   f'<span class="flex gap-1">{sides["L"]}</span><span class="w-4"></span>'
                   f'<span class="flex gap-1">{sides["R"]}</span></div>')
    return '<div class="flex flex-col gap-1">' + "".join(out) + "</div>"
