#!/usr/bin/env python3
"""Draw the seats of your cabin camera once, by clicking their corners.

    python scripts/draw_seat_zones.py --source data/cabin.mp4 --frame 50     # a frame of a video
    python scripts/draw_seat_zones.py --source 1                              # a live camera (index)
    python scripts/draw_seat_zones.py --source cabin.jpg                      # a still

Click the corners of one seat (4 is normal: the seat back and cushion as the camera sees them), press ENTER to keep it.
Seats are numbered R<row>L<n> / R<row>R<n>, row 1 nearest the camera, left to right.

    ENTER finish seat    U undo corner / seat    L / R side of the aisle    N next row    S save    Q quit

Then set in config.yaml:   passenger.seats.polygons_file: seat_zones.yaml
The camera must stay fixed: zones drawn on a moving camera are wrong the moment it moves.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from passenger.zone_editor import ZoneEditor  # noqa: E402


def grab(source: str, frame_no: int) -> np.ndarray:
    p = Path(source)
    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        img = cv2.imread(str(p))
        if img is None:
            raise SystemExit(f"cannot read image {p}")
        return img
    cap = cv2.VideoCapture(int(source) if source.isdigit() else str(p))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {source}")
    if not source.isdigit():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read a frame from {source}")
    return img


def render(base: np.ndarray, ed: ZoneEditor) -> np.ndarray:
    img = base.copy()
    for sid, pts in ed.seats.items():
        px = (np.array(pts) * [ed.w, ed.h]).astype(np.int32)
        cv2.polylines(img, [px], True, (0, 220, 0), 2)
        cx, cy = px.mean(axis=0).astype(int)
        cv2.putText(img, sid, (cx - 18, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    pts = [(int(x * ed.w), int(y * ed.h)) for x, y in ed.points]
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, (0, 140, 255), 2)
    for pt in pts:
        cv2.circle(img, pt, 4, (0, 140, 255), -1)
    cv2.putText(img, f"next: {ed.next_id}   ENTER=keep U=undo L/R=side N=row S=save Q=quit", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Draw seat zones for the passenger monitor")
    ap.add_argument("--source", required=True, help="video file, image, or camera index")
    ap.add_argument("--frame", type=int, default=0, help="video frame to draw on")
    ap.add_argument("--out", default="seat_zones.yaml")
    a = ap.parse_args(argv)

    base = grab(a.source, a.frame)
    ed = ZoneEditor(base.shape[1], base.shape[0])
    win = "seat zones"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, lambda ev, x, y, *_: ed.click(x, y) if ev == cv2.EVENT_LBUTTONDOWN else None)
    while True:
        cv2.imshow(win, render(base, ed))
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 10):
            ed.commit()
        elif k in (ord("u"), 8):
            ed.undo()
        elif k == ord("l"):
            ed.set_side("L")
        elif k == ord("r"):
            ed.set_side("R")
        elif k == ord("n"):
            ed.next_row()
        elif k == ord("s"):
            ed.save(a.out)
            print(f"saved {len(ed.seats)} seats to {a.out}")
        elif k in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
