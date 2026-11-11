#!/usr/bin/env python3
"""Per-frame cost of the passenger code itself, excluding the neural network.

    python scripts/bench_passenger_overhead.py

Uses a fake detector with 40 people on a 1280x720 frame, so the numbers move only when *our* code
changes (seat assignment, overlay, JPEG encode, dashboard payload). The model's own latency is
measured by ``python -m passenger.evaluate`` and the live dashboard.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from passenger import overlay
from passenger.classifier import Attribute, AttributeService
from passenger.hub import PassengerHub, build_payload
from passenger.pipeline import PassengerPipeline
from passenger.seats import Detection, SeatMap, perspective_grid


def _ms(fn: Callable[[], object], n: int) -> float:
    fn()                                              # warm-up
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0


def measure(n: int = 100, people: int = 40) -> Dict[str, float]:
    """Milliseconds per frame for each step, plus ``total`` (the steps the loop runs on every frame)."""
    seats = SeatMap(perspective_grid(10, 2, 3))
    ids = seats.seat_ids
    rng = np.random.default_rng(0)

    def box(sid):
        pts = seats.polygons[sid]
        return (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))

    dets = [Detection(box(ids[i]), 0.9, track_id=int(i)) for i in rng.choice(len(ids), people, replace=False)]

    class FakeDetector:
        def detect(self, frame, track=True):
            return dets

    class FakeClassifier:
        def classify(self, frame, b):
            return Attribute("male", 0.9)

    frame = np.zeros((720, 1280, 3), np.uint8)
    pipe = PassengerPipeline(FakeDetector(), seats, attributes=AttributeService(FakeClassifier(), budget=3))
    hub = PassengerHub()
    state = pipe.process(frame)

    out = {
        "pipeline": _ms(lambda: pipe.process(frame), n),      # includes seat assignment and classifier scheduling
        "overlay": _ms(lambda: overlay.draw(frame, state, seats, True), n),
        "jpeg": _ms(lambda: hub.publish_frame(frame), n),
        "payload": _ms(lambda: build_payload(state, fps=10, latency_ms=5, classifier_on=True), n),
    }
    out["total"] = sum(out.values())
    return out


if __name__ == "__main__":
    result = measure()
    for name, ms in result.items():
        print(f"{name:<10}{ms:8.2f} ms")
    print("(frame budget at 10 FPS = 100 ms)")
