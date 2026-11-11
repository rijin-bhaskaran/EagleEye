"""The passenger frame loop: read -> process -> annotate -> publish, paced to a target FPS.

Pacing matters on an edge device. The camera delivers 30 FPS but the target is
10-12, so the loop sleeps off the spare time instead of burning CPU the driver
monitor needs. If inference is slower than the target it simply runs as fast as
it can and reports the real rate; it never builds a backlog.

A video file is read at its own pace, so ``fps / src_fps`` of the frames are
skipped to keep playback in real time instead of slow motion.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import cv2

from . import overlay
from .hub import PassengerHub
from .pipeline import PassengerPipeline


class PassengerRunner:
    def __init__(self, stream, pipeline: PassengerPipeline, hub: PassengerHub, *, fps: float = 10.0,
                 draw_seats: bool = True, loop: bool = False, max_consecutive_errors: int = 30) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.stream, self.pipeline, self.hub = stream, pipeline, hub
        self.fps, self.draw_seats, self.loop = fps, draw_seats, loop
        self._skip = max(round(stream.src_fps / fps) - 1, 0) if stream.is_file else 0
        self._done: deque = deque(maxlen=30)           # completion times of recent frames
        self.frames = 0
        self.errors = 0                                # frames that failed, in total
        self._max_errors, self._streak = max_consecutive_errors, 0

    @property
    def measured_fps(self) -> float:
        if len(self._done) < 2:
            return 0.0
        return (len(self._done) - 1) / max(self._done[-1] - self._done[0], 1e-6)

    def step(self) -> bool:
        """Process one frame. Returns False when a (non-looping) video file has ended."""
        for _ in range(self._skip):
            self.stream.cap.grab()
        ok, frame, _ = self.stream.read()
        if not ok:
            if not self.stream.is_file:
                return True                            # camera hiccup: try again
            if self.loop:
                self.stream.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.stream.frame_index = 0
                return True
            return False

        try:
            t0 = time.perf_counter()
            state = self.pipeline.process(frame)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            overlay.draw(frame, state, self.pipeline.seatmap, self.draw_seats)
            self.hub.publish_frame(frame)
            self._done.append(time.perf_counter())
            self.frames += 1
            extra = {"classifier_label": self.pipeline.classifier_label} if hasattr(self.pipeline, "classifier_label") else {}
            self.hub.publish(state, fps=self.measured_fps, latency_ms=latency_ms,
                             classifier_on=self.pipeline.attributes is not None, **extra)
        except Exception as exc:                       # one bad frame must not end passenger monitoring
            self.errors += 1
            self._streak += 1
            if self._streak == 1 or self._streak % 10 == 0:
                print(f"[passenger] frame error ({self.errors} so far): {type(exc).__name__}: {exc}", flush=True)
            if self._streak >= self._max_errors:       # failing every frame is a fault, not a hiccup
                raise
        else:
            self._streak = 0
        return True

    def run(self, stop: threading.Event) -> None:
        period = 1.0 / self.fps
        next_at = time.perf_counter()
        while not stop.is_set():
            if not self.step():
                break
            next_at += period
            spare = next_at - time.perf_counter()
            if spare > 0:
                stop.wait(spare)                       # interruptible sleep
            else:
                next_at = time.perf_counter()          # fell behind: drop the debt, don't sprint
