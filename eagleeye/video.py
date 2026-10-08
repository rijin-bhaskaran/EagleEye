"""Video input.

For a webcam we read in a background thread and always hand out the newest
frame. If inference is slower than the camera, a queued reader would build
up latency until the alerts refer to something that happened seconds ago -
for a safety system, dropping stale frames is strictly correct.

For a video *file* we read synchronously so that no frames are skipped and
offline feature extraction is deterministic.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class VideoStream:
    def __init__(self, source, width=640, height=480, fps=30, flip=False) -> None:
        self.is_file = isinstance(source, str) and Path(source).exists()
        self.flip = flip and not self.is_file

        self.cap = self._open(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source!r}")

        if not self.is_file:
            # MJPG lets most webcams deliver 30-60 FPS; raw YUY2 is often
            # capped at ~15-30 FPS over USB.
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # A webcam left in manual exposure (by another app or an earlier
            # session) gives a near-black image and, with long exposures, a
            # lower frame rate. 0.75 = "auto" on DirectShow, 3 on V4L2.
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if sys.platform.startswith("linux") else 0.75)

        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
        self.frame_index = 0
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._read_seq = -1
        self._lock = threading.Lock()
        # Wakes readers the instant a frame lands. Polling with time.sleep(1ms)
        # is really ~15 ms on Windows, which is most of the added latency.
        self._new_frame = threading.Condition(self._lock)
        self.capture_fps = 0.0
        self._stopped = False
        self._thread: Optional[threading.Thread] = None

        if not self.is_file:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            for _ in range(50):          # wait for the first frame
                if self._frame is not None:
                    break
                time.sleep(0.02)

    def _open(self, source) -> "cv2.VideoCapture":
        if self.is_file:
            return cv2.VideoCapture(source)
        index = int(source)
        if sys.platform == "win32":
            # DirectShow gives lower-latency, steadier 30 FPS MJPG capture than
            # the default Media Foundation backend; fall back if it won't open.
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
            cap.release()
        return cv2.VideoCapture(index)

    def _loop(self) -> None:
        stamp, count = time.perf_counter(), 0
        while not self._stopped:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            if self.flip:
                frame = cv2.flip(frame, 1)
            with self._new_frame:
                self._frame = frame
                self._seq += 1
                self._new_frame.notify_all()

            count += 1
            now = time.perf_counter()
            if now - stamp >= 1.0:
                self.capture_fps = count / (now - stamp)
                stamp, count = now, 0

    def wait_latest(self, after_seq: int, timeout: float = 0.1):
        """Block until a frame newer than ``after_seq`` exists.

        Returns ``(seq, frame_bgr_copy)`` or ``(after_seq, None)`` on timeout.
        Does not disturb ``read()``'s own bookkeeping, so a render thread can
        follow the camera at full rate while inference runs slower.
        """
        with self._new_frame:
            self._new_frame.wait_for(lambda: self._seq != after_seq or self._stopped, timeout)
            if self._frame is None or self._seq == after_seq:
                return after_seq, None
            return self._seq, self._frame.copy()

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Return ``(ok, frame_bgr, timestamp_seconds)``."""
        if self.is_file:
            ok, frame = self.cap.read()
            if not ok:
                return False, None, 0.0
            ts = self.frame_index / max(self.src_fps, 1.0)
            self.frame_index += 1
            return True, frame, ts

        # Wait (briefly) for a *new* frame so we never re-process a duplicate.
        with self._new_frame:
            self._new_frame.wait_for(lambda: self._seq != self._read_seq or self._stopped, 0.1)
            frame = None if self._frame is None else self._frame.copy()
            self._read_seq = self._seq
        if frame is None:
            return False, None, time.time()
        self.frame_index += 1
        return True, frame, time.time()

    def release(self) -> None:
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()
