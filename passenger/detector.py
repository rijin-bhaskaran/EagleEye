"""Pretrained YOLO person detector with optional ByteTrack IDs.

Only the COCO ``person`` class is used; no retraining is needed for detection.
Boxes are returned normalised to 0-1 so everything downstream is resolution
independent.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .seats import Detection

_PERSON = 0

# Ultralytics' stock bytetrack.yaml. `passenger.detector.tracker` in config.yaml overrides single entries of it.
_BYTETRACK = {"tracker_type": "bytetrack", "track_high_thresh": 0.25, "track_low_thresh": 0.1, "new_track_thresh": 0.25,
              "track_buffer": 30, "match_thresh": 0.8, "fuse_score": True}


def tracker_yaml(overrides: Dict[str, object]) -> str:
    """Path of a ByteTrack config made of the stock one with ``overrides`` applied (ultralytics wants a file).

    The file name depends only on the settings, so the same config always reuses the same file. A misspelt or
    non-numeric setting is an error, not a silently ignored line."""
    known = set(_BYTETRACK) - {"tracker_type"}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(f"passenger.detector.tracker: unknown setting {unknown[0]!r}; known: {sorted(known)}")
    for key, value in overrides.items():
        ok = isinstance(value, bool) if key == "fuse_score" else isinstance(value, (int, float)) and not isinstance(value, bool)
        if not ok:
            raise ValueError(f"passenger.detector.tracker.{key} must be {'true/false' if key == 'fuse_score' else 'a number'}, got {value!r}")
    merged = {**_BYTETRACK, **overrides}
    name = "eagleeye_bytetrack_" + hashlib.sha1(json.dumps(merged, sort_keys=True).encode()).hexdigest()[:10] + ".yaml"
    path = Path(tempfile.gettempdir()) / name
    lines = [f"{k}: {str(v).lower() if isinstance(v, bool) else v}" for k, v in merged.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def openvino_model(weights: str, imgsz: int) -> str:
    """Path of the OpenVINO export of ``weights``, exporting it first if it is not there yet.

    OpenVINO runs the same network about 3x faster than PyTorch on an Intel CPU
    (measured: 31 vs 11 FPS detector-only), which is what lets the passenger feed
    hold 10-12 FPS while the driver monitor shares the processor.
    """
    from ultralytics import YOLO

    out = Path(weights).with_suffix("")
    out = out.with_name(out.name + "_openvino_model")
    if not out.exists():
        print(f"[passenger] exporting {weights} to OpenVINO (one-off, a few seconds)", flush=True)
        YOLO(weights).export(format="openvino", imgsz=imgsz, verbose=False)
    return str(out)


class PersonDetector:
    tracker = "bytetrack.yaml"            # the stock config; a class default so a detector built without __init__ still works
    iou = None                            # None = the model's own overlap limit for duplicate boxes (0.7)

    def __init__(self, model: str = "yolov8n.pt", conf: float = 0.30, imgsz: int = 960,
                 classes: Optional[Sequence[int]] = (_PERSON,), backend: str = "torch",
                 tracker: Optional[Dict[str, object]] = None, iou: Optional[float] = None) -> None:
        """``classes`` are the model's class ids that mean "a person". The pretrained COCO model uses ``(0,)``;
        a model fine-tuned on male / female / child has only person classes, so pass ``None`` (all classes)."""
        from ultralytics import YOLO          # lazy: tests and tooling don't need the weights

        if backend == "openvino" and model.endswith(".pt"):
            try:
                model = openvino_model(model, imgsz)
            except Exception as exc:                   # fall back rather than lose passenger monitoring
                print(f"[passenger] OpenVINO unavailable ({type(exc).__name__}: {exc}); using PyTorch", flush=True)
        elif backend not in ("torch", "openvino"):
            raise ValueError(f"unknown detector backend {backend!r} (torch or openvino)")
        self._model = YOLO(model, task="detect")
        self.conf, self.imgsz = conf, imgsz
        self.classes = None if classes is None else list(classes)
        if tracker:
            self.tracker = tracker_yaml(dict(tracker))
        if iou is not None:
            if isinstance(iou, bool) or not isinstance(iou, (int, float)) or not 0.0 < iou <= 1.0:
                raise ValueError(f"passenger.detector.iou must be a number above 0 and at most 1, got {iou!r}")
            self.iou = float(iou)

    def detect(self, frame_bgr: np.ndarray, track: bool = True) -> List[Detection]:
        """Detect people. ``track=True`` keeps IDs stable across calls (live video);
        use ``track=False`` for unrelated stills such as an evaluation set."""
        kwargs = dict(conf=self.conf, imgsz=self.imgsz, verbose=False)
        if self.classes is not None:
            kwargs["classes"] = self.classes
        if self.iou is not None:
            kwargs["iou"] = self.iou
        if track:
            res = self._model.track(frame_bgr, persist=True, tracker=self.tracker, **kwargs)[0]
        else:
            res = self._model.predict(frame_bgr, **kwargs)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxyn.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else [None] * len(xyxy)
        return [Detection(tuple(map(float, b)), float(c), None if i is None else int(i))
                for b, c, i in zip(xyxy, confs, ids)]
