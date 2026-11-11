"""Score the detector + seat logic against labelled frames.

    python -m passenger.evaluate --split test

Ground truth is the YOLO labels managed by ``DataManager``. Ground-truth seat
occupancy is derived from the labelled boxes with the *same* ``SeatMap`` the
model uses, so occupancy scores measure detection quality given the seat
layout, not a second set of hand-drawn labels. It also means the numbers are
only as meaningful as the seat polygons: calibrate them before trusting results.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import cv2

from . import metrics
from .data_manager import DataManager
from .classifier import PassengerClassifier
from .pipeline import Detector
from .seats import Detection, SeatMap


def _read_boxes(label_path) -> List[Tuple[metrics.Box, int]]:
    """(box, class id) per labelled person."""
    boxes = []
    for line in label_path.read_text().splitlines():
        if line.strip():
            cls, cx, cy, w, h = map(float, line.split())
            boxes.append(((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), int(cls)))
    return boxes


def evaluate(dm: DataManager, detector: Detector, seatmap: SeatMap, split: Optional[str] = "test",
             warmup: int = 2, classifier: Optional[PassengerClassifier] = None,
             source: Optional[str] = None) -> Dict[str, object]:
    """Score the model on every labelled frame of ``split`` (``source`` limits it to "videos" or one dataset id).

    Class ids in the labels index ``dm.classes``. The classifier (optional) is scored on every detected person
    that matches a labelled one, one frame at a time with no temporal smoothing. Seat occupancy is scored only on
    your own cabin footage: the seat map describes the cabin camera, so on external photos it would be noise."""
    bad = dm.validate_labels()
    if bad:
        raise ValueError(f"{len(bad)} invalid label lines, first: {bad[0]}")
    frames: List[metrics.Frame] = []
    cls_pred: List[str] = []
    cls_gt: List[str] = []
    pred_counts, gt_counts, pred_occ, gt_occ, times = [], [], [], [], []
    per_source: Counter = Counter()
    for fr in dm.iter_frames(split, source):
        image = cv2.imread(str(fr.image))
        if image is None:
            continue
        t0 = time.perf_counter()
        dets = detector.detect(image, track=False)
        if len(frames) >= warmup:                         # first frames include model warm-up
            times.append(time.perf_counter() - t0)

        gt_labelled = _read_boxes(fr.label)
        gt = [b for b, _ in gt_labelled]
        preds = [(d.box, d.conf) for d in dets]
        frames.append((preds, gt))
        per_source[fr.source] += 1
        pred_counts.append(len(dets))
        gt_counts.append(len(gt))
        if fr.source == "videos":
            pred_occ.append(set(seatmap.assign(dets)))
            gt_occ.append(set(seatmap.assign([Detection(b, 1.0) for b in gt])))
        if classifier is not None:
            for pi, gi in metrics.match_pairs(preds, gt):
                cls_pred.append(classifier.classify(image, dets[pi].box).label)
                cls_gt.append(dm.classes[gt_labelled[gi][1]])

    if not frames:
        raise ValueError(f"no labelled frames in split {split!r}")
    report = {
        "split": split,
        "frames": len(frames),
        "detection": {**metrics.detection_prf(frames), "ap50": metrics.average_precision(frames),
                      "map50_95": metrics.map_50_95(frames)},
        "count": metrics.count_metrics(pred_counts, gt_counts),
        "sources": dict(per_source),
        "occupancy": metrics.occupancy_metrics(pred_occ, gt_occ, seatmap.seat_ids) if pred_occ else None,
        "speed": metrics.latency_summary(times),
    }
    if classifier is not None:
        report["classification"] = metrics.classification_metrics(cls_pred, cls_gt, dm.classes)
    return report


def main(argv: Optional[List[str]] = None) -> None:
    from eagleeye.config import load_config
    from .detector import PersonDetector

    ap = argparse.ArgumentParser(description="Evaluate passenger detection on labelled frames")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data", default="passenger_data")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--classifier", action="store_true", help="also score Male/Female/Child")
    ap.add_argument("--source", default=None, help='only "videos" (your footage) or one external dataset id')
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    det = cfg.passenger.detector
    seatmap = SeatMap.from_config(cfg.passenger.seats, cfg.passenger.occupancy.min_overlap)
    report = evaluate(DataManager(args.data), PersonDetector(det.model, det.conf, det.imgsz, det.classes), seatmap, None if args.split == "all" else args.split,
                      source=args.source, classifier=PassengerClassifier.from_config(cfg) if args.classifier else None)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
