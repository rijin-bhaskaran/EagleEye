"""Evaluation metrics for the passenger system.

Everything here is pure functions over plain data (boxes, sets of seat ids), so
it is testable without a camera or model. Boxes are ``(x1, y1, x2, y2)``.

Detection metrics are class-agnostic ("is there a person here"): the pretrained
detector only knows ``person``; Male/Female/Child are scored separately.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

Box = Tuple[float, float, float, float]
#: One frame: predictions as (box, confidence) and ground-truth boxes.
Frame = Tuple[Sequence[Tuple[Box, float]], Sequence[Box]]


def iou(a: Box, b: Box) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _prf(tp: float, fp: float, fn: float) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


# -- detection -----------------------------------------------------------
def match_pairs(preds: Sequence[Tuple[Box, float]], gts: Sequence[Box], thr: float = 0.5) -> List[Tuple[int, int]]:
    """Greedy one-to-one matching in confidence order; returns (prediction index, ground-truth index) pairs."""
    used: Set[int] = set()
    pairs: List[Tuple[int, int]] = []
    for i in sorted(range(len(preds)), key=lambda i: -preds[i][1]):
        best, best_j = thr, -1
        for j, g in enumerate(gts):
            if j in used:
                continue
            v = iou(preds[i][0], g)
            if v >= best:
                best, best_j = v, j
        if best_j >= 0:
            used.add(best_j)
            pairs.append((i, best_j))
    return pairs


def _match(preds: Sequence[Tuple[Box, float]], gts: Sequence[Box], thr: float) -> List[bool]:
    """Per prediction (input order): is it a true positive?"""
    matched = {i for i, _ in match_pairs(preds, gts, thr)}
    return [i in matched for i in range(len(preds))]


def detection_prf(frames: Iterable[Frame], iou_thr: float = 0.5) -> Dict[str, float]:
    tp = fp = fn = 0
    for preds, gts in frames:
        hit = _match(preds, gts, iou_thr)
        tp += sum(hit)
        fp += len(hit) - sum(hit)
        fn += len(gts) - sum(hit)
    return {**_prf(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn}


def average_precision(frames: Iterable[Frame], iou_thr: float = 0.5) -> float:
    """AP at one IoU threshold (all-point interpolation, as in PASCAL VOC / COCO).

    mAP@0.5:0.95 is the mean of this over thresholds 0.50, 0.55 ... 0.95.
    """
    scored: List[Tuple[float, bool]] = []
    n_gt = 0
    for preds, gts in frames:
        n_gt += len(gts)
        scored += [(preds[i][1], h) for i, h in enumerate(_match(preds, gts, iou_thr))]
    if n_gt == 0 or not scored:
        return 0.0
    scored.sort(key=lambda t: -t[0])
    tp = np.cumsum([h for _, h in scored])
    fp = np.cumsum([not h for _, h in scored])
    recall = tp / n_gt
    precision = tp / (tp + fp)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def map_50_95(frames: Sequence[Frame]) -> float:
    frames = list(frames)
    return float(np.mean([average_precision(frames, t) for t in np.arange(0.5, 0.96, 0.05)]))


# -- count ---------------------------------------------------------------
def count_metrics(pred: Sequence[int], gt: Sequence[int]) -> Dict[str, float]:
    if len(pred) != len(gt):
        raise ValueError("pred and gt must be the same length")
    if not gt:
        return {"mae": 0.0, "exact": 0.0, "bias": 0.0}
    err = np.asarray(pred, float) - np.asarray(gt, float)
    return {"mae": float(np.abs(err).mean()), "exact": float((err == 0).mean()), "bias": float(err.mean())}


# -- seat occupancy ------------------------------------------------------
def occupancy_metrics(pred: Sequence[Set[str]], gt: Sequence[Set[str]], seat_ids: Sequence[str]) -> Dict[str, object]:
    """Seat-level scores over all (frame, seat) pairs, plus per-seat F1 to expose weak seats (e.g. the back row)."""
    if len(pred) != len(gt):
        raise ValueError("pred and gt must be the same length")
    tp = {s: 0 for s in seat_ids}
    fp = dict(tp)
    fn = dict(tp)
    exact = 0
    for p, g in zip(pred, gt):
        exact += p == g
        for s in seat_ids:
            if s in p and s in g:
                tp[s] += 1
            elif s in p:
                fp[s] += 1
            elif s in g:
                fn[s] += 1
    per_seat = {s: _prf(tp[s], fp[s], fn[s])["f1"] for s in seat_ids}
    # A seat that is never occupied in the ground truth and never predicted has no F1; exclude it from the macro mean.
    active = [per_seat[s] for s in seat_ids if tp[s] + fp[s] + fn[s] > 0]
    return {
        **_prf(sum(tp.values()), sum(fp.values()), sum(fn.values())),
        "macro_f1": float(np.mean(active)) if active else 0.0,
        "frame_exact": exact / len(gt) if gt else 0.0,
        "per_seat_f1": per_seat,
    }


# -- classification ------------------------------------------------------
def classification_metrics(pred: Sequence[str], gt: Sequence[str],
                           classes: Sequence[str] = ("male", "female", "child"),
                           unknown: str = "unknown") -> Dict[str, object]:
    """Scores for a classifier that may abstain (``unknown``).

    An abstention counts as a miss for recall and as no prediction for precision,
    so a model cannot look good by refusing hard cases. ``coverage`` and
    ``accuracy_answered`` expose that trade-off separately.
    """
    if len(pred) != len(gt):
        raise ValueError("pred and gt must be the same length")
    cols = [*classes, unknown]
    confusion = {g: {c: 0 for c in cols} for g in classes}
    for p_, g in zip(pred, gt):
        if g in confusion:
            confusion[g][p_ if p_ in cols else unknown] += 1
    n = sum(sum(row.values()) for row in confusion.values())
    per_class = {}
    for c in classes:
        tp = confusion[c][c]
        fp = sum(confusion[g][c] for g in classes if g != c)
        fn = sum(confusion[c].values()) - tp
        per_class[c] = {**_prf(tp, fp, fn), "support": tp + fn}
    answered = sum(confusion[g][c] for g in classes for c in classes)
    correct = sum(confusion[c][c] for c in classes)
    present = [per_class[c]["f1"] for c in classes if per_class[c]["support"] > 0]
    return {
        "confusion": confusion,
        "per_class": per_class,
        "macro_f1": float(np.mean(present)) if present else 0.0,
        "coverage": answered / n if n else 0.0,
        "accuracy_answered": correct / answered if answered else 0.0,
        "accuracy_overall": correct / n if n else 0.0,
    }


# -- speed ---------------------------------------------------------------
def latency_summary(seconds: Sequence[float]) -> Dict[str, float]:
    if not len(seconds):
        return {"mean_ms": 0.0, "p95_ms": 0.0, "max_fps": 0.0}
    a = np.asarray(seconds, float)
    return {"mean_ms": float(a.mean() * 1000), "p95_ms": float(np.percentile(a, 95) * 1000),
            "max_fps": float(1.0 / a.mean()) if a.mean() > 0 else 0.0}
