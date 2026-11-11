"""Choose the child / confidence thresholds for a face-attribute model, on data that is not the benchmark.

    python -m passenger.tune_classifier --version v1

Collects the face crop of every detected person that matches a labelled one in the open-data **train** and **val**
splits (never test, never FairFace validation), then sweeps ``teen_weight`` (how much of the ambiguous 10-19 age group
counts as child), ``child_threshold`` and ``min_conf`` and keeps the combination with the best male/female/child
macro-F1. The choice is stored in the registry entry, which also clears its old scores: they were for different thresholds.
Few people are involved (a few hundred faces), so prefer coarse grids and do not over-read small differences.
"""
from __future__ import annotations

import argparse
import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import metrics
from .data_manager import DataManager
from .evaluate import _read_boxes
from .registry import FACE_KEY_METRICS, Registry

TEEN_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
CHILD_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MIN_CONFS = (0.5, 0.6, 0.7, 0.8)


def collect_faces(dm: DataManager, splits: Sequence[str], clf, detector) -> List[Tuple[str, np.ndarray, str]]:
    """(split, face crop, true label) for every matched person whose face the pipeline can find."""
    out = []
    for split in splits:
        for fr in dm.iter_frames(split):
            img = cv2.imread(str(fr.image))
            if img is None:
                continue
            dets = detector.detect(img, track=False)
            gtl = _read_boxes(fr.label)
            for pi, gi in metrics.match_pairs([(d.box, d.conf) for d in dets], [b for b, _ in gtl]):
                face = clf.face_crop(img, dets[pi].box)
                if face is not None:
                    out.append((split, face, dm.classes[gtl[gi][1]]))
    return out


def decide(p_male: float, p_female: float, p_child: float, child_threshold: float, min_conf: float) -> str:
    if p_child >= child_threshold:
        return "child"
    label, conf = ("male", p_male) if p_male >= p_female else ("female", p_female)
    return label if conf >= min_conf else "unknown"


def sweep(pg: np.ndarray, pa: Optional[np.ndarray], p_child_fixed: Optional[np.ndarray], truth: Sequence[str],
          splits: Sequence[str]) -> List[dict]:
    """Every (teen_weight, child_threshold, min_conf) combination, scored on all faces and on each split.
    ``pa`` is the age-group distribution (trained model) or ``p_child_fixed`` a ready P(child) (pretrained nets)."""
    from .face_model import CHILD_GROUPS, TEEN_GROUP
    results = []
    tws = TEEN_WEIGHTS if pa is not None else (0.0,)
    for tw, ct, mc in itertools.product(tws, CHILD_THRESHOLDS, MIN_CONFS):
        pc = (pa[:, list(CHILD_GROUPS)].sum(1) + tw * pa[:, TEEN_GROUP]).clip(max=1.0) if pa is not None else p_child_fixed
        pred = [decide(pg[i, 0], pg[i, 1], pc[i], ct, mc) for i in range(len(truth))]
        row = {"teen_weight": tw, "child_threshold": ct, "min_conf": mc,
               "macro_f1": metrics.classification_metrics(pred, truth)["macro_f1"]}
        for s in set(splits):
            idx = [i for i, sp in enumerate(splits) if sp == s]
            row[s] = metrics.classification_metrics([pred[i] for i in idx], [truth[i] for i in idx])["macro_f1"]
        results.append(row)
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    from eagleeye.config import load_config
    from .classifier import make_face_finder, PassengerClassifier
    from .detector import PersonDetector
    from .face_eval import load_model

    ap = argparse.ArgumentParser(description="Tune the child / confidence thresholds on open-data train+val")
    ap.add_argument("--version", required=True)
    ap.add_argument("--data", default="passenger_data")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    args = ap.parse_args(argv)

    cfg = load_config("config.yaml")
    c, det = cfg.passenger.classifier, cfg.passenger.detector
    reg = Registry("models/passenger/face/registry.json", FACE_KEY_METRICS)
    model = load_model(reg.models[args.version])
    clf = PassengerClassifier(model, make_face_finder(c), min_person_px=c.min_person_px, min_face_px=c.min_face_px,
                              face_margin=c.get("face_margin", 0.4))
    dm = DataManager(args.data)
    faces = collect_faces(dm, args.splits, clf, PersonDetector(det.model, det.conf, det.imgsz, det.classes))
    if len(faces) < 30:
        raise SystemExit(f"only {len(faces)} faces found: too few to tune on")
    splits, imgs, truth = zip(*faces)

    if hasattr(model, "raw_probs"):
        pg, pa = model.raw_probs(list(imgs))
        results = sweep(pg, pa, None, truth, splits)
    else:
        probs = np.array([model.predict(f) for f in imgs])
        results = sweep(probs[:, :2], None, probs[:, 2], truth, splits)
    results.sort(key=lambda r: -r["macro_f1"])
    print(f"[tune] {len(faces)} faces from splits {sorted(set(splits))}; best of {len(results)} combinations:")
    for r in results[:5]:
        print("   " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()))
    best = results[0]
    decision = {"teen_weight": best["teen_weight"], "child_threshold": best["child_threshold"], "min_conf": best["min_conf"]}
    reg.set_decision(args.version, decision, tuned_on="+".join(sorted(set(splits))))
    print(f"[tune] stored for {args.version}: {decision}")


if __name__ == "__main__":
    main()
