"""Train the car-cabin model and score it on the held-out test split.

    python -m passenger.datasets.car_cabin add ...                  # once per source dataset (see that module)
    python -m passenger.train_car --epochs 60 --imgsz 640 --device 0 --name v1
    python -m passenger.train_car --score-only models/car_cabin/v1/run/weights/best.pt --name v1

Reported on val and test (the test split is scored once, after training):
* per-class precision / recall / AP50 / AP50-95 (``model.val``) and the confidence that maximises mean F1;
* the cabin checks that the product is judged on, per image at that operating point:
  - child agreement: does "a child is in the picture" match the labels (precision / recall)?
  - child-seat count error: mean absolute error of installed child seats, and share exactly right;
  - belt recall: of the labelled seatbelts, how many were found (``seatbelt`` AP is the per-box version).
Everything is written to ``models/car_cabin/<name>/metrics.json``; copy the weights to
``models/car_cabin/champion.pt`` (the default of ``passenger.car.weights``) to go live.
A class with no test boxes is reported as ``null`` and listed under ``untested_classes``: it was not measured.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .car_logic import CAR_CLASSES


def _read_gt(label: Path) -> List[int]:
    return [int(l.split()[0]) for l in label.read_text().splitlines() if l.strip()]


def cabin_checks(model, data: Path, split: str, conf: float, imgsz: int) -> dict:
    """Frame-level child / child-seat / belt accuracy at confidence ``conf``."""
    ci, cs, sb = (CAR_CLASSES.index(n) for n in ("child", "child_seat", "seatbelt"))
    images = sorted(p for p in (data / "images" / split).glob("*.*"))
    tp = fp = fn = 0
    seat_err: List[int] = []
    belts_gt = belts_found = 0
    for img in images:
        lab = data / "labels" / split / f"{img.stem}.txt"
        gt = _read_gt(lab) if lab.is_file() else []
        pred = model.predict(str(img), conf=conf, imgsz=imgsz, verbose=False)[0].boxes.cls.cpu().numpy().astype(int).tolist()
        g_child, p_child = ci in gt, ci in pred
        tp += g_child and p_child
        fp += (not g_child) and p_child
        fn += g_child and (not p_child)
        if cs in gt or cs in pred:
            seat_err.append(abs(pred.count(cs) - gt.count(cs)))
        belts_gt += gt.count(sb)
        belts_found += min(pred.count(sb), gt.count(sb))
    div = lambda a, b: round(a / b, 4) if b else None      # noqa: E731
    return {"images": len(images),
            "child_present": {"precision": div(tp, tp + fp), "recall": div(tp, tp + fn), "positives": int(tp + fn)},
            "child_seat_count": {"mean_abs_error": round(float(np.mean(seat_err)), 3) if seat_err else None,
                                 "exact_share": round(float(np.mean(np.array(seat_err) == 0)), 3) if seat_err else None,
                                 "images_with_child_seat": len(seat_err)},
            "belt_recall": div(belts_found, belts_gt), "belts_labelled": belts_gt}


def best_f1(res) -> dict:
    """Confidence that maximises mean F1. ultralytics' curves have one row per class *present in the split*
    (``ap_class_index``), so a class with no boxes has no row: it is None here and left out of the mean."""
    f1 = np.asarray(res.box.f1_curve)
    rows = {int(c): r for r, c in enumerate(res.box.ap_class_index)}
    x = np.linspace(0, 1, f1.shape[1])
    i = int(f1.mean(0).argmax())
    return {"conf": round(float(x[i]), 3), "f1_mean": round(float(f1.mean(0)[i]), 4),
            "f1": {n: (round(float(f1[rows[k]][i]), 4) if k in rows else None) for k, n in enumerate(CAR_CLASSES)}}


def score(weights: Path, data: Path, split: str, imgsz: int, device: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    res = model.val(data=str(data / "data.yaml"), split=split, imgsz=imgsz, device=device, batch=8, plots=False,
                    verbose=False, conf=0.001)
    present = set(int(c) for c in res.box.ap_class_index)         # classes that have boxes in this split
    per_class: Dict[str, Optional[dict]] = {}
    for k, n in enumerate(CAR_CLASSES):
        if k not in present:
            per_class[n] = None
            continue
        j = list(res.box.ap_class_index).index(k)
        per_class[n] = {"precision": round(float(res.box.p[j]), 4), "recall": round(float(res.box.r[j]), 4),
                        "ap50": round(float(res.box.ap50[j]), 4), "ap50_95": round(float(res.box.ap[j]), 4)}
    op = best_f1(res)
    return {"split": split, "map50": round(float(res.box.map50), 4), "map50_95": round(float(res.box.map), 4),
            "per_class": per_class, "untested_classes": [n for n, v in per_class.items() if v is None],
            "operating_point": op, "cabin": cabin_checks(model, data, split, op["conf"], imgsz),
            "speed_ms_per_image": {k: round(float(v), 1) for k, v in res.speed.items()}}


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description="Train and score the car-cabin model")
    ap.add_argument("--data", type=Path, default=Path("car_data"))
    ap.add_argument("--base", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--fraction", type=float, default=1.0, help="share of the training set to use (timing runs)")
    ap.add_argument("--name", default="v1")
    ap.add_argument("--score-only", type=Path, default=None, metavar="WEIGHTS")
    ap.add_argument("--out", type=Path, default=Path("models/car_cabin"))
    a = ap.parse_args(argv)

    for s in ("train", "val", "test"):
        if not any((a.data / "images" / s).glob("*.*")):
            raise SystemExit(f"[train_car] {a.data / 'images' / s} has no images: build the set with passenger.datasets.car_cabin first")
    run_dir = (a.out / a.name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    params = {k: str(v) for k, v in vars(a).items()}
    if a.score_only is not None:
        out = {"params": params, "note": "scored without training", "val": score(a.score_only, a.data, "val", a.imgsz, a.device),
               "test": score(a.score_only, a.data, "test", a.imgsz, a.device)}
        best = a.score_only
    else:
        from ultralytics import YOLO

        t0 = time.time()
        YOLO(a.base).train(data=str((a.data / "data.yaml").resolve()), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                           device=a.device, project=str(run_dir), name="run", exist_ok=True, plots=True, workers=4,
                           patience=a.patience, fraction=a.fraction, seed=0, verbose=False)
        best = run_dir / "run" / "weights" / "best.pt"
        out = {"params": params, "train_minutes": round((time.time() - t0) / 60, 1),
               "val": score(best, a.data, "val", a.imgsz, a.device), "test": score(best, a.data, "test", a.imgsz, a.device)}
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["test"], indent=2))
    if out["test"]["untested_classes"]:
        print(f"[train_car] NOT MEASURED (no test boxes): {out['test']['untested_classes']}")
    print(f"[train_car] weights {best}\n[train_car] metrics {run_dir / 'metrics.json'}\n"
          f"[train_car] to go live: copy the weights to models/car_cabin/champion.pt and set passenger.vehicle_type: car")


if __name__ == "__main__":
    main()
