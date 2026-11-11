"""Train the seat-state detector (Occupied / Unoccupied) and score it on the held-out test clips.

    python -m passenger.seat_data "<dataset dir>" --out seat_data          # once: clip-level split
    python -m passenger.train_seats --epochs 40 --imgsz 960 --device cpu    # train + score

The test split is read only here, once, after training. Reported:
* per-class precision / recall / AP50 / AP50-95 from ultralytics (``model.val(split="test")``);
* the operating point: confidence that maximises F1, because 0.25 is just a default;
* per-frame counting error: how many seats per image are mis-counted as Occupied / Unoccupied;
* the same numbers per recorder stream, so a weak camera is visible instead of averaged away.
Everything is written to ``models/seat_state/<run>/metrics.json``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .seat_data import NAMES


def _read_gt(label: Path) -> List[int]:
    return [int(l.split()[0]) for l in label.read_text().splitlines() if l.strip()]


def count_errors(model, data: Path, split: str, conf: float, imgsz: int) -> dict:
    """Per-image error in the number of Occupied / Unoccupied seats, overall and per recorder."""
    images = sorted((data / "images" / split).glob("*.jpg"))
    err = {n: [] for n in NAMES}
    per_cam: Dict[str, Dict[str, list]] = {}
    for img in images:
        gt = _read_gt(data / "labels" / split / f"{img.stem}.txt")
        pred = model.predict(str(img), conf=conf, imgsz=imgsz, verbose=False)[0].boxes.cls.cpu().numpy().astype(int).tolist()
        cam = img.stem.split("_")[1]
        for k, n in enumerate(NAMES):
            e = abs(pred.count(k) - gt.count(k))
            err[n].append(e)
            per_cam.setdefault(cam, {m: [] for m in NAMES})[n].append(e)
    summ = lambda d: {n: {"mean_abs_error": round(float(np.mean(v)), 3), "exact_share": round(float(np.mean(np.array(v) == 0)), 3)}  # noqa: E731
                      for n, v in d.items()}
    return {"frames": len(images), "overall": summ(err), "per_recorder": {c: {"frames": len(v[NAMES[0]]), **summ(v)} for c, v in per_cam.items()}}


def best_f1(res) -> dict:
    """Confidence that maximises mean F1 over classes, from ultralytics' curves."""
    f1 = np.asarray(res.box.f1_curve)                       # (classes, 1000)
    x = np.linspace(0, 1, f1.shape[1])
    i = int(f1.mean(0).argmax())
    return {"conf": round(float(x[i]), 3), "f1_mean": round(float(f1.mean(0)[i]), 4),
            "f1": {n: round(float(f1[k][i]), 4) for k, n in enumerate(NAMES)}}


def score(weights: Path, data: Path, split: str, imgsz: int, device: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    res = model.val(data=str(data / "data.yaml"), split=split, imgsz=imgsz, device=device, batch=8, plots=False,
                    verbose=False, conf=0.001)
    op = best_f1(res)
    per_class = {n: {"precision": round(float(res.box.p[k]), 4), "recall": round(float(res.box.r[k]), 4),
                     "ap50": round(float(res.box.ap50[k]), 4), "ap50_95": round(float(res.box.ap[k]), 4)}
                 for k, n in enumerate(NAMES)}
    return {"split": split, "map50": round(float(res.box.map50), 4), "map50_95": round(float(res.box.map), 4),
            "per_class": per_class, "operating_point": op,
            "counting": count_errors(model, data, split, op["conf"], imgsz),
            "speed_ms_per_image": {k: round(float(v), 1) for k, v in res.speed.items()}}


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description="Train and score the seat-state detector")
    ap.add_argument("--data", type=Path, default=Path("seat_data"))
    ap.add_argument("--base", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--patience", type=int, default=4, help="stop after this many epochs without a better validation score")
    ap.add_argument("--target-map50", type=float, default=0.90,
                    help="stop as soon as validation mAP50 reaches this (0 = never); best.pt is kept either way")
    ap.add_argument("--target-precision", type=float, default=0.0,
                    help="stop as soon as validation precision (mean of both classes) reaches this (0 = never)")
    ap.add_argument("--fraction", type=float, default=1.0, help="share of the training set to use (timing runs)")
    ap.add_argument("--name", default="v1")
    ap.add_argument("--score-only", type=Path, default=None, metavar="WEIGHTS",
                    help="skip training: score these weights on val and test and write metrics.json")
    ap.add_argument("--out", type=Path, default=Path("models/seat_state"))
    a = ap.parse_args(argv)

    from ultralytics import YOLO

    run_dir = (a.out / a.name).resolve()
    if a.score_only is not None:
        out = {"params": {k: str(v) for k, v in vars(a).items()}, "note": "scored without finishing the training schedule",
               "val": score(a.score_only, a.data, "val", a.imgsz, a.device),
               "test": score(a.score_only, a.data, "test", a.imgsz, a.device)}
        (run_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out["test"], indent=2))
        print(f"[train_seats] metrics {run_dir / 'metrics.json'}")
        return
    t0 = time.time()
    model = YOLO(a.base)

    def stop_when_accurate(trainer) -> None:
        got = float(trainer.metrics.get("metrics/mAP50(B)", 0.0))
        if a.target_map50 > 0 and got >= a.target_map50:
            print(f"[train_seats] validation mAP50 {got:.3f} >= target {a.target_map50:.2f}: stopping at epoch {trainer.epoch + 1}", flush=True)
            trainer.stop = True
        prec = float(trainer.metrics.get("metrics/precision(B)", 0.0))
        if a.target_precision > 0 and prec >= a.target_precision:
            print(f"[train_seats] validation precision {prec:.3f} >= target {a.target_precision:.2f}: stopping at epoch {trainer.epoch + 1}", flush=True)
            trainer.stop = True

    model.add_callback("on_fit_epoch_end", stop_when_accurate)
    model.train(data=str((a.data / "data.yaml").resolve()), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                       device=a.device, project=str(run_dir), name="run", exist_ok=True, plots=True, workers=4,
                       patience=a.patience, fraction=a.fraction, seed=0, verbose=False)
    train_min = (time.time() - t0) / 60
    best = run_dir / "run" / "weights" / "best.pt"
    out = {"params": {k: str(v) for k, v in vars(a).items()}, "train_minutes": round(train_min, 1),
           "val": score(best, a.data, "val", a.imgsz, a.device),
           "test": score(best, a.data, "test", a.imgsz, a.device)}
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["test"], indent=2))
    print(f"[train_seats] weights {best}\n[train_seats] metrics {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
