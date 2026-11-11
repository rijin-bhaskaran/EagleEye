"""Retrain the passenger detector whenever new data arrives.

    # 1. new data arrives (own footage + CVAT labels, or an open dataset)
    python -m passenger.datasets.open_images --split validation --max-images 400     # example source
    # 2. train from the current champion on everything we have, then score on the frozen benchmark
    python -m passenger.train --dataset passengers_v2 --epochs 30 --device 0
    # 3. promote only if nothing regressed
    python -m passenger.registry promote v2

What makes this safe to repeat:
* it trains on the union of all data (nothing already learned is dropped), starting from the champion;
* the **test split is never trained on** (ultralytics only reads train/val), and each image's split is stable,
  so the benchmark is the same next time (the registry checks its fingerprint before comparing scores);
* every model records its lineage (base version, dataset fingerprints, parameters) in the registry;
* the raw data is never touched: the training set is rebuilt from raw/ + labels on every run.

Train on a machine with a GPU, not on the Raspberry Pi; copy ``models/passenger/champion.pt`` (or its ONNX
export) to the device afterwards.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Callable, Optional

from .data_manager import SPLITS, DataManager
from .registry import Registry, summarize

Trainer = Callable[..., Path]      # (base, data_yaml, epochs, imgsz, batch, device, out_dir) -> best weights


def ultralytics_trainer(base: str, data_yaml: Path, epochs: int, imgsz: int, batch: int, device: str,
                        out_dir: Path) -> Path:
    from ultralytics import YOLO

    YOLO(base).train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, device=device,
                     project=str(out_dir.resolve()), name="run", exist_ok=True, plots=False, workers=2,
                     verbose=False)
    return out_dir / "run" / "weights" / "best.pt"


def train(dm: DataManager, registry: Registry, dataset: str, *, base: str = "champion", epochs: int = 30,
          imgsz: int = 640, batch: int = 8, device: str = "cpu", trainer: Trainer = ultralytics_trainer,
          evaluator: Optional[Callable[[Path, Optional[list]], dict]] = None, default_base: str = "yolov8n.pt") -> str:
    """Export the dataset, train a candidate from ``base``, score it on the benchmark and register it.
    Returns the new version. Promotion is a separate, explicit step (``passenger.registry promote``)."""
    out = dm.export_yolo(dataset)
    counts = {s: len(list((out / "images" / s).glob("*.*"))) if (out / "images" / s).exists() else 0 for s in SPLITS}
    if counts["train"] == 0:
        raise ValueError("no labelled training images: register data and labels first")
    if counts["test"] == 0:
        raise ValueError("no labelled test images: there would be no benchmark to judge the new model on")
    if counts["val"] == 0:
        raise ValueError("no labelled validation images: training needs them to pick the best epoch")

    base_version: Optional[str] = None
    if base == "champion":
        champ = registry.champion_weights()
        base_weights = str(champ) if champ else default_base
        base_version = registry.champion
    else:
        base_weights = base

    version = registry.next_version()
    run_dir = registry.dir / version
    run_dir.mkdir(parents=True, exist_ok=True)
    best = trainer(base_weights, out / "data.yaml", epochs, imgsz, batch, device, run_dir)
    kept = run_dir / "best.pt"
    shutil.copy2(best, kept)

    registry.add(kept, base=base_version, classes=None, version=version,
                 dataset={"name": dataset, "counts": counts, **{f"fingerprint_{s}": dm.fingerprint(s) for s in SPLITS}},
                 params={"base_weights": base_weights, "epochs": epochs, "imgsz": imgsz, "batch": batch, "device": device})
    if evaluator is not None:
        registry.record_metrics(version, dm.fingerprint("test"), summarize(evaluator(kept, None)))
    return version


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description="Retrain the passenger detector on all available data")
    ap.add_argument("--data", default="passenger_data")
    ap.add_argument("--dataset", required=True, help="name for the exported training set, e.g. passengers_v2")
    ap.add_argument("--base", default="champion", help='"champion" (default), or a weights path such as yolov8n.pt')
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu", help='"cpu" or a GPU index such as "0"')
    args = ap.parse_args(argv)

    from eagleeye.config import load_config
    from .detector import PersonDetector
    from .evaluate import evaluate
    from .seats import SeatMap

    cfg = load_config("config.yaml")
    dm = DataManager(args.data)
    seatmap = SeatMap.from_config(cfg.passenger.seats)

    def evaluator(weights: Path, classes):
        return evaluate(dm, PersonDetector(str(weights), cfg.passenger.detector.conf, args.imgsz, classes), seatmap, "test")

    reg = Registry()
    v = train(dm, reg, args.dataset, base=args.base, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
              device=args.device, evaluator=evaluator)
    ok, why = reg.compare(v)
    print(f"[train] {v} registered (base {reg.models[v]['base']}). Safe to promote: {ok} ({'; '.join(why)})")
    print(f"[train] to go live: python -m passenger.registry promote {v}")


if __name__ == "__main__":
    main()
