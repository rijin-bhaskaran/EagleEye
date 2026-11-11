"""Model registry: which weights exist, how they were made, how they scored, and which one is live.

    python -m passenger.registry list
    python -m passenger.registry baseline --weights yolov8n.pt      # score the starting model on the benchmark
    python -m passenger.registry promote v2                         # only succeeds if v2 does not regress

``models/passenger/registry.json`` records, for every version: the weights, its base version (lineage), the
dataset it was trained on (with fingerprints of each split), the training parameters and its benchmark scores.
A candidate becomes the **champion** (copied to ``models/passenger/champion.pt``, which ``config.yaml`` can
point at) only if it does not score worse than the current champion on the *same* benchmark. Because
retraining on new data can quietly make a model worse at something it used to do, this gate is what makes it
safe to retrain whenever data arrives.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: metric -> True if higher is better. A drop beyond the tolerance blocks promotion.
KEY_METRICS: Dict[str, bool] = {"recall": True, "precision": True, "ap50": True, "map50_95": True,
                                "count_mae": False, "cls_macro_f1": True, "occ_f1": True}
#: The face-attribute (male / female / child) model is judged on its own benchmark numbers and on the
#: whole pipeline's macro-F1 (the one that matters on the dashboard).
FACE_KEY_METRICS: Dict[str, bool] = {"gender_acc": True, "child_auc": True, "cls_macro_f1": True}


def summarize(report: dict) -> Dict[str, float]:
    """The numbers promotion is decided on, taken from a ``passenger.evaluate`` report."""
    d = report["detection"]
    out = {"recall": d["recall"], "precision": d["precision"], "ap50": d["ap50"], "map50_95": d["map50_95"],
           "count_mae": report["count"]["mae"]}
    if report.get("classification"):
        out["cls_macro_f1"] = report["classification"]["macro_f1"]
    if report.get("occupancy"):
        out["occ_f1"] = report["occupancy"]["f1"]
    return {k: float(v) for k, v in out.items()}


class Registry:
    def __init__(self, path: str | Path = "models/passenger/registry.json",
                 key_metrics: Optional[Dict[str, bool]] = None) -> None:
        self.path = Path(path)
        self.dir = self.path.parent
        self.key_metrics = KEY_METRICS if key_metrics is None else key_metrics
        self._d: dict = json.loads(self.path.read_text()) if self.path.exists() else {"models": {}, "champion": None}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._d, indent=2))
        tmp.replace(self.path)

    @property
    def models(self) -> Dict[str, dict]:
        return self._d["models"]

    @property
    def champion(self) -> Optional[str]:
        return self._d["champion"]

    def champion_weights(self) -> Optional[Path]:
        return Path(self.models[self.champion]["weights"]) if self.champion else None

    def next_version(self) -> str:
        nums = [int(v[1:]) for v in self.models if v[1:].isdigit()]
        return f"v{max(nums, default=-1) + 1}"

    def add(self, weights: str | Path, *, base: Optional[str], classes: Optional[List[int]], dataset: dict,
            params: dict, version: Optional[str] = None) -> str:
        version = version or self.next_version()
        if version in self.models:
            raise ValueError(f"version {version} already exists")
        self.models[version] = {"weights": str(weights), "base": base, "classes": classes, "dataset": dataset,
                                "params": params, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "benchmark": None, "metrics": None, "status": "candidate"}
        self._save()
        return version

    def record_metrics(self, version: str, benchmark: str, metrics: Dict[str, float]) -> None:
        self.models[version]["benchmark"] = benchmark
        self.models[version]["metrics"] = metrics
        self._save()

    def set_decision(self, version: str, decision: dict, tuned_on: str) -> None:
        """Store the decision thresholds chosen for a model (and what they were chosen on)."""
        self.models[version]["params"]["decision"] = decision
        self.models[version]["params"]["tuned_on"] = tuned_on
        self.models[version]["metrics"], self.models[version]["benchmark"] = None, None     # old scores no longer apply
        self._save()

    def compare(self, version: str, tolerance: float = 0.02, mae_tolerance: float = 0.05) -> Tuple[bool, List[str]]:
        """Would promoting ``version`` be safe? Returns (ok, reasons)."""
        cand = self.models[version]
        if not cand["metrics"]:
            return False, [f"{version} has no benchmark scores yet: evaluate it first"]
        if not self.champion:
            return True, ["no champion yet"]
        champ = self.models[self.champion]
        if not champ["metrics"]:
            return False, [f"champion {self.champion} has no benchmark scores: evaluate it first"]
        if cand["benchmark"] != champ["benchmark"]:
            return False, [f"benchmarks differ ({cand['benchmark']} vs {champ['benchmark']}): "
                           f"re-evaluate {self.champion} on the current test set so the scores are comparable"]
        reasons = []
        for key, higher_better in self.key_metrics.items():
            if key in cand["metrics"] and key in champ["metrics"]:
                delta = cand["metrics"][key] - champ["metrics"][key]
                tol = mae_tolerance if key == "count_mae" else tolerance
                if (delta < -tol) if higher_better else (delta > tol):
                    reasons.append(f"{key} regressed {champ['metrics'][key]:.3f} -> {cand['metrics'][key]:.3f}")
        return (not reasons), reasons or ["no regression on any key metric"]

    def adopt(self, version: str) -> None:
        """Make ``version`` the champion without copying weights: for a starting model that is not a single
        file (for example the pretrained age/gender nets), so later candidates have something to be compared to."""
        if self.champion and self.champion != version:
            self.models[self.champion]["status"] = "retired"
        self.models[version]["status"] = "champion"
        self._d["champion"] = version
        self._save()

    def promote(self, version: str, tolerance: float = 0.02, mae_tolerance: float = 0.05,
                force: bool = False) -> Tuple[bool, List[str]]:
        ok, reasons = self.compare(version, tolerance, mae_tolerance)
        if not ok and not force:
            self.models[version]["status"] = "rejected"
            self._save()
            return False, reasons
        shutil.copy2(self.models[version]["weights"], self.dir / "champion.pt")
        if self.champion and self.champion != version:
            self.models[self.champion]["status"] = "retired"
        self.models[version]["status"] = "champion"
        self._d["champion"] = version
        self._save()
        return True, (["forced past: " + "; ".join(reasons)] if (force and not ok) else reasons)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="EagleEye passenger model registry")
    ap.add_argument("--face", action="store_true", help="use the face-attribute registry (models/passenger/face/)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    b = sub.add_parser("baseline", help="register + score a starting model (e.g. the pretrained yolov8n.pt)")
    b.add_argument("--weights", default="yolov8n.pt")
    b.add_argument("--data", default="passenger_data")
    b.add_argument("--classes", type=int, nargs="*", default=[0], help="model class ids that mean a person (omit value list for all)")
    sub.add_parser("rescore", help="re-score the champion on the current test set (after test data was added)")
    p = sub.add_parser("promote")
    p.add_argument("version")
    p.add_argument("--force", action="store_true", help="promote despite a regression (will be recorded)")
    args = ap.parse_args(argv)

    reg = Registry("models/passenger/face/registry.json", FACE_KEY_METRICS) if args.face else Registry()
    if args.cmd == "list":
        for v, m in reg.models.items():
            mark = "*" if v == reg.champion else " "
            met = m["metrics"] or {}
            shown = ", ".join(f"{k}={x:.3f}" for k, x in met.items())
            print(f"{mark} {v:<5}{m['status']:<10}base={m['base']}  data={m['dataset'].get('name', '-')}  {shown}")
    elif args.cmd == "baseline":
        from eagleeye.config import load_config
        from .data_manager import DataManager
        from .detector import PersonDetector
        from .evaluate import evaluate
        from .seats import SeatMap
        cfg = load_config("config.yaml")
        dm = DataManager(args.data)
        det = cfg.passenger.detector
        report = evaluate(dm, PersonDetector(args.weights, det.conf, det.imgsz, args.classes or None),
                          SeatMap.from_config(cfg.passenger.seats), "test")
        v = reg.add(args.weights, base=None, classes=args.classes or None, dataset={"name": "pretrained"}, params={})
        reg.record_metrics(v, dm.fingerprint("test"), summarize(report))
        ok, why = reg.promote(v)
        print(f"{v}: {'champion' if ok else 'not promoted'} {why}")
    elif args.cmd == "rescore":
        from eagleeye.config import load_config
        from .data_manager import DataManager
        from .detector import PersonDetector
        from .evaluate import evaluate
        from .seats import SeatMap
        cfg, dm = load_config("config.yaml"), DataManager("passenger_data")
        champ = reg.models[reg.champion]
        det = cfg.passenger.detector
        report = evaluate(dm, PersonDetector(champ["weights"], det.conf, det.imgsz, champ["classes"]),
                          SeatMap.from_config(cfg.passenger.seats), "test")
        reg.record_metrics(reg.champion, dm.fingerprint("test"), summarize(report))
        print(f"{reg.champion} re-scored on {report['frames']} test frames: "
              + ", ".join(f"{k}={x:.3f}" for k, x in reg.models[reg.champion]['metrics'].items()))
    else:
        ok, why = reg.promote(args.version, force=args.force)
        print(("PROMOTED " if ok else "REJECTED ") + args.version + ": " + "; ".join(why))
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
