"""Score a face-attribute model two ways, the same way for every model.

    python -m passenger.face_eval --baseline          # the pretrained Levi-Hassner nets: becomes the starting champion
    python -m passenger.face_eval --version v1        # a trained candidate from models/passenger/face/

1. **Benchmark**: a fixed, seeded sample of the official FairFace *validation* file (never trained on):
   gender accuracy and child-vs-adult AUC, on clean aligned faces.
2. **Pipeline**: the whole chain (detect -> crop -> face -> model -> decision) on the frozen Open Images test
   photos: male / female / child macro-F1. This is the number that reaches the dashboard.

Both go into the registry so a candidate is promoted only if it does not regress on either.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

from .data_manager import DataManager
from .datasets import fairface
from .face_model import CHILD_GROUPS

ADULT_FROM = 3                           # FairFace age index 3 = 20-29: clearly adult (10-19 is ambiguous and left out of child AUC)
BENCH_N, BENCH_SEED = 3000, 0


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()           # average ranks for ties
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def attr_metrics(probs: np.ndarray, ages: np.ndarray, genders: np.ndarray, child_threshold: float = 0.5) -> Dict[str, float]:
    """``probs`` rows are (p_male, p_female, p_child). Gender label 0 = male. Child = age groups 0-2 and 3-9."""
    pred_gender = (probs[:, 1] > probs[:, 0]).astype(int)
    adult = ages >= ADULT_FROM
    child = np.isin(ages, CHILD_GROUPS)
    scored = child | adult
    p_child = probs[:, 2]
    flagged = p_child >= child_threshold
    return {
        "n": int(len(ages)),
        "gender_acc": float((pred_gender == genders).mean()),
        "gender_acc_adult": float((pred_gender[adult] == genders[adult]).mean()) if adult.any() else float("nan"),
        "child_auc": _auc(p_child[child], p_child[adult]),
        "child_recall": float(flagged[child].mean()) if child.any() else float("nan"),
        "child_false_alarm": float(flagged[adult].mean()) if adult.any() else float("nan"),
        "n_child": int(child.sum()), "n_adult": int(adult.sum()), "n_scored": int(scored.sum()),
    }


def benchmark_rows(dm: DataManager, n: int = BENCH_N, seed: int = BENCH_SEED):
    rows = fairface.load_rows(fairface.split_files(dm, "validation"))
    idx = np.random.default_rng(seed).choice(len(rows), size=min(n, len(rows)), replace=False)
    return [rows[i] for i in sorted(idx)]


def benchmark_id(dm: DataManager, n: int = BENCH_N, seed: int = BENCH_SEED) -> str:
    """Identity of the whole benchmark (test photos + FairFace sample): scores are comparable only if equal."""
    val = next((v["sha256"] for k, v in dm.raw_files[fairface.DATASET_ID]["files"].items() if "validation" in k), "none")
    return f"{dm.fingerprint('test')}+ff{val[:8]}n{n}s{seed}"


def predict_triplets(predict: Callable[[np.ndarray], Tuple[float, float, float]], rows: Sequence) -> np.ndarray:
    out = []
    for jpeg, _, _ in rows:
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        out.append(predict(img))
    return np.array(out)


def benchmark_metrics(model, rows: Sequence) -> Dict[str, float]:
    """FairFace benchmark numbers for any model with ``predict(face_bgr)`` (batch-predict if it has it)."""
    ages, genders = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    if hasattr(model, "predict_batch"):
        probs = []
        for i in range(0, len(rows), 128):
            faces = [cv2.imdecode(np.frombuffer(r[0], np.uint8), cv2.IMREAD_COLOR) for r in rows[i:i + 128]]
            probs.extend(model.predict_batch(faces))
        probs = np.array(probs)
    else:
        probs = predict_triplets(model.predict, rows)
    return attr_metrics(probs, ages, genders)


def pipeline_report(dm: DataManager, model, decision: Dict[str, float], split: Optional[str] = "test",
                    source: Optional[str] = None) -> dict:
    """The whole pipeline on the frozen photos, with this model and these decision thresholds."""
    from eagleeye.config import load_config
    from .classifier import make_face_finder, PassengerClassifier
    from .detector import PersonDetector
    from .evaluate import evaluate
    from .seats import SeatMap

    cfg = load_config("config.yaml")
    c, det = cfg.passenger.classifier, cfg.passenger.detector
    clf = PassengerClassifier(model, make_face_finder(c), min_person_px=c.min_person_px,
                              min_face_px=c.min_face_px, face_margin=c.get("face_margin", 0.4), child_threshold=decision["child_threshold"],
                              min_conf=decision["min_conf"])
    return evaluate(dm, PersonDetector(det.model, det.conf, det.imgsz, det.classes),
                    SeatMap.from_config(cfg.passenger.seats), split, classifier=clf, source=source)


def score(dm: DataManager, model, decision: Dict[str, float]) -> Tuple[Dict[str, float], dict, dict]:
    """(registry metrics, benchmark metrics, pipeline report)."""
    bench = benchmark_metrics(model, benchmark_rows(dm))
    report = pipeline_report(dm, model, decision)
    cls = report["classification"]
    metrics = {"gender_acc": bench["gender_acc"], "child_auc": bench["child_auc"], "cls_macro_f1": cls["macro_f1"]}
    return metrics, bench, report


def load_model(entry: dict):
    """The model object for a registry entry: the pretrained nets or a trained checkpoint."""
    if entry["params"].get("kind") == "levi_hassner":
        from .classifier import AgeGenderNet
        return AgeGenderNet(entry["params"].get("model_dir", "models/passenger"))
    from .face_model import TorchAgeGender
    return TorchAgeGender(entry["weights"], entry["params"]["decision"]["teen_weight"])


def main(argv: Optional[Sequence[str]] = None) -> None:
    from eagleeye.config import load_config
    from .registry import FACE_KEY_METRICS, Registry

    ap = argparse.ArgumentParser(description="Score a face-attribute model on the shared benchmark")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--baseline", action="store_true", help="register + score the pretrained Levi-Hassner nets as v0 (champion)")
    g.add_argument("--version", help="score a trained candidate")
    ap.add_argument("--data", default="passenger_data")
    args = ap.parse_args(argv)

    dm = DataManager(args.data)
    reg = Registry("models/passenger/face/registry.json", FACE_KEY_METRICS)
    if args.baseline:
        c = load_config("config.yaml").passenger.classifier
        decision = {"teen_weight": 0.0, "child_threshold": float(c.child_threshold), "min_conf": float(c.min_conf)}
        v = reg.add("models/passenger/gender_net.caffemodel", base=None, classes=None,
                    dataset={"name": "pretrained (Levi-Hassner, Adience)"},
                    params={"kind": "levi_hassner", "model_dir": str(c.model_dir), "decision": decision})
    else:
        v = args.version
        decision = reg.models[v]["params"]["decision"]
    metrics, bench, report = score(dm, load_model(reg.models[v]), decision)
    reg.record_metrics(v, benchmark_id(dm), metrics)
    if args.baseline:
        reg.adopt(v)
    cls = report["classification"]
    print(f"[{v}] FairFace benchmark (n={bench['n']}): gender acc {bench['gender_acc']:.3f} "
          f"(adults {bench['gender_acc_adult']:.3f}), child AUC {bench['child_auc']:.3f}, "
          f"child recall {bench['child_recall']:.2f} / false alarm {bench['child_false_alarm']:.2f}")
    print(f"[{v}] pipeline on {report['frames']} test photos: macro-F1 {cls['macro_f1']:.3f}, coverage {cls['coverage']:.2f}, "
          f"accuracy on answered {cls['accuracy_answered']:.3f}; per-class recall "
          + ", ".join(f"{k} {m['recall']:.2f}" for k, m in cls["per_class"].items()))
    print(f"[{v}] decision {decision}")


if __name__ == "__main__":
    main()
