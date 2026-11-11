"""Train the male / female / child classifier on person crops (stage B of the car model).

    python -m passenger.train_crops --data passenger_data/annotations/crops_mfc_v1 \
        --pretrain passenger_data/annotations/crops_mfc_v1/pretrain_pa100k --out models/passenger/crops/v1

Optional pretraining on adult male/female crops (PA-100K), then fine-tuning on ``<data>/{train,val,test}/{male,female,child}``.
The best epoch is chosen on val macro-F1; test is scored once at the end and never used to choose anything
(``score_test=False`` skips it for ablation runs).
``balanced`` draws every class equally often per epoch; ``regularise`` adds label smoothing, weight decay, cabin-like colour
jitter and random erasing, a frozen backbone for the first epochs and early stopping.
Writes ``model.pt`` and ``metrics.json`` only; it does not touch the registry or config.yaml.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .train_faces import augment, center

CLASSES = ("male", "female", "child")
SMOOTHING, WEIGHT_DECAY, PATIENCE, FREEZE_EPOCHS, BACKBONE_LR = 0.1, 0.02, 8, 3, 0.1     # the regularised recipe
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_folder(root: str | Path, classes: Sequence[str]) -> Tuple[List[Path], List[int]]:
    """Files and class indices from ``root/<class>/*``; a folder that is not a known class is an error."""
    root = Path(root)
    unknown = [d.name for d in root.iterdir() if d.is_dir() and d.name not in classes]
    if unknown:
        raise ValueError(f"unknown class folders in {root}: {unknown}")
    files: List[Path] = []
    labels: List[int] = []
    for i, name in enumerate(classes):
        for f in sorted((root / name).glob("*")):
            if f.suffix.lower() in IMAGE_SUFFIXES:
                files.append(f)
                labels.append(i)
    return files, labels


def build_net(num_classes: int, imagenet: bool = False):
    from torch import nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if imagenet else None)
    return nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(576, num_classes))


def balanced_order(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One epoch of indices with every class drawn equally often (rare classes with replacement)."""
    classes = np.unique(labels)
    per = len(labels) // len(classes)
    parts = [rng.choice(idx, per, replace=per > len(idx)) for idx in (np.flatnonzero(labels == c) for c in classes)]
    return rng.permutation(np.concatenate(parts))


def cabin_augment(jpeg: bytes, rng: np.random.Generator) -> np.ndarray:
    """``augment`` plus colour-cast / saturation jitter and random erasing (cabin cameras differ in colour and occlude people)."""
    x = augment(jpeg, rng)
    gray = x.mean(0, keepdims=True)
    x = gray + (x - gray) * rng.uniform(0.5, 1.3)
    x = x * rng.uniform(0.85, 1.15, (3, 1, 1)).astype(np.float32) + rng.uniform(-0.25, 0.25, (3, 1, 1)).astype(np.float32)
    if rng.random() < 0.5:
        h, w = x.shape[1:]
        area, ratio = rng.uniform(0.02, 0.2), rng.uniform(0.5, 2.0)
        eh, ew = min(h, max(1, round(h * np.sqrt(area * ratio)))), min(w, max(1, round(w * np.sqrt(area / ratio))))
        y0, x0 = int(rng.integers(0, h - eh + 1)), int(rng.integers(0, w - ew + 1))
        x[:, y0:y0 + eh, x0:x0 + ew] = 0.0
    return x.astype(np.float32)


def _batches(data: Sequence[bytes], labels: np.ndarray, size: int, train: bool, epoch: int, seed: int, pool,
             balanced: bool = False, aug=augment):
    if not train:
        order = np.arange(len(data))
    elif balanced:
        order = balanced_order(labels, np.random.default_rng([seed, epoch]))
    else:
        order = np.random.default_rng([seed, epoch]).permutation(len(data))
    for i in range(0, len(order), size):
        idx = order[i:i + size]
        if train and len(idx) < 2:
            continue                                            # BatchNorm needs two samples
        if train:
            xs = list(pool.map(lambda j: aug(data[j], np.random.default_rng([seed, epoch, int(j)])), idx))
        else:
            xs = list(pool.map(lambda j: center(data[j]), idx))
        yield np.stack(xs), labels[idx]


def _logits(net, data, labels, size, pool) -> np.ndarray:
    import torch

    net.eval()
    out = []
    with torch.no_grad():
        for x, _ in _batches(data, labels, size, False, 0, 0, pool):
            out.append(net(torch.from_numpy(x)).numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def _predict(net, data, labels, size, pool) -> np.ndarray:
    return _logits(net, data, labels, size, pool).argmax(1)


def _ce(logits: np.ndarray, y: np.ndarray) -> float:
    z = logits - logits.max(1, keepdims=True)
    return float(-(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean()) if len(y) else float("nan")


def score(y: np.ndarray, pred: np.ndarray, classes: Sequence[str]) -> dict:
    k = len(classes)
    cm = np.zeros((k, k), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    per = {}
    for i, name in enumerate(classes):
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i].sum() - cm[i, i]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[name] = {"precision": round(float(prec), 4), "recall": round(float(rec), 4),
                     "f1": round(float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0, 4), "n": int(cm[i].sum())}
    rec = [v["recall"] for v in per.values()]
    return {"accuracy": round(float((y == pred).mean()), 4) if len(y) else 0.0,
            "macro_f1": round(float(np.mean([v["f1"] for v in per.values()])), 4),
            "balanced_accuracy": round(float(np.mean(rec)), 4), "recall_spread": round(float(max(rec) - min(rec)), 4),
            "per_class": per, "confusion": cm.tolist()}


def _fit(net, train, val, classes, epochs, batch_size, lr, seed, pool, label, balanced=False, regularise=False):
    """Train ``net``; returns (best-epoch copy by val macro-F1 (train loss if there is no val), per-epoch history)."""
    import torch
    from torch import nn

    x, y = train
    counts = np.bincount(y, minlength=len(classes)).clip(1)
    weight = None if balanced else torch.tensor(len(y) / (len(classes) * counts), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=SMOOTHING if regularise else 0.0)
    if regularise:
        opt = torch.optim.AdamW([{"params": net[0].parameters(), "lr": lr * BACKBONE_LR},
                                 {"params": net[1:].parameters(), "lr": lr}], weight_decay=WEIGHT_DECAY)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    best, best_score, stale, history = copy.deepcopy(net), -1.0, 0, []
    for ep in range(epochs):
        frozen = regularise and ep < FREEZE_EPOCHS                  # head first, then the whole net at the lower backbone LR
        for p in net[0].parameters():
            p.requires_grad_(not frozen)
        net.train()
        if frozen:
            net[0].eval()                                           # keep BatchNorm statistics fixed too
        t0, total, n = time.time(), 0.0, 0
        for xb, yb in _batches(x, y, batch_size, True, ep, seed, pool, balanced, cabin_augment if regularise else augment):
            opt.zero_grad()
            loss = loss_fn(net(torch.from_numpy(xb)), torch.from_numpy(yb))
            loss.backward()
            opt.step()
            total, n = total + loss.item() * len(yb), n + len(yb)
        sched.step()
        row = {"epoch": ep + 1, "train_loss": round(total / max(n, 1), 4)}
        if val:
            lg = _logits(net, val[0], val[1], batch_size, pool)
            sc = score(val[1], lg.argmax(1), classes)
            row.update(val_loss=round(_ce(lg, val[1]), 4), val_macro_f1=sc["macro_f1"], val_recall_spread=sc["recall_spread"])
            s = sc["macro_f1"]
        else:
            s = -total / max(n, 1)
        history.append(row)
        print(f"[{label}] epoch {ep + 1}/{epochs} " + " ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "epoch") + f" {time.time() - t0:.0f}s", flush=True)
        if s > best_score:
            best, best_score, stale = copy.deepcopy(net), s, 0
        else:
            stale += 1
            if regularise and stale >= PATIENCE:
                print(f"[{label}] early stop: no val improvement for {PATIENCE} epochs", flush=True)
                break
    return best, history


def _read(root: Path, split: str, classes: Sequence[str]):
    files, labels = load_folder(root / split, classes)
    return [f.read_bytes() for f in files], np.array(labels, np.int64)      # torch wants int64; numpy's default int is 32-bit on Windows


def _adults(root: Path, per_class: int, seed: int):
    """``per_class`` random male and female crops from ``root/train`` (PA-100K adults)."""
    files, labels = load_folder(root / "train", CLASSES[:2])
    labels = np.array(labels)
    rng = np.random.default_rng(seed)
    pick = np.concatenate([rng.choice(np.flatnonzero(labels == c), per_class, replace=False) for c in (0, 1)])
    return [files[i].read_bytes() for i in pick], labels[pick].astype(np.int64)


def train_crops(data: str | Path, out: str | Path, epochs: int = 25, batch_size: int = 32, threads: int = 4, lr: float = 1e-3,
                seed: int = 0, pretrain: str | Path | None = None, pretrain_epochs: int = 8, imagenet: bool = False,
                balanced: bool = False, regularise: bool = False, adult_mix: int = 0,
                pretrain_cache: str | Path | None = None, score_test: bool = True) -> dict:
    """``adult_mix`` adds that many PA-100K male and female crops (from ``pretrain``) to the fine-tune set;
    ``pretrain_cache`` stores/reuses the pretrained net so ablation runs do not repeat 20 minutes of pretraining."""
    import torch
    from torch import nn

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"classes": list(CLASSES), "imagenet_init": imagenet, "epochs": epochs, "seed": seed, "balanced": balanced,
            "regularise": regularise, "adult_mix": adult_mix}
    with ThreadPoolExecutor(max_workers=max(2, threads)) as pool:
        net = build_net(2 if pretrain else len(CLASSES), imagenet)
        pre_n = 0
        if pretrain:
            pre = Path(pretrain)
            cache = Path(pretrain_cache) if pretrain_cache else None
            if cache and cache.exists():
                ck = torch.load(cache, weights_only=True)
                net.load_state_dict(ck["state"])
                pre_n, hold_score = ck["n"], ck["holdout"]
            else:
                px, py = _read(pre, "train", CLASSES[:2])
                pre_n = len(px)
                hold = _read(pre, "holdout", CLASSES[:2]) if (pre / "holdout").is_dir() else None
                net, _ = _fit(net, (px, py), hold, CLASSES[:2], pretrain_epochs, batch_size, lr, seed, pool, "pretrain")
                hold_score = score(hold[1], _predict(net, hold[0], hold[1], batch_size, pool), CLASSES[:2]) if hold else None
                if cache:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"state": net.state_dict(), "n": pre_n, "holdout": hold_score}, cache)
            if hold_score:
                meta["pretrain_holdout"] = hold_score
            net[-1] = nn.Linear(576, len(CLASSES))              # new 3-way head; the backbone keeps what it learned
        train, val = (_read(data, s, CLASSES) for s in ("train", "val"))
        fit_set = train
        if adult_mix:
            ax, ay = _adults(Path(pretrain), adult_mix, seed)
            fit_set = (train[0] + ax, np.concatenate([train[1], ay]))
        net, history = _fit(net, fit_set, val, CLASSES, epochs, batch_size, lr * (0.3 if pretrain and not regularise else 1.0), seed, pool,
                            "finetune", balanced, regularise)
        metrics = {**meta, "pretrained_on": pre_n, "train_images": len(fit_set[1]), "val_images": len(val[1]), "test_images": None,
                   "history": history,
                   "train": score(train[1], _predict(net, train[0], train[1], batch_size, pool), CLASSES),     # clean cabin-train crops
                   "val": score(val[1], _predict(net, val[0], val[1], batch_size, pool), CLASSES)}
        if score_test:
            test = _read(data, "test", CLASSES)
            metrics.update(test_images=len(test[1]), test=score(test[1], _predict(net, test[0], test[1], batch_size, pool), CLASSES))
    torch.save({"state": net.state_dict(), "meta": meta}, out / "model.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pretrain", default=None)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--pretrain-epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--balanced", action="store_true", help="draw every class equally often per epoch")
    ap.add_argument("--regularise", action="store_true", help="label smoothing, weight decay, cabin augmentation, frozen start, early stopping")
    ap.add_argument("--adult-mix", type=int, default=0, help="PA-100K male and female crops (each) added to the fine-tune set")
    ap.add_argument("--pretrain-cache", default=None)
    ap.add_argument("--no-test", action="store_true", help="skip scoring test (selection runs)")
    ap.add_argument("--imagenet", action=argparse.BooleanOptionalAction, default=True, help="start from ImageNet weights (downloads once)")
    a = ap.parse_args(argv)
    m = train_crops(a.data, a.out, a.epochs, a.batch_size, a.threads, pretrain=a.pretrain, pretrain_epochs=a.pretrain_epochs, imagenet=a.imagenet,
                    balanced=a.balanced, regularise=a.regularise, adult_mix=a.adult_mix, pretrain_cache=a.pretrain_cache, score_test=not a.no_test)
    print(json.dumps({k: m[k] for k in ("train_images", "val_images", "test_images")}), json.dumps(m["val"]["per_class"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
