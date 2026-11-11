"""Train the face-attribute model (gender + age group) on FairFace.

    python -m passenger.datasets.fairface                       # once: download + register the data
    python -m passenger.train_faces --epochs 6                  # train; registers a candidate in models/passenger/face/
    python -m passenger.tune_classifier --version v1            # choose the child/confidence thresholds on open-data train/val
    python -m passenger.face_eval --version v1                  # score it: FairFace validation + the whole pipeline on the test photos
    python -m passenger.registry --face promote v1              # live only if nothing regressed

Rules that keep the result honest:
* the official FairFace **validation** file is the benchmark: never trained on, never used to pick anything;
* a small slice of the training file is held out to choose the best epoch;
* augmentation imitates what a cabin camera gives (loose crops, tilt, low-resolution faces, brightness changes),
  because FairFace faces are clean 224 px crops and the back rows will not be;
* the raw files are only read.
"""
from __future__ import annotations

import argparse
import queue
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .data_manager import DataManager
from .datasets import fairface
from .face_eval import attr_metrics
from .face_model import INPUT, build_net, save_checkpoint, to_input
from .registry import FACE_KEY_METRICS, Registry

Row = Tuple[bytes, int, int]


def _decode(jpeg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def center(jpeg: bytes) -> np.ndarray:
    return to_input(_decode(jpeg))


def augment(jpeg: bytes, rng: np.random.Generator) -> np.ndarray:
    """Random zoom / shift / tilt / flip / low-resolution / brightness, then the normal network input."""
    img = _decode(jpeg)
    h, w = img.shape[:2]
    zoom = rng.uniform(0.8, 1.25)                           # >1 shows more context (looser crop), <1 is tighter
    angle = rng.uniform(-12, 12)
    shift = rng.uniform(-0.08, 0.08, 2) * [w, h]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0 / zoom)
    m[:, 2] += shift
    img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.5:                                  # far-away face: few pixels, blown back up
        s = int(rng.integers(18, 72))
        img = cv2.resize(np.ascontiguousarray(img), (s, s), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.2:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(np.ascontiguousarray(img), (k, k), 0)
    img = cv2.convertScaleAbs(np.ascontiguousarray(img), alpha=rng.uniform(0.7, 1.3), beta=rng.uniform(-25, 25))
    return to_input(img)


def batches(rows: Sequence[Row], batch_size: int, *, train: bool, seed: int = 0, epoch: int = 0,
            workers: int = 8, prefetch: int = 3) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Decode (and augment) on a thread pool while the model trains: OpenCV releases the GIL, and threads
    avoid pickling the whole dataset into worker processes (which Windows would do)."""
    order = np.random.default_rng([seed, epoch]).permutation(len(rows)) if train else np.arange(len(rows))
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if train and chunks and len(chunks[-1]) < max(2, batch_size // 4):
        chunks.pop()                                        # a tiny last batch makes BatchNorm noisy

    def one(i: int) -> np.ndarray:
        jpeg = rows[i][0]
        return augment(jpeg, np.random.default_rng([seed, epoch, int(i)])) if train else center(jpeg)

    q: "queue.Queue" = queue.Queue(prefetch)

    def producer() -> None:
        with ThreadPoolExecutor(workers) as pool:
            for idx in chunks:
                x = np.stack(list(pool.map(one, idx)))
                q.put((x, np.array([rows[i][2] for i in idx], dtype=np.int64), np.array([rows[i][1] for i in idx], dtype=np.int64)))   # int64: PyTorch wants Long (NumPy's default int is 32-bit on Windows)
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


def predict_rows(net, rows: Sequence[Row], batch_size: int = 256, teen_weight: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(p_male, p_female, p_child) for every row, plus true ages and genders, using centre crops."""
    import torch

    net.eval()
    probs, ages, genders = [], [], []
    from .face_model import CHILD_GROUPS, TEEN_GROUP
    with torch.inference_mode():
        for x, g, a in batches(rows, batch_size, train=False):
            lg, la = net(torch.from_numpy(x))
            pg, pa = lg.softmax(1).numpy(), la.softmax(1).numpy()
            child = pa[:, list(CHILD_GROUPS)].sum(1) + teen_weight * pa[:, TEEN_GROUP]
            probs.append(np.column_stack([pg[:, 0], pg[:, 1], np.minimum(child, 1.0)]))
            ages.append(a)
            genders.append(g)
    return np.concatenate(probs), np.concatenate(ages), np.concatenate(genders)


def train_faces(dm: DataManager, registry: Registry, *, epochs: int = 6, batch: int = 128, lr: float = 2e-3,
                max_train: int = 0, seed: int = 0, pretrained: bool = True, rows: Optional[List[Row]] = None,
                holdout_every: int = 33, workers: int = 8, init: Optional[str] = None, dataset_name: Optional[str] = None,
                log: Callable[[str], None] = print) -> str:
    """Train on the FairFace training file(s); returns the registered (unscored) candidate version."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    if rows is None:
        paths = fairface.split_files(dm, "train")
        if not paths:
            raise ValueError("FairFace training files not found: run python -m passenger.datasets.fairface first")
        rows = fairface.load_rows(paths)
    hold = [r for i, r in enumerate(rows) if i % holdout_every == 0]
    fit = [r for i, r in enumerate(rows) if i % holdout_every != 0]
    if max_train:
        fit = fit[:max_train]
    if not fit or not hold:
        raise ValueError("need training rows and a held-out slice")
    log(f"[faces] {len(fit)} training rows, {len(hold)} held out for choosing the best epoch")

    ages = np.array([r[1] for r in fit])
    freq = np.maximum(np.bincount(ages, minlength=9), 1) / len(ages)
    age_w = torch.tensor((1 / np.sqrt(freq)) / (1 / np.sqrt(freq)).mean(), dtype=torch.float32)   # rare groups (children) count more

    if init:                                                # fine-tune an existing model instead of starting from ImageNet
        from .face_model import load_checkpoint
        net = load_checkpoint(init)[0]
    else:
        net = build_net(pretrained)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    steps = max(epochs * (len(fit) // batch + 1), 20)       # OneCycleLR divides by zero for a run of only a few steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.2)
    ce_g, ce_a = nn.CrossEntropyLoss(label_smoothing=0.05), nn.CrossEntropyLoss(weight=age_w, label_smoothing=0.05)

    version = registry.next_version()
    out = registry.dir / version
    out.mkdir(parents=True, exist_ok=True)
    best, best_score, started = None, -1.0, time.time()
    for epoch in range(epochs):
        net.train()
        seen, loss_sum = 0, 0.0
        for step, (x, g, a) in enumerate(batches(fit, batch, train=True, seed=seed, epoch=epoch, workers=workers)):
            lg, la = net(torch.from_numpy(x))
            loss = ce_g(lg, torch.from_numpy(g)) + ce_a(la, torch.from_numpy(a))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched.last_epoch < steps - 1:
                sched.step()
            seen += len(g)
            loss_sum += loss.item() * len(g)
            if step % 50 == 0:
                log(f"[faces] epoch {epoch + 1}/{epochs} step {step} loss {loss_sum / seen:.3f} ({time.time() - started:.0f}s)")
        probs, ha, hg = predict_rows(net, hold)
        m = attr_metrics(probs, ha, hg)
        score = m["gender_acc"] + m["child_auc"]
        log(f"[faces] epoch {epoch + 1}: held-out gender acc {m['gender_acc']:.3f}, child AUC {m['child_auc']:.3f}, "
            f"loss {loss_sum / max(seen, 1):.3f}")
        if score > best_score:
            best_score = score
            save_checkpoint(net, out / "best.pt", {"epoch": epoch + 1, "holdout": m, "input": INPUT})

    registry.add(out / "best.pt", base=None, classes=None, version=version,
                 dataset={"name": dataset_name or fairface.DATASET_ID, "rows_train": len(fit), "rows_holdout": len(hold),
                          "benchmark_file": "validation (never trained on)"},
                 params={"epochs": epochs, "batch": batch, "lr": lr, "seed": seed, "pretrained": pretrained,
                         "input": INPUT, "decision": {"teen_weight": 0.0, "child_threshold": 0.5, "min_conf": 0.6}})
    return version


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description="Train the face gender/age model on FairFace")
    ap.add_argument("--data", default="passenger_data")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-train", type=int, default=0, help="use only the first N training rows (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--utkface", nargs="*", default=[], metavar="TAR", help="UTKFace part*.tar.gz files to add to the training data")
    ap.add_argument("--fairface-sample", type=int, default=0, help="with --utkface: keep only N FairFace rows mixed in (0 = all), so a fine-tune is quick")
    ap.add_argument("--init", default=None, help="fine-tune from this checkpoint instead of ImageNet weights")
    args = ap.parse_args(argv)
    reg = Registry("models/passenger/face/registry.json", FACE_KEY_METRICS)
    rows, name = None, None
    if args.utkface:
        from .datasets import utkface
        ff = fairface.load_rows(fairface.split_files(DataManager(args.data), "train"))
        if args.fairface_sample:
            ff = [ff[i] for i in np.random.default_rng(0).permutation(len(ff))[:args.fairface_sample]]
        utk = utkface.load_rows([Path(p) for p in args.utkface])
        rows = [(ff + utk)[i] for i in np.random.default_rng(1).permutation(len(ff) + len(utk))]
        name = f"fairface+utkface ({len(utk)} UTKFace rows)"
        print(f"[faces] mixed data: {len(ff)} FairFace + {len(utk)} UTKFace")
    v = train_faces(DataManager(args.data), reg, epochs=args.epochs, batch=args.batch, lr=args.lr,
                    max_train=args.max_train, workers=args.workers, rows=rows, init=args.init, dataset_name=name)
    print(f"[faces] registered {v}. Next: python -m passenger.tune_classifier --version {v}")


if __name__ == "__main__":
    main()
