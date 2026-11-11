"""Open Images V7 -> male / female / child person boxes.

    python -m passenger.datasets.open_images --data passenger_data --split validation --max-images 400

Open Images has boxes for ``Man``, ``Woman``, ``Boy`` and ``Girl``, which map directly onto this project's
classes (Boy and Girl -> child). Licences: annotations CC BY 4.0, images CC BY 2.0 (author and licence per
image are kept in ``metadata/attribution.csv``; give attribution when you redistribute anything derived).

The data are general photos, not bus cabins, so they give licence-clean, real-photo evaluation and
a way to exercise retraining, not cabin accuracy. Images from vehicle scenes are ranked first.

Everything downloaded is kept under ``raw/external/<dataset_id>/`` exactly as received (original CSVs
included); the YOLO labels are derived from it and can always be rebuilt.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import pandas as pd

from ..data_manager import DataManager

GCS = "https://storage.googleapis.com/openimages"
IMAGE_HOST = "https://open-images-dataset.s3.amazonaws.com"
_FILES = {   # split -> the four CSVs, all under GCS
    "validation": {"bbox": "v5/validation-annotations-bbox.csv",
                   "labels": "v5/validation-annotations-human-imagelabels-boxable.csv",
                   "images": "2018_04/validation/validation-images-with-rotation.csv"},
    "test": {"bbox": "v5/test-annotations-bbox.csv",
             "labels": "v5/test-annotations-human-imagelabels-boxable.csv",
             "images": "2018_04/test/test-images-with-rotation.csv"},
}
_CLASSES_CSV = "v7/oidv7-class-descriptions-boxable.csv"
#: Open Images class name -> this project's class id (male 0, female 1, child 2).
CLASS_MAP = {"Man": 0, "Woman": 1, "Boy": 2, "Girl": 2}
_VEHICLE = ("Bus", "Train", "Vehicle", "Land vehicle", "Car", "Van", "Taxi")
_CONTEXT_RANK = {"bus": 0, "vehicle": 1, "general": 2}
CITATION = ("Kuznetsova et al., The Open Images Dataset V4, IJCV 2020. "
            "Annotations CC BY 4.0; images CC BY 2.0.")

Fetch = Callable[[str, Path], None]


@dataclass
class Selected:
    image_id: str
    context: str                                        # bus / vehicle / general
    boxes: List[Tuple[int, float, float, float, float]]  # (class id, cx, cy, w, h), normalised


def load_class_ids(classes_csv: Path) -> Dict[str, str]:
    """Class display name -> MID."""
    df = pd.read_csv(classes_csv, header=None, names=["mid", "name"])
    return dict(zip(df["name"], df["mid"]))


def _iou(a, b) -> float:
    iw, ih = min(a[1], b[1]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[2], b[2])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter)


def select_images(bbox: pd.DataFrame, image_labels: pd.DataFrame, mids: Dict[str, str],
                  max_images: int = 400, min_people: int = 1) -> List[Selected]:
    """Pick images whose people are *completely* labelled as male / female / child.

    An image is kept only if it has no group boxes, no depictions (drawings, statues), and every generic
    ``Person`` box is covered by a Man / Woman / Boy / Girl box (IoU >= 0.5). Otherwise an unlabelled person
    would count as a false positive and make every precision number look worse than it is.
    Vehicle scenes come first (bus, then other vehicles), then the rest; ties are broken by a hash of the
    image id so the choice is deterministic.
    """
    attr_mid = {mids[n]: CLASS_MAP[n] for n in CLASS_MAP}
    person_mid = mids["Person"]
    people = bbox[bbox["LabelName"].isin([*attr_mid, person_mid])]

    ctx_mid = {mids[n]: n for n in _VEHICLE if n in mids}
    lab = image_labels[(image_labels["Confidence"] == 1) & image_labels["LabelName"].isin(ctx_mid)]
    context: Dict[str, str] = {}
    for img, names in lab.groupby("ImageID")["LabelName"]:
        context[img] = "bus" if mids["Bus"] in set(names) else "vehicle"

    out: List[Selected] = []
    for img, g in people.groupby("ImageID"):
        if (g["IsGroupOf"] == 1).any() or (g["IsDepiction"] == 1).any():
            continue
        attr = g[g["LabelName"].isin(attr_mid)]
        if len(attr) < min_people:
            continue
        a_boxes = attr[["XMin", "XMax", "YMin", "YMax"]].values
        p_boxes = g[g["LabelName"] == person_mid][["XMin", "XMax", "YMin", "YMax"]].values
        if not all(any(_iou(p, a) >= 0.5 for a in a_boxes) for p in p_boxes):
            continue
        boxes = []
        for _, r in attr.iterrows():
            w, h = r["XMax"] - r["XMin"], r["YMax"] - r["YMin"]
            if w > 0 and h > 0:
                boxes.append((attr_mid[r["LabelName"]], (r["XMin"] + r["XMax"]) / 2, (r["YMin"] + r["YMax"]) / 2, w, h))
        if boxes:
            out.append(Selected(img, context.get(img, "general"), boxes))

    out.sort(key=lambda s: (_CONTEXT_RANK[s.context], hashlib.sha1(s.image_id.encode()).hexdigest()))
    return out[:max_images]


def _http_get(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last: Optional[Exception] = None
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)
            return
        except OSError as exc:
            last = exc
            time.sleep(1 + attempt)
    tmp.unlink(missing_ok=True)
    raise OSError(f"download failed: {url}: {last}")


def ensure_metadata(meta_dir: Path, split: str, fetch: Fetch = _http_get) -> Dict[str, Path]:
    """Download the original CSVs once; they stay in raw/ untouched."""
    files = _FILES[split]
    paths = {"classes": meta_dir / Path(_CLASSES_CSV).name,
             **{k: meta_dir / Path(v).name for k, v in files.items()}}
    for key, rel in (("classes", _CLASSES_CSV), *files.items()):
        if not paths[key].exists() or paths[key].stat().st_size == 0:
            fetch(f"{GCS}/{rel}", paths[key])
    return paths


def download_images(ids: Sequence[str], split: str, dest: Path, fetch: Fetch = _http_get,
                    workers: int = 8) -> List[str]:
    """Fetch the images (skipping ones already present). Returns the ids that are usable JPEGs."""
    dest.mkdir(parents=True, exist_ok=True)

    def one(image_id: str) -> Optional[str]:
        path = dest / f"{image_id}.jpg"
        try:
            if not path.exists() or path.stat().st_size == 0:
                fetch(f"{IMAGE_HOST}/{split}/{image_id}.jpg", path)
            return image_id if cv2.imread(str(path)) is not None else None
        except OSError:
            return None

    with ThreadPoolExecutor(workers) as pool:
        return [i for i in pool.map(one, ids) if i]


def write_attribution(images_csv: Path, ids: Set[str], dest: Path) -> int:
    """Per-image author / licence / source page for the images we kept (needed for CC BY attribution)."""
    df = pd.read_csv(images_csv, usecols=["ImageID", "Author", "OriginalLandingURL", "License", "Title"])
    df = df[df["ImageID"].isin(ids)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False, quoting=csv.QUOTE_MINIMAL)
    return len(df)


def ingest(dm: DataManager, split: str = "validation", max_images: int = 400, min_people: int = 1,
           fetch: Fetch = _http_get) -> Tuple[str, int, Dict[str, int]]:
    """Download, select, convert and register. Returns (dataset id, images registered, images per context)."""
    if split not in _FILES:
        raise ValueError(f"split must be one of {sorted(_FILES)} (train needs a 2 GB CSV: not supported here)")
    dataset_id = f"open_images_v7_{split}"
    raw = dm.external_raw / dataset_id
    meta_dir = raw / "metadata"
    paths = ensure_metadata(meta_dir, split, fetch)

    mids = load_class_ids(paths["classes"])
    bbox = pd.read_csv(paths["bbox"], usecols=["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsGroupOf", "IsDepiction"])
    labels = pd.read_csv(paths["labels"], usecols=["ImageID", "LabelName", "Confidence"])
    chosen = select_images(bbox, labels, mids, max_images, min_people)

    ok = set(download_images([s.image_id for s in chosen], split, raw / "images", fetch))
    kept = [s for s in chosen if s.image_id in ok]
    write_attribution(paths["images"], {s.image_id for s in kept}, meta_dir / "attribution.csv")

    meta = {
        "source": "Open Images V7", "url": "https://storage.googleapis.com/openimages/web/index.html",
        "annotations_license": "CC BY 4.0", "images_license": "CC BY 2.0 (per image, see metadata/attribution.csv)",
        "citation": CITATION, "split_used": split, "retrieved": time.strftime("%Y-%m-%d"),
        "class_map": CLASS_MAP, "selection": {"max_images": max_images, "min_people": min_people,
                                              "rule": "complete gender/child labels, no groups, no depictions; vehicle scenes first"},
    }
    (raw / "SOURCE.json").write_text(json.dumps(meta, indent=2))
    n = dm.add_external_dataset(
        dataset_id, meta,
        [{"name": s.image_id, "image": f"images/{s.image_id}.jpg", "boxes": s.boxes, "context": s.context} for s in kept])
    contexts: Dict[str, int] = {}
    for s in kept:
        contexts[s.context] = contexts.get(s.context, 0) + 1
    return dataset_id, n, contexts


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Ingest Open Images people (male / female / child) into passenger_data")
    ap.add_argument("--data", default="passenger_data")
    ap.add_argument("--split", default="validation", choices=sorted(_FILES))
    ap.add_argument("--max-images", type=int, default=400)
    ap.add_argument("--min-people", type=int, default=1)
    args = ap.parse_args(argv)
    dm = DataManager(args.data)
    dataset_id, n, contexts = ingest(dm, args.split, args.max_images, args.min_people)
    print(f"[open_images] {dataset_id}: {n} images registered {contexts}")
    stats = dm.stats()["external"][dataset_id]
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
