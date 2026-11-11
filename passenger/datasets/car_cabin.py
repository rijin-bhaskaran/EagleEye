"""Build the car-cabin training set from one or more public YOLO-format datasets.

No single public dataset has occupants, seatbelts and child seats together, so the set is assembled from several
whose classes are *remapped* onto ``CAR_CLASSES`` (male, female, child, seatbelt, no_seatbelt, child_seat):

    python -m passenger.datasets.car_cabin add "C:/dl/belts_export" --name belts_rf --license "CC BY 4.0" \\
        --url https://universe.roboflow.com/... --map seatbelt=seatbelt "no belt=no_seatbelt" person=drop
    python -m passenger.datasets.car_cabin add "C:/dl/cabin_kids" --name kids --license "..." \\
        --map child=child adult=drop "child seat=child_seat"
    python -m passenger.datasets.car_cabin stats

``--map src=dst`` renames a source class to a car class; ``dst`` of ``drop`` discards its boxes. A source class
with no ``--map`` entry is an error (silently dropping a class is how a seatbelt model ends up with no belts),
unless ``--drop-unmapped`` is given. Sources are read, never modified.

Splits are decided per *group*, not per image: frames of the same clip/vehicle/person look alike, and a test
frame next to a train frame measures memorisation. ``--group-regex`` extracts the group from the file stem
(default: the stem without Roboflow's ``.rf.<hash>`` suffix and without a trailing frame counter). A group goes
to exactly one split, chosen by a hash, so adding data later never moves existing images between splits.

Output (``car_data/`` by default): ``images/<split>/<name>__<stem>.jpg``, ``labels/<split>/...txt``,
``data.yaml`` and ``sources.json`` (name, licence, URL, class map, counts per split: provenance for every image).
Re-adding a dataset name replaces only that dataset's files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..car_logic import CAR_CLASSES

SPLITS = ("train", "val", "test")
_SHARES = (0.70, 0.15, 0.15)
_IMG = {".jpg", ".jpeg", ".png", ".bmp"}
_RF = re.compile(r"[._-]rf[._-][0-9a-f]+$|_(?:jpg|jpeg|png)\.rf\.[0-9a-f]+$")
_FRAME = re.compile(r"[_-]?(?:frame|f)?[_-]?\d+$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,40}$")


def group_of(stem: str, pattern: Optional[str] = None) -> str:
    """Group key of an image: same group = same clip / scene, so never split apart."""
    if pattern:
        m = re.search(pattern, stem)
        if not m:
            raise ValueError(f"--group-regex {pattern!r} does not match {stem!r}")
        return m.group(1) if m.groups() else m.group(0)
    s = _RF.sub("", stem)
    t = _FRAME.sub("", s)
    return t or s


def split_of(group: str, dataset: str) -> str:
    """Deterministic split from a hash: the same group always lands in the same split."""
    x = int(hashlib.sha1(f"{dataset}:{group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return SPLITS[0] if x < _SHARES[0] else SPLITS[1] if x < _SHARES[0] + _SHARES[1] else SPLITS[2]


def parse_map(pairs: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--map entry {p!r} must look like source=target")
        src, dst = (x.strip() for x in p.rsplit("=", 1))
        if dst != "drop" and dst not in CAR_CLASSES:
            raise ValueError(f"--map {p!r}: target must be one of {list(CAR_CLASSES)} or 'drop'")
        out[src.lower()] = dst
    return out


def source_names(dataset: Path) -> List[str]:
    import yaml

    f = next((dataset / n for n in ("data.yaml", "dataset.yaml") if (dataset / n).is_file()), None)
    if f is None:
        raise FileNotFoundError(f"{dataset}: no data.yaml (expected a YOLO export)")
    names = yaml.safe_load(f.read_text(encoding="utf-8")).get("names")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names)]
    if isinstance(names, list):
        return [str(n) for n in names]
    raise ValueError(f"{f}: no class names")


def _pairs(dataset: Path):
    """(image, label) for every labelled image, whether the export is split/images|labels or images|labels/split."""
    for img in sorted(dataset.rglob("*")):
        if img.suffix.lower() not in _IMG or "images" not in img.parts:
            continue
        parts = list(img.parts)
        k = len(parts) - 1 - parts[::-1].index("images")
        label = Path(*parts[:k], "labels", *parts[k + 1:]).with_suffix(".txt")
        if label.is_file():
            yield img, label


def _link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _safe_stem(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def add(dataset: Path, out: Path, name: str, mapping: Dict[str, str], *, license: str, url: str = "",
        drop_unmapped: bool = False, group_regex: Optional[str] = None) -> dict:
    if not _NAME.match(name):
        raise ValueError("--name must be one short word of letters, digits, '-' or '_'")
    if not license.strip():
        raise ValueError("--license is required: record where the data may be used before it is trained on")
    src_names = source_names(dataset)
    unmapped = [n for n in src_names if n.lower() not in mapping]
    if unmapped and not drop_unmapped:
        raise ValueError(f"source classes {unmapped} have no --map entry (map them, or 'x=drop', or pass --drop-unmapped)")
    to_new = {i: mapping.get(n.lower(), "drop") for i, n in enumerate(src_names)}

    for s in SPLITS:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
        for old in list((out / "images" / s).glob(f"{name}__*")) + list((out / "labels" / s).glob(f"{name}__*")):
            old.unlink()                                  # only this dataset's own earlier files

    seen: set = set()
    counts = {s: Counter() for s in SPLITS}
    images = {s: 0 for s in SPLITS}
    skipped = 0
    for img, label in _pairs(dataset):
        key = _RF.sub("", img.stem)
        if key in seen:                                   # Roboflow augmented copies of one source frame
            continue
        seen.add(key)
        rows = []
        for line in label.read_text(encoding="utf-8").splitlines():
            f = line.split()
            if len(f) < 5:
                continue
            try:
                new = to_new[int(f[0])]
            except (ValueError, KeyError):
                raise ValueError(f"{label}: class id {f[0]!r} is not in {dataset / 'data.yaml'}")
            if new != "drop":
                rows.append((CAR_CLASSES.index(new), f[1:]))
        if not rows:
            skipped += 1                                  # nothing we care about: a negative image would need a deliberate choice
            continue
        sp = split_of(group_of(img.stem, group_regex), name)
        stem = f"{name}__{_safe_stem(key)}"
        _link(img, out / "images" / sp / f"{stem}{img.suffix.lower()}")
        (out / "labels" / sp / f"{stem}.txt").write_text("\n".join(f"{c} {' '.join(v)}" for c, v in rows) + "\n", encoding="utf-8")
        images[sp] += 1
        counts[sp].update(CAR_CLASSES[c] for c, _ in rows)

    entry = {"license": license, "url": url, "source_dir": str(dataset), "map": {n: to_new[i] for i, n in enumerate(src_names)},
             "images": images, "boxes": {s: dict(counts[s]) for s in SPLITS}, "skipped_without_target_classes": skipped}
    meta_path = out / "sources.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    meta[name] = entry
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(CAR_CLASSES)}\nnames: {list(CAR_CLASSES)}\n", encoding="utf-8")
    return entry


def stats(out: Path) -> dict:
    """Boxes per class per split over everything in ``out``: shows which class the model cannot learn yet."""
    boxes = {s: Counter() for s in SPLITS}
    for s in SPLITS:
        for lab in (out / "labels" / s).glob("*.txt"):
            boxes[s].update(CAR_CLASSES[int(l.split()[0])] for l in lab.read_text().splitlines() if l.strip())
    return {"images": {s: len(list((out / "images" / s).glob("*.*"))) for s in SPLITS},
            "boxes": {s: {c: boxes[s].get(c, 0) for c in CAR_CLASSES} for s in SPLITS}}


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description="Assemble the car-cabin training set")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="add a YOLO-format dataset, remapping its classes")
    a.add_argument("dataset", type=Path)
    a.add_argument("--name", required=True)
    a.add_argument("--license", required=True)
    a.add_argument("--url", default="")
    a.add_argument("--map", nargs="+", default=[], metavar="SRC=DST")
    a.add_argument("--drop-unmapped", action="store_true")
    a.add_argument("--group-regex", default=None, help="regex whose first group (or whole match) is the clip/scene id in the file stem")
    a.add_argument("--out", type=Path, default=Path("car_data"))
    s = sub.add_parser("stats", help="box counts per class and split")
    s.add_argument("--out", type=Path, default=Path("car_data"))
    args = ap.parse_args(argv)
    if args.cmd == "add":
        print(json.dumps(add(args.dataset, args.out, args.name, parse_map(args.map), license=args.license, url=args.url,
                             drop_unmapped=args.drop_unmapped, group_regex=args.group_regex), indent=2))
    print(json.dumps(stats(args.out), indent=2))
    thin = [c for c, n in stats(args.out)["boxes"]["train"].items() if n < 200]
    if thin:
        print(f"[car_cabin] WARNING: fewer than 200 training boxes for {thin}: the model will be weak or blind on them")


if __name__ == "__main__":
    main()
