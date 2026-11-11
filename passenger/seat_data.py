"""Seat-state dataset (Occupied / Unoccupied seats), re-split so the test score can be trusted.

The Roboflow export splits *frames* at random, so 93-96% of its valid/test frames sit within 10 s of a
train frame (same people, same seats) and train holds each frame three times as augmented copies. Scores
on that split measure memorisation. This module rebuilds it:

* one image per source frame (the augmented ``.rf.<hash>`` copies are dropped; ultralytics augments anyway);
* frames are grouped into *clips* (same camera, gap to the previous frame <= ``gap_s``) and a whole clip
  goes to one split, deterministically, so the benchmark never drifts and never leaks.

    python -m passenger.seat_data "C:/Users/.../busCabin dataset" --out seat_data
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

NAMES = ("Occupied", "Unoccupied")
SPLITS = ("train", "val", "test")
_STEM = re.compile(r"^(\d{14})\d*_([0-9a-f]{32})_(\w+?)(?:_mp4-\d+)?_jpg\.rf\.[0-9a-f]+$")
_SOURCE_SPLITS = {"train": "train", "valid": "val", "test": "test"}


@dataclass(frozen=True)
class Item:
    image: Path
    label: Path
    source_split: str          # where Roboflow had put it
    time: dt.datetime
    camera: str                # recorder id taken from the file name
    frame_key: str             # identical for the augmented copies of one source frame


def scan(dataset: Path) -> List[Item]:
    items: List[Item] = []
    for sp in _SOURCE_SPLITS:
        for img in sorted((dataset / sp / "images").glob("*.jpg")):
            m = _STEM.match(img.stem)
            if not m:
                raise ValueError(f"unrecognised file name {img.name}: cannot recover its time and camera")
            label = dataset / sp / "labels" / f"{img.stem}.txt"
            if not label.is_file():
                continue
            items.append(Item(img, label, sp, dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S"), m.group(2)[:6],
                              img.stem.split("_jpg.rf")[0]))
    return items


def one_per_frame(items: Sequence[Item]) -> List[Item]:
    """Collapse augmented copies. Prefer a copy from valid/test (never augmented) when there is one."""
    by_key: Dict[str, List[Item]] = defaultdict(list)
    for it in items:
        by_key[it.frame_key].append(it)
    rank = lambda it: (it.source_split == "train", it.image.name)      # noqa: E731 - unaugmented first, then stable
    return sorted((min(v, key=rank) for v in by_key.values()), key=lambda i: (i.camera, i.time))


def clips(items: Sequence[Item], gap_s: float) -> List[List[Item]]:
    out: List[List[Item]] = []
    for it in sorted(items, key=lambda i: (i.camera, i.time)):
        if out and out[-1][-1].camera == it.camera and (it.time - out[-1][-1].time).total_seconds() <= gap_s:
            out[-1].append(it)
        else:
            out.append([it])
    return out


def assign(groups: Sequence[Sequence[Item]], fractions=(0.70, 0.15, 0.15), seed: int = 0) -> Dict[int, str]:
    """Greedy, seeded: shuffle clips, give each to the split furthest below its target share of *frames*."""
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    total = sum(len(g) for g in groups)
    have = {s: 0 for s in SPLITS}
    out: Dict[int, str] = {}
    for i in order:
        deficit = {s: fractions[k] * total - have[s] for k, s in enumerate(SPLITS)}
        s = max(deficit, key=deficit.get)
        out[i] = s
        have[s] += len(groups[i])
    return out


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"{dst.name} already exists: two frames map to one output name")
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build(dataset: Path, out: Path, gap_s: float = 120.0, seed: int = 0) -> dict:
    items = one_per_frame(scan(dataset))
    groups = clips(items, gap_s)
    where = assign(groups, seed=seed)
    if out.exists():
        shutil.rmtree(out)                                  # always rebuilt from the untouched originals
    for s in SPLITS:
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)
    members: Dict[str, List[Item]] = {s: [] for s in SPLITS}
    for gi, g in enumerate(groups):
        for it in g:
            members[where[gi]].append(it)
            name = f"{it.time:%Y%m%d%H%M%S}_{it.camera}_{hashlib.sha1(it.frame_key.encode()).hexdigest()[:10]}"   # unique per frame
            _link(it.image, out / "images" / where[gi] / f"{name}.jpg")
            _link(it.label, out / "labels" / where[gi] / f"{name}.txt")
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(NAMES)}\nnames: {list(NAMES)}\n", encoding="utf-8")

    def boxes(sp: str) -> Counter:
        c: Counter = Counter()
        for lab in (out / "labels" / sp).glob("*.txt"):
            c.update(int(l.split()[0]) for l in lab.read_text().splitlines() if l.strip())
        return c

    report = {
        "source": str(dataset), "gap_s": gap_s, "seed": seed, "unique_frames": len(items), "clips": len(groups),
        "splits": {s: {"frames": len(members[s]),
                       "clips": sum(1 for gi in where if where[gi] == s),
                       "cameras": dict(Counter(i.camera for i in members[s])),
                       "boxes": {NAMES[k]: v for k, v in sorted(boxes(s).items())}} for s in SPLITS},
    }
    (out / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def leakage(out: Path, within_s: float = 10.0) -> Dict[str, int]:
    """Frames in val/test within ``within_s`` of a train frame on the same camera. Must be 0 for a trustworthy score."""
    def times(sp: str):
        r = []
        for p in (out / "images" / sp).glob("*.jpg"):
            t, cam = p.stem.split("_")[:2]
            r.append((cam, dt.datetime.strptime(t, "%Y%m%d%H%M%S")))
        return r
    tr = times("train")
    return {sp: sum(1 for c, t in times(sp) if any(c == c2 and abs((t - t2).total_seconds()) <= within_s for c2, t2 in tr))
            for sp in ("val", "test")}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Rebuild the seat-state dataset with a clip-level split")
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--out", type=Path, default=Path("seat_data"))
    ap.add_argument("--gap", type=float, default=120.0, help="seconds between frames that still belong to one clip")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rep = build(a.dataset, a.out, a.gap, a.seed)
    print(json.dumps(rep, indent=2))
    print("frames within 10 s of a train frame (val, test):", leakage(a.out))


if __name__ == "__main__":
    main()
