#!/usr/bin/env python3
"""Build a demo video for the passenger channel from the photos already in passenger_data/.

    python scripts/make_demo_video.py                       # data/passenger_demo.mp4
    python scripts/make_demo_video.py --seconds 90 --out data/my_test.mp4

A slideshow of real photos with people (vehicle scenes first), 1280x720, so the passenger dashboard, counting,
tracking and classification can be tried end to end before any cabin footage exists. It is NOT a bus cabin:
use it to test the software, not to judge accuracy. The output is a derived file and can be rebuilt any time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from passenger.data_manager import DataManager

W, H = 1280, 720
_RANK = {"bus": 0, "vehicle": 1}


def letterbox(img: np.ndarray, w: int = W, h: int = H) -> np.ndarray:
    s = min(w / img.shape[1], h / img.shape[0])
    resized = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), np.uint8)
    y, x = (h - resized.shape[0]) // 2, (w - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def pick_images(dm: DataManager, count: int) -> List[Path]:
    """Photos from external datasets, bus then vehicle scenes first, with a stable order."""
    items = []
    for entry in dm.external.values():
        for name, item in entry.get("items", {}).items():
            if item.get("context"):
                items.append((_RANK.get(item["context"], 2), name, dm.root / item["image"]))
    return [p for _, _, p in sorted(items)][:count]


def build(dm: DataManager, out: Path, seconds: int = 60, fps: int = 10, seconds_per_photo: float = 4.0) -> int:
    """Write the video; returns the number of photos used."""
    images = pick_images(dm, max(1, int(seconds / seconds_per_photo)))
    if not images:
        raise SystemExit("no photos found in passenger_data/: run python -m passenger.datasets.open_images first")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    used = 0
    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            continue
        frame = letterbox(img)
        for _ in range(int(seconds_per_photo * fps)):
            writer.write(frame)
        used += 1
    writer.release()
    return used


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build a demo video for the passenger channel")
    ap.add_argument("--data", default=str(ROOT / "passenger_data"))
    ap.add_argument("--out", default=str(ROOT / "data" / "passenger_demo.mp4"))
    ap.add_argument("--seconds", type=int, default=60)
    args = ap.parse_args(argv)
    used = build(DataManager(args.data), Path(args.out), args.seconds)
    print(f"wrote {args.out} ({used} photos, {args.seconds} s). Test it with: python run_all.py")


if __name__ == "__main__":
    main()
