#!/usr/bin/env python3
"""Download the age/gender weights into models/passenger/ and verify them.

    python scripts/fetch_passenger_models.py

Files that already match their SHA-256 are left alone; a truncated or corrupt
file is downloaded again. The weights are the Levi & Hassner (2015) Adience
CNNs: research licence, so review it before commercial use.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master"
FILES = {
    "gender_deploy.prototxt": "c1961acc32e6e9ce855f6ec4973e9a939cc2d49089a8aaefeafa0e100fb110cc",
    "age_deploy.prototxt": "af2a268d12bda7821140461319828a8560ce7739db9568d6b3827a60487a6333",
    "age_net.caffemodel": "6dde5d07df5ca1d66ff39e525693f05ccfb9d2c437e188fdd1a10d42e57fabd6",
    "gender_net.caffemodel": "ac7571b281ae078817764b645a20541bd6aa1babeac20a45e6d8de7d61ba0e50",
}

#: Face detector (OpenCV Zoo, MIT licence), used when passenger.classifier.face_detector is yunet.
YUNET_BASE = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet"
YUNET_FILES = {"face_detection_yunet_2023mar.onnx": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid(path: Path, expected: str) -> bool:
    return path.is_file() and sha256(path) == expected


def fetch(dest_dir: Path, files=FILES, base: str = BASE, retries: int = 3) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in files.items():
        target = dest_dir / name
        if is_valid(target, expected):
            print(f"ok       {name}")
            continue
        for attempt in range(1, retries + 1):
            print(f"fetching {name} (attempt {attempt})")
            tmp = target.with_suffix(target.suffix + ".part")
            try:
                urllib.request.urlretrieve(f"{base}/{name}", tmp)
            except OSError as exc:
                print(f"  failed: {exc}")
                continue
            if is_valid(tmp, expected):
                tmp.replace(target)
                break
            print("  checksum mismatch (truncated download?)")
        else:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"could not fetch a valid {name}")


if __name__ == "__main__":
    dest = Path(__file__).resolve().parents[1] / "models" / "passenger"
    fetch(dest)
    fetch(dest, YUNET_FILES, YUNET_BASE)
