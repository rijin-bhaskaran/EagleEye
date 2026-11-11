"""FairFace (Kärkkäinen & Joo, 2021): about 108k face crops with gender and age group labels.

    python -m passenger.datasets.fairface --data passenger_data

Licence: CC BY 4.0 (https://github.com/joojs/fairface). Used here for training a licence-clean face-attribute
model. Faces are 224 px crops with 0.25 padding, from Flickr, balanced across 7 race groups.

The parquet files (via the Hugging Face mirror ``HuggingFaceM4/FairFace``) are kept untouched under
``raw/external/fairface_hf_025/files/`` with their checksums; training reads them directly.
The official **validation** file is the benchmark and is never trained on.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import pyarrow.parquet as pq

from ..data_manager import DataManager
from .open_images import _http_get

DATASET_ID = "fairface_hf_025"
BASE = "https://huggingface.co/datasets/HuggingFaceM4/FairFace/resolve/main/0.25"
#: file -> expected size in bytes (checked after download; a cut-off download must not pass for the real file)
FILES: Dict[str, int] = {
    "train-00000-of-00002-d405faba4f4b9b85.parquet": 250030031,
    "train-00001-of-00002-dd3cb68164727418.parquet": 250217804,
    "validation-00000-of-00001-951dbd63c8724ee1.parquet": 63189799,
}
AGE_NAMES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
GENDER_NAMES = ["Male", "Female"]                       # label 0 = Male, 1 = Female
CITATION = "Kärkkäinen & Joo, FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age, WACV 2021 (CC BY 4.0)."

Fetch = Callable[[str, Path], None]


def files_dir(dm: DataManager) -> Path:
    return dm.external_raw / DATASET_ID / "files"


def split_files(dm: DataManager, split: str) -> List[Path]:
    """``train`` or ``validation`` parquet files."""
    return sorted(p for p in files_dir(dm).glob(f"{split}-*.parquet"))


def check_labels(path: Path) -> None:
    """The label order is baked into this project's metrics: refuse a file that disagrees."""
    import json
    meta = (pq.ParquetFile(path).schema_arrow.metadata or {}).get(b"huggingface", b"{}")
    feats = json.loads(meta).get("info", {}).get("features", {})
    if feats.get("age", {}).get("names") != AGE_NAMES or feats.get("gender", {}).get("names") != GENDER_NAMES:
        raise ValueError(f"{path.name}: unexpected FairFace label names; expected {AGE_NAMES} / {GENDER_NAMES}")


def iter_rows(path: Path, batch_size: int = 1024) -> Iterator[Tuple[bytes, int, int]]:
    """Yield ``(jpeg bytes, age group index, gender index)``."""
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["image", "age", "gender"]):
        d = batch.to_pydict()
        for img, age, gender in zip(d["image"], d["age"], d["gender"]):
            yield img["bytes"], int(age), int(gender)


def load_rows(paths: List[Path]) -> List[Tuple[bytes, int, int]]:
    return [row for p in paths for row in iter_rows(p)]


def ingest(dm: DataManager, fetch: Fetch = _http_get, sizes: Optional[Dict[str, int]] = None) -> int:
    """Download any missing file, check sizes and labels, and register everything with checksums."""
    sizes = FILES if sizes is None else sizes
    d = files_dir(dm)
    d.mkdir(parents=True, exist_ok=True)
    for name, size in sizes.items():
        path = d / name
        if not path.exists() or path.stat().st_size != size:
            path.unlink(missing_ok=True)
            fetch(f"{BASE}/{name}", path)
            if path.stat().st_size != size:
                raise OSError(f"{name}: got {path.stat().st_size} bytes, expected {size}")
    for name in sizes:
        check_labels(d / name)
    meta = {
        "source": "FairFace (Hugging Face mirror HuggingFaceM4/FairFace, padding 0.25)",
        "url": "https://github.com/joojs/fairface", "license": "CC BY 4.0", "citation": CITATION,
        "retrieved": time.strftime("%Y-%m-%d"), "age_groups": AGE_NAMES, "genders": GENDER_NAMES,
        "use": "face gender/age classifier; official validation file is the benchmark and is never trained on",
    }
    return dm.register_raw_files(DATASET_ID, meta, [f"files/{n}" for n in sizes])


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Register FairFace (downloads missing files first)")
    ap.add_argument("--data", default="passenger_data")
    args = ap.parse_args(argv)
    n = ingest(DataManager(args.data))
    print(f"[fairface] {n} files registered under raw/external/{DATASET_ID}/")


if __name__ == "__main__":
    main()
