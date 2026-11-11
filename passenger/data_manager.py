"""Single entry point for all passenger data.

Layout under ``root`` (default ``passenger_data/``)::

    raw/                                  ORIGINALS. Nothing in this class ever edits or deletes them;
                                          ``verify_raw`` proves they are intact (checksums).
      videos/<vehicle>/<id>.mp4             your own cabin footage
      external/<dataset_id>/                downloaded open datasets: images/, metadata/, SOURCE.json
    processed/                            derived, can be rebuilt from raw/
      frames/<video_id>/*.jpg               extracted frames, named <video_id>_<frame>.jpg
    annotations/
      labels/*.txt                          YOUR hand-made YOLO labels (same stem as the frame). Precious.
      external/<dataset_id>/labels/*.txt    labels converted from an open dataset (rebuildable from raw/)
      <export_name>/                        training sets built by ``export_yolo`` (+ data.yaml), rebuildable
    manifest.json                         index of videos, external datasets, frames, splits, provenance

Splits: a *video* is assigned as a whole (consecutive frames are nearly identical, so a frame-level
split would leak test data into training). External images are independent photos, so each gets a
stable split from a hash of its name: the same image always lands in the same split, whatever else is
added later, so a benchmark built from it never drifts.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

CLASSES = ("male", "female", "child")
SPLITS = ("train", "val", "test")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")      # a single path segment, never "." or ".."


@dataclass(frozen=True)
class Frame:
    """One labelled image, wherever it came from."""
    stem: str
    image: Path
    label: Path
    source: str            # "videos" or an external dataset id
    split: str


def _check_name(kind: str, name: str, reserved: Sequence[str] = ()) -> str:
    """Names become directory names, and some are later deleted, so they must not be paths."""
    if not _SAFE_NAME.match(name or "") or name in reserved:
        raise ValueError(f"invalid {kind} name {name!r}: use letters, digits, '_', '-' or '.' only")
    return name


class DataManager:
    def __init__(self, root: str | Path = "passenger_data", classes: Sequence[str] = CLASSES) -> None:
        self.root = Path(root)
        self.classes = tuple(classes)
        self.raw = self.root / "raw" / "videos"
        self.frames_dir = self.root / "processed" / "frames"
        self.labels_dir = self.root / "annotations" / "labels"
        self.external_raw = self.root / "raw" / "external"
        self.external_labels = self.root / "annotations" / "external"
        self._manifest_path = self.root / "manifest.json"
        for d in (self.raw, self.frames_dir, self.labels_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._m: Dict[str, dict] = (
            json.loads(self._manifest_path.read_text()) if self._manifest_path.exists() else {"videos": {}}
        )
        self._m.setdefault("external", {})
        self._m.setdefault("raw_files", {})

    # -- manifest -------------------------------------------------------
    def _save(self) -> None:
        tmp = self._manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._m, indent=2))
        for attempt in range(5):
            try:
                tmp.replace(self._manifest_path)
                return
            except PermissionError:                 # Windows: antivirus/indexer briefly holds the file
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))

    @property
    def videos(self) -> Dict[str, dict]:
        return self._m["videos"]

    @property
    def external(self) -> Dict[str, dict]:
        return self._m["external"]

    @property
    def raw_files(self) -> Dict[str, dict]:
        """Datasets that are just files (for example face-attribute parquet), not images with boxes."""
        return self._m["raw_files"]

    # -- 1. register ----------------------------------------------------
    def register_video(self, src: str | Path, vehicle: str = "unknown", group: Optional[str] = None,
                       rights: Optional[str] = None) -> str:
        """Copy ``src`` into raw/ and index it. Returns its id; re-registering the same file is a no-op.

        ``group`` names the people/scene the clip belongs to: clips of one group are always put in the same
        split (see ``assign_splits``). ``rights`` is a free-text licence note kept with the video."""
        src = Path(src)
        _check_name("vehicle", vehicle)
        if group is not None:
            _check_name("group", group)
        if not src.is_file():
            raise FileNotFoundError(src)
        video_id = _file_id(src)
        if video_id in self.videos:
            return video_id

        cap = cv2.VideoCapture(str(src))
        try:
            if not cap.isOpened():
                raise ValueError(f"cannot decode video: {src}")
            info = {
                "fps": cap.get(cv2.CAP_PROP_FPS) or 0.0,
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
        finally:
            cap.release()
        if info["frame_count"] <= 0 or info["fps"] <= 0:
            raise ValueError(f"video has no readable frames: {src}")

        dest_dir = self.raw / vehicle
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{video_id}{src.suffix.lower()}"
        shutil.copy2(src, dest)
        self.videos[video_id] = {
            "path": dest.relative_to(self.root).as_posix(),
            "vehicle": vehicle,
            "source_name": src.name,
            "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "split": None,
            "frames": [],
            **info,
        }
        if group is not None:
            self.videos[video_id]["group"] = group
        if rights is not None:
            self.videos[video_id]["rights"] = rights
        self._save()
        return video_id

    def set_group(self, video_id: str, group: str) -> None:
        """Record the group of an already registered video. Never changes its split."""
        _check_name("group", group)
        self.videos[video_id]["group"] = group
        self._save()

    # -- 2. extract -----------------------------------------------------
    def extract_frames(self, video_id: str, target_fps: float = 2.0, dedupe_threshold: float = 2.0) -> int:
        """Sample the video at ``target_fps`` and drop near-duplicate frames.

        ``dedupe_threshold`` is the mean absolute grey difference (0-255) on a 32x32
        thumbnail below which a frame counts as a repeat of the last kept one
        (an empty parked bus produces hundreds of identical frames). Re-running
        replaces the earlier extraction. Returns the number of frames kept.
        """
        meta = self.videos[video_id]
        step = max(meta["fps"] / target_fps, 1.0)
        out = self.frames_dir / video_id
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)

        cap = cv2.VideoCapture(str(self.root / meta["path"]))
        kept: List[int] = []
        last_thumb = None
        next_pick, idx = 0.0, 0
        try:
            while True:
                if idx < next_pick:
                    if not cap.grab():
                        break
                    idx += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                next_pick += step
                thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)
                if last_thumb is None or float(np.abs(thumb - last_thumb).mean()) >= dedupe_threshold:
                    cv2.imwrite(str(out / f"{video_id}_{idx:06d}.jpg"), frame)
                    kept.append(idx)
                    last_thumb = thumb
                idx += 1
        finally:
            cap.release()

        meta["frames"] = kept
        meta["extract_fps"] = target_fps
        self._save()
        return len(kept)

    # -- 3. splits ------------------------------------------------------
    def assign_splits(self, ratios: Sequence[float] = (0.7, 0.15, 0.15), seed: int = 0) -> Dict[str, List[str]]:
        """Assign whole videos (and whole groups) to train/val/test, deterministically.

        Only videos without a split are assigned; existing assignments never move. Otherwise adding
        footage would silently shuffle old test videos into training and invalidate every earlier
        metric. Videos of one ``group`` (same people or scene) always share a split: a pending member of a
        group that already has a split joins it, and an all-pending group is placed as one unit. Each new
        unit goes to the split furthest below its target share, and an empty split (with a non-zero ratio)
        is filled as soon as there are just enough units left to fill them.
        """
        if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError("ratios must be three non-negative numbers summing to 1")
        counts = {s: sum(m["split"] == s for m in self.videos.values()) for s in SPLITS}
        units: Dict[str, List[str]] = {}
        for v, m in sorted(self.videos.items()):
            if m["split"] is not None:
                continue
            g = m.get("group")
            known = {o["split"] for o in self.videos.values() if g and o.get("group") == g and o["split"]}
            if known:                                       # group already placed: follow it
                split = sorted(known)[0]
                m["split"] = split
                counts[split] += 1
            else:
                units.setdefault(g or f"video:{v}", []).append(v)
        pending = sorted(units)
        random.Random(seed).shuffle(pending)
        wanted = [s for s, r in zip(SPLITS, ratios) if r > 0]
        for left, key in zip(range(len(pending), 0, -1), pending):
            empty = [s for s in wanted if counts[s] == 0]
            total = sum(counts.values()) + len(units[key])
            pool = empty if empty and left <= len(empty) else wanted
            split = max(pool, key=lambda s: ratios[SPLITS.index(s)] * total - counts[s])
            for vid in units[key]:
                self.videos[vid]["split"] = split
            counts[split] += len(units[key])
        self._save()
        return {s: sorted(v for v, m in self.videos.items() if m["split"] == s) for s in SPLITS}

    def assign_group(self, group: str, split: str) -> List[str]:
        """Put every video of ``group`` into ``split``. Refuses to move anything already assigned elsewhere."""
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        members = sorted(v for v, m in self.videos.items() if m.get("group") == group)
        if not members:
            raise KeyError(group)
        if any(self.videos[v]["split"] not in (None, split) for v in members):
            raise ValueError(f"group {group} already has a member in another split; splits never move")
        for v in members:
            self.videos[v]["split"] = split
        self._save()
        return members

    # -- 4. labels ------------------------------------------------------
    def import_labels(self, src_dir: str | Path) -> int:
        """Copy YOLO ``*.txt`` files (e.g. a CVAT export) whose stem matches an extracted frame.

        If the export contains an ``obj.names`` file its class order must match ours exactly: a swapped
        order would silently turn every male into a female (or a child), so it is refused."""
        for names_file in Path(src_dir).rglob("obj.names"):
            names = tuple(line.strip() for line in names_file.read_text().splitlines() if line.strip())
            if names != self.classes:
                raise ValueError(f"class order in {names_file} is {list(names)} but this project uses "
                                 f"{list(self.classes)}: recreate the labels in CVAT in exactly that order")
        known = {p.stem for p in self.frames_dir.glob("*/*.jpg")}
        count = 0
        for txt in Path(src_dir).rglob("*.txt"):
            if txt.stem in known:
                shutil.copy2(txt, self.labels_dir / txt.name)
                count += 1
        return count

    def validate_labels(self) -> List[str]:
        """Return a list of problems (empty = clean): bad field count, class id, or out-of-range coords."""
        problems: List[str] = []
        label_files = [*self.labels_dir.glob("*.txt"), *self.external_labels.glob("*/labels/*.txt")]
        for txt in sorted(label_files):
            for n, line in enumerate(txt.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split()
                try:
                    cls = int(parts[0])
                    vals = [float(v) for v in parts[1:]]
                except ValueError:
                    problems.append(f"{txt.name}:{n} non-numeric field")
                    continue
                if len(parts) != 5:
                    problems.append(f"{txt.name}:{n} expected 5 fields, got {len(parts)}")
                elif not 0 <= cls < len(self.classes):
                    problems.append(f"{txt.name}:{n} class id {cls} out of range")
                elif any(not 0.0 <= v <= 1.0 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                    problems.append(f"{txt.name}:{n} coordinates out of range")
        return problems

    # -- 5. export ------------------------------------------------------
    def export_yolo(self, dataset: str = "passengers_v1", source: Optional[str] = None) -> Path:
        """Build ``annotations/<dataset>/`` (images, labels, data.yaml) from labelled frames in assigned splits.

        Your own footage and every registered external dataset are merged; ``source`` restricts it to one
        ("videos" or a dataset id). Only the originals in raw/ and the labels are read, so any export can
        be deleted and rebuilt."""
        _check_name("dataset", dataset, reserved=("labels",))      # this directory is deleted and rebuilt
        bad = self.validate_labels()
        if bad:
            raise ValueError(f"{len(bad)} invalid label lines, first: {bad[0]}")
        out = self.root / "annotations" / dataset
        if out.exists():
            shutil.rmtree(out)
        for fr in self.iter_frames(source=source):           # labelled frames only, in assigned splits
            for kind, src in (("images", fr.image), ("labels", fr.label)):
                d = out / kind / fr.split
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, d / f"{fr.stem}{src.suffix}")
        yaml_lines = [f"path: {out.resolve().as_posix()}", "train: images/train", "val: images/val", "test: images/test",
                      f"nc: {len(self.classes)}", "names:"]
        yaml_lines += [f"  {i}: {name}" for i, name in enumerate(self.classes)]
        out.mkdir(parents=True, exist_ok=True)
        (out / "data.yaml").write_text("\n".join(yaml_lines) + "\n")
        return out

    # -- external datasets ------------------------------------------------
    def add_external_dataset(self, dataset_id: str, meta: dict, items: Sequence[dict],
                             ratios: Sequence[float] = (0.7, 0.15, 0.15)) -> int:
        """Index an open dataset whose originals already sit under ``raw/external/<dataset_id>/``.

        ``meta`` records provenance (source URL, licence, attribution, retrieval date); each item is
        ``{"name", "image" (path under raw/external/<id>/), "boxes": [(class_id, cx, cy, w, h), ...], "context"}``.
        Labels are written to annotations/external/<id>/labels/ (derived data); raw files are only read
        and checksummed. Re-adding an id updates its items but never changes an existing item's split.
        Returns the number of items indexed.
        """
        _check_name("dataset id", dataset_id)
        raw_dir = self.external_raw / dataset_id
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"raw files must be in {raw_dir} first")
        if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError("ratios must be three numbers summing to 1")
        label_dir = self.external_labels / dataset_id / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        entry = self.external.setdefault(dataset_id, {"items": {}})
        entry.update({k: v for k, v in meta.items() if k != "items"})

        for it in items:
            name = _check_name("item", it["name"])
            image = raw_dir / it["image"]
            if not image.is_file():
                raise FileNotFoundError(image)
            lines = []
            for cls, cx, cy, w, h in it["boxes"]:
                if not (0 <= int(cls) < len(self.classes) and 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ValueError(f"{dataset_id}/{name}: bad box {(cls, cx, cy, w, h)}")
                lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            (label_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            old = entry["items"].get(name, {})
            entry["items"][name] = {
                "image": image.relative_to(self.root).as_posix(),
                "split": old.get("split") or _hash_split(f"{dataset_id}/{name}", ratios),
                "context": it.get("context", ""),
                "sha256": _sha256(image),
            }
        self._save()
        return len(items)

    def register_raw_files(self, dataset_id: str, meta: dict, files: Sequence[str]) -> int:
        """Record provenance and checksums for originals that sit under ``raw/external/<dataset_id>/``.

        For datasets that are not images-with-boxes (so they have no frames to index), for example the
        FairFace face-attribute files. They get the same guarantee as everything else in raw/: ``verify_raw``
        re-checks them. Returns the number of files registered."""
        _check_name("dataset id", dataset_id)
        raw_dir = self.external_raw / dataset_id
        entry = {k: v for k, v in meta.items() if k != "files"}
        entry["files"] = {}
        for rel in files:
            path = raw_dir / rel
            if not path.is_file():
                raise FileNotFoundError(path)
            entry["files"][rel] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
        self.raw_files[dataset_id] = entry
        self._save()
        return len(entry["files"])

    def iter_frames(self, split: Optional[str] = None, source: Optional[str] = None) -> Iterator[Frame]:
        """Every labelled frame in an assigned split, from your footage and from external datasets."""
        if source in (None, "videos"):
            for vid, meta in sorted(self.videos.items()):
                if meta["split"] is None or split not in (None, meta["split"]):
                    continue
                for idx in meta["frames"]:
                    stem = f"{vid}_{idx:06d}"
                    label, image = self.labels_dir / f"{stem}.txt", self.frames_dir / vid / f"{stem}.jpg"
                    if label.exists() and image.exists():
                        yield Frame(stem, image, label, "videos", meta["split"])
        for ds, entry in sorted(self.external.items()):
            if source not in (None, ds):
                continue
            for name, item in sorted(entry["items"].items()):
                if split not in (None, item["split"]):
                    continue
                label = self.external_labels / ds / "labels" / f"{name}.txt"
                image = self.root / item["image"]
                if label.exists() and image.exists():
                    yield Frame(f"{ds}__{name}", image, label, ds, item["split"])

    def verify_raw(self) -> List[str]:
        """Check that every original is still present and byte-identical to when it was registered.

        Returns a list of problems (empty = intact). Run it before trusting any result and after copying
        the data anywhere: raw/ is the one thing that cannot be regenerated."""
        problems: List[str] = []
        for vid, meta in self.videos.items():
            path = self.root / meta["path"]
            if not path.is_file():
                problems.append(f"video {vid}: missing {meta['path']}")
            elif _file_id(path) != vid:
                problems.append(f"video {vid}: content changed (checksum mismatch)")
        for ds, entry in self.external.items():
            for name, item in entry["items"].items():
                path = self.root / item["image"]
                if not path.is_file():
                    problems.append(f"{ds}/{name}: missing {item['image']}")
                elif _sha256(path) != item["sha256"]:
                    problems.append(f"{ds}/{name}: content changed (checksum mismatch)")
        for ds, entry in self.raw_files.items():
            for rel, info in entry["files"].items():
                path = self.external_raw / ds / rel
                if not path.is_file():
                    problems.append(f"{ds}/{rel}: missing")
                elif path.stat().st_size != info["bytes"] or _sha256(path) != info["sha256"]:
                    problems.append(f"{ds}/{rel}: content changed (checksum mismatch)")
        return problems

    def fingerprint(self, split: str, source: Optional[str] = None) -> str:
        """Identity of a split: changes if any image or label in it is added, removed or edited.
        Two evaluations are only comparable when this is equal."""
        h = hashlib.sha256()
        for fr in self.iter_frames(split, source):
            h.update(fr.stem.encode())
            h.update(_sha256(fr.label).encode())
        return h.hexdigest()[:16]

    # -- 6. stats -------------------------------------------------------
    def stats(self) -> dict:
        """Frames and per-class box counts per split. Check class balance here before trusting any metric."""
        per_split = {s: {"videos": 0, "frames": 0, "labelled": 0, "boxes": Counter()} for s in (*SPLITS, None)}
        for vid, meta in self.videos.items():
            s = per_split[meta["split"]]
            s["videos"] += 1
            for idx in meta["frames"]:
                s["frames"] += 1
                label = self.labels_dir / f"{vid}_{idx:06d}.txt"
                if label.exists():
                    s["labelled"] += 1
                    for line in label.read_text().splitlines():
                        if line.strip():
                            cid = int(float(line.split()[0]))
                            s["boxes"][self.classes[cid] if 0 <= cid < len(self.classes) else "invalid"] += 1
        out = {("unassigned" if k is None else k): {**v, "boxes": dict(v["boxes"])} for k, v in per_split.items()}
        ext: Dict[str, dict] = {}
        for ds, entry in self.external.items():
            per = {s: {"images": 0, "boxes": Counter()} for s in SPLITS}
            for name, item in entry["items"].items():
                label = self.external_labels / ds / "labels" / f"{name}.txt"
                per[item["split"]]["images"] += 1
                for line in (label.read_text().splitlines() if label.exists() else []):
                    if line.strip():
                        cid = int(float(line.split()[0]))
                        per[item["split"]]["boxes"][self.classes[cid] if 0 <= cid < len(self.classes) else "invalid"] += 1
            ext[ds] = {s: {"images": v["images"], "boxes": dict(v["boxes"])} for s, v in per.items()}
        out["external"] = ext
        return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line for the whole annotation workflow: see docs/ANNOTATION.md."""
    import argparse

    ap = argparse.ArgumentParser(prog="python -m passenger.data_manager", description="Manage passenger_data")
    ap.add_argument("--data", default="passenger_data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register", help="copy a cabin video into raw/ and index it")
    r.add_argument("video")
    r.add_argument("--vehicle", default="unknown")
    r.add_argument("--group", default=None, help="people/scene group: clips of one group share a split")
    r.add_argument("--rights", default=None, help="licence note kept with the video")
    e = sub.add_parser("extract", help="extract frames to label from a registered video")
    e.add_argument("video_id")
    e.add_argument("--fps", type=float, default=2.0)
    sub.add_parser("splits", help="assign train/val/test to videos that have none")
    i = sub.add_parser("import-labels", help="import a CVAT 'YOLO 1.1' export")
    i.add_argument("folder")
    sub.add_parser("validate", help="check every label file")
    x = sub.add_parser("export", help="build a YOLO training set")
    x.add_argument("name")
    x.add_argument("--source", default=None)
    sub.add_parser("stats", help="frames and boxes per split and class")
    sub.add_parser("verify", help="check that every original in raw/ is intact")
    args = ap.parse_args(argv)

    dm = DataManager(args.data)
    if args.cmd == "register":
        vid = dm.register_video(args.video, args.vehicle, args.group, args.rights)
        print(f"registered {args.video} as {vid}")
    elif args.cmd == "extract":
        n = dm.extract_frames(args.video_id, args.fps)
        print(f"{n} frames in {dm.frames_dir / args.video_id}  (upload these to CVAT)")
    elif args.cmd == "splits":
        print(json.dumps({s: len(v) for s, v in dm.assign_splits().items()}))
    elif args.cmd == "import-labels":
        print(f"imported {dm.import_labels(args.folder)} label files")
    elif args.cmd == "validate":
        problems = dm.validate_labels()
        print("\n".join(problems[:20]) or "labels OK")
        return 1 if problems else 0
    elif args.cmd == "export":
        print(f"exported to {dm.export_yolo(args.name, args.source)}")
    elif args.cmd == "stats":
        print(json.dumps(dm.stats(), indent=2))
    elif args.cmd == "verify":
        problems = dm.verify_raw()
        print("\n".join(problems) or "raw data intact")
        return 1 if problems else 0
    return 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_split(key: str, ratios: Sequence[float]) -> str:
    """Stable split from a hash of ``key``: independent of what else exists, so it never drifts."""
    bucket = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0x100000000
    return SPLITS[0] if bucket < ratios[0] else SPLITS[1] if bucket < ratios[0] + ratios[1] else SPLITS[2]


def _file_id(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


if __name__ == "__main__":
    raise SystemExit(main())
