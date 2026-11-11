"""Car-cabin videos -> frames -> pseudo-labels -> human review -> a YOLO dataset for ``passenger.datasets.car_cabin``.

    python -m passenger.car_autolabel run --videos <dir> --run first            # frames + auto-labels + review sheets
    python -m passenger.car_autolabel sheets --run first --which queue          # (re)draw the review sheets
    python -m passenger.car_autolabel apply  --run first corrections.csv        # fix labels from a CSV
    python -m passenger.car_autolabel export --run first --out <dir>            # reviewed + high-confidence frames only
    python -m passenger.car_autolabel stats  --run first

These are PSEUDO-LABELS, not ground truth. Every box says where it came from (``source``) and how sure that source was.
Nothing is guessed: a person whose face cannot be read, a belt that cannot be judged, a child seat the detector is unsure
about all go to the review queue (the box has ``cls: null``), stay out of the label files, and keep their frame out of the
export until a human resolves them (``apply``). The importer treats a missing box as background, so an unresolved box in an
exported frame would teach the model that a real person is not one.

Output (never inside the repo, default ``C:/Users/RijinBhaskaran/Downloads/car_open_data/autolabel/<run>/``):
``images/<videoid>_<frameno>.jpg``  frame names keep the video id first, so the importer's group-by-stem logic puts every
frame of one video in one split; ``labels/*.txt`` YOLO, normalised, ids from ``CAR_CLASSES``; ``review/`` contact sheets;
``manifest.json`` per frame (source video, licence, status auto|reviewed|rejected, per box class/confidence/source);
``data.yaml``.

Box semantics (what the importer and ``car_logic`` expect): the occupant boxes are the whole visible person, ``seatbelt`` is
the strap itself, ``no_seatbelt`` is the occupant's torso, ``child_seat`` is the seat.

Downloaded video is untrusted: file names become frame ids only after being reduced to ``[A-Za-z0-9_-]``, and the only
deletion is inside the run directory (``--force`` clears the run's own images/labels/review/manifest, nothing else).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .car_logic import CAR_CLASSES, PERSON_CLASSES, assign_items, area, inside_fraction

DEFAULT_ROOT = Path(r"C:\Users\RijinBhaskaran\Downloads\car_open_data")
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
STATUSES = ("auto", "reviewed", "rejected")
_RUN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,60}$")
DEV_ONLY = "DEV-ONLY"                  # licence prefix for tooling-test footage: such frames can never be exported
_THUMB = (32, 18)                      # frames are compared at this size to drop near-duplicates

# Queue reasons (why a box has no class yet)
Q_UNKNOWN_PERSON = "person_unreadable"         # no usable face / low gender confidence
Q_BELT_UNSURE = "belt_unsure"
Q_SEAT_UNSURE = "child_seat_unsure"

Box = Tuple[float, float, float, float]


class AutolabelError(ValueError):
    """Bad input the user can fix (unsafe name, unreadable file, bad CSV row)."""


# -- names and paths ------------------------------------------------------------
def safe_id(name: str, taken: Iterable[str] = (), limit: int = 48) -> str:
    """A file-name-safe id from a video name: only ``[A-Za-z0-9_-]``, never empty, unique among ``taken``.

    Videos come from the internet; their names must not be able to carry ``..``, separators or drive letters into a path."""
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(str(name)).stem).strip("_-")[:limit].strip("_-") or "video"
    taken = set(taken)
    if s in taken:
        s = f"{s[:limit - 7]}_{hashlib.sha1(str(name).encode('utf-8', 'replace')).hexdigest()[:6]}"
        n = 2
        while s in taken:
            s, n = f"{s}{n}", n + 1
    return s


def frame_id(video_id: str, frameno: int) -> str:
    return f"{video_id}_{int(frameno):06d}"


def run_path(root: Path, run: str) -> Path:
    """``<root>/autolabel/<run>``; the run name is a single safe component, so the result cannot leave ``<root>/autolabel``."""
    if not isinstance(run, str) or not _RUN.match(run) or ".." in run:
        raise AutolabelError(f"run name {run!r} must be letters, digits, '_', '-', '.' (no separators, no '..')")
    base = (Path(root) / "autolabel").resolve()
    p = (base / run).resolve()
    if p.parent != base:
        raise AutolabelError(f"run directory {p} is outside {base}")
    return p


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def reset_run(run_dir: Path) -> None:
    """Clear a run's own outputs. Only the known sub-paths of ``run_dir`` are touched, and only if they really lie inside it."""
    for sub in ("images", "labels", "review"):
        p = run_dir / sub
        if p.is_symlink():
            p.unlink()
        elif p.is_dir() and _inside(p, run_dir):
            shutil.rmtree(p)
    for f in ("manifest.json", "data.yaml"):
        p = run_dir / f
        if p.is_file() or p.is_symlink():
            p.unlink()


# -- licence table ----------------------------------------------------------------
def read_licenses(manifest_csv: Optional[Path]) -> Dict[str, dict]:
    """``MANIFEST.csv`` -> {file stem: {license, url}}. Column names are matched loosely; a missing file gives {}."""
    out: Dict[str, dict] = {}
    if manifest_csv is None or not Path(manifest_csv).is_file():
        return out
    with open(manifest_csv, newline="", encoding="utf-8-sig", errors="replace") as fh:
        for row in csv.DictReader(fh):
            low = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            fname = next((low[k] for k in ("file", "filename", "file_name", "video", "path", "local_path") if low.get(k)), "")
            if not fname:
                continue
            lic = next((low[k] for k in ("license", "licence", "license_name", "licence_name") if low.get(k)), "")
            url = next((low[k] for k in ("url", "source_url", "source url", "page", "page_url", "source") if low.get(k)), "")
            out[Path(fname.replace("\\", "/")).stem.lower()] = {"license": lic or "unknown", "url": url}
    return out


# -- frames ---------------------------------------------------------------------
def _thumb(img: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), _THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)


def is_duplicate(prev: Optional[np.ndarray], cur: np.ndarray, threshold: float) -> bool:
    """Mean absolute grey difference (0-255) of two thumbnails below ``threshold``: the cabin has not changed."""
    return prev is not None and float(np.abs(prev - cur).mean()) < threshold


def extract_frames(video: Path, out_dir: Path, vid: str, *, fps: float = 1.5, max_frames: int = 200,
                   dedup: float = 2.0, max_side: int = 1280, jpeg_quality: int = 92) -> List[dict]:
    """Sample ``video`` at about ``fps`` frames per second, drop near-duplicates, save ``<vid>_<frameno>.jpg`` in ``out_dir``.

    Returns one dict per saved frame (``id``, ``frameno``, ``file``, ``width``, ``height``). Raises ``AutolabelError`` when the
    file cannot be opened; a video that opens but yields nothing returns ``[]``. Frames larger than ``max_side`` are shrunk."""
    if fps <= 0 or max_frames <= 0:
        raise AutolabelError("fps and max_frames must be positive")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise AutolabelError(f"cannot open video {video}")
    try:
        native = cap.get(cv2.CAP_PROP_FPS)
        native = native if native and native > 1 and native < 1000 else 25.0
        step = max(int(round(native / fps)), 1)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: List[dict] = []
        prev = None
        n = 0
        while len(saved) < max_frames:
            ok = cap.grab()
            if not ok:
                break
            if n % step == 0:
                ok, img = cap.retrieve()
                if ok and img is not None and img.size:
                    t = _thumb(img)
                    if not is_duplicate(prev, t, dedup):
                        prev = t
                        h, w = img.shape[:2]
                        if max(h, w) > max_side:
                            s = max_side / max(h, w)
                            img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
                            h, w = img.shape[:2]
                        fid = frame_id(vid, n)
                        path = out_dir / f"{fid}.jpg"
                        if cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                            saved.append({"id": fid, "frameno": n, "file": path.name, "width": w, "height": h})
            n += 1
        return saved
    finally:
        cap.release()


def find_videos(src: Path) -> List[Path]:
    """Video files in ``src`` (a directory, searched recursively, or a single file), sorted."""
    src = Path(src)
    if src.is_file():
        return [src] if src.suffix.lower() in VIDEO_EXT else []
    if not src.is_dir():
        raise AutolabelError(f"{src} is not a file or directory")
    return sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT)


# -- boxes, labels, manifest ------------------------------------------------------
def new_box(kind: str, box: Sequence[float], cls: Optional[str], conf: float, source: str, queue: Optional[str] = None,
            note: str = "") -> dict:
    """One manifest box. ``cls`` None = unresolved (in the review queue, not written to the label file)."""
    if cls is not None and cls not in CAR_CLASSES:
        raise AutolabelError(f"unknown class {cls!r}")
    x1, y1, x2, y2 = (min(max(float(v), 0.0), 1.0) for v in box)
    return {"kind": kind, "cls": cls, "box": [x1, y1, x2, y2], "conf": round(float(conf), 3), "source": source,
            "queue": queue, "note": note, "deleted": False}


def live_boxes(frame: dict) -> List[Tuple[int, dict]]:
    """(index, box) of every box that has not been deleted. Indices never shift: deleting leaves a tombstone."""
    return [(i, b) for i, b in enumerate(frame["boxes"]) if not b.get("deleted")]


def unresolved(frame: dict) -> List[Tuple[int, dict]]:
    return [(i, b) for i, b in live_boxes(frame) if b["cls"] is None]


def yolo_lines(frame: dict) -> List[str]:
    """YOLO rows (``cls cx cy w h``, normalised) for the resolved, non-degenerate boxes of a frame."""
    rows = []
    for _, b in live_boxes(frame):
        if b["cls"] is None:
            continue
        x1, y1, x2, y2 = b["box"]
        if x2 - x1 <= 1e-4 or y2 - y1 <= 1e-4:
            continue
        rows.append(f"{CAR_CLASSES.index(b['cls'])} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    return rows


def write_label(run_dir: Path, fid: str, frame: dict) -> None:
    (run_dir / "labels").mkdir(parents=True, exist_ok=True)
    rows = yolo_lines(frame)
    (run_dir / "labels" / f"{fid}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def data_yaml_text() -> str:
    return f"path: .\ntrain: images\nval: images\nnc: {len(CAR_CLASSES)}\nnames: {list(CAR_CLASSES)}\n"


def load_manifest(run_dir: Path) -> dict:
    p = run_dir / "manifest.json"
    if not p.is_file():
        raise AutolabelError(f"{p} not found: run `python -m passenger.car_autolabel run ...` first")
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AutolabelError(f"{p} is not valid JSON: {exc}") from exc
    if not isinstance(m, dict) or not isinstance(m.get("frames"), dict):
        raise AutolabelError(f"{p} has no 'frames'")
    return m


def save_manifest(run_dir: Path, m: dict) -> None:
    tmp = run_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(m, indent=1), encoding="utf-8")
    tmp.replace(run_dir / "manifest.json")


def frame_needs_review(frame: dict) -> bool:
    return frame["status"] == "auto" and bool(unresolved(frame))


# -- review sheets --------------------------------------------------------------
_COLORS = {"male": (255, 140, 0), "female": (180, 0, 255), "child": (0, 200, 255), "seatbelt": (0, 220, 0),
           "no_seatbelt": (0, 0, 255), "child_seat": (0, 140, 255), None: (0, 255, 255)}


def _text(img: np.ndarray, s: str, org: Tuple[int, int], color, scale: float) -> None:
    th = max(int(round(scale * 2)), 1)
    (w, h), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = org
    x, y = max(min(x, img.shape[1] - w - 2), 0), max(y, h + 2)
    cv2.rectangle(img, (x, y - h - 3), (x + w + 4, y + base), (0, 0, 0), -1)
    cv2.putText(img, s, (x + 2, y - 1), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def draw_frame(img: np.ndarray, frame: dict, width: int = 640) -> np.ndarray:
    """Copy of ``img`` scaled to ``width`` with each live box drawn and captioned ``index class conf source``."""
    h, w = img.shape[:2]
    s = width / w
    out = cv2.resize(img, (width, int(round(h * s))), interpolation=cv2.INTER_AREA)
    H, W = out.shape[:2]
    scale = max(W / 1100.0, 0.45)
    for i, b in live_boxes(frame):
        x1, y1, x2, y2 = (int(b["box"][0] * W), int(b["box"][1] * H), int(b["box"][2] * W), int(b["box"][3] * H))
        col = _COLORS.get(b["cls"], _COLORS[None])
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2 if b["cls"] else 1)
        name = b["cls"] or f"? {b['queue'] or ''}"
        _text(out, f"{i} {name} {b['conf']:.2f}", (x1, y1 + int(18 * scale) + 2), col, scale)
    return out


def contact_sheet(tiles: Sequence[np.ndarray], cols: int, caption: Sequence[str]) -> np.ndarray:
    """Tiles (any size) on a grid, each with a caption bar above it. Empty ``tiles`` gives a small blank sheet."""
    if not tiles:
        return np.zeros((40, 200, 3), np.uint8)
    tw = max(t.shape[1] for t in tiles)
    th = max(t.shape[0] for t in tiles)
    bar = 26
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * (th + bar), cols * tw, 3), 30, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        y, x = r * (th + bar), c * tw
        sheet[y + bar:y + bar + t.shape[0], x:x + t.shape[1]] = t
        cv2.putText(sheet, caption[k][:70], (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return sheet


def make_sheets(run_dir: Path, which: str = "queue", per_sheet: int = 6, cols: int = 3, width: int = 640) -> List[str]:
    """Write ``review/sheet_NNN.jpg`` (several drawn frames per image) and ``review/index.json``; returns the sheet names.

    ``which``: ``queue`` frames with unresolved boxes, ``auto`` every still-unreviewed frame, ``all`` every frame."""
    if which not in ("queue", "auto", "all"):
        raise AutolabelError("which must be queue, auto or all")
    if per_sheet < 1 or cols < 1:
        raise AutolabelError("per_sheet and cols must be at least 1")
    m = load_manifest(run_dir)
    rev = run_dir / "review"
    rev.mkdir(exist_ok=True)
    for old in rev.glob("sheet_*.jpg"):
        old.unlink()
    pick = [(fid, f) for fid, f in sorted(m["frames"].items())
            if (which == "all" or (which == "auto" and f["status"] == "auto") or (which == "queue" and frame_needs_review(f)))]
    names, index = [], {}
    for s in range(0, len(pick), per_sheet):
        chunk = pick[s:s + per_sheet]
        tiles, caps = [], []
        for fid, f in chunk:
            img = cv2.imread(str(run_dir / "images" / f["file"]))
            if img is None:
                continue
            tiles.append(draw_frame(img, f, width))
            caps.append(f"{fid} [{f['status']}]")
        name = f"sheet_{s // per_sheet + 1:03d}.jpg"
        cv2.imwrite(str(rev / name), contact_sheet(tiles, cols, caps), [cv2.IMWRITE_JPEG_QUALITY, 90])
        names.append(name)
        index[name] = [fid for fid, _ in chunk]
    (rev / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return names


def box_sheet(run_dir: Path, items: Sequence[Tuple[str, int]], out_path: Path, cols: int = 4, size: int = 300) -> None:
    """Crops of individual boxes (frame id, box index) with their label, for judging box-level precision."""
    m = load_manifest(run_dir)
    tiles, caps = [], []
    for k, (fid, bi) in enumerate(items):
        f = m["frames"][fid]
        b = f["boxes"][bi]
        img = cv2.imread(str(run_dir / "images" / f["file"]))
        if img is None:
            continue
        H, W = img.shape[:2]
        x1, y1, x2, y2 = b["box"]
        px = int(0.15 * (x2 - x1) * W) + 4
        py = int(0.15 * (y2 - y1) * H) + 4
        crop = img[max(int(y1 * H) - py, 0):min(int(y2 * H) + py, H), max(int(x1 * W) - px, 0):min(int(x2 * W) + px, W)]
        if crop.size == 0:
            continue
        s = size / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(int(crop.shape[1] * s), 1), max(int(crop.shape[0] * s), 1)))
        tile = np.zeros((size, size, 3), np.uint8)
        tile[:crop.shape[0], :crop.shape[1]] = crop
        tiles.append(tile)
        caps.append(f"#{k} {b['cls'] or '?'} {b['conf']:.2f}")
    cv2.imwrite(str(out_path), contact_sheet(tiles, cols, caps), [cv2.IMWRITE_JPEG_QUALITY, 90])


# -- corrections ----------------------------------------------------------------
def apply_corrections(run_dir: Path, csv_path: Path) -> dict:
    """Apply a corrections CSV and mark the touched frames ``reviewed``.

    Columns: ``frame, box_index, new_class[, x1, y1, x2, y2]``.
      * ``new_class`` one of the six classes: relabel box ``box_index``. A box that is a person can become male/female/child
        (the queued ``?`` boxes are resolved this way); the same for seatbelt / no_seatbelt / child_seat.
      * ``delete``: remove the box. ``keep``: accept an unresolved-free frame as correct (marks it reviewed, no change).
      * ``add`` (``box_index`` blank or ``add``) with ``new_class`` and normalised x1..y2: add a box.
      * ``reject`` in ``new_class`` (any ``box_index``): drop the whole frame from the dataset.
    All rows are checked before anything is changed: one bad row aborts the file and nothing is written."""
    m = load_manifest(run_dir)
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        if rd.fieldnames is None or not {"frame", "box_index", "new_class"} <= {c.strip().lower() for c in rd.fieldnames}:
            raise AutolabelError("corrections CSV needs the columns frame, box_index, new_class")
        rows = [{(k or "").strip().lower(): (v or "").strip() for k, v in r.items()} for r in rd]

    plan = []
    for n, r in enumerate(rows, start=2):
        where = f"{csv_path} line {n}"
        fid, cmd = r["frame"], r["new_class"].lower()
        if fid not in m["frames"]:
            raise AutolabelError(f"{where}: unknown frame {fid!r}")
        f = m["frames"][fid]
        idx = r["box_index"]
        if cmd in ("reject", "keep") or idx.lower() in ("", "add"):
            if cmd not in CAR_CLASSES + ("reject", "keep"):
                raise AutolabelError(f"{where}: new_class {cmd!r} is not a class, delete, keep or reject")
            if cmd in CAR_CLASSES:
                try:
                    box = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
                except (KeyError, ValueError):
                    raise AutolabelError(f"{where}: adding a {cmd} box needs numeric x1,y1,x2,y2 (normalised 0-1)") from None
                if not (all(0.0 <= v <= 1.0 for v in box) and box[2] > box[0] and box[3] > box[1]):
                    raise AutolabelError(f"{where}: box {box} must be inside 0-1 with x2>x1, y2>y1")
                plan.append((fid, "add", cmd, box))
            else:
                plan.append((fid, cmd, None, None))
            continue
        try:
            i = int(idx)
        except ValueError:
            raise AutolabelError(f"{where}: box_index {idx!r} is not a number") from None
        if not 0 <= i < len(f["boxes"]) or f["boxes"][i].get("deleted"):
            raise AutolabelError(f"{where}: frame {fid} has no live box {i}")
        if cmd == "delete":
            plan.append((fid, "delete", i, None))
        elif cmd in CAR_CLASSES:
            plan.append((fid, "relabel", i, cmd))
        else:
            raise AutolabelError(f"{where}: new_class {cmd!r} is not a class, delete, keep or reject")

    touched = set()
    counts = {"relabel": 0, "delete": 0, "add": 0, "reject": 0, "keep": 0}
    for fid, op, a, b in plan:
        f = m["frames"][fid]
        touched.add(fid)
        counts[op] += 1
        if op == "relabel":
            bx = f["boxes"][a]
            bx.update(cls=b, conf=1.0, source="human", queue=None)
        elif op == "delete":
            f["boxes"][a]["deleted"] = True
        elif op == "add":
            f["boxes"].append(new_box(_kind_of(a), b, a, 1.0, "human"))
        elif op == "reject":
            f["status"], f["reject_reason"] = "rejected", "human"
    for fid in touched:
        f = m["frames"][fid]
        if f["status"] != "rejected":
            f["status"] = "reviewed"
        write_label(run_dir, fid, f)
    save_manifest(run_dir, m)
    return {"frames_touched": len(touched), **counts}


def _kind_of(cls: str) -> str:
    return "person" if cls in PERSON_CLASSES else "child_seat" if cls == "child_seat" else "belt"


# -- export ---------------------------------------------------------------------
def export_dataset(run_dir: Path, out_dir: Path, *, min_auto_conf: float = 0.8, include_auto: bool = True,
                   allow_unknown_license: bool = False) -> dict:
    """Write a YOLO dataset (``images/``, ``labels/``, ``data.yaml``, ``provenance.json``) of trustworthy frames only.

    Included: ``reviewed`` frames with every live box resolved, and (``include_auto``) ``auto`` frames with no unresolved box
    where every box has confidence >= ``min_auto_conf``. Everything else is counted by reason in the result. ``out_dir`` must
    be new or empty, and may not be the run dir or inside it. Frames whose licence is unknown are skipped unless
    ``allow_unknown_license``; frames marked ``DEV-ONLY`` (tooling tests on stock footage) are never exported."""
    m = load_manifest(run_dir)
    out_dir = Path(out_dir)
    if _inside(out_dir, run_dir) or _inside(run_dir, out_dir):
        raise AutolabelError("--out must not be the run directory or inside it")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise AutolabelError(f"{out_dir} is not empty: export into a new folder")
    skipped: Dict[str, int] = {}
    keep: List[Tuple[str, dict, List[str]]] = []
    for fid, f in sorted(m["frames"].items()):
        why = None
        lic = str(f.get("license", "unknown"))
        if lic.upper().startswith(DEV_ONLY):
            why = "dev_only_license"
        elif f["status"] == "rejected":
            why = "rejected"
        elif lic.strip().lower() in ("", "unknown") and not allow_unknown_license:
            why = "unknown_license"
        elif unresolved(f):
            why = "unresolved_boxes"
        elif f["status"] == "auto" and not include_auto:
            why = "auto_not_included"
        elif f["status"] == "auto" and any(b["conf"] < min_auto_conf for _, b in live_boxes(f)):
            why = "low_confidence"
        rows = yolo_lines(f)
        if why is None and not rows:
            why = "no_boxes"
        if why:
            skipped[why] = skipped.get(why, 0) + 1
        else:
            keep.append((fid, f, rows))
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    prov, boxes = {}, {c: 0 for c in CAR_CLASSES}
    for fid, f, rows in keep:
        shutil.copy2(run_dir / "images" / f["file"], out_dir / "images" / f["file"])
        (out_dir / "labels" / f"{fid}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
        for _, b in live_boxes(f):
            boxes[b["cls"]] += 1
        prov[fid] = {"video": f["video"], "license": f["license"], "url": f.get("url", ""), "status": f["status"],
                     "label_sources": sorted({b["source"] for _, b in live_boxes(f)})}
    (out_dir / "data.yaml").write_text(data_yaml_text(), encoding="utf-8")
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    licenses = sorted({p["license"] for p in prov.values()})
    return {"frames": len(keep), "boxes": boxes, "skipped": skipped, "licenses": licenses}


def run_stats(run_dir: Path) -> dict:
    m = load_manifest(run_dir)
    fr = m["frames"]
    st = {s: 0 for s in STATUSES}
    boxes = {c: 0 for c in CAR_CLASSES}
    queue: Dict[str, int] = {}
    in_queue = 0
    for f in fr.values():
        st[f["status"]] = st.get(f["status"], 0) + 1
        if frame_needs_review(f):
            in_queue += 1
        for _, b in live_boxes(f):
            if b["cls"]:
                boxes[b["cls"]] += 1
            else:
                queue[b["queue"] or "unknown"] = queue.get(b["queue"] or "unknown", 0) + 1
    n = len(fr)
    return {"frames": n, "status": st, "frames_in_review_queue": in_queue,
            "review_queue_share": round(in_queue / n, 3) if n else 0.0, "boxes": boxes, "unresolved_boxes_by_reason": queue,
            "videos": len(m.get("videos", {}))}


# -- labelling stages -----------------------------------------------------------
def iou(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = w * h if (w > 0 and h > 0) else 0.0
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def refine_people(dets: Sequence[Tuple[Box, float]], *, min_area: float = 0.004, nms_iou: float = 0.6,
                  contain: float = 0.9, max_aspect: float = 2.5) -> List[Tuple[Box, float]]:
    """Clean raw detections of a cabin frame (boxes normalised, ``(box, conf)`` pairs).

    Cabin boxes are small, overlap each other and are cut off by seats, so beyond plain NMS this drops: boxes that are tiny
    (``min_area`` of the frame), boxes far wider than tall (a limb, ``max_aspect``), and a weaker box that lies almost
    entirely (``contain``) inside a stronger one (a second box on the same person's torso)."""
    ok = []
    for box, conf in dets:
        x1, y1, x2, y2 = (min(max(float(v), 0.0), 1.0) for v in box)
        b = (x1, y1, x2, y2)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0 or area(b) < min_area or w / h > max_aspect:
            continue
        ok.append((b, float(conf)))
    ok.sort(key=lambda d: -d[1])
    kept: List[Tuple[Box, float]] = []
    for b, c in ok:
        if any(iou(b, k) > nms_iou or inside_fraction(b, k) >= contain for k, _ in kept):
            continue
        kept.append((b, c))
    return kept


def strap_evidence(crop_bgr: np.ndarray) -> Tuple[float, Optional[Box]]:
    """Look for a seat-belt strap in a person crop: ``(score 0-1, strap box normalised to the crop)``.

    A strap is two long, nearly parallel edges 2-16% of the person's width apart that run diagonally from the upper to the
    lower half of the person, with a band between them whose grey level differs from both sides. Score = the shorter edge's
    length over the crop height. MEASURED to be weak (docs/CAR_MODEL.md): plaid shirts, headrests and door frames pass it,
    dark belts on dark clothes do not. It is a hint for the reviewer, not a verdict."""
    import math

    h, w = crop_bgr.shape[:2]
    if h < 80 or w < 40:
        return 0.0, None
    scale = 240.0 / h
    im = cv2.resize(crop_bgr, (max(int(w * scale), 1), 240))
    h2, w2 = im.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = np.zeros_like(g)
    band = slice(int(0.18 * h2), int(0.95 * h2))
    edges[band] = cv2.Canny(g, 40, 110)[band]
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=int(0.22 * h2), maxLineGap=int(0.05 * h2))
    if lines is None:
        return 0.0, None
    segs = []
    for x1, y1, x2, y2 in lines[:, 0]:
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        if 28 <= ang <= 80 or 100 <= ang <= 152:
            segs.append((float(x1), float(y1), float(x2), float(y2), ang))
    best, pair = 0.0, None
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            a, b = segs[i], segs[j]
            d = abs(a[4] - b[4])
            if min(d, 180 - d) > 9:
                continue
            dx, dy = a[2] - a[0], a[3] - a[1]
            n = math.hypot(dx, dy)
            mx, my = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            dist = abs(dy * (mx - a[0]) - dx * (my - a[1])) / n
            if not 0.02 * w2 <= dist <= 0.16 * w2:
                continue
            sc = min(n, math.hypot(b[2] - b[0], b[3] - b[1])) / h2
            ys = [a[1], a[3], b[1], b[3]]
            if sc <= best or min(ys) / h2 > 0.5 or max(ys) / h2 < 0.5:
                continue
            nx, ny = -dy / n, dx / n
            sgn = 1 if ((mx - a[0]) * nx + (my - a[1]) * ny) > 0 else -1

            def sample(off: float) -> Optional[float]:
                vals = []
                for t in np.linspace(0.1, 0.9, 15):
                    px, py = a[0] + dx * t + nx * sgn * off, a[1] + dy * t + ny * sgn * off
                    if 0 <= int(px) < w2 and 0 <= int(py) < h2:
                        vals.append(float(g[int(py), int(px)]))
                return float(np.mean(vals)) if len(vals) >= 8 else None

            inner, o1, o2 = sample(dist / 2), sample(-dist * 0.8), sample(dist * 1.8)
            if inner is None or o1 is None or o2 is None or min(abs(inner - o1), abs(inner - o2)) < 12:
                continue
            best, pair = sc, (a, b)
    if pair is None:
        return 0.0, None
    xs = [pair[0][0], pair[0][2], pair[1][0], pair[1][2]]
    ys = [pair[0][1], pair[0][3], pair[1][1], pair[1][3]]
    return float(min(best, 1.0)), (min(xs) / w2, min(ys) / h2, max(xs) / w2, max(ys) / h2)


@dataclass
class LabelerConfig:
    min_person_conf: float = 0.25
    belt_mode: str = "queue"            # queue: every person's belt goes to the review queue | auto: a strong strap is labelled
    belt_auto_score: float = 0.5        # strap score that counts as 'worn' in auto mode
    belt_hint_score: float = 0.35       # strap score below which no suggested belt box is offered
    seat_queue_conf: float = 0.10       # child-seat candidate confidence that goes to the queue
    seat_auto_conf: float = 0.50        # ... that is labelled without a human (YOLO-World rarely gets here: measured)
    min_person_px: int = 60             # person crops shorter than this are not judged for belts


class AutoLabeler:
    """Frame -> manifest boxes. The models are injected, so the decision logic is testable without weights.

    ``find_people(img) -> [(box, conf)]``, ``classify(img, box) -> Attribute`` (classifier.PassengerClassifier),
    ``find_seats(img) -> [(box, conf)]`` or None (no child-seat detector), ``strap(crop) -> (score, box)``."""

    def __init__(self, find_people: Callable, classify: Callable, find_seats: Optional[Callable] = None,
                 strap: Callable = strap_evidence, cfg: Optional[LabelerConfig] = None) -> None:
        self.find_people, self.classify, self.find_seats, self.strap = find_people, classify, find_seats, strap
        self.cfg = cfg or LabelerConfig()
        if self.cfg.belt_mode not in ("queue", "auto"):
            raise AutolabelError("belt_mode must be queue or auto")

    def label(self, img: np.ndarray) -> List[dict]:
        people = refine_people([(b, c) for b, c in self.find_people(img) if c >= self.cfg.min_person_conf])
        boxes: List[dict] = []
        tallest = max((b[3] - b[1] for b, _ in people), default=0.0)
        for pb, pc in people:
            a = self.classify(img, pb)
            if a.label in PERSON_CLASSES:
                boxes.append(new_box("person", pb, a.label, a.conf, "face"))
                boxes.extend(self._belt(img, pb, a.label))
            else:
                small = tallest > 0 and (pb[3] - pb[1]) < 0.6 * tallest
                boxes.append(new_box("person", pb, None, pc, "face", Q_UNKNOWN_PERSON,
                                     "small next to the others: possibly a child" if small else "no readable face"))
        return boxes + self._seats(img, [b for b in boxes if b["cls"] == "child"])

    def _belt(self, img: np.ndarray, pb: Box, who: str) -> List[dict]:
        H, W = img.shape[:2]
        x1, y1, x2, y2 = int(pb[0] * W), int(pb[1] * H), int(pb[2] * W), int(pb[3] * H)
        if y2 - y1 < self.cfg.min_person_px or x2 - x1 < 20:
            return []                               # too small to judge a belt on
        score, sb = self.strap(img[y1:y2, x1:x2])
        sbox = None
        if sb is not None:
            pw, ph = pb[2] - pb[0], pb[3] - pb[1]
            sbox = (pb[0] + sb[0] * pw, pb[1] + sb[1] * ph, pb[0] + sb[2] * pw, pb[1] + sb[3] * ph)
        if self.cfg.belt_mode == "auto" and sbox is not None and score >= self.cfg.belt_auto_score:
            return [new_box("belt", sbox, "seatbelt", score, "strap_heuristic")]
        # never an automatic no_seatbelt: an absent strap is as often a hidden or dark one
        hint = sbox is not None and score >= self.cfg.belt_hint_score
        torso = (pb[0], pb[1] + 0.2 * (pb[3] - pb[1]), pb[2], pb[1] + 0.8 * (pb[3] - pb[1]))
        return [new_box("belt", sbox if hint else torso, None, score, "strap_heuristic", Q_BELT_UNSURE,
                        f"{who}: strap suggested" if hint else f"{who}: no strap found, box is a torso placeholder")]

    def _seats(self, img: np.ndarray, kids: List[dict]) -> List[dict]:
        out: List[dict] = []
        cand = [(b, c) for b, c in (self.find_seats(img) if self.find_seats else []) if c >= self.cfg.seat_queue_conf]
        for b, c in refine_people(cand, min_area=0.002, max_aspect=3.0):
            if c >= self.cfg.seat_auto_conf:
                out.append(new_box("child_seat", b, "child_seat", c, "yolo_world"))
            else:
                out.append(new_box("child_seat", b, None, c, "yolo_world", Q_SEAT_UNSURE, "candidate child seat"))
        # a child with no seat box near is a question for the reviewer, not an answer
        for k in kids:
            if not any(inside_fraction(tuple(k["box"]), tuple(s["box"])) >= 0.4 for s in out):
                out.append(new_box("child_seat", k["box"], None, 0.0, "rule", Q_SEAT_UNSURE,
                                   "child with no seat found: set to child_seat (fix the box) if seated in one, else delete"))
        return out


def frame_status(boxes: Sequence[dict]) -> Tuple[str, Optional[str]]:
    """New frames are ``auto``, or ``rejected`` when no occupant was found at all (nothing to label)."""
    if not any(b["kind"] == "person" for b in boxes):
        return "rejected", "no_person"
    return "auto", None


# -- models (built lazily: tests and the review tools never load weights) -----------
REPO = Path(__file__).resolve().parents[1]
SEAT_PROMPTS = ("child car seat", "baby car seat", "infant carrier", "booster seat")


def build_people_finder(model: str = "yolov8s.pt", conf: float = 0.25, imgsz: int = 960, backend: str = "torch") -> Callable:
    """COCO-person detector (the repo's ``PersonDetector``, torch or openvino) as ``img -> [(box, conf)]``."""
    from .detector import PersonDetector

    det = PersonDetector(model, conf=conf, imgsz=imgsz, backend=backend)
    return lambda img: [(d.box, d.conf) for d in det.detect(img, track=False)]


def build_classifier(config: Path = REPO / "config.yaml") -> Callable:
    """The repo's face classifier (YuNet + FairFace champion) from ``passenger.classifier``, with weight paths made absolute."""
    from eagleeye.config import load_config

    from .classifier import PassengerClassifier

    cfg = load_config(config)
    c = cfg.passenger.classifier
    for key in ("weights", "yunet_weights", "model_dir"):
        if c.get(key) and not Path(c[key]).is_absolute():
            c[key] = str(REPO / c[key])
    return PassengerClassifier.from_config(cfg).classify


def build_seat_finder(weights: Path, prompts: Sequence[str] = SEAT_PROMPTS, conf: float = 0.05) -> Optional[Callable]:
    """Open-vocabulary child-seat finder (ultralytics YOLO-World, needs the ``clip`` package and its text weights).

    Returns None, with a printed warning, when it cannot be built: the run then has no seat candidates, only the rule that
    asks about every child."""
    try:
        from ultralytics import YOLOWorld

        model = YOLOWorld(str(weights))
        model.set_classes(list(prompts))
    except Exception as exc:
        print(f"[autolabel] child-seat detector unavailable ({type(exc).__name__}: {exc}); continuing without it", flush=True)
        return None

    def find(img: np.ndarray) -> List[Tuple[Box, float]]:
        r = model.predict(img, conf=conf, imgsz=640, verbose=False, agnostic_nms=True)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        return [(tuple(map(float, b)), float(c)) for b, c in zip(r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy())]

    return find


# -- the run ----------------------------------------------------------------------
def run_pipeline(videos: Sequence[Path], run_dir: Path, labeler: AutoLabeler, *, licenses: Optional[Dict[str, dict]] = None,
                 default_license: str = "unknown", fps: float = 1.5, max_frames: int = 200, dedup: float = 2.0,
                 force: bool = False, log: Callable[[str], None] = print) -> dict:
    """Videos -> frames -> auto-labels in ``run_dir``. A video that cannot be read is recorded and skipped, not fatal."""
    if (run_dir / "manifest.json").exists() and not force:
        raise AutolabelError(f"{run_dir} already holds a run: pick a new --run name or pass --force to clear that run")
    if force:
        reset_run(run_dir)
    (run_dir / "images").mkdir(parents=True, exist_ok=True)
    (run_dir / "labels").mkdir(exist_ok=True)
    licenses = licenses or {}
    m: dict = {"run": run_dir.name, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "classes": list(CAR_CLASSES), "note": "pseudo-labels, not ground truth",
               "settings": {"fps": fps, "dedup": dedup, "belt_mode": labeler.cfg.belt_mode,
                            "seat_detector": labeler.find_seats is not None}, "videos": {}, "frames": {}}
    for video in videos:
        vid = safe_id(video.name, m["videos"])
        info = licenses.get(video.stem.lower(), {})
        lic, url = info.get("license") or default_license, info.get("url", "")
        entry = m["videos"][vid] = {"source": str(video), "license": lic, "url": url, "frames": 0, "error": None}
        try:
            saved = extract_frames(video, run_dir / "images", vid, fps=fps, max_frames=max_frames, dedup=dedup)
        except AutolabelError as exc:
            entry["error"] = str(exc)
            log(f"[autolabel] {video.name}: {exc}")
            continue
        if not saved:
            entry["error"] = "no frames could be read"
        for s in saved:
            fid = s["id"]
            frame = {"video": vid, "frameno": s["frameno"], "file": s["file"], "width": s["width"], "height": s["height"],
                     "license": lic, "url": url, "status": "auto", "reject_reason": None, "boxes": []}
            img = cv2.imread(str(run_dir / "images" / s["file"]))
            if img is None:
                frame["status"], frame["reject_reason"] = "rejected", "unreadable_image"
            else:
                try:
                    frame["boxes"] = labeler.label(img)
                    frame["status"], frame["reject_reason"] = frame_status(frame["boxes"])
                except Exception as exc:                  # a model failure on one frame must not lose the rest of the run
                    frame["boxes"] = []
                    frame["status"], frame["reject_reason"] = "rejected", f"label_error: {type(exc).__name__}: {exc}"[:200]
            write_label(run_dir, fid, frame)
            m["frames"][fid] = frame
        entry["frames"] = len(saved)
        log(f"[autolabel] {video.name}: {len(saved)} frames")
        save_manifest(run_dir, m)
    (run_dir / "data.yaml").write_text(data_yaml_text(), encoding="utf-8")
    save_manifest(run_dir, m)
    return run_stats(run_dir)


# -- command line ---------------------------------------------------------------------
def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Car-cabin videos -> frames -> pseudo-labels -> review -> YOLO dataset")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="car_open_data folder; runs live in <root>/autolabel/<run>")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="extract frames, auto-label them, draw the review sheets")
    r.add_argument("--videos", type=Path, required=True, help="a video file or a folder of them")
    r.add_argument("--run", required=True)
    r.add_argument("--manifest-csv", type=Path, default=None, help="MANIFEST.csv with per-video licences (default: <videos>/MANIFEST.csv or <root>/MANIFEST.csv)")
    r.add_argument("--default-license", default="unknown", help=f"for videos not in the manifest; '{DEV_ONLY}-...' marks tooling-test footage that can never be exported")
    r.add_argument("--fps", type=float, default=1.5)
    r.add_argument("--max-frames", type=int, default=200, help="per video")
    r.add_argument("--dedup", type=float, default=2.0, help="drop a frame whose 32x18 grey thumbnail differs from the last kept one by less than this (0-255)")
    r.add_argument("--detector", default=None, help="person detector weights (default <root>/_models/yolov8s.pt if present, else yolov8s.pt: ultralytics downloads it into the current folder)")
    r.add_argument("--det-conf", type=float, default=0.25)
    r.add_argument("--imgsz", type=int, default=960)
    r.add_argument("--backend", default="torch", choices=("torch", "openvino"))
    r.add_argument("--belt-mode", default="queue", choices=("queue", "auto"))
    r.add_argument("--world-weights", type=Path, default=None, help="YOLO-World weights (default <root>/_models/yolov8s-worldv2.pt)")
    r.add_argument("--no-seat-detector", action="store_true")
    r.add_argument("--force", action="store_true", help="clear this run's own images/labels/review/manifest first")

    s = sub.add_parser("sheets", help="(re)draw review contact sheets")
    s.add_argument("--run", required=True)
    s.add_argument("--which", default="queue", choices=("queue", "auto", "all"))
    s.add_argument("--per-sheet", type=int, default=6)

    a = sub.add_parser("apply", help="apply a corrections CSV (frame, box_index, new_class[, x1, y1, x2, y2])")
    a.add_argument("--run", required=True)
    a.add_argument("csv", type=Path)

    e = sub.add_parser("export", help="write a YOLO dataset of reviewed / high-confidence frames only")
    e.add_argument("--run", required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--min-auto-conf", type=float, default=0.8)
    e.add_argument("--reviewed-only", action="store_true")
    e.add_argument("--allow-unknown-license", action="store_true")

    t = sub.add_parser("stats", help="counts, review-queue share")
    t.add_argument("--run", required=True)

    args = ap.parse_args(argv)
    try:
        run_dir = run_path(args.root, args.run)
        if args.cmd == "run":
            man = args.manifest_csv or next((p for p in (args.videos / "MANIFEST.csv", args.root / "MANIFEST.csv", args.root / "videos_open" / "MANIFEST.csv") if p.is_file()), None)
            vids = find_videos(args.videos)
            if not vids:
                raise AutolabelError(f"no video files in {args.videos}")
            weights = args.world_weights or args.root / "_models" / "yolov8s-worldv2.pt"
            labeler = AutoLabeler(build_people_finder(args.detector or str(next((p for p in [args.root / "_models" / "yolov8s.pt"] if p.is_file()), "yolov8s.pt")), args.det_conf, args.imgsz, args.backend), build_classifier(),
                                  None if args.no_seat_detector else build_seat_finder(weights),
                                  cfg=LabelerConfig(belt_mode=args.belt_mode))
            out = run_pipeline(vids, run_dir, labeler, licenses=read_licenses(man), default_license=args.default_license,
                               fps=args.fps, max_frames=args.max_frames, dedup=args.dedup, force=args.force)
            out["sheets"] = len(make_sheets(run_dir, "queue"))
        elif args.cmd == "sheets":
            out = {"sheets": make_sheets(run_dir, args.which, per_sheet=args.per_sheet)}
        elif args.cmd == "apply":
            out = apply_corrections(run_dir, args.csv)
        elif args.cmd == "export":
            out = export_dataset(run_dir, args.out, min_auto_conf=args.min_auto_conf, include_auto=not args.reviewed_only,
                                 allow_unknown_license=args.allow_unknown_license)
        else:
            out = run_stats(run_dir)
    except (AutolabelError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
