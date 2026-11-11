#!/usr/bin/env python3
"""Where do faces get lost? A per-stage funnel for the passenger classifier, over a video.

    python scripts/diagnose_faces.py data/vid-9.mp4
    python scripts/diagnose_faces.py data/vid-9.mp4 --every 10 --max-frames 300 --save-misses misses/

For every sampled frame and every detected person it records the first stage that stops a label:

    person_too_short   box shorter than passenger.classifier.min_person_px (the face is never searched for)
    no_face            the face finder (per person crop) returned nothing
    face_too_small     a face was found but under min_face_px
    low_confidence     face read, but gender confidence < min_conf (answer would be 'unknown')
    labelled           male / female / child

Uses the settings in config.yaml (including passenger.classifier.face_detector), so run it once per detector
to compare them. No setting is changed. Read-only apart from --save-misses.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STAGES = ("person_too_short", "no_face", "face_too_small", "low_confidence", "labelled")


def stage_of(person_h_px: int, face: Optional[Tuple[int, int, int, int]], conf: Optional[float],
             *, min_person_px: int, min_face_px: int, min_conf: float, child: bool = False) -> str:
    """The first stage that stops a label for one person (see the module docstring)."""
    if person_h_px < min_person_px:
        return "person_too_short"
    if face is None:
        return "no_face"
    if min(face[2], face[3]) < min_face_px:
        return "face_too_small"
    if not child and (conf is None or conf < min_conf):
        return "low_confidence"
    return "labelled"


def summarise(stages: Sequence[str]) -> Dict[str, Tuple[int, float]]:
    """stage -> (count, share of all people)."""
    n = max(len(stages), 1)
    c = Counter(stages)
    return {s: (c.get(s, 0), c.get(s, 0) / n) for s in STAGES}


def run(video: str, every: int, max_frames: int, save_misses: Optional[Path]) -> int:
    import cv2

    from eagleeye.config import load_config
    from passenger.classifier import PassengerClassifier
    from passenger.detector import PersonDetector

    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    c = cfg.passenger.classifier
    d = cfg.passenger.detector
    clf = PassengerClassifier.from_config(cfg)
    detector = PersonDetector(d.model, d.conf, d.imgsz, d.classes, backend=d.get("backend", "torch"))

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"cannot open {video}")
        return 2
    stages: List[str] = []
    heights: List[int] = []
    face_px: List[int] = []
    n_persons = n_in_crops = 0
    sampled = idx = 0
    if save_misses:
        save_misses.mkdir(parents=True, exist_ok=True)

    while sampled < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if (idx - 1) % every:
            continue
        sampled += 1
        H, W = frame.shape[:2]
        dets = detector.detect(frame, track=False)
        for p in dets:
            x1, y1 = max(int(p.box[0] * W), 0), max(int(p.box[1] * H), 0)
            x2, y2 = min(int(np.ceil(p.box[2] * W)), W), min(int(np.ceil(p.box[3] * H)), H)
            ph = y2 - y1
            n_persons += 1
            heights.append(ph)
            face = clf.find_face(frame[y1:y2, x1:x2]) if ph >= c.min_person_px and x2 > x1 else None
            conf, child = None, False
            if face is not None:
                n_in_crops += 1
                face_px.append(min(face[2], face[3]))
                img = clf.face_crop(frame, p.box)
                if img is not None:
                    pm, pf, pc = clf.model.predict(img)
                    child = pc >= clf.child_threshold
                    conf = max(pm, pf)
            st = stage_of(ph, face, conf, min_person_px=c.min_person_px, min_face_px=c.min_face_px,
                          min_conf=c.min_conf, child=child)
            stages.append(st)
            if save_misses and st in ("no_face", "face_too_small") and ph >= c.min_person_px:
                cv2.imwrite(str(save_misses / f"f{idx:05d}_{st}_{x1}_{y1}.jpg"), frame[y1:y2, x1:x2])
    cap.release()

    print(f"\n{video}: {sampled} frames sampled (every {every}), {n_persons} person detections")
    print(f"settings: min_person_px={c.min_person_px} min_face_px={c.min_face_px} min_conf={c.min_conf} "
          f"face_detector={c.get('face_detector', 'mediapipe')}")
    print("\nstage                count   share")
    for s, (k, share) in summarise(stages).items():
        print(f"{s:<18} {k:>7} {share:>7.1%}")
    if heights:
        q = np.percentile(heights, [10, 50, 90]).astype(int)
        print(f"\nperson height px  p10/p50/p90 = {q[0]}/{q[1]}/{q[2]}")
    if face_px:
        q = np.percentile(face_px, [10, 50, 90]).astype(int)
        print(f"face size px      p10/p50/p90 = {q[0]}/{q[1]}/{q[2]}   (shortest side, faces found in crops)")
    print(f"\nfaces found in person crops: {n_in_crops} of {n_persons} people (before the size gate)")
    print("\nHow to read it: the biggest non-'labelled' row is where to act. person_too_short/face_too_small -> "
          "resolution or thresholds; no_face -> a stronger face detector (face_detector: yunet); low_confidence -> classifier/data.")
    return 0


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Per-stage funnel of where faces are lost")
    ap.add_argument("video")
    ap.add_argument("--every", type=int, default=10, help="analyse every Nth frame")
    ap.add_argument("--max-frames", type=int, default=200, help="stop after this many analysed frames")
    ap.add_argument("--save-misses", type=Path, default=None, help="folder to save person crops where no usable face was found")
    a = ap.parse_args(argv)
    return run(a.video, max(a.every, 1), a.max_frames, a.save_misses)


if __name__ == "__main__":
    sys.exit(main())
