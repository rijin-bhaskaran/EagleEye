"""Small, testable frame-cleaning primitives for new cabin footage (see docs/ANNOTATION.md and WORKFLOW.md).

Used by the data-cleaning scripts, never on the live per-frame path. Everything works on BGR uint8 frames
(or grey float32 thumbnails) and has no model dependency."""
from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np


def thumb(img: np.ndarray, size: int = 32) -> np.ndarray:
    """Grey ``size`` x ``size`` thumbnail used for near-duplicate checks."""
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two thumbnails (1.0 = same picture). Two flat images count as identical."""
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return 1.0 if d == 0 else float((a * b).sum() / d)


def sharpness(img: np.ndarray, width: int = 640) -> float:
    """Variance of the Laplacian, measured at <= ``width`` px so clips of different resolution are comparable."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if g.shape[1] > width:
        g = cv2.resize(g, (width, int(g.shape[0] * width / g.shape[1])), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def exposure(img: np.ndarray) -> float:
    """Mean grey level 0-255 (very low = black / night, very high = blown out)."""
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())


def static_border(grays: Sequence[np.ndarray], thr: float = 3.0, max_frac: float = 0.15,
                  edge: float = 12.0) -> Tuple[int, int, int, int]:
    """Detect UI bars / letterboxing in a screen recording: ``(top, bottom, left, right)`` pixels to cut.

    A border is a strip along the frame edge whose pixels hardly change over the clip (temporal std < ``thr``)
    AND that ends in a sharp edge in the median image. A static part of the cabin has no such edge, and
    nothing wider than ``max_frac`` of the frame is ever cut. ``grays`` are equal-size grey frames."""
    G = np.stack([np.asarray(g, np.float32) for g in grays])
    sd, med = G.std(0), np.median(G, 0)
    H, W = med.shape
    drow = np.abs(np.diff(med, axis=0)).mean(1)
    dcol = np.abs(np.diff(med, axis=1)).mean(0)

    def run(sd_axis: np.ndarray, edges: np.ndarray, limit: int) -> int:
        n = 0
        for v in sd_axis:
            if v < thr and n < limit:
                n += 1
            else:
                break
        if n == 0:
            return 0
        lo, hi = max(n - 2, 0), min(n + 2, len(edges))
        return n if edges[lo:hi].max() > edge else 0

    rs, cs = sd.mean(1), sd.mean(0)
    return (run(rs, drow, int(H * max_frac)), run(rs[::-1], drow[::-1], int(H * max_frac)),
            run(cs, dcol, int(W * max_frac)), run(cs[::-1], dcol[::-1], int(W * max_frac)))


def person_thumb(img: np.ndarray, boxes: Sequence[Sequence[float]], size: int = 32) -> np.ndarray:
    """Thumbnail of the union of normalised person boxes (x1, y1, x2, y2). Pose changes show up here even when the
    unchanged cabin makes whole-frame thumbnails correlate above 0.97."""
    H, W = img.shape[:2]
    x1 = int(min(b[0] for b in boxes) * W)
    y1 = int(min(b[1] for b in boxes) * H)
    x2 = int(max(b[2] for b in boxes) * W)
    y2 = int(max(b[3] for b in boxes) * H)
    return thumb(img[y1:max(y2, y1 + 2), x1:max(x2, x1 + 2)], size)


def select_diverse(feats: np.ndarray, times: Sequence[float], cap: int, min_gap_s: float = 1.0) -> list:
    """Greedy farthest-point sampling: up to ``cap`` row indices that differ most from each other, no two closer
    than ``min_gap_s`` seconds. Starts from the row nearest the mean. Returns sorted indices."""
    F = np.asarray(feats, np.float32)
    n = len(F)
    if n == 0 or cap <= 0:
        return []
    T = np.asarray(times, np.float32)
    first = int(np.argmin(((F - F.mean(0)) ** 2).sum(1)))
    picked = [first]
    d = ((F - F[first]) ** 2).sum(1)
    while len(picked) < min(cap, n):
        cand = d.copy()
        for p in picked:
            cand[np.abs(T - T[p]) < min_gap_s - 1e-6] = -1
            cand[p] = -1
        j = int(np.argmax(cand))
        if cand[j] < 0:
            break
        picked.append(j)
        d = np.minimum(d, ((F - F[j]) ** 2).sum(1))
    return sorted(picked)
