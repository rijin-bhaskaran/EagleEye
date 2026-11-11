"""Tests for passenger.frame_quality (frame cleaning primitives for new footage)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from passenger import frame_quality as fq


def textured(h=120, w=160, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_correlation_identical_inverse_and_flat():
    a = fq.thumb(textured(seed=1))
    assert fq.correlation(a, a) == 1.0 or abs(fq.correlation(a, a) - 1.0) < 1e-6
    assert fq.correlation(a, 255 - a) < -0.99
    flat = np.full((32, 32), 7, np.float32)
    assert fq.correlation(flat, flat) == 1.0                      # no variance: identical, never a division error
    assert abs(fq.correlation(a, fq.thumb(textured(seed=2)))) < 0.3


def test_sharpness_drops_with_blur_and_is_resolution_normalised():
    img = textured(240, 320)
    assert fq.sharpness(cv2.GaussianBlur(img, (15, 15), 6)) < 0.2 * fq.sharpness(img)
    big = cv2.resize(img, (1280, 960), interpolation=cv2.INTER_NEAREST)
    assert fq.sharpness(big) > 0                                   # large frames are measured at <= 640 px wide


def test_exposure_is_mean_grey():
    assert fq.exposure(np.zeros((8, 8, 3), np.uint8)) == 0
    assert fq.exposure(np.full((8, 8, 3), 200, np.uint8)) == 200


def _clip(n=12, h=100, w=160, bar=0, side=0):
    frames = []
    for i in range(n):
        g = cv2.cvtColor(textured(h, w, seed=i), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if bar:
            g[-bar:, :] = 40                                       # static taskbar, sharp edge to the content
            g[-bar:, 20:60] = 200
        if side:
            g[:, :side] = 0
        frames.append(g)
    return frames


def test_static_border_finds_a_ui_bar_at_the_bottom():
    assert fq.static_border(_clip(bar=12)) == (0, 12, 0, 0)


def test_static_border_finds_black_side_bars():
    top, bot, left, right = fq.static_border(_clip(side=10))
    assert (top, bot, right) == (0, 0, 0) and left == 10


def test_static_border_ignores_clean_footage_and_unbounded_static_areas():
    assert fq.static_border(_clip()) == (0, 0, 0, 0)
    still = [np.full((100, 160), 90, np.float32)] * 6               # everything static, no edge: not a border
    assert fq.static_border(still) == (0, 0, 0, 0)


def test_static_border_never_removes_more_than_the_limit():
    clip = _clip(bar=40)                                            # 40 of 100 rows is more than max_frac 0.15
    assert fq.static_border(clip, max_frac=0.15)[1] == 0


def test_person_thumb_covers_the_union_of_boxes():
    img = np.zeros((100, 200, 3), np.uint8)
    img[20:60, 50:100] = 255
    t = fq.person_thumb(img, [(0.25, 0.2, 0.5, 0.6)])
    assert t.shape == (32, 32) and t.min() > 200                    # only the bright person region
    t2 = fq.person_thumb(img, [(0.25, 0.2, 0.3, 0.3), (0.45, 0.5, 0.5, 0.6)])
    assert t2.shape == (32, 32)


def test_select_diverse_respects_cap_spacing_and_spreads():
    feats = np.array([[0.0], [0.01], [0.02], [5.0], [5.01], [10.0]], np.float32)
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    picked = fq.select_diverse(feats, times, cap=3, min_gap_s=1.0)
    assert picked == sorted(picked) and len(picked) == 3
    assert {3 if p in (3, 4) else p for p in picked} >= {5}          # the far outlier is chosen
    ts = [times[p] for p in picked]
    assert all(b - a >= 1.0 - 1e-6 for a, b in zip(ts, ts[1:]))
    assert fq.select_diverse(feats[:0], [], cap=3) == []
    assert fq.select_diverse(feats, times, cap=100, min_gap_s=0.0) == list(range(6))
