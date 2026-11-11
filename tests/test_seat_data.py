"""The seat-state split must never put near-identical frames in different splits."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from passenger import seat_data
from passenger.seat_data import Item, assign, build, clips, leakage, one_per_frame

T0 = dt.datetime(2024, 1, 29, 7, 0, 0)


def item(sec, cam="aaaaaa", key=None, split="train", name=None):
    key = key or f"{sec}"
    return Item(Path(name or f"{key}_{split}.jpg"), Path(f"{key}.txt"), split, T0 + dt.timedelta(seconds=sec), cam, key)


def test_augmented_copies_collapse_and_the_unaugmented_one_wins():
    out = one_per_frame([item(0, key="f", split="train", name="a.jpg"), item(0, key="f", split="valid", name="b.jpg"),
                         item(0, key="f", split="train", name="c.jpg"), item(5, key="g")])
    assert len(out) == 2 and out[0].image.name == "b.jpg"


def test_clips_break_on_time_gap_and_on_camera():
    items = [item(0), item(10), item(20), item(500), item(510), item(0, cam="bbbbbb")]
    groups = clips(items, gap_s=120)
    assert sorted(len(g) for g in groups) == [1, 2, 3]
    assert all(len({i.camera for i in g}) == 1 for g in groups)


def test_a_clip_goes_to_exactly_one_split_and_every_clip_is_assigned():
    groups = clips([item(s) for s in range(0, 100_000, 30)], gap_s=60)           # many one-frame clips
    where = assign(groups, seed=1)
    assert set(where) == set(range(len(groups))) and set(where.values()) <= set(seat_data.SPLITS)


def test_assignment_is_deterministic_and_close_to_the_target_shares():
    groups = [[item(i * 1000 + j) for j in range(10)] for i in range(60)]
    assert assign(groups, seed=3) == assign(groups, seed=3)
    where = assign(groups, seed=3)
    share = {s: sum(len(groups[i]) for i, w in where.items() if w == s) / 600 for s in seat_data.SPLITS}
    assert share["train"] == pytest.approx(0.70, abs=0.05) and share["val"] == pytest.approx(0.15, abs=0.05)


def _make_dataset(root: Path, n_clips=12, per_clip=4):
    """A fake Roboflow export: frames of clips scattered across Roboflow's own splits, train frames tripled."""
    k = 0
    for c in range(n_clips):
        for j in range(per_clip):
            t = T0 + dt.timedelta(seconds=c * 3600 + j * 5)
            sp = ("train", "valid", "test")[(c + j) % 3]                  # Roboflow-style frame-level mix
            copies = 3 if sp == "train" else 1
            for h in range(copies):
                stem = f"{t:%Y%m%d%H%M%S}000_{'9' * 32}_AA1_jpg.rf.{k:032x}"
                k += 1
                (root / sp / "images").mkdir(parents=True, exist_ok=True)
                (root / sp / "labels").mkdir(parents=True, exist_ok=True)
                (root / sp / "images" / f"{stem}.jpg").write_bytes(b"x")
                (root / sp / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n")


def test_build_removes_leakage_that_the_original_split_had(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _make_dataset(src)
    rep = build(src, out, gap_s=120, seed=0)
    assert rep["unique_frames"] == 48                                      # copies collapsed
    assert sum(s["frames"] for s in rep["splits"].values()) == 48
    assert sum(len(list((out / "images" / sp).glob("*.jpg"))) for sp in seat_data.SPLITS) == 48   # nothing overwritten on disk
    assert leakage(out) == {"val": 0, "test": 0}
    assert all(s["frames"] > 0 for s in rep["splits"].values())
    assert (out / "data.yaml").read_text().count("Unoccupied") == 1


def test_an_unreadable_file_name_fails_loudly(tmp_path):
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "train" / "labels").mkdir(parents=True)
    (tmp_path / "train" / "images" / "cat.jpg").write_bytes(b"x")
    with pytest.raises(ValueError, match="cannot recover"):
        seat_data.scan(tmp_path)
