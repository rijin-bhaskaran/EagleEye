"""Tests for passenger.data_manager. Run with:  pytest -q"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from passenger.data_manager import DataManager


def make_video(path: Path, static: int = 30, moving: int = 30, fps: int = 10, seed: int = 0) -> Path:
    """`static` identical frames followed by `moving` visibly different ones."""
    rng = np.random.default_rng(seed)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    base = np.full((48, 64, 3), 100, np.uint8)
    for _ in range(static):
        w.write(base)
    for i in range(moving):
        w.write(np.clip(base.astype(int) + (i % 2) * 90 + rng.integers(0, 5), 0, 255).astype(np.uint8))
    w.release()
    return path


@pytest.fixture
def dm(tmp_path):
    return DataManager(tmp_path / "pd")


def test_register_copies_and_indexes(dm, tmp_path):
    vid = dm.register_video(make_video(tmp_path / "a.mp4"), vehicle="bus1")
    meta = dm.videos[vid]
    assert (dm.root / meta["path"]).is_file()
    assert meta["vehicle"] == "bus1" and meta["frame_count"] == 60 and meta["split"] is None


def test_register_same_file_twice_is_idempotent(dm, tmp_path):
    src = make_video(tmp_path / "a.mp4")
    assert dm.register_video(src) == dm.register_video(src)
    assert len(dm.videos) == 1


def test_register_rejects_missing_and_undecodable(dm, tmp_path):
    with pytest.raises(FileNotFoundError):
        dm.register_video(tmp_path / "nope.mp4")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        dm.register_video(bad)


def test_manifest_persists_across_instances(dm, tmp_path):
    vid = dm.register_video(make_video(tmp_path / "a.mp4"))
    assert vid in DataManager(dm.root).videos


def test_extract_dedupes_static_frames(dm, tmp_path):
    vid = dm.register_video(make_video(tmp_path / "a.mp4"))
    kept = dm.extract_frames(vid, target_fps=10)          # sample every frame
    assert 1 <= kept < 60                                  # static run collapsed
    assert len(list((dm.frames_dir / vid).glob("*.jpg"))) == kept
    assert kept == len(dm.videos[vid]["frames"])


def test_extract_respects_target_fps(dm, tmp_path):
    vid = dm.register_video(make_video(tmp_path / "a.mp4", static=0, moving=60))
    assert dm.extract_frames(vid, target_fps=10, dedupe_threshold=0) == 60
    assert dm.extract_frames(vid, target_fps=5, dedupe_threshold=0) == 30   # rerun replaces


def test_assign_splits_is_by_video_deterministic_and_complete(dm, tmp_path):
    for i in range(10):
        dm.register_video(make_video(tmp_path / f"v{i}.mp4", static=i + 1))
    a = dm.assign_splits(seed=1)
    assert sum(len(v) for v in a.values()) == 10
    assert len({v for ids in a.values() for v in ids}) == 10        # no video in two splits
    assert all(a[s] for s in a)
    assert dm.assign_splits(seed=1) == a


def test_assign_splits_small_and_invalid(dm, tmp_path):
    for i in range(3):
        dm.register_video(make_video(tmp_path / f"v{i}.mp4", static=i + 1))
    assert all(len(v) == 1 for v in dm.assign_splits().values())    # 3 videos -> one each
    with pytest.raises(ValueError):
        dm.assign_splits(ratios=(0.5, 0.5, 0.5))


def _prepare(dm, tmp_path, n=3):
    ids = [dm.register_video(make_video(tmp_path / f"v{i}.mp4", static=i + 1)) for i in range(n)]
    for vid in ids:
        dm.extract_frames(vid, target_fps=10, dedupe_threshold=0)
    dm.assign_splits()
    return ids


def _label_all(dm, lines="0 0.5 0.5 0.2 0.3\n1 0.2 0.2 0.1 0.1\n"):
    src = dm.root / "incoming"
    src.mkdir(exist_ok=True)
    for jpg in dm.frames_dir.glob("*/*.jpg"):
        (src / f"{jpg.stem}.txt").write_text(lines)
    return src


def test_import_labels_only_matches_known_frames(dm, tmp_path):
    _prepare(dm, tmp_path)
    src = _label_all(dm)
    (src / "stranger_000001.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    n = dm.import_labels(src)
    assert n == len(list(dm.labels_dir.glob("*.txt"))) and not (dm.labels_dir / "stranger_000001.txt").exists()


@pytest.mark.parametrize("line", ["0 0.5 0.5 0.2", "7 0.5 0.5 0.2 0.2", "0 1.5 0.5 0.2 0.2", "0 0.5 0.5 0 0.2", "x 0.5 0.5 0.2 0.2"])
def test_validate_labels_flags_bad_lines(dm, tmp_path, line):
    (dm.labels_dir / "f.txt").write_text(line + "\n")
    assert len(dm.validate_labels()) == 1


def test_validate_labels_clean(dm):
    (dm.labels_dir / "f.txt").write_text("2 0.5 0.5 0.2 0.2\n\n")
    assert dm.validate_labels() == []


def test_export_yolo_builds_dataset_with_yaml(dm, tmp_path):
    _prepare(dm, tmp_path)
    dm.import_labels(_label_all(dm))
    out = dm.export_yolo("ds")
    yaml = (out / "data.yaml").read_text()
    assert "nc: 3" in yaml and "0: male" in yaml and "2: child" in yaml
    for split in ("train", "val", "test"):
        imgs = {p.stem for p in (out / "images" / split).glob("*.jpg")}
        lbls = {p.stem for p in (out / "labels" / split).glob("*.txt")}
        assert imgs and imgs == lbls


def test_export_skips_unlabelled_and_unassigned_and_rejects_bad_labels(dm, tmp_path):
    _prepare(dm, tmp_path)
    out = dm.export_yolo("empty")                          # nothing labelled -> no images, yaml still valid
    assert not list(out.rglob("*.jpg")) and (out / "data.yaml").exists()
    (dm.labels_dir / "bad.txt").write_text("9 0.5 0.5 0.2 0.2\n")
    with pytest.raises(ValueError):
        dm.export_yolo("x")


def test_stats_counts_boxes_per_class_and_split(dm, tmp_path):
    _prepare(dm, tmp_path)
    dm.import_labels(_label_all(dm))
    s = dm.stats()
    total = sum(s[k]["frames"] for k in ("train", "val", "test"))
    assert total == sum(len(m["frames"]) for m in dm.videos.values())
    assert s["train"]["boxes"]["male"] == s["train"]["labelled"]
    assert s["train"]["boxes"]["female"] == s["train"]["labelled"]
    assert "child" not in s["train"]["boxes"]


def test_manifest_save_retries_transient_permission_error(dm, tmp_path, monkeypatch):
    real, calls = Path.replace, {"n": 0}

    def flaky(self, target):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("file is locked")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", flaky)
    vid = dm.register_video(make_video(tmp_path / "a.mp4"))
    assert calls["n"] == 3 and vid in DataManager(dm.root).videos


def test_manifest_save_gives_up_on_persistent_lock(dm, tmp_path, monkeypatch):
    def locked(self, target):
        raise PermissionError("file is locked")

    monkeypatch.setattr(Path, "replace", locked)
    with pytest.raises(PermissionError):
        dm.register_video(make_video(tmp_path / "a.mp4"))


# -- groups: people/scenes that must never be split across train / val / test --------------------------
def _gvideos(dm, tmp_path, n, group=None, start=0):
    return [dm.register_video(make_video(tmp_path / f"g{start + i}.mp4", static=start + i + 1), group=group) for i in range(n)]


def test_register_records_group_and_rights_only_when_given(dm, tmp_path):
    a = dm.register_video(make_video(tmp_path / "a.mp4"))
    b = dm.register_video(make_video(tmp_path / "b.mp4", static=5), group="g_pair", rights="unverified, internal only")
    assert dm.videos[a].get("group") is None and dm.videos[a].get("rights") is None
    assert dm.videos[b]["group"] == "g_pair" and dm.videos[b]["rights"] == "unverified, internal only"
    with pytest.raises(ValueError):
        dm.register_video(make_video(tmp_path / "c.mp4", static=7), group="../x")


def test_set_group_on_existing_video_never_touches_split(dm, tmp_path):
    vid = dm.register_video(make_video(tmp_path / "a.mp4"))
    dm.videos[vid]["split"] = "val"
    dm.set_group(vid, "g_family")
    assert dm.videos[vid]["group"] == "g_family" and dm.videos[vid]["split"] == "val"
    with pytest.raises(ValueError):
        dm.set_group(vid, "bad name")
    with pytest.raises(KeyError):
        dm.set_group("nope", "g_family")


def test_assign_splits_keeps_a_group_together(dm, tmp_path):
    groups = {}
    for gi in range(6):
        groups[f"g{gi}"] = _gvideos(dm, tmp_path, 3, group=f"g{gi}", start=gi * 3)
    dm.assign_splits(seed=3)
    for ids in groups.values():
        assert len({dm.videos[v]["split"] for v in ids}) == 1


def test_assign_splits_new_member_follows_its_assigned_group_and_old_splits_stay(dm, tmp_path):
    old = _gvideos(dm, tmp_path, 2, group="g_old")
    solo = _gvideos(dm, tmp_path, 4, start=10)
    dm.videos[old[0]]["split"] = "test"
    dm.videos[solo[0]]["split"] = "val"
    before = {v: m["split"] for v, m in dm.videos.items() if m["split"]}
    dm.assign_splits(seed=1)
    assert dm.videos[old[1]]["split"] == "test"
    assert all(dm.videos[v]["split"] == s for v, s in before.items())


def test_assign_group_sets_whole_group_and_refuses_conflicts(dm, tmp_path):
    ids = _gvideos(dm, tmp_path, 3, group="g_a")
    assert sorted(dm.assign_group("g_a", "test")) == sorted(ids)
    assert {dm.videos[v]["split"] for v in ids} == {"test"}
    with pytest.raises(ValueError):
        dm.assign_group("g_a", "val")                      # already assigned elsewhere: never moves
    with pytest.raises(ValueError):
        dm.assign_group("g_a", "dev")                      # not a split
    with pytest.raises(KeyError):
        dm.assign_group("g_missing", "train")
    mixed = _gvideos(dm, tmp_path, 2, group="g_m", start=20)
    dm.videos[mixed[0]]["split"] = "train"
    with pytest.raises(ValueError):
        dm.assign_group("g_m", "test")                     # a member is already in train
