"""Tests for the male/female/child person-crop trainer (offline, tiny synthetic crops)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import passenger.train_crops as tc
from passenger.train_crops import CLASSES, balanced_order, cabin_augment, load_folder, score, train_crops


def make_split(root: Path, split: str, classes, n: int = 6) -> None:
    for ci, name in enumerate(classes):
        d = root / split / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = np.full((48, 40, 3), 40 + 80 * ci, np.uint8)       # class = brightness, so it is learnable
            img += np.random.default_rng(i).integers(0, 20, img.shape, dtype=np.uint8)
            cv2.imwrite(str(d / f"{name}_{i}.jpg"), img)


def test_load_folder_orders_classes_and_skips_non_images(tmp_path):
    make_split(tmp_path, "train", CLASSES, n=3)
    (tmp_path / "train" / "male" / "notes.txt").write_text("not an image")
    files, labels = load_folder(tmp_path / "train", CLASSES)
    assert len(files) == 9 and sorted(set(labels)) == [0, 1, 2]
    assert all(f.suffix == ".jpg" for f in files)


def test_load_folder_unknown_class_folder_is_refused(tmp_path):
    make_split(tmp_path, "train", ("male", "female", "giraffe"), n=2)
    with pytest.raises(ValueError):
        load_folder(tmp_path / "train", CLASSES)


def test_train_writes_checkpoint_and_honest_metrics(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    for s in ("train", "val", "test"):
        make_split(data, s, CLASSES)
    m = train_crops(data, out, epochs=2, batch_size=8, threads=1)
    assert (out / "model.pt").exists() and (out / "metrics.json").exists()
    assert set(m["test"]["per_class"]) == set(CLASSES)
    assert m["train_images"] == 18 and m["test_images"] == 18
    assert 0.0 <= m["test"]["macro_f1"] <= 1.0


def test_pretrain_folder_with_two_classes_is_accepted(tmp_path):
    data, pre, out = tmp_path / "data", tmp_path / "pre", tmp_path / "out"
    for s in ("train", "val", "test"):
        make_split(data, s, CLASSES)
    make_split(pre, "train", ("male", "female"))
    m = train_crops(data, out, epochs=1, batch_size=8, threads=1, pretrain=pre, pretrain_epochs=1)
    assert m["pretrained_on"] == 12


def test_score_reports_balanced_accuracy_and_recall_spread():
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)
    pred = np.array([0] * 10 + [1] * 5 + [0] * 5 + [2] * 10)          # recalls 1.0, 0.5, 1.0
    s = score(y, pred, CLASSES)
    assert s["balanced_accuracy"] == pytest.approx(0.8333, abs=1e-3)
    assert s["recall_spread"] == pytest.approx(0.5)


def test_balanced_order_gives_every_class_the_same_share():
    y = np.array([0] * 3 + [1] * 30 + [2] * 300)
    order = balanced_order(y, np.random.default_rng(0))
    assert len(order) == len(y)
    assert np.bincount(y[order], minlength=3).tolist() == [111, 111, 111]
    assert set(order) <= set(range(len(y)))
    assert np.array_equal(order, balanced_order(y, np.random.default_rng(0)))        # same seed, same epoch


def test_cabin_augment_matches_input_shape_and_is_seeded():
    jpeg = cv2.imencode(".jpg", np.random.default_rng(0).integers(0, 255, (60, 50, 3), dtype=np.uint8))[1].tobytes()
    a = cabin_augment(jpeg, np.random.default_rng(1))
    assert a.shape == tc.center(jpeg).shape and np.isfinite(a).all()
    assert np.array_equal(a, cabin_augment(jpeg, np.random.default_rng(1)))
    assert not np.array_equal(a, cabin_augment(jpeg, np.random.default_rng(2)))


def test_regularised_balanced_run_records_the_bias_variance_evidence(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    for s in ("train", "val", "test"):
        make_split(data, s, CLASSES)
    m = train_crops(data, out, epochs=3, batch_size=8, threads=1, balanced=True, regularise=True, score_test=False)
    assert "test" not in m and m["test_images"] is None                 # selection runs never look at test
    assert len(m["history"]) >= 1 and {"train_loss", "val_loss", "val_macro_f1"} <= set(m["history"][0])
    assert {"macro_f1", "balanced_accuracy", "recall_spread"} <= set(m["train"]) and "recall_spread" in m["val"]
    assert m["balanced"] and m["regularise"]


def test_early_stopping_ends_before_the_epoch_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "PATIENCE", 1)
    monkeypatch.setattr(tc, "FREEZE_EPOCHS", 0)
    data = tmp_path / "data"
    for s in ("train", "val", "test"):
        make_split(data, s, CLASSES)
    m = train_crops(data, tmp_path / "out", epochs=12, batch_size=8, threads=1, regularise=True, score_test=False)
    assert len(m["history"]) < 12


def test_backbone_stays_frozen_during_the_frozen_epochs(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(tc, "FREEZE_EPOCHS", 99)
    data, out = tmp_path / "data", tmp_path / "out"
    for s in ("train", "val", "test"):
        make_split(data, s, CLASSES)
    train_crops(data, out, epochs=2, batch_size=8, threads=1, regularise=True, score_test=False)
    torch.manual_seed(0)
    fresh = tc.build_net(3).state_dict()
    saved = torch.load(out / "model.pt", weights_only=True)["state"]
    backbone = [k for k in fresh if k.startswith("0.")]
    assert backbone and all(torch.equal(fresh[k], saved[k]) for k in backbone)
    assert not torch.equal(fresh["4.weight"], saved["4.weight"])        # the head did learn


def test_adult_mix_adds_capped_male_female_crops_from_the_pretrain_folder(tmp_path):
    data, pre, out = tmp_path / "data", tmp_path / "pre", tmp_path / "out"
    for s in ("train", "val", "test"):
        make_split(data, s, CLASSES)
    make_split(pre, "train", ("male", "female"), n=10)
    m = train_crops(data, out, epochs=1, batch_size=8, threads=1, pretrain=pre, pretrain_epochs=1, adult_mix=4, score_test=False)
    assert m["train_images"] == 18 + 8 and m["adult_mix"] == 4


def test_pretrain_cache_is_reused_instead_of_retrained(tmp_path, capsys):
    data, pre, cache = tmp_path / "data", tmp_path / "pre", tmp_path / "pre.pt"
    for s in ("train", "val", "test"):
        make_split(data, s, CLASSES)
    make_split(pre, "train", ("male", "female"))
    train_crops(data, tmp_path / "o1", epochs=1, batch_size=8, threads=1, pretrain=pre, pretrain_epochs=1, pretrain_cache=cache, score_test=False)
    assert cache.exists() and "[pretrain]" in capsys.readouterr().out
    train_crops(data, tmp_path / "o2", epochs=1, batch_size=8, threads=1, pretrain=pre, pretrain_epochs=1, pretrain_cache=cache, score_test=False)
    assert "[pretrain]" not in capsys.readouterr().out
