"""Tests for the model registry (promotion gate), the retraining flow and the detector class filter."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from passenger.data_manager import DataManager
from passenger.detector import PersonDetector
from passenger.registry import Registry, summarize
from passenger.train import train

GOOD = {"recall": 0.80, "precision": 0.85, "ap50": 0.80, "map50_95": 0.60, "count_mae": 0.40}


def report(**over):
    base = {"detection": {"recall": 0.8, "precision": 0.85, "ap50": 0.8, "map50_95": 0.6},
            "count": {"mae": 0.4}, "classification": None, "occupancy": None}
    for k, v in over.items():
        base[k] = v
    return base


@pytest.fixture
def reg(tmp_path):
    return Registry(tmp_path / "models" / "registry.json")


def weights(tmp_path, name="w.pt", data=b"weights"):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def add(reg, tmp_path, metrics, benchmark="bench1", base=None):
    v = reg.add(weights(tmp_path, f"{reg.next_version()}.pt"), base=base, classes=None, dataset={"name": "d"}, params={})
    if metrics is not None:
        reg.record_metrics(v, benchmark, metrics)
    return v


# -- summarize ------------------------------------------------------------------
def test_summarize_includes_optional_sections_only_when_present():
    assert set(summarize(report())) == {"recall", "precision", "ap50", "map50_95", "count_mae"}
    full = report(classification={"macro_f1": 0.5}, occupancy={"f1": 0.9})
    assert summarize(full)["cls_macro_f1"] == 0.5 and summarize(full)["occ_f1"] == 0.9


# -- registry ---------------------------------------------------------------------
def test_versions_increment_and_persist(reg, tmp_path):
    assert reg.next_version() == "v0"
    v0 = add(reg, tmp_path, GOOD)
    v1 = add(reg, tmp_path, GOOD)
    assert (v0, v1) == ("v0", "v1")
    reloaded = Registry(reg.path)
    assert set(reloaded.models) == {"v0", "v1"} and reloaded.models["v1"]["status"] == "candidate"
    with pytest.raises(ValueError):
        reg.add(weights(tmp_path), base=None, classes=None, dataset={}, params={}, version="v0")


def test_first_model_becomes_champion_and_weights_are_copied(reg, tmp_path):
    v0 = add(reg, tmp_path, GOOD)
    ok, why = reg.promote(v0)
    assert ok and reg.champion == "v0" and (reg.dir / "champion.pt").read_bytes() == b"weights"
    assert reg.champion_weights() == Path(reg.models["v0"]["weights"])


def test_better_model_replaces_champion_and_retires_the_old_one(reg, tmp_path):
    v0 = add(reg, tmp_path, GOOD)
    reg.promote(v0)
    v1 = add(reg, tmp_path, {**GOOD, "ap50": 0.85, "recall": 0.83}, base="v0")
    (reg.dir / "v1.pt").write_bytes(b"new")
    reg.models[v1]["weights"] = str(reg.dir / "v1.pt")
    ok, why = reg.promote(v1)
    assert ok and reg.champion == "v1" and reg.models["v0"]["status"] == "retired"
    assert (reg.dir / "champion.pt").read_bytes() == b"new"


@pytest.mark.parametrize("metric,value", [("recall", 0.70), ("ap50", 0.70), ("precision", 0.70), ("count_mae", 0.60)])
def test_a_regression_on_any_key_metric_blocks_promotion(reg, tmp_path, metric, value):
    reg.promote(add(reg, tmp_path, GOOD))
    v1 = add(reg, tmp_path, {**GOOD, metric: value})
    ok, why = reg.promote(v1)
    assert not ok and any(metric in w for w in why)
    assert reg.champion == "v0" and reg.models[v1]["status"] == "rejected"


def test_small_fluctuations_within_tolerance_are_allowed(reg, tmp_path):
    reg.promote(add(reg, tmp_path, GOOD))
    ok, _ = reg.promote(add(reg, tmp_path, {**GOOD, "recall": 0.79, "count_mae": 0.43}))
    assert ok


def test_scores_from_different_benchmarks_are_never_compared(reg, tmp_path):
    reg.promote(add(reg, tmp_path, GOOD, benchmark="old"))
    v1 = add(reg, tmp_path, {**GOOD, "ap50": 0.99}, benchmark="new")
    ok, why = reg.promote(v1)
    assert not ok and "benchmarks differ" in why[0]


def test_unscored_models_cannot_be_promoted(reg, tmp_path):
    ok, why = reg.promote(add(reg, tmp_path, None))
    assert not ok and "no benchmark scores" in why[0]
    reg.promote(add(reg, tmp_path, GOOD))
    ok, why = reg.promote(add(reg, tmp_path, None))
    assert not ok


def test_force_promotes_despite_regression_and_says_so(reg, tmp_path):
    reg.promote(add(reg, tmp_path, GOOD))
    v1 = add(reg, tmp_path, {**GOOD, "recall": 0.5})
    ok, why = reg.promote(v1, force=True)
    assert ok and reg.champion == v1 and "forced past" in why[0]


def test_classification_regression_blocks_promotion(reg, tmp_path):
    reg.promote(add(reg, tmp_path, {**GOOD, "cls_macro_f1": 0.60}))
    ok, why = reg.promote(add(reg, tmp_path, {**GOOD, "cls_macro_f1": 0.40}))
    assert not ok and "cls_macro_f1" in why[0]


# -- retraining flow (fake trainer, real data manager) ----------------------------------------
def make_dataset(tmp_path, n=40) -> DataManager:
    dm = DataManager(tmp_path / "pd")
    raw = dm.external_raw / "demo" / "images"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(0)
    items = []
    for i in range(n):
        cv2.imwrite(str(raw / f"i{i}.jpg"), rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        items.append({"name": f"i{i}", "image": f"images/i{i}.jpg", "boxes": [(i % 3, 0.5, 0.5, 0.4, 0.5)]})
    dm.add_external_dataset("demo", {"source": "synthetic"}, items)
    return dm


class FakeTrainer:
    def __init__(self):
        self.calls = []

    def __call__(self, base, data_yaml, epochs, imgsz, batch, device, out_dir):
        self.calls.append({"base": base, "yaml": Path(data_yaml), "epochs": epochs})
        best = out_dir / "run" / "weights" / "best.pt"
        best.parent.mkdir(parents=True, exist_ok=True)
        best.write_bytes(f"trained from {base}".encode())
        return best


def test_train_exports_trains_registers_lineage_and_scores(tmp_path, reg):
    dm = make_dataset(tmp_path)
    trainer = FakeTrainer()
    v = train(dm, reg, "ds1", epochs=3, trainer=trainer, evaluator=lambda w, c: report())
    assert v == "v0" and trainer.calls[0]["base"] == "yolov8n.pt" and trainer.calls[0]["epochs"] == 3
    entry = reg.models[v]
    assert entry["status"] == "candidate" and entry["base"] is None
    assert entry["dataset"]["name"] == "ds1" and entry["dataset"]["counts"]["train"] > 0
    assert entry["benchmark"] == dm.fingerprint("test") and entry["metrics"]["ap50"] == 0.8
    assert Path(entry["weights"]).read_bytes() == b"trained from yolov8n.pt"
    assert dm.verify_raw() == []                                   # training never touches the originals


def test_second_round_starts_from_the_champion_and_records_lineage(tmp_path, reg):
    dm = make_dataset(tmp_path)
    trainer = FakeTrainer()
    v0 = train(dm, reg, "ds1", trainer=trainer, evaluator=lambda w, c: report())
    reg.promote(v0)
    v1 = train(dm, reg, "ds2", trainer=trainer, evaluator=lambda w, c: report())
    assert trainer.calls[1]["base"] == str(reg.models[v0]["weights"])
    assert reg.models[v1]["base"] == v0 and reg.models[v1]["dataset"]["name"] == "ds2"


def test_explicit_base_weights_override_the_champion(tmp_path, reg):
    dm = make_dataset(tmp_path)
    trainer = FakeTrainer()
    reg.promote(train(dm, reg, "ds1", trainer=trainer, evaluator=lambda w, c: report()))
    train(dm, reg, "ds2", base="custom.pt", trainer=trainer, evaluator=lambda w, c: report())
    assert trainer.calls[1]["base"] == "custom.pt"


def test_the_test_split_is_never_given_to_training_and_the_benchmark_is_stable(tmp_path, reg):
    dm = make_dataset(tmp_path)
    trainer = FakeTrainer()
    fp_before = dm.fingerprint("test")
    train(dm, reg, "ds1", trainer=trainer)
    yaml = trainer.calls[0]["yaml"].read_text()
    assert "train: images/train" in yaml and "val: images/val" in yaml
    out = trainer.calls[0]["yaml"].parent
    test_names = {p.stem for p in (out / "images" / "test").glob("*.jpg")}
    assert test_names and not test_names & {p.stem for p in (out / "images" / "train").glob("*.jpg")}
    # more data arriving later must not move anything already in the test split
    raw = dm.external_raw / "demo" / "images"
    items = []
    for i in range(40, 80):
        cv2.imwrite(str(raw / f"i{i}.jpg"), np.full((64, 64, 3), i, np.uint8))
        items.append({"name": f"i{i}", "image": f"images/i{i}.jpg", "boxes": [(0, .5, .5, .4, .5)]})
    dm.add_external_dataset("demo", {}, items)
    old_test = {fr.stem for fr in dm.iter_frames("test") if int(fr.stem.split("__i")[1]) < 40}
    assert old_test == test_names
    assert dm.fingerprint("test") != fp_before                     # but the benchmark changed, so old scores are not comparable


def test_train_refuses_when_there_is_nothing_to_learn_or_judge_from(tmp_path, reg):
    empty = DataManager(tmp_path / "empty")
    with pytest.raises(ValueError, match="no labelled training images"):
        train(empty, reg, "ds", trainer=FakeTrainer())
    dm = make_dataset(tmp_path, n=40)
    for fr in list(dm.iter_frames("test")):                       # leave no benchmark
        fr.label.unlink()
    with pytest.raises(ValueError, match="no labelled test images"):
        train(dm, reg, "ds", trainer=FakeTrainer())


# -- detector class filter -------------------------------------------------------------------------
class FakeYolo:
    def __init__(self):
        self.kwargs = None

    def predict(self, frame, **kw):
        self.kwargs = kw

        class R:
            boxes = None
        return [R()]


def detector_with(classes):
    det = PersonDetector.__new__(PersonDetector)
    det._model, det.conf, det.imgsz = FakeYolo(), 0.3, 320
    det.classes = None if classes is None else list(classes)
    return det


def test_detector_filters_to_person_for_coco_and_to_nothing_for_finetuned_models():
    coco = detector_with((0,))
    coco.detect(np.zeros((8, 8, 3), np.uint8), track=False)
    assert coco._model.kwargs["classes"] == [0]
    tuned = detector_with(None)
    tuned.detect(np.zeros((8, 8, 3), np.uint8), track=False)
    assert "classes" not in tuned._model.kwargs


# -- real training smoke test (tiny, CPU) --------------------------------------------------------------
@pytest.mark.skipif(not (ROOT / "yolov8n.pt").exists(), reason="needs yolov8n.pt in the repo root")
def test_real_ultralytics_training_runs_end_to_end(tmp_path, reg, monkeypatch):
    monkeypatch.chdir(ROOT)
    dm = make_dataset(tmp_path, n=40)
    from passenger.train import ultralytics_trainer
    v = train(dm, reg, "tiny", base=str(ROOT / "yolov8n.pt"), epochs=1, imgsz=64, batch=4, device="cpu",
              trainer=ultralytics_trainer)
    assert Path(reg.models[v]["weights"]).stat().st_size > 1_000_000       # a real checkpoint
    tuned = PersonDetector(reg.models[v]["weights"], conf=0.01, imgsz=64, classes=None)
    assert isinstance(tuned.detect(np.zeros((64, 64, 3), np.uint8), track=False), list)


def test_rescore_refreshes_the_champion_on_the_current_test_set(tmp_path, monkeypatch, capsys):
    """After test data is added the benchmark fingerprint changes; rescoring lets later candidates be compared again."""
    from passenger import registry
    import passenger.evaluate as ev
    import passenger.detector as dt
    monkeypatch.chdir(tmp_path)
    reg = Registry("models/passenger/registry.json")
    w = tmp_path / "w.pt"
    w.write_bytes(b"w")
    v = reg.add(w, base=None, classes=[0], dataset={}, params={})
    reg.record_metrics(v, "old-benchmark", GOOD)
    reg.promote(v)
    seen = {}
    monkeypatch.setattr(dt, "PersonDetector", lambda weights, conf, imgsz, classes: seen.update(w=weights, c=classes) or object())
    monkeypatch.setattr(ev, "evaluate", lambda *a, **k: {**report(), "frames": 7})
    registry.main(["rescore"])
    entry = Registry("models/passenger/registry.json").models[v]
    assert entry["benchmark"] != "old-benchmark" and entry["metrics"]["ap50"] == 0.8
    assert seen == {"w": str(w), "c": [0]}                           # scored with the champion's own weights and class filter
    assert "re-scored on 7 test frames" in capsys.readouterr().out
