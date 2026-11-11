"""Tests for the face-attribute model, FairFace adapter, trainer, evaluation metrics and threshold tuning (offline)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from passenger import face_eval, tune_classifier
from passenger.classifier import PassengerClassifier
from passenger.data_manager import DataManager
from passenger.datasets import fairface
from passenger.face_model import INPUT, TorchAgeGender, build_net, load_checkpoint, save_checkpoint, to_input
from passenger.registry import FACE_KEY_METRICS, Registry
from passenger.train_faces import augment, batches, center, predict_rows, train_faces


def jpeg(seed: int, size: int = 64) -> bytes:
    img = np.random.default_rng(seed).integers(0, 255, (size, size, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def write_parquet(path: Path, n: int, seed: int = 0, labels_ok: bool = True) -> int:
    rng = np.random.default_rng(seed)
    table = pa.table({
        "image": pa.array([{"bytes": jpeg(seed * 1000 + i), "path": f"{i}.jpg"} for i in range(n)],
                          type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        "age": pa.array(rng.integers(0, 9, n), type=pa.int64()),
        "gender": pa.array(rng.integers(0, 2, n), type=pa.int64()),
        "race": pa.array(np.zeros(n, dtype=np.int64)), "service_test": pa.array([False] * n),
    })
    names = fairface.AGE_NAMES if labels_ok else list(reversed(fairface.AGE_NAMES))
    feats = {"age": {"_type": "ClassLabel", "names": names}, "gender": {"_type": "ClassLabel", "names": fairface.GENDER_NAMES}}
    table = table.replace_schema_metadata({"huggingface": json.dumps({"info": {"features": feats}})})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, row_group_size=16)
    return path.stat().st_size


@pytest.fixture
def dm(tmp_path):
    return DataManager(tmp_path / "pd")


def make_fairface(dm, n_train=60, n_val=40):
    d = fairface.files_dir(dm)
    sizes = {"train-0.parquet": write_parquet(d / "train-0.parquet", n_train, 1),
             "validation-0.parquet": write_parquet(d / "validation-0.parquet", n_val, 2)}
    return fairface.ingest(dm, fetch=lambda url, dest: (_ for _ in ()).throw(AssertionError("no download expected")), sizes=sizes)


# -- metrics ---------------------------------------------------------------------------------------
def test_auc_perfect_inverted_and_ties():
    assert face_eval._auc(np.array([.9, .8]), np.array([.1, .2])) == 1.0
    assert face_eval._auc(np.array([.1]), np.array([.9])) == 0.0
    assert face_eval._auc(np.array([.5, .5]), np.array([.5, .5])) == 0.5
    assert np.isnan(face_eval._auc(np.array([]), np.array([.5])))


def test_attr_metrics_on_a_perfect_and_a_blind_model():
    ages = np.array([0, 1, 3, 4, 5, 8, 2])
    genders = np.array([0, 1, 0, 1, 0, 1, 0])
    perfect = np.array([[.9, .1, .9], [.1, .9, .9], [.9, .1, .1], [.1, .9, .1], [.9, .1, .1], [.1, .9, .1], [.9, .1, .5]])
    m = face_eval.attr_metrics(perfect, ages, genders)
    assert m["gender_acc"] == 1.0 and m["gender_acc_adult"] == 1.0 and m["child_auc"] == 1.0
    assert m["child_recall"] == 1.0 and m["child_false_alarm"] == 0.0 and m["n_child"] == 2 and m["n_adult"] == 4
    blind = perfect.copy()
    blind[:, 2] = 0.5                                              # no child signal at all
    assert face_eval.attr_metrics(blind, ages, genders)["child_auc"] == 0.5


def test_the_ambiguous_teen_group_is_not_scored_as_child_or_adult():
    ages, genders = np.array([2, 2, 0, 5]), np.array([0, 0, 0, 0])
    probs = np.array([[.9, .1, .99], [.9, .1, .01], [.9, .1, .9], [.9, .1, .1]])
    m = face_eval.attr_metrics(probs, ages, genders)
    assert m["n_child"] == 1 and m["n_adult"] == 1 and m["child_auc"] == 1.0      # the two 10-19 rows are ignored


# -- network --------------------------------------------------------------------------------------------
def test_input_conversion_shape_and_range():
    x = to_input(np.random.default_rng(0).integers(0, 255, (50, 70, 3), dtype=np.uint8))
    assert x.shape == (3, INPUT, INPUT) and x.dtype == np.float32 and -3 < x.min() and x.max() < 3.5


def test_checkpoint_round_trip_and_prediction_contract(tmp_path):
    import torch
    torch.manual_seed(0)
    net = build_net(pretrained=False)
    save_checkpoint(net, tmp_path / "m.pt", {"epoch": 3})
    loaded, meta = load_checkpoint(tmp_path / "m.pt")
    assert meta["epoch"] == 3
    x = torch.randn(2, 3, INPUT, INPUT)
    net.eval()
    with torch.no_grad():
        assert torch.allclose(net(x)[0], loaded(x)[0], atol=1e-6) and net(x)[1].shape == (2, 9)
    model = TorchAgeGender(tmp_path / "m.pt")
    face = np.random.default_rng(1).integers(0, 255, (60, 60, 3), dtype=np.uint8)
    pm, pf, pc = model.predict(face)
    assert pm + pf == pytest.approx(1.0, abs=1e-5) and 0.0 <= pc <= 1.0
    assert model.predict_batch([face, face])[0] == pytest.approx(model.predict(face))


def test_teen_weight_only_ever_raises_p_child_and_is_capped_at_one(tmp_path):
    save_checkpoint(build_net(False), tmp_path / "m.pt", {})
    face = np.random.default_rng(2).integers(0, 255, (60, 60, 3), dtype=np.uint8)
    low = TorchAgeGender(tmp_path / "m.pt", teen_weight=0.0).predict(face)[2]
    high = TorchAgeGender(tmp_path / "m.pt", teen_weight=1.0).predict(face)[2]
    assert high >= low and high <= 1.0


# -- FairFace adapter -----------------------------------------------------------------------------------
def test_ingest_registers_checksums_and_verify_raw_catches_tampering(dm):
    assert make_fairface(dm) == 2
    assert dm.verify_raw() == []
    entry = dm.raw_files[fairface.DATASET_ID]
    assert entry["license"] == "CC BY 4.0" and set(entry["files"]) == {"files/train-0.parquet", "files/validation-0.parquet"}
    victim = fairface.files_dir(dm) / "train-0.parquet"
    victim.write_bytes(victim.read_bytes() + b"x")
    assert any("checksum mismatch" in p for p in dm.verify_raw())
    victim.unlink()
    assert any("missing" in p for p in dm.verify_raw())


def test_ingest_downloads_missing_files_and_rejects_cut_off_ones(dm, tmp_path):
    src = tmp_path / "src"
    size = write_parquet(src / "train-0.parquet", 8, 5)
    fetched = []

    def fetch(url, dest):
        fetched.append(url)
        dest.write_bytes((src / "train-0.parquet").read_bytes()[: size // 2])    # cut off
    with pytest.raises(OSError, match="expected"):
        fairface.ingest(dm, fetch=fetch, sizes={"train-0.parquet": size})
    assert fetched and not dm.raw_files


def test_check_labels_rejects_a_reordered_label_set(tmp_path):
    good, bad = tmp_path / "g.parquet", tmp_path / "b.parquet"
    write_parquet(good, 4)
    write_parquet(bad, 4, labels_ok=False)
    fairface.check_labels(good)
    with pytest.raises(ValueError, match="unexpected FairFace label names"):
        fairface.check_labels(bad)


def test_rows_and_split_files(dm):
    make_fairface(dm, 30, 20)
    assert len(fairface.load_rows(fairface.split_files(dm, "train"))) == 30
    jpeg_bytes, age, gender = next(fairface.iter_rows(fairface.split_files(dm, "validation")[0]))
    assert jpeg_bytes[:2] == b"\xff\xd8" and 0 <= age < 9 and gender in (0, 1)


def test_benchmark_sample_is_fixed_and_its_id_changes_with_the_data(dm):
    make_fairface(dm, 10, 80)
    a, b = face_eval.benchmark_rows(dm, n=25, seed=0), face_eval.benchmark_rows(dm, n=25, seed=0)
    assert [r[1:] for r in a] == [r[1:] for r in b] and len(a) == 25
    bid = face_eval.benchmark_id(dm, n=25)
    assert bid == face_eval.benchmark_id(dm, n=25) and bid != face_eval.benchmark_id(dm, n=30)


# -- augmentation and batching ------------------------------------------------------------------------------
def test_augment_is_deterministic_per_seed_and_varies_across_seeds():
    j = jpeg(1, 224)
    a1 = augment(j, np.random.default_rng(3))
    assert a1.shape == (3, INPUT, INPUT) and a1.dtype == np.float32
    assert np.array_equal(a1, augment(j, np.random.default_rng(3)))
    assert not np.array_equal(a1, augment(j, np.random.default_rng(4)))
    assert np.array_equal(center(j), center(j))


def test_batches_shapes_order_and_epoch_shuffling():
    rows = [(jpeg(i), i % 9, i % 2) for i in range(70)]
    out = list(batches(rows, 16, train=True, seed=0, epoch=0, workers=2))
    assert all(x.shape[1:] == (3, INPUT, INPUT) for x, _, _ in out) and sum(len(g) for _, g, _ in out) <= 70
    ages0 = np.concatenate([a for _, _, a in out])
    again = np.concatenate([a for _, _, a in batches(rows, 16, train=True, seed=0, epoch=0, workers=2)])
    other = np.concatenate([a for _, _, a in batches(rows, 16, train=True, seed=0, epoch=1, workers=2)])
    assert np.array_equal(ages0, again) and not np.array_equal(ages0, other)
    ev = list(batches(rows, 32, train=False, workers=2))
    assert np.concatenate([a for _, _, a in ev]).tolist() == [r[1] for r in rows]       # evaluation keeps the order and every row


# -- training (tiny, real, offline) ----------------------------------------------------------------------------------
def test_train_faces_runs_end_to_end_and_registers_an_unscored_candidate(dm, tmp_path):
    make_fairface(dm, 120, 30)
    reg = Registry(tmp_path / "face" / "registry.json", FACE_KEY_METRICS)
    logs = []
    v = train_faces(dm, reg, epochs=1, batch=16, max_train=64, pretrained=False, holdout_every=10, workers=2, log=logs.append)
    entry = reg.models[v]
    assert v == "v0" and entry["status"] == "candidate" and entry["metrics"] is None
    assert Path(entry["weights"]).stat().st_size > 100_000
    assert entry["params"]["decision"] == {"teen_weight": 0.0, "child_threshold": 0.5, "min_conf": 0.6}
    assert any("held-out" in line for line in logs)
    probs, ages, genders = predict_rows(load_checkpoint(entry["weights"])[0], face_eval.benchmark_rows(dm, n=20))
    assert probs.shape == (20, 3) and np.all((probs >= 0) & (probs <= 1.0001))


def test_train_faces_needs_data(dm, tmp_path):
    reg = Registry(tmp_path / "face" / "registry.json", FACE_KEY_METRICS)
    with pytest.raises(ValueError, match="run python -m passenger.datasets.fairface"):
        train_faces(dm, reg, epochs=1)


# -- threshold tuning ----------------------------------------------------------------------------------------------------
def test_decide_rules():
    assert tune_classifier.decide(.9, .1, .96, 0.95, 0.6) == "child"
    assert tune_classifier.decide(.9, .1, .50, 0.95, 0.6) == "male"
    assert tune_classifier.decide(.2, .8, .0, 0.95, 0.6) == "female"
    assert tune_classifier.decide(.55, .45, .0, 0.95, 0.6) == "unknown"


def test_sweep_finds_a_perfect_setting_when_one_exists_and_reports_per_split():
    n = 24
    truth = (["male"] * 8 + ["female"] * 8 + ["child"] * 8)
    pg = np.array([[.9, .1]] * 8 + [[.1, .9]] * 8 + [[.9, .1]] * 8)
    pa = np.zeros((n, 9))
    pa[:16, 4] = 1.0                                              # adults: age group 30-39
    pa[16:, 1] = 1.0                                              # children: age group 3-9
    splits = ["train"] * 12 + ["val"] * 12
    res = tune_classifier.sweep(pg, pa, None, truth, splits)
    assert len(res) == len(tune_classifier.TEEN_WEIGHTS) * len(tune_classifier.CHILD_THRESHOLDS) * len(tune_classifier.MIN_CONFS)
    best = max(res, key=lambda r: r["macro_f1"])
    assert best["macro_f1"] == 1.0 and {"train", "val"} <= set(best)


def test_sweep_for_fixed_probabilities_has_no_teen_axis():
    truth = ["male", "female"]
    res = tune_classifier.sweep(np.array([[.9, .1], [.1, .9]]), None, np.array([0.0, 0.0]), truth, ["train", "val"])
    assert {r["teen_weight"] for r in res} == {0.0}


# -- registry additions ------------------------------------------------------------------------------------------------------
def test_face_registry_adopt_set_decision_and_gate(tmp_path):
    reg = Registry(tmp_path / "face" / "registry.json", FACE_KEY_METRICS)
    base = reg.add("gender_net.caffemodel", base=None, classes=None, dataset={}, params={"kind": "levi_hassner", "decision": {}})
    reg.record_metrics(base, "bench", {"gender_acc": 0.77, "child_auc": 0.62, "cls_macro_f1": 0.33})
    reg.adopt(base)
    assert reg.champion == base and not (reg.dir / "champion.pt").exists()      # nothing copied for a multi-file baseline
    w = tmp_path / "w.pt"
    w.write_bytes(b"w")
    cand = reg.add(w, base=None, classes=None, dataset={}, params={"decision": {"child_threshold": .5}})
    reg.record_metrics(cand, "bench", {"gender_acc": 0.90, "child_auc": 0.85, "cls_macro_f1": 0.30})
    ok, why = reg.compare(cand)
    assert not ok and "cls_macro_f1" in why[0]                                  # better on the benchmark, worse on the pipeline: blocked
    reg.set_decision(cand, {"child_threshold": .9}, tuned_on="train+val")
    assert reg.models[cand]["metrics"] is None and reg.models[cand]["params"]["tuned_on"] == "train+val"
    reg.record_metrics(cand, "bench", {"gender_acc": 0.90, "child_auc": 0.85, "cls_macro_f1": 0.45})
    ok, _ = reg.promote(cand)
    assert ok and reg.champion == cand and (reg.dir / "champion.pt").read_bytes() == b"w"


def test_registry_cli_face_flag_lists_the_face_registry(tmp_path, monkeypatch, capsys):
    from passenger import registry
    monkeypatch.chdir(tmp_path)
    reg = Registry("models/passenger/face/registry.json", FACE_KEY_METRICS)
    v = reg.add("x.pt", base=None, classes=None, dataset={"name": "ff"}, params={})
    reg.record_metrics(v, "b", {"gender_acc": 0.9})
    registry.main(["--face", "list"])
    assert "gender_acc=0.900" in capsys.readouterr().out


# -- classifier face_crop ---------------------------------------------------------------------------------------------------------
def test_face_crop_returns_the_model_input_or_none():
    class Never:
        def predict(self, f):
            raise AssertionError

    frame = np.full((200, 200, 3), 50, np.uint8)
    clf = PassengerClassifier(Never(), lambda crop: (40, 40, 40, 40), min_person_px=60, min_face_px=24, face_margin=0.5)
    assert clf.face_crop(frame, (0, 0, 1, 1)).shape[:2] == (80, 80)
    assert clf.face_crop(frame, (0, 0, 1, 0.1)) is None                          # person too short
    assert PassengerClassifier(Never(), lambda c: None).face_crop(frame, (0, 0, 1, 1)) is None
    assert PassengerClassifier(Never(), lambda c: (1, 1, 5, 5), min_face_px=24).face_crop(frame, (0, 0, 1, 1)) is None


# -- data manager: raw files ---------------------------------------------------------------------------------------------------------
def test_register_raw_files_validates_names_and_files(dm):
    with pytest.raises(ValueError):
        dm.register_raw_files("../x", {}, [])
    (dm.external_raw / "ds").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        dm.register_raw_files("ds", {}, ["nope.bin"])
    (dm.external_raw / "ds" / "a.bin").write_bytes(b"abc")
    assert dm.register_raw_files("ds", {"license": "x"}, ["a.bin"]) == 1 and dm.verify_raw() == []
    assert DataManager(dm.root).raw_files["ds"]["files"]["a.bin"]["bytes"] == 3          # persisted


def test_classifier_from_config_selects_the_trained_model(tmp_path, monkeypatch):
    from eagleeye.config import load_config
    import passenger.classifier as C
    save_checkpoint(build_net(False), tmp_path / "w.pt", {})
    monkeypatch.setattr(C, "MediaPipeFaceFinder", lambda conf: (lambda crop: None))   # no MediaPipe needed for this check
    cfg = load_config("config.yaml")
    monkeypatch.setattr(C, "AgeGenderNet", lambda model_dir: ("pretrained nets", model_dir))
    cfg.passenger.classifier.model = "levi_hassner"
    assert C.PassengerClassifier.from_config(cfg).model[0] == "pretrained nets"          # selectable as a fallback
    cfg.passenger.classifier.model = "torch"
    cfg.passenger.classifier.weights = str(tmp_path / "w.pt")
    cfg.passenger.classifier.teen_weight = 0.5
    clf = C.PassengerClassifier.from_config(cfg)
    assert isinstance(clf.model, TorchAgeGender) and clf.model.teen_weight == 0.5


def test_missing_trained_weights_give_an_actionable_error(tmp_path):
    from eagleeye.config import load_config
    import passenger.classifier as C
    cfg = load_config("config.yaml")
    cfg.passenger.classifier.model = "torch"
    cfg.passenger.classifier.weights = str(tmp_path / "nope.pt")
    with pytest.raises(FileNotFoundError, match="train_faces"):
        C.PassengerClassifier.from_config(cfg)
