"""Male / female / child from a person crop: the ensemble that ``passenger.classifier.model: crops`` selects."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from eagleeye.config import load_config
from passenger import classifier as clf_mod
from passenger.classifier import UNKNOWN, AttributeService, make_classifier
from passenger.crop_model import CropAttributeClassifier, load_crop_nets, person_crop
from passenger.seats import Detection
from passenger.train_crops import build_net            # the trainer's own network: what model.pt files hold


def save_net(path: Path, bias) -> Path:
    """A network whose answer ignores the picture: logits = ``bias``. Saved the way ``train_crops`` saves."""
    net = build_net(3)
    with torch.no_grad():
        net[-1].weight.zero_()
        net[-1].bias.copy_(torch.tensor(bias, dtype=torch.float32))
    torch.save({"state": net.state_dict(), "meta": {}}, path)
    return path


def nets(tmp_path, *biases):
    return load_crop_nets([save_net(tmp_path / f"n{i}.pt", b) for i, b in enumerate(biases)])


FRAME = np.zeros((200, 100, 3), np.uint8)
BOX = (0.1, 0.1, 0.9, 0.9)                          # 160 px tall, well above min_person_px


# ---- crop geometry (must match how crops_mfc_v4 was built: top 60% of the box, square, edge-replicate, 112 px) ----
def test_person_crop_is_the_top_60_percent_of_the_box_padded_to_a_112_square():
    frame = np.zeros((200, 100, 3), np.uint8)
    frame[:, :, 0] = np.arange(100, dtype=np.uint8)[None, :]          # blue = column
    frame[:, :, 1] = np.arange(200, dtype=np.uint8)[:, None]          # green = row
    crop = person_crop(frame, (0.2, 0.1, 0.6, 0.9))                   # box: columns 20-60, rows 20-180
    assert crop.shape == (112, 112, 3)
    assert 19 <= crop[:, :, 1].min() and crop[:, :, 1].max() <= 118   # rows 20-116 only: the lower 40% is not in it
    assert crop[:, :, 1].max() >= 105                                 # ... but the whole upper part is
    assert abs(int(crop[50, 0, 0]) - 20) <= 1 and abs(int(crop[50, -1, 0]) - 59) <= 1   # sides padded by repeating the edge


def test_person_crop_of_a_degenerate_box_is_none():
    assert person_crop(FRAME, (0.5, 0.5, 0.5, 0.9)) is None
    assert person_crop(FRAME, (0.2, 0.9, 0.6, 0.9)) is None


# ---- loading ----------------------------------------------------------------------------------------
def test_a_checkpoint_from_the_trainer_loads_into_the_runtime_network(tmp_path):
    reference = build_net(3)
    torch.save({"state": reference.state_dict(), "meta": {}}, tmp_path / "m.pt")
    (net,) = load_crop_nets([tmp_path / "m.pt"])
    x = torch.randn(2, 3, 112, 112)
    reference.eval()
    with torch.no_grad():
        assert torch.allclose(net(x), reference(x), atol=1e-5)       # same architecture, same weights, same answer


def test_missing_weights_say_how_to_make_them(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_crops"):
        load_crop_nets([tmp_path / "nope.pt"])


# ---- decisions --------------------------------------------------------------------------------------
def test_log_offsets_decide_between_equally_likely_classes(tmp_path):
    n = nets(tmp_path, (0, 0, 0))
    assert CropAttributeClassifier(n).classify(FRAME, BOX).label == "male"                       # a tie goes to the first class
    assert CropAttributeClassifier(n, log_offsets=(0.0, 2.0, 0.5)).classify(FRAME, BOX).label == "female"
    assert CropAttributeClassifier(n, log_offsets=(0.0, 0.5, 2.0)).classify(FRAME, BOX).label == "child"


def test_the_networks_are_averaged_before_the_decision(tmp_path):
    n = nets(tmp_path, (4, 0, 0), (4, 0, 0), (0, 4, 0))                # two for male, one for female: mean is male 0.65, female 0.33
    assert CropAttributeClassifier(n).classify(FRAME, BOX).label == "male"
    assert CropAttributeClassifier(n, log_offsets=(0.0, 2.0, 0.0)).classify(FRAME, BOX).label == "female"


def test_confidence_is_the_probability_after_the_offsets_and_min_prob_can_refuse(tmp_path):
    n = nets(tmp_path, (0, 0, 5))                                      # child with probability about 0.987
    a = CropAttributeClassifier(n).classify(FRAME, BOX)
    assert a.label == "child" and a.conf == pytest.approx(0.9867, abs=1e-3)
    assert CropAttributeClassifier(n, min_prob=0.98).classify(FRAME, BOX).label == "child"
    assert CropAttributeClassifier(n, min_prob=0.99).classify(FRAME, BOX) == clf_mod.Attribute(UNKNOWN, a.conf)


def test_a_person_too_small_to_read_is_unknown_without_running_the_networks(tmp_path):
    n = nets(tmp_path, (5, 0, 0))
    tiny = (0.1, 0.1, 0.4, 0.2)                                        # 20 px tall
    assert CropAttributeClassifier(n, min_person_px=60).classify(FRAME, tiny).label == UNKNOWN


def test_constructor_refuses_no_networks_and_bad_offsets(tmp_path):
    with pytest.raises(ValueError):
        CropAttributeClassifier([])
    with pytest.raises(ValueError):
        CropAttributeClassifier(nets(tmp_path, (0, 0, 0)), log_offsets=(0.0, 1.0))


def test_it_plugs_into_the_per_person_smoothing(tmp_path):
    svc = AttributeService(CropAttributeClassifier(nets(tmp_path, (0, 3, 0))), window=3, min_votes=1)
    assert svc.update(FRAME, [Detection(BOX, 0.9, 7)]) == {7: "female"}


# ---- configuration ----------------------------------------------------------------------------------
def crops_config(tmp_path, weights, **crops):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"passenger": {"classifier": {"model": "crops", "min_person_px": 60,
                                                              "crops": {"weights": [str(w) for w in weights], **crops}}}}))
    return load_config(str(p))


def test_from_config_builds_the_ensemble_with_the_configured_offsets(tmp_path):
    w = [save_net(tmp_path / f"n{i}.pt", (0, 0, 0)) for i in range(3)]
    cfg = crops_config(tmp_path, w, log_offsets={"male": 0.0, "female": 2.0, "child": 0.5}, min_prob=None)
    c = CropAttributeClassifier.from_config(cfg)
    assert len(c.nets) == 3 and c.log_offsets == (0.0, 2.0, 0.5)
    assert c.classify(FRAME, BOX).label == "female"


def test_from_config_without_weights_or_with_a_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="crops.weights"):
        CropAttributeClassifier.from_config(crops_config(tmp_path, []))
    with pytest.raises(FileNotFoundError, match="train_crops"):
        CropAttributeClassifier.from_config(crops_config(tmp_path, [tmp_path / "nope.pt"]))


def test_log_offsets_must_name_all_three_classes(tmp_path):
    w = [save_net(tmp_path / "n.pt", (0, 0, 0))]
    with pytest.raises(ValueError, match="log_offsets"):
        CropAttributeClassifier.from_config(crops_config(tmp_path, w, log_offsets={"male": 0.0, "female": 1.0}))


def test_make_classifier_picks_the_model_named_in_the_config(tmp_path, monkeypatch):
    monkeypatch.setattr(clf_mod.PassengerClassifier, "from_config", classmethod(lambda cls, cfg: "face classifier"))
    w = [save_net(tmp_path / "n.pt", (0, 0, 0))]
    assert isinstance(make_classifier(crops_config(tmp_path, w)), CropAttributeClassifier)
    face = crops_config(tmp_path, w)
    face.passenger.classifier["model"] = "torch"
    assert make_classifier(face) == "face classifier"


# ---- warm-up: the first pass through the networks is slow; do it before the video starts ----------------
def test_warm_up_runs_one_pass_and_from_config_does_it(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(CropAttributeClassifier, "warm_up", lambda self: calls.append(1))
    w = [save_net(tmp_path / "n.pt", (0, 0, 0))]
    CropAttributeClassifier.from_config(crops_config(tmp_path, w))
    assert calls == [1]


def test_warm_up_itself_is_harmless(tmp_path):
    c = CropAttributeClassifier(nets(tmp_path, (0, 3, 0)))
    assert c.warm_up() is None
    assert c.classify(FRAME, BOX).label == "female"                    # still answers normally afterwards


# ---- an unrecognised classifier name must not quietly become another model -------------------------------
def test_the_face_classifier_refuses_models_it_cannot_build(tmp_path):
    cfg = crops_config(tmp_path, [save_net(tmp_path / "n.pt", (0, 0, 0))])            # classifier.model: crops
    with pytest.raises(ValueError, match="make_classifier"):
        clf_mod.PassengerClassifier.from_config(cfg)
    cfg.passenger.classifier["model"] = "turbo"                                        # a typo used to select the research-licence nets
    with pytest.raises(ValueError, match="turbo"):
        clf_mod.PassengerClassifier.from_config(cfg)
