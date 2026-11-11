"""``passenger.vehicles.<bus|car>`` in config.yaml: one switch (``vehicle_type``) selects the weights, the static video and
every other setting that differs between the two vehicles."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eagleeye.config import load_config
from passenger.vehicle import vehicle_type


def write(tmp_path, passenger: dict) -> str:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump({"passenger": passenger}))
    return str(p)


PROFILES = {"bus": {"camera": {"video_path": "data/bus/b.mp4"}, "detector": {"conf": 0.40}},
            "car": {"camera": {"video_path": "data/car/c.mp4"}, "detector": {"conf": 0.35, "imgsz": 416},
                    "classifier": {"model": "crops"}}}


def test_the_selected_vehicle_overrides_the_shared_settings(tmp_path):
    p = load_config(write(tmp_path, {"vehicle_type": "car", "camera": {"fps": 9, "video_path": "base.mp4"},
                                     "vehicles": PROFILES})).passenger
    assert p.camera.video_path == "data/car/c.mp4"
    assert p.camera.fps == 9                                   # shared settings the profile does not name are kept
    assert p.detector.conf == 0.35 and p.detector.imgsz == 416
    assert p.detector.model == "yolov8n.pt"                    # ... including defaults, because the merge is deep
    assert p.classifier.model == "crops"


def test_switching_vehicle_type_switches_weights_and_video_together(tmp_path):
    cfg = {"camera": {"video_path": "base.mp4"}, "vehicles": PROFILES}
    bus = load_config(write(tmp_path, {**cfg, "vehicle_type": "bus"})).passenger
    car = load_config(write(tmp_path, {**cfg, "vehicle_type": "car"})).passenger
    assert (bus.camera.video_path, bus.detector.conf, bus.classifier.model) == ("data/bus/b.mp4", 0.40, "levi_hassner")
    assert (car.camera.video_path, car.detector.conf, car.classifier.model) == ("data/car/c.mp4", 0.35, "crops")


def test_the_vehicle_name_is_not_case_sensitive(tmp_path):
    p = load_config(write(tmp_path, {"vehicle_type": " Car ", "vehicles": PROFILES})).passenger
    assert p.camera.video_path == "data/car/c.mp4"


def test_a_vehicle_without_a_profile_keeps_the_shared_settings(tmp_path):
    p = load_config(write(tmp_path, {"vehicle_type": "car", "camera": {"video_path": "base.mp4"},
                                     "vehicles": {"bus": PROFILES["bus"]}})).passenger
    assert p.camera.video_path == "base.mp4"


def test_no_vehicles_block_means_nothing_changes(tmp_path):
    p = load_config(write(tmp_path, {"vehicle_type": "car", "camera": {"video_path": "base.mp4"}})).passenger
    assert p.camera.video_path == "base.mp4" and p.detector.conf == 0.40


def test_an_empty_or_null_profile_is_harmless(tmp_path):
    for profile in ({}, None):
        p = load_config(write(tmp_path, {"vehicle_type": "bus", "camera": {"video_path": "base.mp4"},
                                         "vehicles": {"bus": profile}})).passenger
        assert p.camera.video_path == "base.mp4"


def test_a_typo_in_vehicle_type_is_still_rejected_by_the_vehicle_check(tmp_path):
    cfg = load_config(write(tmp_path, {"vehicle_type": "truck", "vehicles": PROFILES}))
    with pytest.raises(ValueError, match="bus.*car"):
        vehicle_type(cfg)


# ---- the shipped config.yaml ----------------------------------------------------------------------
def shipped(tmp_path, vehicle: str):
    data = yaml.safe_load((ROOT / "config.yaml").read_text())
    data["passenger"]["vehicle_type"] = vehicle
    p = tmp_path / f"{vehicle}.yaml"
    p.write_text(yaml.safe_dump(data))
    return load_config(str(p)).passenger


def test_shipped_config_has_a_profile_for_both_vehicles(tmp_path):
    vehicles = yaml.safe_load((ROOT / "config.yaml").read_text())["passenger"]["vehicles"]
    assert set(vehicles) == {"bus", "car"}


def test_shipped_config_bus_profile_uses_the_bus_video_and_the_face_models(tmp_path):
    p = shipped(tmp_path, "bus")
    assert p.camera.mode == "video" and p.camera.video_path.startswith("data/bus/")
    assert (ROOT / p.camera.video_path).is_file()
    assert p.classifier.model in ("torch", "levi_hassner")
    assert p.seat_state.enabled and "seat_state" in p.seat_state.weights


def test_shipped_config_car_profile_uses_the_car_video_and_the_crop_models(tmp_path):
    p = shipped(tmp_path, "car")
    assert p.camera.mode == "video" and p.camera.video_path.startswith("data/car/")
    assert (ROOT / p.camera.video_path).is_file()
    assert p.car.pipeline == "two_stage" and p.classifier.model == "crops"
    weights = p.classifier.crops.weights
    assert len(weights) == 3 and all(w.endswith("model.pt") for w in weights)
    assert set(p.classifier.crops.log_offsets) == {"male", "female", "child"}
    assert p.seat_state.enabled is False                       # the bus seat model does not apply to a car
    assert p.car.seats >= 1


def test_the_two_profiles_do_not_share_a_video_or_a_classifier(tmp_path):
    bus, car = shipped(tmp_path, "bus"), shipped(tmp_path, "car")
    assert bus.camera.video_path != car.camera.video_path
    assert bus.classifier.model != car.classifier.model


def test_shipped_car_profile_calming_settings_are_valid(tmp_path):
    """The new blocks are real settings: a typo in config.yaml must fail here, not silently on the dashboard."""
    from passenger.detector import tracker_yaml
    from passenger.stability import TrackStabiliser

    p = shipped(tmp_path, "car")
    st = {k: v for k, v in p.car.stability.items() if k != "enabled"}
    assert TrackStabiliser(**st).hold_frames == st["hold_frames"]
    assert Path(tracker_yaml(dict(p.detector.tracker))).is_file()
    assert p.detector.tracker.new_track_thresh >= p.detector.tracker.track_low_thresh     # a weak box may continue a track, not start one
    assert p.detector.conf <= p.detector.tracker.track_low_thresh                          # the tracker can only use boxes the detector passes on


def test_the_bus_profile_has_no_calming_settings_so_the_bus_is_unchanged(tmp_path):
    p = shipped(tmp_path, "bus")
    assert "stability" not in p.car or not p.car.get("stability")
    assert not p.detector.get("tracker") and p.detector.conf == 0.40


def test_shipped_car_profile_limits_duplicate_boxes_and_the_bus_does_not(tmp_path):
    assert 0.0 < shipped(tmp_path, "car").detector.iou < 0.7                                # tighter than the model's own 0.7
    assert shipped(tmp_path, "bus").detector.get("iou") is None                             # the bus detector is untouched


def test_a_caller_can_force_the_vehicle_whatever_the_file_says(tmp_path):
    """Benchmarks and tools that mean 'the bus pipeline' must not change with the vehicle the owner has toggled."""
    path = write(tmp_path, {"vehicle_type": "car", "camera": {"video_path": "base.mp4"}, "vehicles": PROFILES})
    forced = load_config(path, vehicle_type="bus").passenger
    assert forced.vehicle_type == "bus" and forced.camera.video_path == "data/bus/b.mp4" and forced.classifier.model == "levi_hassner"
    assert load_config(path).passenger.vehicle_type == "car"                       # the default still follows the file
    assert load_config(path, vehicle_type="Car").passenger.camera.video_path == "data/car/c.mp4"
