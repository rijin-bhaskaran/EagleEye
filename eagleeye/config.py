"""Configuration loading.

The defaults below mirror ``config.yaml`` so the package still runs if the
YAML file is missing. A user file is deep-merged on top of the defaults,
which means a partial YAML (overriding only two thresholds) is valid.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


DEFAULTS: Dict[str, Any] = {
    "camera": {"source": 1, "width": 640, "height": 480, "fps": 30, "flip": True},
    "face_mesh": {
        "max_num_faces": 1,
        "refine_landmarks": True,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    },
    "eye": {
        "ear_threshold": 0.21,
        "use_calibration": True,
        "calib_ratio": 0.75,
        "perclos_window_s": 60.0,
        "perclos_warn": 0.15,
        "perclos_critical": 0.30,
        "blink_min_s": 0.06,
        "blink_max_s": 0.50,
        "microsleep_s": 0.80,
    },
    "mouth": {
        "mar_threshold": 0.55,
        "use_calibration": True,
        "calib_multiplier": 2.2,
        "yawn_min_s": 1.2,
        "yawn_window_s": 120.0,
    },
    "head": {
        "yaw_threshold_deg": 25.0,
        "pitch_nod_deg": 15.0,
        "nod_min_s": 0.5,
        "off_road_min_s": 2.0,
        "nod_memory_s": 20.0,
    },
    "gaze": {
        "h_range": [0.36, 0.64],
        "v_range": [0.28, 0.75],
        "window_s": 10.0,
        "off_ratio_warn": 0.40,
    },
    "objects": {
        "enabled": True,
        "model": "yolov8n.pt",
        "conf": 0.35,
        "every_n_frames": 5,
        "phone_persist_s": 1.0,
        "hands": {
            "enabled": True,
            "max_num_hands": 2,
            "min_detection_confidence": 0.5,
            "wheel_roi": [0.20, 0.55, 0.90, 1.00],
            "hands_off_min_s": 2.0,
        },
    },
    "fusion": {
        "mode": "rule",
        "drowsy_threshold": 0.45,
        "distract_threshold": 0.45,
        "confirm_frames": 8,
        "release_frames": 20,
        "min_state_s": 1.5,
        "weights": {
            "drowsy": {"perclos": 0.40, "microsleep": 0.20, "yawn": 0.20, "nod": 0.20},
            "distract": {"phone": 0.35, "gaze": 0.25, "head": 0.25, "hands": 0.15},
        },
    },
    "temporal": {"checkpoint": "models/eagleeye_lstm.pt", "window": 30, "blend": 0.5},
    "alerts": {
        "enabled": True,
        "cooldown_s": 5.0,
        "escalate_s": 8.0,
        "volume": 0.35,
        "tts": False,
    },
    "logging": {
        "dir": "logs",
        "frame_log": True,
        "db": "logs/eagleeye.db",
        "flush_every": 100,
    },
    "calibration": {"duration_s": 10.0, "file": "calibration.json"},
    "passenger": {
        "vehicle_type": "bus",
        "threads": 4,
        "vehicles": {},
        "car": {"pipeline": "cabin_model", "weights": "models/car_cabin/champion.pt", "conf": 0.35, "imgsz": 640, "backend": "torch", "seats": 5,
                "rules": {"window": 15, "min_votes": 3, "belt_overlap": 0.5, "belt_on_ratio": 0.3,
                          "child_seat_overlap": 0.5, "seat_window": 15}},
        "camera": {"mode": "video", "index": 1, "video_path": "data/passenger_demo.mp4", "loop": True, "width": 1280, "height": 720, "capture_fps": 30, "fps": 10},
        "server": {"port": 8100},
        "display": {"draw_seats": False},
        "detector": {"model": "yolov8n.pt", "conf": 0.40, "imgsz": 640, "classes": [0], "backend": "openvino"},
        "seats": {"rows": 10, "left": 2, "right": 3, "polygons": {}, "polygons_file": None},
        "occupancy": {"min_overlap": 0.30, "on_frames": 3, "off_frames": 15, "count_window": 5},
        "seat_state": {"enabled": False, "weights": "models/seat_state/v2b/run/weights/best.pt", "conf": 0.35,
                       "imgsz": 640, "every": 3, "window": 5, "count_reads": 9, "hold_reads": 3, "on_reads": 2, "off_reads": 5},
        "classifier": {
            "model": "levi_hassner", "weights": "models/passenger/face/champion.pt", "teen_weight": 0.0,
            "model_dir": "models/passenger", "face_confidence": 0.5, "face_detector": "mediapipe",
            "yunet_weights": "models/passenger/face_detection_yunet_2023mar.onnx", "yunet_score": 0.6, "face_margin": 0.4, "min_person_px": 60,
            "min_face_px": 24, "child_threshold": 0.95, "min_conf": 0.6,
            "window": 15, "min_votes": 3, "budget_per_frame": 3, "background": True,
        },
    },
}


class Cfg(dict):
    """A dict that also supports attribute access (``cfg.eye.ear_threshold``)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Cfg({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _apply_vehicle_profile(data: Dict[str, Any], vehicle_type: str | None = None) -> Dict[str, Any]:
    """``passenger.vehicles.<vehicle_type>`` replaces, setting by setting, the shared ``passenger`` settings of the same name.

    So one switch (``passenger.vehicle_type``) selects the weights, the static video and anything else that differs between
    a bus and a car. A vehicle with no profile keeps the shared settings."""
    p = data.get("passenger")
    if not isinstance(p, dict):
        return data
    if vehicle_type is not None:                       # a caller that means one vehicle whatever the file says (benchmarks, tools)
        p["vehicle_type"] = vehicle_type
    name = str(p.get("vehicle_type", "bus")).strip().lower()
    profiles = {str(k).strip().lower(): v for k, v in (p.get("vehicles") or {}).items()}
    if isinstance(profiles.get(name), dict):
        data["passenger"] = _deep_merge(p, profiles[name])
    return data


def load_config(path: str | Path | None = None, vehicle_type: str | None = None) -> Cfg:
    """Load ``config.yaml`` merged over the built-in defaults, then apply the selected vehicle's profile.

    ``vehicle_type`` overrides ``passenger.vehicle_type`` from the file for this call only."""
    data = copy.deepcopy(DEFAULTS)
    if path is not None:
        p = Path(path)
        if p.exists():
            if yaml is None:
                raise RuntimeError("PyYAML is required to read a config file")
            user = yaml.safe_load(p.read_text()) or {}
            data = _deep_merge(data, user)
    return _wrap(_apply_vehicle_profile(data, vehicle_type))


def load_calibration(path: str | Path) -> Dict[str, float] | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def save_calibration(path: str | Path, values: Dict[str, float]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(values, indent=2))
