"""Tests for choosing the passenger input (camera or video) and the demo-video builder."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_all
import run_passenger
from eagleeye.config import Cfg, load_config
from passenger.data_manager import DataManager
from passenger.source import resolve_source


def video_file(path: Path) -> Path:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
    for _ in range(3):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    return path


def cam(**kw) -> Cfg:
    return Cfg({"mode": "video", "index": 1, "video_path": "clip.mp4", **kw})


# -- resolve_source ------------------------------------------------------------------
def test_camera_mode_returns_the_device_index():
    assert resolve_source(cam(mode="camera", index=2)) == (2, False)
    assert resolve_source(cam(mode="CAMERA", index="3")) == (3, False)                 # case and numeric strings tolerated


def test_video_mode_returns_an_existing_file_relative_to_the_base(tmp_path):
    clip = video_file(tmp_path / "clip.mp4")
    assert resolve_source(cam(), base=tmp_path) == (str(clip), True)
    assert resolve_source(cam(video_path=str(clip)), base=Path("/somewhere/else")) == (str(clip), True)   # absolute paths ignore the base


def test_a_missing_video_gives_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_demo_video"):
        resolve_source(cam(), base=tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_source(cam(video_path=""), base=tmp_path)                              # empty path is not a file


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="'camera' or 'video'"):
        resolve_source(cam(mode="webcam"))


def test_command_line_override_wins_over_the_config(tmp_path):
    clip = video_file(tmp_path / "other.mp4")
    assert resolve_source(cam(mode="camera"), override="2") == (2, False)             # a number is a camera
    assert resolve_source(cam(mode="camera"), override=str(clip)) == (str(clip), True)  # anything else is a video
    with pytest.raises(FileNotFoundError):
        resolve_source(cam(), override=str(tmp_path / "nope.mp4"))


def test_the_shipped_config_selects_video_with_a_video_path():
    c = load_config(str(ROOT / "config.yaml")).passenger.camera
    assert c.mode == "video" and c.video_path.lower().endswith((".mp4", ".avi", ".webm")) and c.loop is True and c.index == 1
    assert (ROOT / c.video_path).is_file()                          # the configured clip is really there


# -- launcher: video mode never clashes with the driver camera ---------------------------------
def args(**kw):
    ns = run_all.parse_args([])
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def cfg_with(tmp_path, **camera):
    cfg = load_config(str(ROOT / "config.yaml"))
    cfg.passenger.camera.update(camera)
    return cfg


def test_a_passenger_video_can_run_beside_the_driver_webcam(tmp_path):
    clip = video_file(tmp_path / "clip.mp4")
    cfg = cfg_with(tmp_path, mode="video", video_path=str(clip))
    assert run_all.same_camera(args(driver_source="0"), cfg) is None
    assert run_all.same_camera(args(driver_source="1"), cfg) is None                   # same index number, but it is a video


def test_two_webcams_on_one_index_are_still_refused(tmp_path):
    cfg = cfg_with(tmp_path, mode="camera", index=1)
    assert run_all.same_camera(args(driver_source="1"), cfg) == "1"
    assert run_all.same_camera(args(driver_source="0"), cfg) is None


def test_run_all_stops_early_with_a_clear_message_when_the_video_is_missing(tmp_path, capsys):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("passenger:\n  camera:\n    mode: video\n    video_path: definitely/not/here.mp4\n")
    assert run_all.main(["--config", str(cfg_file)]) == 2
    out = capsys.readouterr().out
    assert "passenger video not found" in out and "make_demo_video" in out


def test_run_passenger_exits_cleanly_when_the_video_is_missing(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("passenger:\n  camera:\n    mode: video\n    video_path: definitely/not/here.mp4\n")
    with pytest.raises(SystemExit) as exc:
        run_passenger.main(["--config", str(cfg_file), "--port", "0"])
    assert "passenger video not found" in str(exc.value)


# -- demo video builder ------------------------------------------------------------------------
def test_demo_video_is_built_from_the_photos_in_passenger_data(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_demo_video as mdv

    dm = DataManager(tmp_path / "pd")
    raw = dm.external_raw / "ds" / "images"
    raw.mkdir(parents=True)
    items = []
    for i, ctx in enumerate(["general", "bus", "vehicle"]):
        cv2.imwrite(str(raw / f"i{i}.jpg"), np.full((90, 160, 3), 40 * (i + 1), np.uint8))
        items.append({"name": f"i{i}", "image": f"images/i{i}.jpg", "boxes": [(0, .5, .5, .2, .2)], "context": ctx})
    dm.add_external_dataset("ds", {}, items)
    assert [p.stem for p in mdv.pick_images(dm, 3)] == ["i1", "i2", "i0"]               # bus, then vehicle, then the rest

    out = tmp_path / "demo.mp4"
    assert mdv.build(dm, out, seconds=6, fps=5, seconds_per_photo=2.0) == 3
    cap = cv2.VideoCapture(str(out))
    assert (int(cap.get(3)), int(cap.get(4)), int(cap.get(7))) == (1280, 720, 30)       # 3 photos x 2 s x 5 fps


def test_letterbox_keeps_the_aspect_ratio():
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_demo_video as mdv

    out = mdv.letterbox(np.full((100, 100, 3), 255, np.uint8))
    assert out.shape == (720, 1280, 3) and out[:, :280].sum() == 0 and out[:, 1000:].sum() == 0 and out[360, 640].all()   # a square becomes 720 px wide, centred: black bars either side


def test_demo_builder_refuses_to_run_without_photos(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_demo_video as mdv

    with pytest.raises(SystemExit, match="no photos"):
        mdv.build(DataManager(tmp_path / "empty"), tmp_path / "x.mp4")
