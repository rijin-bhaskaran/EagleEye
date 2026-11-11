"""Smoke tests that launch the real entry points (real model, real process, real sockets).

Slower than the unit tests (model load), so there are only a few. They skip if the weights are absent.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from passenger.data_manager import DataManager
from passenger.evaluate import main as evaluate_main

needs_weights = pytest.mark.skipif(not (ROOT / "yolov8n.pt").exists(), reason="needs yolov8n.pt in the repo root")
NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_video(path: Path, frames=40, fps=10, size=(320, 240), seed=0) -> Path:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    rng = np.random.default_rng(seed)
    for _ in range(frames):
        w.write(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    w.release()
    return path


def config_with_seat_state(tmp_path: Path, enabled: bool) -> Path:
    """A copy of config.yaml with the seat-state model switched on or off, so a test does not depend on the shipped choice."""
    import yaml

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["passenger"]["vehicle_type"] = "bus"                       # the seat model is a bus feature, whichever vehicle is shipped
    cfg["passenger"].setdefault("seat_state", {})["enabled"] = enabled
    cfg["passenger"]["vehicles"]["bus"]["seat_state"]["enabled"] = enabled      # the bus profile owns this setting
    out = tmp_path / "config.yaml"
    out.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return out


@needs_weights
@pytest.mark.parametrize("seat_state", [False, True])
def test_run_passenger_process_serves_telemetry_and_video_then_stops(tmp_path, seat_state):
    video = make_video(tmp_path / "cabin.mp4")
    port = free_port()
    config = config_with_seat_state(tmp_path, seat_state)
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "run_passenger.py"), "--config", str(config), "--source", str(video),
         "--port", str(port), "--no-classifier", "--fps", "5", "--loop"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        data = None
        deadline = time.time() + 90
        while time.time() < deadline and proc.poll() is None:
            try:
                with NO_PROXY.open(f"http://127.0.0.1:{port}/api/passenger", timeout=2) as r:
                    data = json.loads(r.read())
                if data.get("text"):
                    break
            except OSError:
                pass
            time.sleep(0.5)
        assert proc.poll() is None, "passenger process exited early"
        assert data and data["age_s"] < 5
        t = data["text"]
        if seat_state:                                              # seats come from the model: occupied + vacant = total
            assert int(t["paxOccupied"]) + int(t["paxVacant"]) == int(t["paxTotal"])
        else:                                                       # placeholder grid: 10 rows x 5 seats
            assert t["paxTotal"] == "50"
        with NO_PROXY.open(f"http://127.0.0.1:{port}/healthz", timeout=2) as r:
            assert json.loads(r.read()) == {"ok": True}
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def config_for_vehicle(tmp_path: Path, vehicle: str) -> Path:
    import yaml

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["passenger"]["vehicle_type"] = vehicle
    out = tmp_path / f"{vehicle}.yaml"
    out.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return out


@needs_weights
@pytest.mark.skipif(not (ROOT / "models/passenger/crops/v4_seed0/model.pt").exists(), reason="needs the trained crop classifier")
def test_run_passenger_process_runs_the_car_profile_with_its_own_models_and_video(tmp_path):
    """vehicle_type: car launches the two-stage chain from its profile: crop classifier, car video (no --source given)."""
    port = free_port()
    proc = subprocess.Popen([sys.executable, str(ROOT / "run_passenger.py"), "--config", str(config_for_vehicle(tmp_path, "car")),
                             "--port", str(port), "--fps", "5"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        data, deadline = None, time.time() + 120
        while time.time() < deadline and proc.poll() is None:
            try:
                with NO_PROXY.open(f"http://127.0.0.1:{port}/api/passenger", timeout=2) as r:
                    data = json.loads(r.read())
                if data.get("text"):
                    break
            except OSError:
                pass
            time.sleep(0.5)
        assert proc.poll() is None, "passenger process exited early"
        t = data["text"]
        assert t["paxClassifier"] == "Crop model live" and t["paxTotal"] == "5"      # crop models, seats gauge from passenger.car.seats
        assert "car" not in data                                                       # no belt / child-seat block: no cabin model
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@needs_weights
def test_second_passenger_process_on_the_same_port_refuses_to_start(tmp_path):
    video = make_video(tmp_path / "cabin.mp4")
    port = free_port()
    cmd = [sys.executable, str(ROOT / "run_passenger.py"), "--source", str(video), "--port", str(port), "--no-classifier"]
    first = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                NO_PROXY.open(f"http://127.0.0.1:{port}/healthz", timeout=1).close()
                break
            except OSError:
                time.sleep(0.3)
        second = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60)
        assert second.returncode != 0 and "cannot listen" in (second.stdout + second.stderr)
    finally:
        first.terminate()
        first.wait(timeout=10)


@needs_weights
def test_evaluate_cli_runs_end_to_end_on_a_labelled_split(tmp_path, capsys):
    dm = DataManager(tmp_path / "pd")
    for i in range(3):
        vid = dm.register_video(make_video(tmp_path / f"v{i}.mp4", frames=6, fps=5, seed=i), vehicle="bus1")
        dm.extract_frames(vid, target_fps=5, dedupe_threshold=0)
    dm.assign_splits()
    for jpg in dm.frames_dir.glob("*/*.jpg"):
        (dm.labels_dir / f"{jpg.stem}.txt").write_text("0 0.5 0.5 0.2 0.3\n")
    cwd = Path.cwd()
    try:
        import os
        os.chdir(ROOT)                                    # config.yaml and yolov8n.pt are resolved from here
        evaluate_main(["--data", str(tmp_path / "pd"), "--split", "test"])
    finally:
        os.chdir(cwd)
    report = json.loads(capsys.readouterr().out)
    assert report["frames"] > 0 and report["split"] == "test"
    assert {"detection", "count", "occupancy", "speed"} <= set(report)
    assert report["detection"]["fn"] == report["frames"]  # random noise has no people, every label is missed
    assert "classification" not in report
