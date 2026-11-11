#!/usr/bin/env python3
"""Start the driver monitor and the passenger monitor together.

    python run_all.py                                        # inputs from config.yaml: driver = camera.source (laptop webcam),
                                                             # passenger = passenger.camera (mode video -> data/passenger_demo.mp4)
    python run_all.py --passenger-source cabin.mp4           # passenger plays a different video, this run only
    python run_all.py --passenger-source 2                   # passenger from live camera 2, this run only
    python run_all.py --driver-source 0 --passenger-source 2 # two live cameras (they must be different devices)
    python run_all.py --driver-source drv.mp4 --passenger-source cabin.mp4   # two videos (no cameras needed)
    python run_all.py --no-passenger-classifier              # count + seats only (lightest on the CPU)

A number is a camera index; anything else is a video path. Passenger videos loop by default
(passenger.camera.loop). Build a test video with:  python scripts/make_demo_video.py

Each channel runs in its own process, and the dashboard (printed on start) shows
both feeds at once. If the passenger monitor stops, the driver monitor keeps
running; if the driver monitor stops (Ctrl+C or the Stop button), everything stops.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

HERE = Path(__file__).resolve().parent


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run EagleEye driver + passenger monitoring together")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--driver-source", default=None, help="driver camera index or video path")
    p.add_argument("--passenger-source", default=None, help="cabin camera index or video path")
    p.add_argument("--driver-id", default="driver_01")
    p.add_argument("--vehicle-id", default="vehicle_01")
    p.add_argument("--ui-port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--no-passenger-classifier", action="store_true", help="count and seats only (lightest)")
    p.add_argument("--loop", action="store_true", help="restart passenger video files when they end")
    return p.parse_args(argv)


def build_commands(args: argparse.Namespace, python: str = sys.executable) -> Tuple[List[str], List[str]]:
    driver = [python, str(HERE / "run.py"), "--config", args.config, "--driver-id", args.driver_id,
              "--vehicle-id", args.vehicle_id, "--ui-port", str(args.ui_port)]
    if args.driver_source is not None:
        driver += ["--source", str(args.driver_source)]
    if args.no_browser:
        driver.append("--no-browser")

    passenger = [python, str(HERE / "run_passenger.py"), "--config", args.config]
    if args.passenger_source is not None:
        passenger += ["--source", str(args.passenger_source)]
    if args.loop:
        passenger.append("--loop")
    if args.no_passenger_classifier:
        passenger.append("--no-classifier")
    return driver, passenger


def _is_camera(source) -> bool:
    return source is not None and str(source).isdigit()


def same_camera(args: argparse.Namespace, cfg) -> Optional[str]:
    """Two processes cannot share one webcam; return the shared index if they would.
    A passenger *video* never clashes with the driver's camera."""
    from passenger.source import resolve_source

    drv = args.driver_source if args.driver_source is not None else cfg.camera.source
    try:
        pax, is_video = resolve_source(cfg.passenger.camera, args.passenger_source, base=HERE)
    except (FileNotFoundError, ValueError):
        return None                                        # reported separately, with its own message
    return None if is_video else (str(drv) if _is_camera(drv) and str(drv) == str(pax) else None)


def supervise(driver: subprocess.Popen, passenger: subprocess.Popen, poll_s: float = 0.5) -> int:
    """Block until the driver monitor exits; report (but tolerate) a passenger exit."""
    reported = False
    while driver.poll() is None:
        if passenger.poll() is not None and not reported:
            print(f"[run_all] passenger monitor exited (code {passenger.returncode}); "
                  "driver monitoring continues. Dashboard shows the cabin feed offline.")
            reported = True
        time.sleep(poll_s)
    return driver.returncode


def stop(proc: subprocess.Popen, grace_s: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(grace_s)
    except subprocess.TimeoutExpired:
        proc.kill()


def main(argv=None) -> int:
    args = parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)               # logs must not sit in a buffer when piped to a file/journal
    sys.path.insert(0, str(HERE))
    from eagleeye.config import load_config

    cfg = load_config(args.config)
    from passenger.source import resolve_source
    try:
        resolve_source(cfg.passenger.camera, args.passenger_source, base=HERE)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[run_all] ERROR: {exc}")
        return 2
    clash = same_camera(args, cfg)
    if clash is not None:
        print(f"[run_all] ERROR: both monitors are set to camera {clash}, and one webcam cannot be opened "
              "twice. Pass --driver-source / --passenger-source, or set passenger.camera.mode: video in config.yaml.")
        return 2

    driver_cmd, passenger_cmd = build_commands(args)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    passenger = subprocess.Popen(passenger_cmd, cwd=HERE, env=env)   # first: its model load is the slow part
    driver = subprocess.Popen(driver_cmd, cwd=HERE, env=env)
    print(f"[run_all] driver pid {driver.pid}, passenger pid {passenger.pid}; "
          "the dashboard URL is printed by the driver monitor (Ctrl+C stops both)")
    code = 0
    try:
        code = supervise(driver, passenger)
    except KeyboardInterrupt:
        print("\n[run_all] stopping...")
    finally:
        stop(passenger)
        stop(driver)
    return code or 0


if __name__ == "__main__":
    sys.exit(main())
