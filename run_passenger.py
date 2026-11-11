#!/usr/bin/env python3
"""EagleEye passenger monitor: the cabin camera, in its own process.

    python run_passenger.py                          # camera from config.yaml
    python run_passenger.py --source cabin.mp4 --loop
    python run_passenger.py --no-classifier          # count + seats only (lightest)

It serves its feed on a loopback port (passenger.server.port); the driver
dashboard proxies it, so you normally start both with ``python run_all.py``.
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path

from eagleeye.config import load_config
from eagleeye.video import VideoStream
from passenger.hub import PassengerHub
from passenger.runner import PassengerRunner
from passenger.server import PassengerServer
from passenger.source import resolve_source
from passenger.vehicle import build_pipeline, vehicle_type


ROOT = Path(__file__).resolve().parent


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EagleEye passenger monitoring")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", default=None, help="camera index or video path for this run (default: passenger.camera in config.yaml)")
    p.add_argument("--fps", type=float, default=None, help="target inference FPS (default: passenger.camera.fps)")
    p.add_argument("--port", type=int, default=None, help="loopback port (default: passenger.server.port)")
    p.add_argument("--loop", action="store_true", help="restart a video file when it ends (config: passenger.camera.loop)")
    p.add_argument("--no-classifier", action="store_true", help="skip Male/Female/Child (faster)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    cam = cfg.passenger.camera
    try:
        source, is_video = resolve_source(cam, args.source, base=ROOT)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"[passenger] {exc}")
    loop = args.loop or (is_video and bool(cam.get("loop", True)))
    fps = args.fps or cam.fps
    port = args.port or cfg.passenger.server.port

    import torch
    torch.set_num_threads(max(int(cfg.passenger.get("threads", 4)), 1))   # leave cores for the driver monitor

    hub = PassengerHub()
    try:
        server = PassengerServer(hub, port)
    except OSError as exc:
        sys.exit(f"[passenger] cannot listen on 127.0.0.1:{port} ({exc}). Is another passenger monitor running?")

    try:
        stream = VideoStream(source, cam.width, cam.height, cam.capture_fps, flip=False)
    except RuntimeError as exc:
        server.shutdown()
        sys.exit(f"[passenger] {exc}")

    try:
        pipeline = build_pipeline(cfg, with_classifier=not args.no_classifier)
    except (FileNotFoundError, ValueError) as exc:
        stream.release()
        server.shutdown()
        sys.exit(f"[passenger] {exc}")
    print(f"[passenger] vehicle type: {vehicle_type(cfg)} (detector {cfg.passenger.detector.model}, classifier {cfg.passenger.classifier.model})")
    runner = PassengerRunner(stream, pipeline, hub, fps=fps, draw_seats=cfg.passenger.display.draw_seats,
                             loop=loop)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())     # `kill` / launcher shutdown
    server.start()
    print(f"[passenger] source: {'video ' + str(source) if is_video else 'camera ' + str(source)}"
          f"{' (looping)' if loop else ''}")
    print(f"[passenger] feed on {server.url}/video, target {fps:g} FPS, "
          f"classifier {'off' if args.no_classifier else 'on'}")
    try:
        runner.run(stop)
    except KeyboardInterrupt:
        print("\n[passenger] interrupted")
    finally:
        stream.release()
        server.shutdown()
        print(f"[passenger] {runner.frames} frames processed")


if __name__ == "__main__":
    main()
