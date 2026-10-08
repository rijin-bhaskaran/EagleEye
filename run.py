#!/usr/bin/env python3
"""EagleEye live runner.

    python run.py                          # webcam + browser cockpit
    python run.py --source data/test.mp4   # run against a video file
    python run.py --mode hybrid            # rules + trained LSTM
    python run.py --no-objects             # skip YOLO/hands (low-power mode)
    python run.py --recalibrate            # force a fresh baseline
    python run.py --window                 # also show the OpenCV window
    python run.py --no-ui                  # OpenCV window only, no web UI
    python run.py --headless               # no window, no browser (embedded)

The browser cockpit exposes the same controls as the desktop window:
recalibrate, mute, snapshot and stop.

Keys (OpenCV window): q quit | c recalibrate | m mute | h hide HUD | s save frame
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2

from eagleeye.alerts import AlertManager
from eagleeye.config import load_config
from eagleeye.fusion import DriverState
from eagleeye.logger import SessionLogger
from eagleeye.pipeline import EagleEyePipeline
from eagleeye.video import VideoStream
from eagleeye.pico_alert import close_pico, pico_status, send_to_pico
from eagleeye.visualize import (
    draw_detections,
    draw_hud,
    draw_landmarks,
    draw_pose_axis,
    draw_wheel_roi,
)
from eagleeye.webui import TelemetryHub, WebUI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EagleEye driver monitoring")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--source", default=None, help="camera index or video path")
    p.add_argument("--mode", choices=["rule", "temporal", "hybrid"], default=None)
    p.add_argument("--driver-id", default="driver_01")
    p.add_argument("--vehicle-id", default="vehicle_01")
    p.add_argument("--no-objects", action="store_true", help="disable YOLO + hands")
    p.add_argument("--no-alerts", action="store_true")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--recalibrate", action="store_true")
    p.add_argument("--headless", action="store_true", help="no window and no browser")
    p.add_argument("--record", default=None, help="write annotated video to this path")
    p.add_argument("--no-ui", action="store_true", help="disable the browser cockpit")
    p.add_argument("--ui-port", type=int, default=8000)
    p.add_argument("--ui-host", default="127.0.0.1")
    p.add_argument("--no-browser", action="store_true", help="serve the UI but don't open a tab")
    p.add_argument("--window", action="store_true", help="also show the OpenCV window")
    return p.parse_args()


class LiveRenderer:
    """Streams the camera to the browser at camera rate, independent of inference.

    Inference (face mesh, CNN, fusion) can take longer than one frame interval.
    If the browser feed waited for it, the video would stutter below 30 FPS and
    lag by the full processing time. This thread instead takes every new camera
    frame, paints the most recent overlay onto it and publishes it, so the
    video is live and only the overlay can trail by one inference pass.
    """

    def __init__(self, stream, hub, get_overlay, get_roi, hud_on) -> None:
        self._stream, self._hub = stream, hub
        self._get_overlay, self._get_roi, self._hud_on = get_overlay, get_roi, hud_on
        self._stop = threading.Event()
        self.fps = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        seq, count, stamp = -1, 0, time.perf_counter()
        while not self._stop.is_set():
            seq, frame = self._stream.wait_latest(seq, 0.1)
            if frame is None:
                continue
            if self._hud_on():
                points, pose, detections = self._get_overlay()
                try:
                    if points is not None:
                        draw_landmarks(frame, points)
                        if pose is not None:
                            draw_pose_axis(frame, points, pose)
                    draw_wheel_roi(frame, self._get_roi())
                    draw_detections(frame, detections)
                except Exception:   # a half-updated overlay must never stall the video
                    pass
            self._hub.publish_frame(frame)

            count += 1
            now = time.perf_counter()
            if now - stamp >= 1.0:
                self.fps = count / (now - stamp)
                count, stamp = 0, now


def _timer_resolution(ms: int):
    """Ask Windows for ms-granularity timers (default ~15.6 ms); no-op elsewhere."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(ms)
        return lambda: ctypes.windll.winmm.timeEndPeriod(ms)
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    restore_timer = _timer_resolution(1)
    cfg = load_config(args.config)

    if args.source is not None:
        cfg.camera.source = args.source
    if args.mode is not None:
        cfg.fusion.mode = args.mode
    if args.no_alerts:
        cfg.alerts.enabled = False

    if args.recalibrate:
        Path(cfg.calibration.file).unlink(missing_ok=True)

    stream = VideoStream(
        cfg.camera.source, cfg.camera.width, cfg.camera.height,
        cfg.camera.fps, cfg.camera.flip,
    )
    pipeline = EagleEyePipeline(
        cfg, use_objects=not args.no_objects, async_objects=not stream.is_file
    )
    alerts = AlertManager(cfg.alerts)
    logger = None if args.no_log else SessionLogger(cfg.logging, args.driver_id, args.vehicle_id)

    ui_enabled = not args.no_ui
    show_window = (not args.headless) and (args.window or not ui_enabled)

    hub = None
    webui = None
    if ui_enabled:
        hub = TelemetryHub(args.driver_id, args.vehicle_id)
        if logger:
            hub.session_id = logger.session_id
        webui = WebUI(hub, host=args.ui_host, port=args.ui_port, db_path=str(cfg.logging.db))
        webui.start()
        print(f"[EagleEye] command deck: {webui.url}")
        if not args.headless and not args.no_browser:
            webui.open_browser()

    backend = f"MediaPipe {pipeline.face.backend} · {cfg.fusion.mode}"
    writer = None
    fps_hist = deque(maxlen=30)
    last_tick = None
    show_hud = True
    muted = False
    frames = 0
    t_start = time.time()
    previous_state = None
    running = True
    want_snapshot = False
    warned_low_fps = False
    no_frame_since = None

    overlay = (None, None, [])      # latest (landmarks, pose, detections) for the renderer
    renderer = None
    if hub and not stream.is_file:
        renderer = LiveRenderer(
            stream, hub,
            get_overlay=lambda: overlay,
            get_roi=lambda: pipeline.hands.roi,
            hud_on=lambda: show_hud,
        )
        renderer.start()

    if logger:
        print(f"[EagleEye] session {logger.session_id} started")
    print("[EagleEye] running - press 'q' in the window or Stop in the browser to quit")

    try:
        while running:
            ok, frame, ts = stream.read()
            if not ok:
                if stream.is_file:
                    break                # end of the video file
                # A webcam can be slow to warm up (or briefly stall); keep the
                # UI alive and retry rather than shutting everything down.
                if no_frame_since is None:
                    no_frame_since = time.time()
                    print("[EagleEye] waiting for the camera to deliver frames...")
                elif time.time() - no_frame_since > 15.0:
                    print("[EagleEye] ERROR: no frames from the camera for 15s. Close any "
                          "other app using it (Teams/Zoom/browser), check Windows camera "
                          "privacy settings, or try camera.source: 1 in config.yaml.")
                    break
                continue
            no_frame_since = None

            t0 = time.perf_counter()
            signals, result = pipeline.process(frame, ts)
            if not pipeline.calibrating:
                current_state = (
                    result.state.value
                    if hasattr(result.state, "value")
                    else str(result.state)
                )

                if current_state != previous_state:
                    send_to_pico(current_state)
                    previous_state = current_state
            latency_ms = (time.perf_counter() - t0) * 1000.0
            overlay = (pipeline.last_points, pipeline.last_pose, pipeline.last_detections)
            now = time.perf_counter()
            if last_tick is not None:
                fps_hist.append(1.0 / max(now - last_tick, 1e-3))
            last_tick = now
            fps = sum(fps_hist) / len(fps_hist) if fps_hist else 0.0
            frames += 1

            if not warned_low_fps and frames == 90 and not stream.is_file:
                video_fps = renderer.fps if renderer else stream.capture_fps
                if 0 < video_fps < 28:      # 28 leaves room for 30 FPS jitter
                    print(f"[EagleEye] WARNING: video is running at {video_fps:.0f} FPS "
                          f"(camera delivers {stream.capture_fps:.0f}). Low light makes webcams "
                          "lower their frame rate - add light or disable auto-exposure.")
                warned_low_fps = True

            event = None
            if not muted and not pipeline.calibrating:
                event = alerts.update(ts, result.state, pipeline.time_in_state(ts))

            if logger:
                logger.log_frame(signals, result)
                if event:
                    logger.log_alert(event, result, signals.perclos)
            if event:
                print(f"[ALERT L{event.level}] {event.message}")
                if hub:
                    hub.add_alert(event)

            # -- commands from the browser ---------------------------------
            if hub:
                for action in hub.drain_commands():
                    if action == "recalibrate":
                        pipeline.recalibrate()
                        print("[EagleEye] recalibrating...")
                    elif action in ("mute", "unmute"):
                        muted = action == "mute"
                        print(f"[EagleEye] alerts {'muted' if muted else 'unmuted'}")
                    elif action == "snapshot":
                        want_snapshot = True
                    elif action == "stop":
                        print("[EagleEye] stop requested from the browser")
                        running = False

            calib_progress = (
                pipeline.calibrator.progress(ts) if pipeline.calibrating else -1.0
            )

            if hub:
                hub.publish(
                    signals, result,
                    fps=(renderer.fps or fps) if renderer else fps,
                    latency_ms=latency_ms,
                    calibrating=pipeline.calibrating,
                    calib_progress=max(calib_progress, 0.0),
                    muted=muted, backend=backend, pico=pico_status(),
                )

            # With a renderer thread the browser feed is painted there; the main
            # loop only builds a canvas for the window, recording or a snapshot.
            if show_window or args.record or want_snapshot or (hub and renderer is None):
                canvas = frame.copy()
                if show_hud:
                    if pipeline.last_points is not None:
                        draw_landmarks(canvas, pipeline.last_points)
                        if pipeline.last_pose is not None:
                            draw_pose_axis(canvas, pipeline.last_points, pipeline.last_pose)
                    draw_wheel_roi(canvas, pipeline.hands.roi)
                    draw_detections(canvas, pipeline.last_detections)

                # The browser renders its own metrics, so the painted HUD panel
                # is only worth the pixels on the desktop window / recording.
                if show_hud and (show_window or args.record):
                    canvas = draw_hud(
                        canvas, signals, result, fps,
                        flashing=alerts.is_flashing(ts),
                        calibrating=calib_progress,
                    )

                if hub and renderer is None:
                    hub.publish_frame(canvas)

                if want_snapshot:
                    name = f"snapshot_{int(time.time())}.png"
                    cv2.imwrite(name, canvas)
                    print(f"[EagleEye] saved {name}")
                    want_snapshot = False

                if args.record:
                    if writer is None:
                        h, w = canvas.shape[:2]
                        writer = cv2.VideoWriter(
                            args.record, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h)
                        )
                    writer.write(canvas)

                if show_window:
                    cv2.imshow("EagleEye", canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("c"):
                        pipeline.recalibrate()
                        print("[EagleEye] recalibrating...")
                    if key == ord("m"):
                        muted = not muted
                        print(f"[EagleEye] alerts {'muted' if muted else 'unmuted'}")
                    if key == ord("h"):
                        show_hud = not show_hud
                    if key == ord("s"):
                        want_snapshot = True

    except KeyboardInterrupt:
        print("\n[EagleEye] interrupted")
    finally:
        elapsed = max(time.time() - t_start, 1e-6)
        mean_fps = frames / elapsed
        if renderer is not None:
            renderer.stop()
        stream.release()
        if restore_timer:
            restore_timer()
        pipeline.close()
        close_pico()
        if writer is not None:
            writer.release()
        if logger:
            logger.close(mean_fps)
        if webui is not None:
            webui.shutdown()
        cv2.destroyAllWindows()
        print(f"[EagleEye] {frames} frames in {elapsed:.1f}s ({mean_fps:.1f} FPS)")


if __name__ == "__main__":
    main()
