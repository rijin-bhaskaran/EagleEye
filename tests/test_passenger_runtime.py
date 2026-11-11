"""Tests for the passenger runtime: hub, overlay, loop, both HTTP servers and the launcher.

Every HTTP test talks to a real server on a real socket. Run with:  pytest -q
"""
from __future__ import annotations

import http.client
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_all
import run_passenger
from eagleeye.config import load_config
from eagleeye.webui import TelemetryHub, WebUI
from passenger.hub import PassengerHub, build_payload, seat_grid_html
from passenger.overlay import draw
from passenger.pipeline import PassengerPipeline, PassengerState
from passenger.runner import PassengerRunner
from passenger.seats import Detection, SeatMap, perspective_grid
from passenger.server import PassengerServer

NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get(url, timeout=3):
    """(status, body, content-type); HTTP errors are returned, not raised."""
    try:
        with NO_PROXY.open(url, timeout=timeout) as r:
            return r.status, r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", "")


def post(url, body, raw=False):
    data = body if raw else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    try:
        with NO_PROXY.open(req, timeout=3) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw_body = e.read()
        try:
            return e.code, json.loads(raw_body or b"{}")
        except ValueError:                      # plain-text error pages (404)
            return e.code, raw_body.decode()


def read_mjpeg_frame(url, timeout=3) -> bytes:
    """Connect to an MJPEG endpoint and return the first complete JPEG found in the stream."""
    parts = urllib.parse.urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout)
    conn.request("GET", parts.path)
    resp = conn.getresponse()
    assert resp.status == 200 and "multipart/x-mixed-replace" in resp.getheader("Content-Type")
    buf = b""
    deadline = time.time() + timeout
    while time.time() < deadline:
        buf += resp.read1(65536)
        start = buf.find(b"\xff\xd8")
        end = buf.find(b"\xff\xd9", start + 2)
        if start >= 0 and end > start:
            conn.close()
            return buf[start:end + 2]
    conn.close()
    raise AssertionError("no JPEG frame arrived")


import urllib.parse  # noqa: E402  (used by read_mjpeg_frame)


def sample_state(**kw) -> PassengerState:
    seats = SeatMap(perspective_grid(2, 2, 3))
    ids = seats.seat_ids
    base = dict(count=3, occupied=ids[:3], vacant=ids[3:], total_seats=len(ids),
                seat_labels={ids[0]: "female", ids[1]: "child", ids[2]: "male"},
                composition={"male": 1, "female": 1, "child": 1, "unknown": 0})
    base.update(kw)
    return PassengerState(**base)


def frame(w=320, h=240, value=0):
    return np.full((h, w, 3), value, np.uint8)


# -- hub / payload ------------------------------------------------------------
def test_payload_has_all_dashboard_fields_as_strings():
    p = build_payload(sample_state(), fps=9.87, latency_ms=101.4, classifier_on=True)
    assert p["online"] is True and p["ts"] > 0
    t = p["text"]
    assert (t["paxCount"], t["paxOccupied"], t["paxVacant"], t["paxTotal"]) == ("3", "3", "7", "10")
    assert (t["paxMale"], t["paxFemale"], t["paxChild"], t["paxUnknown"]) == ("1", "1", "1", "0")
    assert t["paxFps"] == "9.9 FPS" and t["paxLatency"] == "101 ms" and t["paxOccPct"] == "30%"
    assert t["paxClassifier"] == "Face model live"
    assert p["width"]["paxOccBar"] == "30%"
    assert all(isinstance(v, str) for v in t.values())
    assert build_payload(sample_state(), fps=1, latency_ms=1, classifier_on=False)["text"]["paxClassifier"] == "Classifier off"


def test_payload_survives_zero_seats():
    st = PassengerState(count=0, occupied=[], vacant=[], total_seats=0)
    assert build_payload(st, fps=0, latency_ms=0, classifier_on=False)["text"]["paxOccPct"] == "0%"


def test_seat_grid_is_front_first_with_colours_for_each_state():
    html = seat_grid_html(sample_state())
    assert html.count("<span title=") == 10                        # every seat drawn once
    assert html.index("R1L1") < html.index("R2L1")                 # row 1 (front) first
    assert "#FF4FA3" in html and "#22B14C" in html and "#2F80ED" in html   # female, child, male
    assert "vacant" in html and "R1L1 · female" in html


def test_seat_grid_occupied_but_unseen_seat_shows_unknown():
    st = sample_state(seat_labels={})
    assert "R1L1 · unknown" in seat_grid_html(st)


def test_seat_grid_custom_ids_fall_back_to_flat_list():
    st = PassengerState(count=1, occupied=["front"], vacant=["rear"], total_seats=2)
    html = seat_grid_html(st)
    assert "front" in html and "rear" in html and "flex-wrap" in html


def test_hub_snapshot_reports_age_and_is_empty_before_first_publish():
    hub = PassengerHub()
    assert hub.snapshot() == {}
    hub.publish(sample_state(), fps=10, latency_ms=5, classifier_on=False)
    time.sleep(0.05)
    snap = hub.snapshot()
    assert 0.0 <= snap["age_s"] < 2.0 and snap["text"]["paxCount"] == "3"


def test_hub_frames_yield_each_frame_once_and_stop_on_close():
    hub = PassengerHub()
    hub.publish_frame(frame(value=10))
    gen = hub.frames()
    first = next(gen)
    assert first[:2] == b"\xff\xd8"
    hub.publish_frame(frame(value=200))
    assert next(gen) != first
    hub.close()
    assert list(gen) == []


# -- overlay ------------------------------------------------------------------
def test_overlay_draws_boxes_and_seat_outlines():
    seats = SeatMap(perspective_grid(2, 2, 3))
    st = sample_state(detections=[Detection((0.3, 0.4, 0.5, 0.8), 0.9, track_id=4)], track_labels={4: "child"})
    img = frame(value=0)
    out = draw(img, st, seats, draw_seats=True)
    assert out is img and out.any()
    plain = frame()
    draw(plain, st, seats, draw_seats=False)
    assert plain.sum() < out.sum()                                 # seat outlines add pixels


def test_overlay_standing_passenger_uses_track_label_and_untracked_is_ok():
    seats = SeatMap(perspective_grid(1, 1, 1))
    st = PassengerState(count=2, occupied=[], vacant=seats.seat_ids, total_seats=2,
                        detections=[Detection((0.1, 0.1, 0.3, 0.6), 0.9, 7), Detection((0.5, 0.1, 0.7, 0.6), 0.9, None)],
                        track_labels={7: "female"})
    assert draw(frame(), st, seats, draw_seats=False).any()


# -- runner --------------------------------------------------------------------
class FakeCap:
    def __init__(self):
        self.grabbed, self.sets = 0, []

    def grab(self):
        self.grabbed += 1
        return True

    def set(self, prop, value):
        self.sets.append((prop, value))


class FakeStream:
    def __init__(self, n=5, is_file=True, src_fps=30.0, endless=False):
        self.is_file, self.src_fps, self.cap = is_file, src_fps, FakeCap()
        self.left, self.endless, self.frame_index = n, endless, 0

    def read(self):
        if self.left <= 0 and not self.endless:
            return False, None, 0.0
        self.left -= 1
        return True, frame(), 0.0


class EmptyDetector:
    def detect(self, f, track=True):
        return []


def make_runner(stream, fps=10.0, loop=False, hub=None):
    seats = SeatMap(perspective_grid(2, 2, 3))
    pipe = PassengerPipeline(EmptyDetector(), seats)
    return PassengerRunner(stream, pipe, hub or PassengerHub(), fps=fps, loop=loop)


def test_runner_rejects_non_positive_fps():
    with pytest.raises(ValueError):
        make_runner(FakeStream(), fps=0)


def test_runner_step_publishes_frame_and_telemetry():
    hub = PassengerHub()
    r = make_runner(FakeStream(), hub=hub)
    assert r.step() is True and r.frames == 1
    assert hub.snapshot()["text"]["paxTotal"] == "10" and next(hub.frames())[:2] == b"\xff\xd8"


def test_runner_skips_frames_of_a_30fps_file_to_hit_10fps():
    stream = FakeStream(src_fps=30.0)
    r = make_runner(stream, fps=10.0)
    r.step()
    assert stream.cap.grabbed == 2                                 # read 1 of every 3 frames


def test_runner_live_camera_never_skips():
    stream = FakeStream(is_file=False, src_fps=30.0)
    make_runner(stream, fps=10.0).step()
    assert stream.cap.grabbed == 0


def test_runner_ends_at_eof_unless_looping():
    r = make_runner(FakeStream(n=1))
    assert r.step() is True and r.step() is False
    stream = FakeStream(n=1)
    looping = make_runner(stream, loop=True)
    looping.step()
    assert looping.step() is True and (cv2.CAP_PROP_POS_FRAMES, 0) in stream.cap.sets


def test_runner_tolerates_camera_hiccup():
    stream = FakeStream(n=0, is_file=False)
    assert make_runner(stream).step() is True


def test_runner_run_paces_to_target_fps_and_stops_at_eof():
    r = make_runner(FakeStream(n=8, src_fps=10.0), fps=40.0)
    t0 = time.perf_counter()
    r.run(threading.Event())
    elapsed = time.perf_counter() - t0
    assert r.frames == 8 and 0.15 < elapsed < 3.0                  # 8 frames at 40 FPS take ~0.2 s


def test_runner_run_stops_promptly_on_event():
    r = make_runner(FakeStream(is_file=False, endless=True), fps=5.0)
    stop = threading.Event()
    th = threading.Thread(target=r.run, args=(stop,))
    th.start()
    time.sleep(0.3)
    stop.set()
    th.join(timeout=2)
    assert not th.is_alive() and r.frames >= 1


# -- passenger server (real socket) ---------------------------------------------
@pytest.fixture
def pax():
    hub = PassengerHub()
    server = PassengerServer(hub, port=0)
    server.start()
    yield hub, server
    server.shutdown()


def test_passenger_server_healthz_and_unknown_routes(pax):
    hub, server = pax
    status, body, ctype = get(server.url + "/healthz")
    assert status == 200 and json.loads(body) == {"ok": True} and "json" in ctype
    assert get(server.url + "/nope")[0] == 404
    assert get(server.url + "/")[0] == 404                         # no UI of its own
    assert get(server.url + "/api/passenger/extra")[0] == 404


def test_passenger_server_telemetry_empty_then_filled_and_ignores_query(pax):
    hub, server = pax
    assert json.loads(get(server.url + "/api/passenger")[1]) == {}
    hub.publish(sample_state(), fps=10, latency_ms=5, classifier_on=True)
    status, body, ctype = get(server.url + "/api/passenger?x=1")
    data = json.loads(body)
    assert status == 200 and "json" in ctype and data["text"]["paxCount"] == "3" and "age_s" in data


def test_passenger_server_streams_mjpeg(pax):
    hub, server = pax
    hub.publish_frame(frame(value=120))
    jpeg = read_mjpeg_frame(server.url + "/video")
    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape == (240, 320, 3)


def test_passenger_server_rejects_non_get(pax):
    _, server = pax
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        req = urllib.request.Request(server.url + "/api/passenger", data=b"{}", method=method)
        with pytest.raises(urllib.error.HTTPError) as err:
            NO_PROXY.open(req, timeout=3)
        assert err.value.code == 405 and err.value.headers["Allow"] == "GET"      # read-only server


def test_unread_request_bodies_never_abort_the_connection(pax):
    """Regression: replying without reading the body made Windows reset the socket (intermittently)."""
    _, server = pax
    big = b"x" * 200_000
    for _ in range(60):
        req = urllib.request.Request(server.url + "/api/passenger", data=big, method="POST")
        with pytest.raises(urllib.error.HTTPError) as err:
            NO_PROXY.open(req, timeout=3)
        assert err.value.code == 405


def test_passenger_server_port_in_use_fails_loudly(pax):
    hub, server = pax
    with pytest.raises(OSError):
        PassengerServer(PassengerHub(), port=server.port)


def test_passenger_server_binds_loopback_only(pax):
    assert pax[1].host == "127.0.0.1"


# -- dashboard server: passenger proxy + existing API contract ------------------
@pytest.fixture
def deck(tmp_path):
    """Driver dashboard wired to a passenger server, plus handles to both hubs."""
    pax_hub = PassengerHub()
    pax_server = PassengerServer(pax_hub, port=0)
    pax_server.start()
    drv_hub = TelemetryHub("driver_x", "vehicle_x")
    ui = WebUI(drv_hub, port=free_port(), db_path=str(tmp_path / "none.db"), passenger_url=pax_server.url)
    ui.start()
    yield ui, drv_hub, pax_hub, pax_server
    ui.shutdown()
    pax_server.shutdown()


def test_proxy_api_passenger_passes_payload_through(deck):
    ui, _, pax_hub, _ = deck
    pax_hub.publish(sample_state(), fps=10, latency_ms=5, classifier_on=True)
    status, body, ctype = get(ui.url + "api/passenger")
    assert status == 200 and "json" in ctype
    data = json.loads(body)
    assert data["text"]["paxVacant"] == "7" and data["age_s"] < 2


def test_proxy_api_passenger_before_first_frame_is_empty_not_offline(deck):
    ui = deck[0]
    assert json.loads(get(ui.url + "api/passenger")[1]) == {}


def test_proxy_reports_offline_when_passenger_stops(deck):
    ui, _, _, pax_server = deck
    pax_server.shutdown()
    status, body, _ = get(ui.url + "api/passenger")
    assert status == 200 and json.loads(body) == {"online": False}
    assert get(ui.url + "video/passenger")[0] == 503


def test_proxy_offline_when_not_configured(tmp_path):
    ui = WebUI(TelemetryHub("d", "v"), port=free_port(), db_path=str(tmp_path / "n.db"))
    ui.start()
    try:
        assert json.loads(get(ui.url + "api/passenger")[1]) == {"online": False}
        assert get(ui.url + "video/passenger")[0] == 503
    finally:
        ui.shutdown()


def test_proxy_streams_passenger_video(deck):
    ui, _, pax_hub, _ = deck
    pax_hub.publish_frame(frame(value=90))
    jpeg = read_mjpeg_frame(ui.url + "video/passenger")
    assert jpeg[:2] == b"\xff\xd8"


def test_both_feeds_stream_at_the_same_time(deck):
    ui, drv_hub, pax_hub, _ = deck
    drv_hub.publish_frame(frame(value=30))
    pax_hub.publish_frame(frame(value=200))
    results = {}

    def grab(name, path):
        results[name] = read_mjpeg_frame(ui.url + path)

    threads = [threading.Thread(target=grab, args=(n, p)) for n, p in (("driver", "video"), ("pax", "video/passenger"))]
    [t.start() for t in threads]
    [t.join(timeout=5) for t in threads]
    assert set(results) == {"driver", "pax"} and results["driver"] != results["pax"]


def test_dashboard_page_contains_passenger_panel(deck):
    status, body, ctype = get(deck[0].url)
    html = body.decode()
    assert status == 200 and "text/html" in ctype
    for element in ("paxFeedSlot", "paxCount", "/api/passenger", "/video/passenger"):
        assert element in html


def test_existing_driver_routes_still_work(deck):
    ui, drv_hub, _, _ = deck
    assert json.loads(get(ui.url + "healthz")[1]) == {"ok": True}
    assert get(ui.url + "api/telemetry")[0] == 200
    assert get(ui.url + "nope")[0] == 404
    status, body, _ = get(ui.url + "api/fleet")
    assert status == 200 and json.loads(body)["sessions"] == []


def test_command_endpoint_validation(deck):
    ui, drv_hub, _, _ = deck
    assert post(ui.url + "api/command", {"action": "mute"}) == (200, {"ok": True, "action": "mute"})
    assert drv_hub.drain_commands() == ["mute"]
    assert post(ui.url + "api/command", {"action": "explode"})[0] == 400
    assert post(ui.url + "api/command", b"{not json", raw=True)[0] == 400
    assert post(ui.url + "api/other", {"action": "mute"})[0] == 404
    assert drv_hub.drain_commands() == []


def test_dashboard_unknown_post_route_with_a_body_always_gets_its_404(deck):
    ui = deck[0]
    big = json.dumps({"pad": "x" * 200_000}).encode()
    for _ in range(60):
        assert post(ui.url + "api/other", big, raw=True)[0] == 404


def test_asset_route_blocks_path_traversal(deck):
    ui = deck[0]
    assert get(ui.url + "assets/eagleeye-logo.png")[0] == 200
    assert get(ui.url + "assets/../server.py")[0] == 404
    assert get(ui.url + "assets/..%2Fserver.py")[0] == 404
    assert get(ui.url + "assets/missing.png")[0] == 404
    assert get(ui.url + "assets/index.html")[0] == 404             # only image types are served


# -- launcher ------------------------------------------------------------------
def args(**kw):
    ns = run_all.parse_args([])
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_build_commands_forwards_options():
    drv, pax = run_all.build_commands(args(driver_source="0", passenger_source="2", no_browser=True,
                                           loop=True, no_passenger_classifier=True, ui_port=9000), python="py")
    assert drv[0] == "py" and drv[1].endswith("run.py")
    assert drv[drv.index("--source") + 1] == "0" and "--no-browser" in drv and drv[drv.index("--ui-port") + 1] == "9000"
    assert pax[1].endswith("run_passenger.py")
    assert pax[pax.index("--source") + 1] == "2" and "--loop" in pax and "--no-classifier" in pax


def test_build_commands_defaults_leave_sources_to_config():
    drv, pax = run_all.build_commands(args(), python="py")
    assert "--source" not in drv and "--source" not in pax and "--loop" not in pax and "--no-browser" not in drv


def test_same_camera_detection():
    cfg = load_config("config.yaml")
    assert run_all.same_camera(args(driver_source="1", passenger_source="1"), cfg) == "1"
    assert run_all.same_camera(args(driver_source="0", passenger_source="2"), cfg) is None
    assert run_all.same_camera(args(driver_source="a.mp4", passenger_source="a.mp4"), cfg) is None   # files can be shared
    assert run_all.same_camera(args(), cfg) is None                # config defaults use different cameras


class FakeProc:
    def __init__(self, polls, returncode=0):
        self._polls, self.returncode, self.terminated, self.killed, self.pid = list(polls), returncode, False, False, 1

    def poll(self):
        return self._polls.pop(0) if len(self._polls) > 1 else self._polls[0]

    def terminate(self):
        self.terminated = True
        self._polls = [self.returncode]

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return self.returncode


def test_supervise_returns_driver_code_and_tolerates_passenger_exit(capsys):
    driver = FakeProc([None, None, 3], returncode=3)
    passenger = FakeProc([None, 1], returncode=1)
    assert run_all.supervise(driver, passenger, poll_s=0) == 3
    assert "passenger monitor exited" in capsys.readouterr().out


def test_stop_terminates_running_and_ignores_finished():
    running, done = FakeProc([None]), FakeProc([0])
    run_all.stop(running)
    run_all.stop(done)
    assert running.terminated and not done.terminated


def test_stop_kills_after_grace_timeout():
    import subprocess

    class Stubborn(FakeProc):
        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("x", timeout)

    p = Stubborn([None])
    run_all.stop(p, grace_s=0)
    assert p.terminated and p.killed


def test_run_passenger_arg_defaults_defer_to_config():
    a = run_passenger.parse_args([])
    assert a.source is None and a.fps is None and a.port is None and not a.loop and not a.no_classifier
    b = run_passenger.parse_args(["--source", "x.mp4", "--fps", "12", "--port", "9", "--loop", "--no-classifier"])
    assert (b.source, b.fps, b.port, b.loop, b.no_classifier) == ("x.mp4", 12.0, 9, True, True)


def test_every_id_the_passenger_payload_fills_exists_in_the_dashboard_page():
    html = (ROOT / "eagleeye" / "webui" / "index.html").read_text(encoding="utf-8")
    import re
    page_ids = set(re.findall(r'id="([A-Za-z0-9_-]+)"', html))
    payload = build_payload(sample_state(), fps=10, latency_ms=5, classifier_on=True)
    filled = set(payload["text"]) | set(payload["width"]) | set(payload["html"])
    assert filled <= page_ids, f"payload ids with no element: {sorted(filled - page_ids)}"
    script = re.findall(r"<script>(.*?)</script>", html, re.S)[-1]
    assert set(re.findall(r'\$\("([A-Za-z0-9_-]+)"\)', script)) <= page_ids
