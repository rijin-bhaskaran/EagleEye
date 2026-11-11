"""Loopback HTTP server for the passenger process.

Only the driver dashboard talks to this (it proxies the two feeds), so the
server binds 127.0.0.1 and has just three routes:

* ``/video``          - annotated frames as an MJPEG stream
* ``/api/passenger``  - the element-id -> value payload (+ ``age_s``)
* ``/healthz``        - liveness
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .hub import PassengerHub

_BOUNDARY = "passengerframe"


class _Handler(BaseHTTPRequestHandler):
    hub: PassengerHub = None            # type: ignore[assignment]
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True

    def log_message(self, *args) -> None:  # noqa: D102 - per-frame logging would drown the console
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass

    def _drain(self) -> None:
        """Read and discard the request body. Replying and closing with unread data makes Windows
        reset the connection, so the client sees an abort instead of our response."""
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if 0 < n <= 1 << 20:
            self.rfile.read(n)
        elif n > 1 << 20:
            self.close_connection = True

    def _method_not_allowed(self) -> None:
        self._drain()
        body = b"read-only server: GET only"
        self.send_response(405)
        self.send_header("Allow", "GET")
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_POST = do_PUT = do_PATCH = do_DELETE = _method_not_allowed  # noqa: N815

    def do_GET(self) -> None:  # noqa: N802
        route = self.path.split("?")[0]
        if route == "/api/passenger":
            self._send(200, json.dumps(self.hub.snapshot()).encode(), "application/json")
        elif route == "/healthz":
            self._send(200, b'{"ok": true}', "application/json")
        elif route == "/video":
            self._stream_video()
        else:
            self._send(404, b"not found", "text/plain")

    def _stream_video(self) -> None:
        self.close_connection = True        # no Content-Length: the stream ends when the socket closes
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Connection", "close")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
        self.end_headers()
        try:
            for jpeg in self.hub.frames():
                self.wfile.write(
                    f"--{_BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode()
                    + jpeg + b"\r\n"
                )
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass


class _Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second process bind a port that is already
    # listening, so two monitors would silently share it. On Linux it is safe and
    # lets a restarted monitor rebind at once instead of waiting out TIME_WAIT.
    allow_reuse_address = sys.platform != "win32"


class PassengerServer:
    def __init__(self, hub: PassengerHub, port: int = 8100, host: str = "127.0.0.1") -> None:
        handler = type("PassengerHandler", (_Handler,), {"hub": hub})
        # A fixed port: the dashboard's proxy is configured with it. If it is taken, fail loudly.
        self._srv = _Server((host, port), handler)
        self._srv.daemon_threads = True
        self.hub, self.host, self.port = hub, host, self._srv.server_address[1]
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        self._thread.start()

    def shutdown(self) -> None:
        self.hub.close()
        if self._thread.is_alive():            # socketserver.shutdown() waits forever if serve_forever never ran
            self._srv.shutdown()
        self._srv.server_close()
