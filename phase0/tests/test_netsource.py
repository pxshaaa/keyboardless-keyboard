"""NetVideoSource against a local fake WideCam server (no Bonjour, no LAN)."""
from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase0.capture import netsource  # noqa: E402
from phase0.capture.netsource import MultipartParser, NetVideoSource  # noqa: E402

W, H = 64, 48


def make_jpeg(i: int) -> bytes:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :, 0] = (i * 7) % 256
    img[:, : (i % W) + 1, 2] = 255
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    return buf.tobytes()


class FakeWideCam(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, n_frames=None, skip_seqs=(), period=0.005):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.n_frames, self.skip_seqs, self.period = n_frames, set(skip_seqs), period
        self.stop_event = threading.Event()
        self.control_calls: list[dict] = []
        self.state = {"lens": "ultrawide", "width": W, "height": H, "fps": 60.0,
                      "exposure": "auto", "clients": 0, "seq": 0, "dropped": 0, "battery": 0.5}
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}/stream"

    def stop(self) -> None:
        self.stop_event.set()
        self.shutdown()
        self.server_close()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self):
        body = json.dumps(self.server.state).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/status":
            return self._json()
        if not self.path.startswith("/stream"):
            self.send_error(404)
            return
        srv = self.server
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        seq = 0
        sent = 0
        try:
            while not srv.stop_event.is_set() and (srv.n_frames is None or sent < srv.n_frames):
                if seq in srv.skip_seqs:
                    seq += 1
                    continue
                jpg = make_jpeg(seq)
                part = (b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpg)}\r\n".encode()
                        + f"X-Seq: {seq}\r\nX-Timestamp: {1000.0 + seq / 60.0:.6f}\r\n".encode()
                        + f"X-Width: {W}\r\nX-Height: {H}\r\n\r\n".encode()
                        + jpg + b"\r\n")
                self.wfile.write(part)
                self.wfile.flush()
                srv.state["seq"] = seq
                seq += 1
                sent += 1
                time.sleep(srv.period)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def do_POST(self):
        u = urlsplit(self.path)
        if u.path != "/control":
            self.send_error(404)
            return
        params = {k: v[0] for k, v in parse_qs(u.query).items()}
        self.server.control_calls.append(params)
        if "fps" in params:
            self.server.state["fps"] = float(params["fps"])
        if "exposure" in params:
            self.server.state["exposure"] = "locked" if params["exposure"] == "lock" else "auto"
        if params.get("preset") == "1080p":
            self.server.state.update(width=1920, height=1080)
        self._json()


@pytest.fixture(autouse=True)
def no_bonjour(monkeypatch):
    monkeypatch.setattr(netsource, "discover", lambda *a, **k: pytest.fail("Bonjour used"))
    monkeypatch.setattr(netsource, "RECONNECT_INTERVAL", 0.1)


@pytest.fixture
def server():
    srv = FakeWideCam(skip_seqs={5})
    yield srv
    srv.stop()


def test_stream_decodes_seqs_increase_pts_and_gap_counted(server):
    src = NetVideoSource(server.url, timeout=0.5, connect_timeout=2.0)
    try:
        assert src.isOpened()
        assert src.name == f"WideCam@127.0.0.1:{server.server_port}"
        assert (src.get(cv2.CAP_PROP_FRAME_WIDTH), src.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (W, H)
        assert src.get(cv2.CAP_PROP_FPS) == 60.0
        seqs, pts = [], []
        deadline = time.monotonic() + 3.0
        while len(seqs) < 40 and time.monotonic() < deadline:
            ok, frame = src.read()
            assert ok and frame.shape == (H, W, 3) and frame.dtype == np.uint8
            seqs.append(src.last_seq)
            pts.append(src.last_pts)
        assert len(seqs) >= 40
        assert all(b > a for a, b in zip(seqs, seqs[1:]))
        assert all(b > a for a, b in zip(pts, pts[1:]))
        assert abs(pts[0] - (1000.0 + seqs[0] / 60.0)) < 1e-4
        assert src.get(cv2.CAP_PROP_POS_FRAMES) == len(seqs)
        assert src.frames_dropped == 1
        assert src.bytes_received > 40 * 100
        assert src.status()["lens"] == "ultrawide"
    finally:
        src.release()


def test_set_fps_posts_control_and_control_kw(server):
    src = NetVideoSource(server.url, timeout=0.5, connect_timeout=2.0)
    try:
        assert src.set(cv2.CAP_PROP_FPS, 30) is True
        assert server.control_calls[-1] == {"fps": "30"}
        assert src.get(cv2.CAP_PROP_FPS) == 30.0
        assert src.control(exposure="lock")["exposure"] == "locked"
        assert src.set(cv2.CAP_PROP_FRAME_WIDTH, W) is True  # already matching, no POST
        assert src.set(cv2.CAP_PROP_FRAME_WIDTH, 1234) is False
        assert src.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) is True
        assert server.control_calls[-1] == {"preset": "1080p"}
    finally:
        src.release()


def test_read_times_out_cleanly_when_server_stops_and_release_idempotent():
    srv = FakeWideCam(n_frames=None)
    src = NetVideoSource(srv.url, timeout=0.3, connect_timeout=0.6)
    try:
        ok, _ = src.read()
        assert ok
        srv.stop()
        t0 = time.monotonic()
        results = [src.read()[0] for _ in range(4)]
        assert results[-1] is False
        assert time.monotonic() - t0 < 4.0
        assert src.read() == (False, None)
        time.sleep(0.9)  # outage > connect_timeout -> reader gives up
        assert src.isOpened() is False
    finally:
        src.release()
        src.release()
        assert src.read() == (False, None)
        assert not src._thread.is_alive()


def test_unreachable_host_is_not_opened_and_does_not_raise():
    srv = FakeWideCam()
    url = srv.url
    srv.stop()
    src = NetVideoSource(url, timeout=0.2, connect_timeout=0.5)
    assert src.isOpened() is False
    assert src.read() == (False, None)
    src.release()


def test_resolve_target_forms(monkeypatch):
    monkeypatch.setattr(netsource, "discover", lambda *a, **k: [
        {"name": "Other", "host": "10.0.0.9", "port": 9999, "txt": {}},
        {"name": "WideCam", "host": "10.0.0.7", "port": 8081, "txt": {"v": "1"}},
    ])
    assert netsource.resolve_target("192.168.1.5") == "http://192.168.1.5:8080/stream"
    assert netsource.resolve_target("192.168.1.5:9000") == "http://192.168.1.5:9000/stream"
    assert netsource.resolve_target("http://x:1/stream") == "http://x:1/stream"
    assert netsource.resolve_target("auto") == "http://10.0.0.7:8081/stream"
    assert netsource.resolve_target("Other") == "http://10.0.0.9:9999/stream"
    monkeypatch.setattr(netsource, "discover", lambda *a, **k: [])
    with pytest.raises(ValueError):
        netsource.resolve_target("auto")


def test_parser_out_of_order_headers_zero_byte_part_and_missing_length():
    p = MultipartParser("frame")
    jpg = make_jpeg(3)
    stream = (
        b"--frame\r\nX-Seq: 7\r\nContent-Length: 0\r\nContent-Type: image/jpeg\r\n\r\n\r\n"
        b"--frame\r\nX-Extra: yes\r\nX-Timestamp: 12.5\r\nContent-Type: image/jpeg\r\n"
        b"X-Seq: 8\r\ncontent-length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
        b"--frame\r\nContent-Type: image/jpeg\r\nX-Seq: 9\r\n\r\n" + b"junk" * 10 + b"\r\n"
        b"--frame\r\nContent-Length: 4\r\nX-Seq: 10\r\n\r\nabcd\r\n"
    )
    parts = []
    for i in range(0, len(stream), 7):  # feed in ragged chunks
        parts += p.feed(stream[i:i + 7])
    assert [h["x-seq"] for h, _ in parts] == ["7", "8", "10"]
    assert parts[0][1] == b""
    assert parts[1][1] == jpg and parts[1][0]["x-extra"] == "yes" and parts[1][0]["x-timestamp"] == "12.5"
    assert parts[2][1] == b"abcd"
    assert p.parts_skipped == 1
