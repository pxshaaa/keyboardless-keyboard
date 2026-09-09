"""cv2.VideoCapture-compatible source for the WideCam iPhone MJPEG stream
(widecam/PROTOCOL.md): Bonjour discovery, multipart parsing, reconnect."""
from __future__ import annotations

import argparse
import http.client
import json
import re
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Optional
from urllib.parse import urlencode, urlsplit

import numpy as np

# avsource imports only cv2/numpy at module import time (AVFoundation is loaded
# lazily inside its functions), so reusing the mailbox is safe here.
from phase0.capture.avsource import (  # noqa: F401
    CAP_PROP_FPS,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_POS_FRAMES,
    FrameMailbox,
)

SERVICE_TYPE = "_widecam._tcp"
DEFAULT_PORT = 8080
DEFAULT_INSTANCE = "WideCam"
BROWSE_SECONDS = 3.0
RECONNECT_INTERVAL = 1.0
_MAX_HEADER_BLOCK = 64 * 1024


# -- multipart/x-mixed-replace parser -----------------------------------------

class MultipartParser:
    """Incremental multipart/x-mixed-replace parser; feed() -> [(headers, body)].
    Keys lower-cased, any order; Content-Length required else part is skipped."""

    _HEADERS, _BODY, _RESYNC = 0, 1, 2

    def __init__(self, boundary: str = "frame") -> None:
        self._delim = b"--" + boundary.strip().strip('"').encode("latin-1")
        self._buf = bytearray()
        self._state = self._HEADERS
        self._need = 0
        self._headers: dict[str, str] = {}
        self.parts_skipped = 0

    def feed(self, data: bytes) -> list[tuple[dict[str, str], bytes]]:
        buf = self._buf
        buf += data
        out: list[tuple[dict[str, str], bytes]] = []
        while True:
            if self._state == self._HEADERS:
                idx = buf.find(b"\r\n\r\n")
                if idx < 0:
                    if len(buf) > _MAX_HEADER_BLOCK:  # garbage; hunt for a boundary
                        self.parts_skipped += 1
                        self._state = self._RESYNC
                        continue
                    break
                block = bytes(buf[:idx])
                del buf[: idx + 4]
                headers: dict[str, str] = {}
                for line in block.split(b"\r\n"):
                    line = line.strip()
                    if not line or line.startswith(b"--") or b":" not in line:
                        continue
                    k, v = line.split(b":", 1)
                    headers[k.strip().decode("latin-1").lower()] = v.strip().decode("latin-1")
                try:
                    need = int(headers["content-length"])
                    if need < 0:
                        raise ValueError
                except (KeyError, ValueError):
                    self.parts_skipped += 1
                    self._state = self._RESYNC
                    continue
                self._headers, self._need = headers, need
                self._state = self._BODY
            elif self._state == self._BODY:
                if len(buf) < self._need:
                    break
                body = bytes(buf[: self._need])
                del buf[: self._need]
                if buf.startswith(b"\r\n"):  # trailing CRLF before next boundary
                    del buf[:2]
                out.append((self._headers, body))
                self._headers = {}
                self._state = self._HEADERS
            else:  # _RESYNC
                idx = buf.find(self._delim)
                if idx < 0:
                    keep = len(self._delim) - 1
                    if len(buf) > keep:
                        del buf[:-keep]
                    break
                del buf[:idx]
                self._state = self._HEADERS
        return out


# -- discovery -----------------------------------------------------------------

def _txt_to_dict(props: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in (props or {}).items():
        if isinstance(k, bytes):
            k = k.decode("utf-8", "replace")
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
        out[str(k)] = "" if v is None else str(v)
    return out


def _discover_zeroconf(timeout: float, want: Optional[str]) -> list[dict[str, Any]]:
    from zeroconf import ServiceBrowser, Zeroconf

    found: list[dict[str, Any]] = []
    hit = threading.Event()
    zc = Zeroconf()

    class _Listener:
        def add_service(self, zc_, stype, name):
            info = zc_.get_service_info(stype, name, timeout=int(timeout * 1000))
            if info is None:
                return
            addrs = info.parsed_addresses()
            if not addrs:
                return
            inst = name[: -len("." + stype)] if name.endswith("." + stype) else name
            rec = {"name": inst, "host": addrs[0], "port": int(info.port),
                   "txt": _txt_to_dict(info.properties)}
            found.append(rec)
            if want is not None and inst == want:
                hit.set()

        def update_service(self, *a):  # pragma: no cover
            pass

        def remove_service(self, *a):  # pragma: no cover
            pass

    try:
        ServiceBrowser(zc, SERVICE_TYPE + ".local.", _Listener())
        hit.wait(timeout)
    finally:
        zc.close()
    return found


_DNSSD_ADD = re.compile(r"^\s*\S+\s+Add\s+\d+\s+\d+\s+\S+\s+(\S+)\s+(.+?)\s*$")
_DNSSD_REACH = re.compile(r"can be reached at\s+(\S+?):(\d+)")


def _discover_dns_sd(timeout: float, want: Optional[str]) -> list[dict[str, Any]]:
    """macOS fallback: `dns-sd -B` to enumerate, `dns-sd -L` to resolve."""
    def run(cmd: list[str], secs: float) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=secs).stdout
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout
            return out.decode("utf-8", "replace") if isinstance(out, bytes) else (out or "")
        except (OSError, ValueError):
            return ""

    t0 = time.monotonic()
    browse = run(["dns-sd", "-B", SERVICE_TYPE], max(0.5, min(timeout, 1.5)))
    names: list[str] = []
    for line in browse.splitlines():
        m = _DNSSD_ADD.match(line)
        if m and m.group(1) == SERVICE_TYPE + "." and m.group(2) not in names:
            names.append(m.group(2))
    if want is not None:
        names.sort(key=lambda n: n != want)
    found: list[dict[str, Any]] = []
    for inst in names:
        remaining = timeout - (time.monotonic() - t0)
        if remaining <= 0.1:
            break
        look = run(["dns-sd", "-L", inst, SERVICE_TYPE], max(0.5, min(remaining, 1.5)))
        m = _DNSSD_REACH.search(look)
        if not m:
            continue
        host, port = m.group(1).rstrip("."), int(m.group(2))
        try:
            host = socket.gethostbyname(host)
        except OSError:
            pass
        txt: dict[str, str] = {}
        for kv in re.findall(r"(\w+)=(\S+)", look.split("can be reached at", 1)[1]):
            txt[kv[0]] = kv[1]
        found.append({"name": inst, "host": host, "port": port, "txt": txt})
        if want is not None and inst == want:
            break
    return found


def discover(timeout: float = BROWSE_SECONDS, want: Optional[str] = DEFAULT_INSTANCE
             ) -> list[dict[str, Any]]:
    """Browse for up to `timeout` s -> [{name, host, port, txt}]; returns early
    once instance `want` shows up (None = collect everything)."""
    try:
        import zeroconf  # noqa: F401
    except ImportError:
        return _discover_dns_sd(timeout, want)
    return _discover_zeroconf(timeout, want)


def resolve_target(target: str, browse_seconds: float = BROWSE_SECONDS) -> str:
    """'auto' / service name -> browse; bare host[:port] -> default URL; URL as is."""
    t = (target or "auto").strip()
    if "://" in t:
        return t
    if "/" in t:
        return f"http://{t}"
    if re.fullmatch(r"[^:\s]+:\d{1,5}", t):
        return f"http://{t}/stream"
    if "." in t or t.lower() == "localhost":
        return f"http://{t}:{DEFAULT_PORT}/stream"
    want = DEFAULT_INSTANCE if t.lower() == "auto" else t
    hits = discover(browse_seconds, want=want)
    if t.lower() != "auto":
        hits = [h for h in hits if h["name"] == want]
    else:
        hits.sort(key=lambda h: h["name"] != want)  # prefer "WideCam", else first seen
    if not hits:
        raise ValueError(
            f"no WideCam found via Bonjour ({SERVICE_TYPE}) in {browse_seconds:.0f}s; "
            "start the app on the iPhone, or pass its IP / URL shown on screen"
        )
    h = hits[0]
    return f"http://{h['host']}:{h['port']}/stream"


# -- the source -----------------------------------------------------------------

class NetVideoSource:
    """cv2.VideoCapture look-alike over the WideCam MJPEG stream."""

    def __init__(self, target: str = "auto", timeout: float = 2.0,
                 connect_timeout: float = 5.0) -> None:
        self.read_timeout = float(timeout)
        self.connect_timeout = float(connect_timeout)
        self.url = resolve_target(target)
        u = urlsplit(self.url)
        self.host = u.hostname or "localhost"
        self.port = u.port or DEFAULT_PORT
        self.path = (u.path or "/stream") + (f"?{u.query}" if u.query else "")
        self._name = f"WideCam@{self.host}:{self.port}"

        self.width = 0
        self.height = 0
        self.fps = 0.0
        self._mailbox = FrameMailbox()
        self._mb_seq = 0
        self._frames_read = 0
        self._last_seq = -1
        self._last_pts = float("nan")
        self._prev_stream_seq = -1
        self.frames_received = 0
        self.frames_dropped = 0
        self.decode_failures = 0
        self.bytes_received = 0
        self.reconnects = 0
        self._sock: Optional[socket.socket] = None
        self._sock_lock = threading.Lock()
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._released = False
        self._opened = False
        self._last_error: Optional[BaseException] = None

        try:
            st = self.status()
            self.width = int(st.get("width") or 0)
            self.height = int(st.get("height") or 0)
            self.fps = float(st.get("fps") or 0.0)
        except Exception:
            pass

        self._thread = threading.Thread(target=self._run, name="netsource-reader", daemon=True)
        self._thread.start()
        self._connected.wait(self.connect_timeout)
        self._opened = self._connected.is_set() and not self._stop.is_set()

    # -- HTTP helpers -------------------------------------------------------------
    def _http_json(self, method: str, path: str, params: Optional[dict] = None) -> dict:
        if params:
            path = path + "?" + urlencode({k: v for k, v in params.items() if v is not None})
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.read_timeout)
        try:
            conn.request(method, path)
            resp = conn.getresponse()
            body = resp.read()
            if resp.status != 200:
                raise RuntimeError(f"{method} {path} -> HTTP {resp.status}")
            return json.loads(body.decode("utf-8")) if body.strip() else {}
        finally:
            conn.close()

    def status(self) -> dict:
        """GET /status as a dict (raises on network/HTTP errors)."""
        return self._http_json("GET", "/status")

    def control(self, **kw: Any) -> dict:
        """POST /control?k=v...; returns the /status JSON after applying and
        refreshes the cached width/height/fps. {} on failure."""
        try:
            st = self._http_json("POST", "/control", kw)
        except Exception:
            return {}
        if isinstance(st, dict):
            self.width = int(st.get("width") or self.width)
            self.height = int(st.get("height") or self.height)
            self.fps = float(st.get("fps") or self.fps)
        return st if isinstance(st, dict) else {}

    # -- reader thread --------------------------------------------------------------
    def _open_stream(self) -> tuple[socket.socket, MultipartParser, bytes]:
        sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        sock.settimeout(self.read_timeout)
        req = (f"GET {self.path} HTTP/1.1\r\nHost: {self.host}:{self.port}\r\n"
               f"User-Agent: phase0-netsource\r\nAccept: multipart/x-mixed-replace\r\n"
               f"Connection: close\r\n\r\n").encode("ascii")
        sock.sendall(req)
        head = bytearray()
        while b"\r\n\r\n" not in head:
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("EOF before HTTP response headers")
            head += chunk
            if len(head) > _MAX_HEADER_BLOCK:
                raise ConnectionError("HTTP response headers too large")
        idx = head.find(b"\r\n\r\n")
        hdr, rest = bytes(head[:idx]), bytes(head[idx + 4:])
        lines = hdr.split(b"\r\n")
        status_line = lines[0].decode("latin-1", "replace")
        parts = status_line.split()
        if len(parts) < 2 or parts[1] != "200":
            raise ConnectionError(f"GET {self.path}: {status_line}")
        boundary = "frame"
        for line in lines[1:]:
            if line.lower().startswith(b"content-type:"):
                m = re.search(rb'boundary="?([^";\s]+)"?', line, re.I)
                if m:
                    boundary = m.group(1).decode("latin-1")
        return sock, MultipartParser(boundary), rest

    def _run(self) -> None:
        outage_start: Optional[float] = None
        while not self._stop.is_set():
            try:
                sock, parser, rest = self._open_stream()
            except (OSError, ConnectionError) as exc:
                now = time.monotonic()
                if outage_start is None:
                    outage_start = now
                if now - outage_start >= self.connect_timeout:
                    self._last_error = exc
                    break
                self._stop.wait(RECONNECT_INTERVAL)
                continue
            if outage_start is not None:
                self.reconnects += 1
            outage_start = None
            with self._sock_lock:
                if self._stop.is_set():
                    sock.close()
                    break
                self._sock = sock
            self._connected.set()
            try:
                self._consume(sock, parser, rest)
            except (OSError, ConnectionError):
                pass
            finally:
                with self._sock_lock:
                    self._sock = None
                try:
                    sock.close()
                except OSError:
                    pass
            if self._stop.is_set():
                break
            outage_start = time.monotonic()
            self._stop.wait(RECONNECT_INTERVAL)
        self._opened = False
        self._mailbox.close()

    def _consume(self, sock: socket.socket, parser: MultipartParser, rest: bytes) -> None:
        import cv2

        data = rest
        while not self._stop.is_set():
            if data:
                self.bytes_received += len(data)
                for headers, body in parser.feed(data):
                    self._handle_part(cv2, headers, body)
            data = sock.recv(256 * 1024)
            if not data:
                raise ConnectionError("stream EOF")

    def _handle_part(self, cv2, headers: dict[str, str], body: bytes) -> None:
        self.frames_received += 1
        try:
            seq = int(headers.get("x-seq", ""))
        except ValueError:
            seq = self._prev_stream_seq + 1
        try:
            pts = float(headers.get("x-timestamp", "nan"))
        except ValueError:
            pts = float("nan")
        if self._prev_stream_seq >= 0 and seq > self._prev_stream_seq + 1:
            self.frames_dropped += seq - self._prev_stream_seq - 1
        self._prev_stream_seq = seq
        if not body:
            self.decode_failures += 1
            return
        frame = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.ndim != 3:
            self.decode_failures += 1
            return
        try:
            self.width = int(headers.get("x-width", frame.shape[1]))
            self.height = int(headers.get("x-height", frame.shape[0]))
        except ValueError:
            self.height, self.width = frame.shape[:2]
        # The mailbox slot carries (frame, X-Seq) so read() sees both atomically.
        self._mailbox.put((frame, seq), pts)  # type: ignore[arg-type]

    # -- cv2.VideoCapture surface ---------------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def last_pts(self) -> float:
        """X-Timestamp (phone CACurrentMediaTime at capture) of the frame most
        recently returned by read(); NaN before the first read."""
        return self._last_pts

    @property
    def last_seq(self) -> int:
        """X-Seq of the frame most recently returned by read(); -1 before."""
        return self._last_seq

    def isOpened(self) -> bool:
        return self._opened and not self._released

    def read(self, timeout: Optional[float] = None) -> tuple[bool, Optional[np.ndarray]]:
        if self._released:
            return False, None
        got = self._mailbox.wait_newer(
            self._mb_seq, self.read_timeout if timeout is None else timeout
        )
        if got is None:
            return False, None
        self._mb_seq, (frame, seq), self._last_pts = got
        self._last_seq = seq
        self._frames_read += 1
        return True, frame

    def grab(self) -> bool:
        ok, _ = self.read()
        return ok

    def get(self, prop) -> float:
        if prop == CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop == CAP_PROP_FPS:
            return float(self.fps)
        if prop == CAP_PROP_POS_FRAMES:
            return float(self._frames_read)
        return 0.0

    def set(self, prop, value) -> bool:
        """FPS / width / height are forwarded to POST /control; True only if the
        phone reports the requested value afterwards."""
        if self._released:
            return False
        if prop == CAP_PROP_FPS:
            want = int(round(float(value)))
            st = self.control(fps=want)
            return bool(st) and abs(float(st.get("fps") or 0) - want) < 0.5
        presets = {CAP_PROP_FRAME_WIDTH: {1280: "720p", 1920: "1080p"},
                   CAP_PROP_FRAME_HEIGHT: {720: "720p", 1080: "1080p"}}
        if prop in presets:
            v = int(value)
            if prop == CAP_PROP_FRAME_WIDTH and v == self.width:
                return True
            if prop == CAP_PROP_FRAME_HEIGHT and v == self.height:
                return True
            preset = presets[prop].get(v)
            if preset is None:
                return False
            st = self.control(preset=preset)
            key = "width" if prop == CAP_PROP_FRAME_WIDTH else "height"
            return bool(st) and int(st.get(key) or 0) == v
        return False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._opened = False
        self._stop.set()
        self._mailbox.close()
        with self._sock_lock:
            s = self._sock
            if s is not None:
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    s.close()
                except OSError:
                    pass
        t = getattr(self, "_thread", None)
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(self.read_timeout + self.connect_timeout + 1.0)

    def __enter__(self) -> "NetVideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.release()
        except Exception:
            pass


# -- CLI ---------------------------------------------------------------------

def _demo(args: argparse.Namespace) -> int:
    import cv2

    try:
        src = NetVideoSource(args.target, timeout=args.timeout, connect_timeout=args.connect_timeout)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"source   : {src.name}  ({src.url})")
    if not src.isOpened():
        print("ERROR: could not connect to the stream", file=sys.stderr)
        src.release()
        return 2
    if args.control:
        kw = dict(kv.split("=", 1) for kv in args.control)
        print(f"control  : {kw} -> {src.control(**kw)}")
    print(f"format   : {src.width}x{src.height} @ {src.fps:.1f} fps (negotiated)")

    writer = None
    n = dup = fails = 0
    prev_gray = None
    first_t = last_t = None
    first_pts = last_pts = None
    bytes0 = src.bytes_received
    t_end = time.monotonic() + args.seconds
    while time.monotonic() < t_end:
        ok, frame = src.read()
        t = time.monotonic()
        if not ok:
            fails += 1
            continue
        if first_t is None:
            first_t, first_pts = t, src.last_pts
        last_t, last_pts = t, src.last_pts
        n += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and prev_gray.shape == gray.shape and np.array_equal(gray, prev_gray):
            dup += 1
        prev_gray = gray
        if args.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"avc1"),
                                         src.fps or 30.0, (w, h))
            writer.write(frame)
    nbytes = src.bytes_received - bytes0
    received = src.frames_received
    dropped = src.frames_dropped
    src.release()
    if writer is not None:
        writer.release()

    span = (last_t - first_t) if (first_t is not None and last_t is not None) else 0.0
    fps = (n - 1) / span if span > 0 and n > 1 else 0.0
    pts_span = (last_pts - first_pts) if (first_pts is not None and last_pts is not None) else float("nan")
    mean_bytes = nbytes / received if received else 0.0
    mbit = (nbytes * 8 / 1e6) / args.seconds if args.seconds > 0 else 0.0
    shape = prev_gray.shape if prev_gray is not None else None
    print(
        f"frames   : {n} read ok, {fails} read timeouts, {received} received on the wire, "
        f"{dropped} seq gaps (phone drops), {src.decode_failures} undecodable\n"
        f"fps      : {fps:.2f} measured from read() over {span:.2f}s (pts span {pts_span:.2f}s)\n"
        f"dupes    : {dup} byte-identical consecutive gray frames (must be 0)\n"
        f"bytes    : {mean_bytes:.0f} mean bytes/frame, ~{mbit:.1f} Mbit/s over {args.seconds:.0f}s\n"
        + (f"shape    : {shape[1]}x{shape[0]}" if shape else "shape    : (no frames)")
    )
    if args.save and n:
        print(f"wrote    : {args.save}")
    return 0 if n else 1


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m phase0.capture.netsource",
                                description="WideCam (iPhone MJPEG over WiFi) source demo")
    p.add_argument("--target", default="auto", help="auto | host/IP | full URL")
    p.add_argument("--list", action="store_true", help="browse Bonjour for _widecam._tcp and exit")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--save", default=None, help="write the received frames to this mp4")
    p.add_argument("--control", action="append", default=[], metavar="K=V",
                   help="POST /control k=v before measuring (repeatable), e.g. exposure=lock")
    p.add_argument("--timeout", type=float, default=2.0, help="read() timeout (s)")
    p.add_argument("--connect-timeout", type=float, default=5.0)
    args = p.parse_args(argv)
    if args.list:
        hits = discover(BROWSE_SECONDS, want=None)
        if not hits:
            print(f"no {SERVICE_TYPE} services found in {BROWSE_SECONDS:.0f}s")
            return 1
        for h in hits:
            print(f"{h['name']}: http://{h['host']}:{h['port']}/stream  txt={h['txt']}")
        return 0
    return _demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
