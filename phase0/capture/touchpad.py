"""Built-in trackpad contacts via private MultitouchSupport -> touches.jsonl on the time.monotonic() clock.
Manual test: PYTHONPATH=. .venv/bin/python -m phase0.capture.touchpad --seconds 10 (tap the pad); --probe needs no touch."""

from __future__ import annotations

import argparse
import ctypes
import math
import queue
import sys
import threading
import time
from ctypes import CFUNCTYPE, POINTER, Structure, c_bool, c_double, c_float, c_int, c_void_p
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

FRAMEWORK = "/System/Library/PrivateFrameworks/MultitouchSupport.framework/MultitouchSupport"
COREFOUNDATION = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"

# MTTouchState. Only MAKE_TOUCH/TOUCHING are glass contact; 1/2/6 are capacitive
# proximity ("in range") without contact and are NOT logged as touches.
STATE_NAMES = {
    0: "not_tracking", 1: "start_in_range", 2: "hover_in_range", 3: "make_touch",
    4: "touching", 5: "break_touch", 6: "linger_in_range", 7: "out_of_range",
}
CONTACT_STATES = frozenset({3, 4})
# Framework timestamps within this of time.monotonic() are taken to share its clock.
SHARED_CLOCK_TOL_S = 2.0
_QUEUE_MAX = 4096


class TouchpadError(RuntimeError):
    pass


# -- ctypes layout (arm64 + x86_64; verified sizeof == 96) -------------------------
class MTPoint(Structure):
    _fields_ = [("x", c_float), ("y", c_float)]


class MTVector(Structure):
    _fields_ = [("position", MTPoint), ("velocity", MTPoint)]


class MTTouch(Structure):
    _fields_ = [
        ("frame", c_int),
        ("timestamp", c_double),
        ("identifier", c_int),      # path id, stable for the life of a contact
        ("state", c_int),           # MTTouchState
        ("finger_id", c_int),       # framework's own finger guess (unreliable)
        ("hand_id", c_int),
        ("normalized", MTVector),   # position 0..1, origin bottom-left (y up, away from user)
        ("z_total", c_float),       # contact size / capacitance sum
        ("_f9", c_int),
        ("angle", c_float),         # ellipse angle, radians
        ("major_axis", c_float),
        ("minor_axis", c_float),
        ("absolute", MTVector),     # framework mm-ish vector (not relied on)
        ("_f14", c_int),
        ("_f15", c_int),
        ("z_density", c_float),     # capacitance density; a pressure proxy, not Force Touch force
    ]


MTTOUCH_SIZE = 96
FrameCallback = CFUNCTYPE(c_int, c_void_p, POINTER(MTTouch), c_int, c_double, c_int)


def touch_to_raw(t: MTTouch) -> dict[str, float]:
    """Copy one MTTouch out of framework memory (only valid during the callback)."""
    return {
        "frame": int(t.frame), "t_dev": float(t.timestamp), "id": int(t.identifier),
        "state": int(t.state), "finger": int(t.finger_id), "hand": int(t.hand_id),
        "x_norm": float(t.normalized.position.x), "y_norm": float(t.normalized.position.y),
        "vx": float(t.normalized.velocity.x), "vy": float(t.normalized.velocity.y),
        "size": float(t.z_total), "angle": float(t.angle),
        "major": float(t.major_axis), "minor": float(t.minor_axis),
        "density": float(t.z_density),
    }


# -- pure logic (hardware-free, unit tested) --------------------------------------
def to_mm(x_norm: float, y_norm: float, width_mm: float, height_mm: float) -> tuple[float, float]:
    """Pad mm with origin at the FAR-LEFT corner as the user sits; y grows toward the user."""
    return x_norm * width_mm, (1.0 - y_norm) * height_mm


class ClockMap:
    """Measured on Mac15,6: t_dev has its own epoch and runs 0.88% slow vs the host, so online t is the callback
    arrival (jitter <= 2 ms) unless the clocks agree ("shared"); map_clock refits t from t_dev offline."""

    def __init__(self) -> None:
        self.mode: Optional[str] = None
        self.first_d: Optional[float] = None
        self.n = 0
        self._x0 = self._y0 = 0.0
        self._sx = self._sy = self._sxx = self._sxy = 0.0

    def __call__(self, t_dev: float, t_rx: float) -> float:
        if self.mode is None:
            self.first_d = t_rx - t_dev
            self.mode = "shared" if abs(self.first_d) < SHARED_CLOCK_TOL_S else "arrival"
            self._x0, self._y0 = t_dev, t_rx
        x, y = t_dev - self._x0, t_rx - self._y0
        self.n += 1
        self._sx += x
        self._sy += y
        self._sxx += x * x
        self._sxy += x * y
        return t_dev if self.mode == "shared" else t_rx

    def rate(self) -> Optional[float]:
        den = self.n * self._sxx - self._sx ** 2
        if self.n < 2 or den <= 1e-12:
            return None
        return (self.n * self._sxy - self._sx * self._sy) / den

    def info(self) -> dict[str, Any]:
        r = self.rate()
        return {
            "clock_mode": self.mode,
            "rx_minus_dev_s": None if self.first_d is None else round(self.first_d, 6),
            "host_per_dev_second": None if r is None else round(r, 6),
        }


def map_clock(t_dev: Iterable[float], t_rx: Iterable[float]) -> list[float]:
    """Offline t: t_dev if shared, else a lower-envelope affine fit t_rx ~ a + b*t_dev (removes drift and jitter)."""
    import numpy as np

    td, tr = np.asarray(list(t_dev), float), np.asarray(list(t_rx), float)
    if not len(td):
        return []
    if abs(tr[0] - td[0]) < SHARED_CLOCK_TOL_S:
        return td.tolist()
    x = td - td[0]
    if np.ptp(x) < 1e-6:
        return tr.tolist()
    b, a = np.polyfit(x, tr, 1)
    for _ in range(4):
        res = tr - (a + b * x)
        low = res <= np.quantile(res, 0.2)
        if low.sum() < 2:
            break
        b, a = np.polyfit(x[low], tr[low], 1)
    return (a + b * x).tolist()


class ContactTracker:
    """Per-frame touch lists -> down/move/up rows by path id; a contact ends when it leaves
    CONTACT_STATES or vanishes from the list (the framework sends an empty frame after the last lift)."""

    def __init__(self, width_mm: float, height_mm: float) -> None:
        self.width_mm = width_mm
        self.height_mm = height_mm
        self.active: dict[int, dict[str, Any]] = {}
        self.downs = 0
        self.ups = 0
        self.out_of_range_xy = 0

    def _row(self, event: str, t: float, t_rx: float, raw: dict[str, Any]) -> dict[str, Any]:
        x_mm, y_mm = to_mm(raw["x_norm"], raw["y_norm"], self.width_mm, self.height_mm)
        if not (-0.05 <= raw["x_norm"] <= 1.05 and -0.05 <= raw["y_norm"] <= 1.05):
            self.out_of_range_xy += 1
        return {
            "t": t, "t_dev": raw["t_dev"], "t_rx": t_rx, "frame": raw["frame"], "event": event,
            "id": raw["id"], "finger": raw["finger"], "hand": raw["hand"], "state": raw["state"],
            "x_norm": round(raw["x_norm"], 5), "y_norm": round(raw["y_norm"], 5),
            "x_mm": round(x_mm, 3), "y_mm": round(y_mm, 3),
            "vx": round(raw["vx"], 4), "vy": round(raw["vy"], 4),
            "major": round(raw["major"], 4), "minor": round(raw["minor"], 4),
            "angle": round(raw["angle"], 4), "size": round(raw["size"], 4),
            "density": round(raw["density"], 4),
        }

    def feed(self, touches: list[dict[str, Any]], t: float, t_rx: float, t_dev: float,
             frame: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[int] = set()
        for raw in touches:
            pid = raw["id"]
            seen.add(pid)
            in_contact = raw["state"] in CONTACT_STATES
            was = pid in self.active
            if in_contact:
                rows.append(self._row("move" if was else "down", t, t_rx, raw))
                self.active[pid] = raw
                if not was:
                    self.downs += 1
            elif was:
                rows.append(self._row("up", t, t_rx, raw))
                del self.active[pid]
                self.ups += 1
        for pid in [p for p in self.active if p not in seen]:
            last = dict(self.active.pop(pid))
            last.update(t_dev=t_dev, frame=frame, state=7)
            rows.append(self._row("up", t, t_rx, last))
            self.ups += 1
        return rows

    def close(self, t: float) -> list[dict[str, Any]]:
        """Synthetic ups for contacts still down at shutdown, flagged truncated."""
        rows = []
        for pid in list(self.active):
            last = self.active.pop(pid)
            row = self._row("up", t, t, last)
            row["truncated"] = True
            rows.append(row)
            self.ups += 1
        return rows


# -- framework binding ------------------------------------------------------------
class Framework:
    """Thin ctypes binding. Kept separate so TouchpadLogger can be tested with a fake."""

    def __init__(self, path: str = FRAMEWORK) -> None:
        if sys.platform != "darwin":
            raise TouchpadError("MultitouchSupport is macOS-only")
        try:
            mt = ctypes.CDLL(path)
            cf = ctypes.CDLL(COREFOUNDATION)
        except OSError as exc:
            raise TouchpadError(f"cannot load {path}: {exc}") from exc
        mt.MTDeviceCreateList.restype = c_void_p
        mt.MTDeviceCreateDefault.restype = c_void_p
        mt.MTDeviceIsBuiltIn.argtypes = [c_void_p]
        mt.MTDeviceIsBuiltIn.restype = c_bool
        mt.MTDeviceGetSensorSurfaceDimensions.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int)]
        mt.MTDeviceGetSensorDimensions.argtypes = [c_void_p, POINTER(c_int), POINTER(c_int)]
        mt.MTDeviceGetFamilyID.argtypes = [c_void_p, POINTER(c_int)]
        mt.MTRegisterContactFrameCallback.argtypes = [c_void_p, FrameCallback]
        mt.MTUnregisterContactFrameCallback.argtypes = [c_void_p, FrameCallback]
        mt.MTDeviceStart.argtypes = [c_void_p, c_int]
        mt.MTDeviceStart.restype = c_int
        mt.MTDeviceStop.argtypes = [c_void_p]
        mt.MTDeviceStop.restype = c_int
        mt.MTDeviceIsRunning.argtypes = [c_void_p]
        mt.MTDeviceIsRunning.restype = c_bool
        cf.CFArrayGetCount.argtypes = [c_void_p]
        cf.CFArrayGetCount.restype = ctypes.c_long
        cf.CFArrayGetValueAtIndex.argtypes = [c_void_p, ctypes.c_long]
        cf.CFArrayGetValueAtIndex.restype = c_void_p
        self.mt, self.cf = mt, cf
        self._list = None  # retained CFArray keeps device refs alive

    def devices(self) -> list[dict[str, Any]]:
        self._list = self.mt.MTDeviceCreateList()
        out = []
        n = self.cf.CFArrayGetCount(self._list) if self._list else 0
        for k in range(n):
            dev = self.cf.CFArrayGetValueAtIndex(self._list, k)
            w, h, rows, cols, fam = c_int(), c_int(), c_int(), c_int(), c_int()
            self.mt.MTDeviceGetSensorSurfaceDimensions(dev, ctypes.byref(w), ctypes.byref(h))
            self.mt.MTDeviceGetSensorDimensions(dev, ctypes.byref(rows), ctypes.byref(cols))
            self.mt.MTDeviceGetFamilyID(dev, ctypes.byref(fam))
            out.append({
                "ref": dev, "index": k, "builtin": bool(self.mt.MTDeviceIsBuiltIn(dev)),
                # surface dimensions are hundredths of a millimetre
                "width_mm": w.value / 100.0, "height_mm": h.value / 100.0,
                "sensor_rows": rows.value, "sensor_cols": cols.value, "family_id": fam.value,
            })
        return out

    def register(self, dev: int, cb: Any) -> None:
        self.mt.MTRegisterContactFrameCallback(dev, cb)

    def unregister(self, dev: int, cb: Any) -> None:
        self.mt.MTUnregisterContactFrameCallback(dev, cb)

    def start(self, dev: int) -> int:
        return int(self.mt.MTDeviceStart(dev, 0))

    def stop(self, dev: int) -> int:
        return int(self.mt.MTDeviceStop(dev))

    def is_running(self, dev: int) -> bool:
        return bool(self.mt.MTDeviceIsRunning(dev))


# -- logger -----------------------------------------------------------------------
class TouchpadLogger:
    """Framework thread only copies touches + arrival time into a queue; the logger thread writes rows."""

    def __init__(self, writer: Any, clock: Callable[[], float] = time.monotonic,
                 framework: Optional[Any] = None, echo: Optional[Callable[[dict], None]] = None) -> None:
        self.writer = writer
        self.clock = clock
        self._fw = framework
        self.echo = echo
        self.q: "queue.Queue[Optional[tuple]]" = queue.Queue(_QUEUE_MAX)
        self.clockmap = ClockMap()
        self.tracker: Optional[ContactTracker] = None
        self.device: Optional[dict[str, Any]] = None
        self.frames = 0
        self.frames_dropped = 0
        self.rows = 0
        self._cb = FrameCallback(self._on_frame)
        self._thread: Optional[threading.Thread] = None
        self._started = False

    # framework thread: keep this minimal
    def _on_frame(self, dev, touches, n, timestamp, frame) -> int:
        t_rx = self.clock()
        try:
            raws = [touch_to_raw(touches[k]) for k in range(n)]
            self.q.put_nowait((float(timestamp), int(frame), t_rx, raws))
        except queue.Full:
            self.frames_dropped += 1
        except Exception:  # never raise into the framework
            self.frames_dropped += 1
        return 0

    def _process(self, item: tuple) -> None:
        t_dev, frame, t_rx, raws = item
        self.frames += 1
        t = self.clockmap(t_dev, t_rx)
        assert self.tracker is not None
        for row in self.tracker.feed(raws, t, t_rx, t_dev, frame):
            self._write(row)

    def _write(self, row: dict[str, Any]) -> None:
        self.writer.write(row)
        self.rows += 1
        if self.echo is not None and row["event"] != "move":
            self.echo(row)

    def _loop(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                break
            self._process(item)

    def start(self) -> dict[str, Any]:
        if self._fw is None:
            self._fw = Framework()
        devs = self._fw.devices()
        if not devs:
            raise TouchpadError("MultitouchSupport lists no multitouch devices")
        dev = next((d for d in devs if d["builtin"]), devs[0])
        if not dev["width_mm"] or not dev["height_mm"]:
            raise TouchpadError(f"device reports no surface size: {dev}")
        self.device = dev
        self.tracker = ContactTracker(dev["width_mm"], dev["height_mm"])
        self._thread = threading.Thread(target=self._loop, name="touchpad-logger", daemon=True)
        self._thread.start()
        self._fw.register(dev["ref"], self._cb)
        rc = self._fw.start(dev["ref"])
        if rc != 0 or not self._fw.is_running(dev["ref"]):
            self._fw.unregister(dev["ref"], self._cb)
            self.q.put(None)
            raise TouchpadError(f"MTDeviceStart returned {rc}, running={self._fw.is_running(dev['ref'])}")
        self._started = True
        return self.info()

    def stop(self) -> dict[str, Any]:
        if self._started and self.device is not None:
            try:
                self._fw.unregister(self.device["ref"], self._cb)
                self._fw.stop(self.device["ref"])
            finally:
                self._started = False
        if self._thread is not None:
            self.q.put(None)
            self._thread.join(timeout=5)
            self._thread = None
            if self.tracker is not None:
                for row in self.tracker.close(self.clock()):
                    self._write(row)
        return self.info()

    def info(self) -> dict[str, Any]:
        d = self.device or {}
        tr = self.tracker
        return {
            "width_mm": d.get("width_mm"), "height_mm": d.get("height_mm"),
            "builtin": d.get("builtin"), "family_id": d.get("family_id"),
            "sensor_rows": d.get("sensor_rows"), "sensor_cols": d.get("sensor_cols"),
            "frames": self.frames, "frames_dropped": self.frames_dropped, "rows": self.rows,
            "downs": tr.downs if tr else 0, "ups": tr.ups if tr else 0,
            "xy_out_of_range": tr.out_of_range_xy if tr else 0,
            **self.clockmap.info(),
        }


# -- CLI --------------------------------------------------------------------------
def _echo(row: dict[str, Any]) -> None:
    print(f"  {row['event']:>4}  id={row['id']:<3} t={row['t']:.4f}  x={row['x_mm']:6.1f}mm y={row['y_mm']:5.1f}mm  "
          f"major={row['major']:.2f} density={row['density']:.3f}", flush=True)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m phase0.capture.touchpad",
                                description="Log built-in trackpad contacts (MultitouchSupport).")
    p.add_argument("--probe", action="store_true", help="list devices, start+stop, exit")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--out", type=Path, default=Path("/tmp/touches_test.jsonl"))
    a = p.parse_args(argv)

    from phase0.capture.recorder import JsonlWriter

    if a.probe:
        fw = Framework()
        for d in fw.devices():
            print({k: v for k, v in d.items() if k != "ref"})
        with JsonlWriter(Path("/dev/null")) as w:
            lg = TouchpadLogger(w, framework=fw)
            print("start:", lg.start())
            time.sleep(1.0)
            print("stop :", lg.stop())
        print(f"sizeof(MTTouch)={ctypes.sizeof(MTTouch)} (expected {MTTOUCH_SIZE})")
        return 0

    a.out.unlink(missing_ok=True)
    with JsonlWriter(a.out, flush_every=1) as w:
        lg = TouchpadLogger(w, echo=_echo)
        info = lg.start()
        print(f"trackpad {info['width_mm']}x{info['height_mm']} mm -- tap it now ({a.seconds:.0f}s)", flush=True)
        try:
            time.sleep(a.seconds)
        except KeyboardInterrupt:
            pass
        info = lg.stop()
    print("summary:", info)
    print(f"wrote {a.out}")
    ok = info["downs"] > 0 and info["xy_out_of_range"] == 0
    print("RESULT:", "PASS" if ok else ("NO TOUCHES SEEN" if info["downs"] == 0 else "CHECK summary"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
