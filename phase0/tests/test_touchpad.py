"""Hardware-free tests for the trackpad logger: struct layout, contact tracking, clock mapping."""

from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase0.capture import touchpad as tp  # noqa: E402
from phase0.capture.recorder import JsonlWriter, build_parser  # noqa: E402

W, H = 124.8, 76.8


def raw(pid=1, state=4, x=0.5, y=0.5, t_dev=100.0, frame=1, **kw):
    r = {"frame": frame, "t_dev": t_dev, "id": pid, "state": state, "finger": 2, "hand": 1,
         "x_norm": x, "y_norm": y, "vx": 0.0, "vy": 0.0, "size": 1.0, "angle": 0.0,
         "major": 8.0, "minor": 7.0, "density": 0.1}
    r.update(kw)
    return r


# --- struct layout -----------------------------------------------------------------

def test_mttouch_layout_matches_framework_abi():
    assert ctypes.sizeof(tp.MTTouch) == tp.MTTOUCH_SIZE == 96
    assert tp.MTTouch.timestamp.offset == 8
    assert tp.MTTouch.normalized.offset == 32
    assert tp.MTTouch.major_axis.offset == 60
    assert tp.MTTouch.z_density.offset == 92


def test_touch_to_raw_reads_a_packed_buffer():
    arr = (tp.MTTouch * 2)()
    arr[1].timestamp = 123.25
    arr[1].identifier = 7
    arr[1].state = 3
    arr[1].normalized.position.x = 0.25
    arr[1].normalized.position.y = 0.75
    arr[1].z_density = 0.5
    r = tp.touch_to_raw(ctypes.cast(arr, ctypes.POINTER(tp.MTTouch))[1])
    assert r["t_dev"] == 123.25 and r["id"] == 7 and r["state"] == 3
    assert r["x_norm"] == 0.25 and r["y_norm"] == 0.75 and r["density"] == 0.5


# --- geometry ---------------------------------------------------------------------

def test_to_mm_origin_is_far_left_and_y_grows_toward_user():
    assert tp.to_mm(0.0, 1.0, W, H) == (0.0, 0.0)
    x, y = tp.to_mm(1.0, 0.0, W, H)
    assert x == pytest.approx(W) and y == pytest.approx(H)


# --- contact tracking -------------------------------------------------------------

def test_down_move_up_sequence():
    tr = tp.ContactTracker(W, H)
    assert [r["event"] for r in tr.feed([raw(state=3)], 1.0, 1.001, 1.0, 1)] == ["down"]
    assert [r["event"] for r in tr.feed([raw(state=4)], 1.008, 1.009, 1.008, 2)] == ["move"]
    rows = tr.feed([raw(state=5)], 1.016, 1.017, 1.016, 3)
    assert [r["event"] for r in rows] == ["up"]
    assert tr.downs == tr.ups == 1 and not tr.active


def test_hover_states_are_not_contacts():
    tr = tp.ContactTracker(W, H)
    for st in (1, 2, 6, 7):
        assert tr.feed([raw(state=st)], 1.0, 1.0, 1.0, 1) == []
    assert tr.downs == 0


def test_vanished_contact_emits_up_with_current_frame_time():
    tr = tp.ContactTracker(W, H)
    tr.feed([raw(pid=1), raw(pid=2, x=0.1)], 1.0, 1.0, 1.0, 1)
    rows = tr.feed([], 2.0, 2.002, 2.0, 9)
    assert sorted((r["id"], r["event"], r["t"], r["frame"]) for r in rows) == [
        (1, "up", 2.0, 9), (2, "up", 2.0, 9)]


def test_two_fingers_are_independent():
    tr = tp.ContactTracker(W, H)
    tr.feed([raw(pid=1)], 1.0, 1.0, 1.0, 1)
    rows = tr.feed([raw(pid=1), raw(pid=2, state=3)], 1.01, 1.01, 1.01, 2)
    assert {(r["id"], r["event"]) for r in rows} == {(1, "move"), (2, "down")}


def test_close_flags_truncated_contacts():
    tr = tp.ContactTracker(W, H)
    tr.feed([raw()], 1.0, 1.0, 1.0, 1)
    rows = tr.close(5.0)
    assert rows[0]["event"] == "up" and rows[0]["truncated"] is True


def test_rows_carry_mm_and_flag_out_of_range_xy():
    tr = tp.ContactTracker(W, H)
    r = tr.feed([raw(x=0.5, y=0.25)], 1.0, 1.0, 1.0, 1)[0]
    assert r["x_mm"] == pytest.approx(62.4) and r["y_mm"] == pytest.approx(57.6)
    tr.feed([raw(pid=3, x=3.0)], 1.0, 1.0, 1.0, 1)
    assert tr.out_of_range_xy == 1


# --- clock ------------------------------------------------------------------------

def test_clockmap_shared_uses_device_time_exactly():
    cm = tp.ClockMap()
    assert cm(1000.0, 1000.004) == 1000.0
    assert cm(1000.01, 1000.012) == 1000.01
    assert cm.info()["clock_mode"] == "shared"


def test_clockmap_foreign_epoch_uses_arrival_and_reports_rate():
    cm = tp.ClockMap()
    for k in range(100):
        rx = 198474.0 + 0.008 * k * 1.0088
        assert cm(212428.0 + 0.008 * k, rx) == rx
    info = cm.info()
    assert info["clock_mode"] == "arrival"
    assert info["host_per_dev_second"] == pytest.approx(1.0088, abs=1e-4)


def test_map_clock_offline_removes_drift_and_jitter():
    rng = np.random.default_rng(0)
    td = 212428.0 + np.sort(rng.uniform(0, 600, 5000))
    true = 198474.0 + 1.0088 * (td - td[0]) + 0.001
    tr = true + rng.exponential(0.001, len(td))
    assert np.abs(np.array(tp.map_clock(td, tr)) - true).max() < 0.0015
    assert tp.map_clock([1.0, 2.0], [1.003, 2.001]) == [1.0, 2.0]
    assert tp.map_clock([], []) == []


# --- logger with a fake framework --------------------------------------------------

class FakeFramework:
    def __init__(self, rc=0):
        self.rc, self.running, self.cb = rc, False, None

    def devices(self):
        return [{"ref": 1, "index": 0, "builtin": True, "width_mm": W, "height_mm": H,
                 "sensor_rows": 18, "sensor_cols": 24, "family_id": 109}]

    def register(self, dev, cb):
        self.cb = cb

    def unregister(self, dev, cb):
        self.cb = None

    def start(self, dev):
        self.running = self.rc == 0
        return self.rc

    def stop(self, dev):
        self.running = False
        return 0

    def is_running(self, dev):
        return self.running


def test_logger_end_to_end_without_hardware(tmp_path):
    clock = iter([10.001, 10.009, 10.017, 11.0]).__next__
    path = tmp_path / "touches.jsonl"
    with JsonlWriter(path, flush_every=1) as w:
        lg = tp.TouchpadLogger(w, clock=clock, framework=FakeFramework())
        info = lg.start()
        assert info["width_mm"] == W
        lg._process((10.0, 1, 10.001, [raw(state=3, t_dev=10.0)]))
        lg._process((10.008, 2, 10.009, [raw(state=4, t_dev=10.008)]))
        lg._process((10.016, 3, 10.017, []))
        info = lg.stop()
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["down", "move", "up"]
    assert rows[0]["t"] == 10.0 and info["downs"] == 1 and info["clock_mode"] == "shared"


def test_logger_start_failure_raises():
    with pytest.raises(tp.TouchpadError):
        tp.TouchpadLogger(open("/dev/null", "w"), framework=FakeFramework(rc=-1)).start()


def test_callback_never_raises_into_framework():
    lg = tp.TouchpadLogger(None, framework=FakeFramework())
    assert lg._on_frame(None, None, 1, 0.0, 0) == 0  # NULL touches -> counted drop
    assert lg.frames_dropped == 1


# --- recorder CLI -----------------------------------------------------------------

def test_recorder_default_has_touchpad_off():
    a = build_parser().parse_args(["--condition", "kbd"])
    assert a.touchpad is False


def test_recorder_accepts_pad_condition_and_flag():
    a = build_parser().parse_args(["--condition", "pad", "--touchpad"])
    assert a.condition == "pad" and a.touchpad is True
