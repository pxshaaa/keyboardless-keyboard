"""Mailbox/sequence tests for AVVideoSource (no camera), plus an optional
real-device smoke test that skips when no Desk View camera is present."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase0.capture.avsource import AVVideoSource, FrameMailbox  # noqa: E402


class FakeSource(AVVideoSource):
    """AVVideoSource with the AVFoundation setup bypassed; read()/release() use
    the real code paths against a FrameMailbox fed by a thread."""

    def __init__(self, read_timeout=0.2):
        self.read_timeout = read_timeout
        self._mailbox = FrameMailbox()
        self._last_seq = 0
        self._last_pts = float("nan")
        self._opened = True
        self._released = False
        self._name = "fake"
        self.width, self.height, self.fps = 4, 3, 30.0
        self._delegate = None
        self._counts = (0, 0)


def _producer(mailbox, n, period, stop):
    for i in range(n):
        if stop.is_set():
            return
        frame = np.full((3, 4, 3), i % 256, dtype=np.uint8)
        mailbox.put(frame, pts=i * period)
        time.sleep(period)


def test_read_never_returns_same_sequence_twice():
    src = FakeSource(read_timeout=0.5)
    stop = threading.Event()
    t = threading.Thread(target=_producer, args=(src._mailbox, 60, 0.004, stop), daemon=True)
    t.start()
    seqs, pts = [], []
    while True:
        ok, frame = src.read()
        if not ok:
            break
        assert frame.shape == (3, 4, 3) and frame.dtype == np.uint8
        seqs.append(src.get(1))  # CAP_PROP_POS_FRAMES == last seq
        pts.append(src.last_pts)
    t.join()
    assert len(seqs) > 5
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert all(b > a for a, b in zip(pts, pts[1:]))
    src.release()


def test_slow_reader_gets_latest_not_backlog():
    src = FakeSource(read_timeout=0.5)
    for i in range(10):
        src._mailbox.put(np.full((3, 4, 3), i, np.uint8), pts=float(i))
    ok, frame = src.read()
    assert ok and int(frame[0, 0, 0]) == 9 and src.last_pts == 9.0
    ok, frame = src.read(timeout=0.05)
    assert (ok, frame) == (False, None)


def test_read_times_out_cleanly_when_producer_stops():
    src = FakeSource(read_timeout=0.1)
    stop = threading.Event()
    t = threading.Thread(target=_producer, args=(src._mailbox, 5, 0.01, stop), daemon=True)
    t.start()
    t.join()
    got = 0
    while src.read()[0]:
        got += 1
    assert got == 1  # only the latest of the 5 was still unread
    t0 = time.monotonic()
    assert src.read() == (False, None)
    assert 0.08 <= time.monotonic() - t0 < 0.5


def test_release_twice_and_read_after_release():
    src = FakeSource()
    src.release()
    src.release()
    assert not src.isOpened()
    assert src.read() == (False, None)


def test_wait_newer_wakes_on_close():
    mb = FrameMailbox()
    threading.Timer(0.05, mb.close).start()
    t0 = time.monotonic()
    assert mb.wait_newer(0, timeout=2.0) is None
    assert time.monotonic() - t0 < 1.0


def _has_desk_view() -> bool:
    try:
        from phase0.capture.devices import list_cameras
        return any("desk view" in c["name"].lower() for c in list_cameras())
    except Exception:
        return False


@pytest.mark.skipif(not _has_desk_view(), reason="no Desk View camera attached")
def test_real_desk_view_delivers_new_frames():
    src = AVVideoSource("Desk View", read_timeout=2.0)
    try:
        assert src.isOpened()
        frames = [f for ok, f in (src.read() for _ in range(5)) if ok]
        assert len(frames) >= 2, "no frames -- camera may be held by another process"
        assert frames[0].shape == (src.height, src.width, 3)
        assert frames[0].flags["C_CONTIGUOUS"]
        assert not np.isnan(src.last_pts)
    finally:
        src.release()
