"""cv2.VideoCapture-compatible source over AVFoundation (pyobjc): reaches devices
OpenCV's backend refuses, e.g. the iPhone Desk View camera (AVF index 3)."""
from __future__ import annotations

import argparse
import sys
import threading
import time
import warnings
from typing import Optional

import numpy as np

try:  # cv2 only needed for CAP_PROP_* constants and the demo; keep import soft.
    import cv2 as _cv2
    CAP_PROP_FRAME_WIDTH = _cv2.CAP_PROP_FRAME_WIDTH
    CAP_PROP_FRAME_HEIGHT = _cv2.CAP_PROP_FRAME_HEIGHT
    CAP_PROP_FPS = _cv2.CAP_PROP_FPS
    CAP_PROP_POS_FRAMES = _cv2.CAP_PROP_POS_FRAMES
except Exception:  # pragma: no cover
    CAP_PROP_POS_FRAMES, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS = 1, 3, 4, 5


class FrameMailbox:
    """Single-slot latest-frame mailbox; put() overwrites, wait_newer() only
    returns seq > after_seq so a reader never sees the same frame twice."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._pts: float = float("nan")
        self._seq: int = 0
        self._closed = False

    @property
    def seq(self) -> int:
        return self._seq

    def put(self, frame: np.ndarray, pts: float = float("nan")) -> int:
        with self._cond:
            self._seq += 1
            self._frame = frame
            self._pts = pts
            self._cond.notify_all()
            return self._seq

    def wait_newer(self, after_seq: int, timeout: float
                   ) -> Optional[tuple[int, np.ndarray, float]]:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._seq <= after_seq and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            if self._closed or self._frame is None:
                return None
            return self._seq, self._frame, self._pts

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()


def _pixel_buffer_to_bgr(img) -> Optional[np.ndarray]:
    """Copy a locked 32BGRA CVPixelBuffer into a contiguous HxWx3 BGR array."""
    import Quartz

    w = Quartz.CVPixelBufferGetWidth(img)
    h = Quartz.CVPixelBufferGetHeight(img)
    bpr = Quartz.CVPixelBufferGetBytesPerRow(img)
    base = Quartz.CVPixelBufferGetBaseAddress(img)
    if base is None or not hasattr(base, "as_buffer"):
        return None
    buf = base.as_buffer(bpr * h)
    a = np.frombuffer(buf, dtype=np.uint8).reshape(h, bpr // 4, 4)[:, :w, :3]
    return np.ascontiguousarray(a)  # copies: the view dies at unlock


def _make_delegate_class():
    """Build the ObjC delegate class lazily so importing this module never
    requires pyobjc (the tests exercise FrameMailbox only)."""
    import objc
    import CoreMedia as CM
    import Quartz

    class _AVSourceDelegate(objc.lookUpClass("NSObject")):
        def initWithMailbox_(self, mailbox):
            self = objc.super(_AVSourceDelegate, self).init()
            if self is None:
                return None
            self._mailbox = mailbox
            self.dropped = 0
            self.received = 0
            return self

        def captureOutput_didOutputSampleBuffer_fromConnection_(self, out, sbuf, conn):
            self.received += 1
            img = CM.CMSampleBufferGetImageBuffer(sbuf)
            if img is None:
                self.dropped += 1
                return
            try:
                pts = CM.CMTimeGetSeconds(CM.CMSampleBufferGetPresentationTimeStamp(sbuf))
            except Exception:
                pts = float("nan")
            Quartz.CVPixelBufferLockBaseAddress(img, 1)  # kCVPixelBufferLock_ReadOnly
            try:
                frame = _pixel_buffer_to_bgr(img)
            finally:
                Quartz.CVPixelBufferUnlockBaseAddress(img, 1)
            if frame is None:
                self.dropped += 1
                return
            self._mailbox.put(frame, pts)

        def captureOutput_didDropSampleBuffer_fromConnection_(self, out, sbuf, conn):
            self.dropped += 1

    return _AVSourceDelegate


def find_device(name_substring: str):
    """Case-insensitive substring over AVF device names; ValueError (listing
    names) on no/ambiguous match -- guessing wrong wastes a session."""
    import AVFoundation as AV
    from phase0.capture.devices import _device_types

    session = AV.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        _device_types(), AV.AVMediaTypeVideo, 0
    )
    devices = list(session.devices())
    names = [str(d.localizedName()) for d in devices]
    needle = name_substring.lower()
    hits = [(d, n) for d, n in zip(devices, names) if needle in n.lower()]
    if not hits:
        raise ValueError(
            f"no camera matching {name_substring!r}. Available: "
            + ", ".join(f"{i}:{n}" for i, n in enumerate(names))
        )
    if len(hits) > 1:
        raise ValueError(
            f"{name_substring!r} is ambiguous, matches: "
            + ", ".join(n for _, n in hits)
        )
    return hits[0]


class AVVideoSource:
    """cv2.VideoCapture look-alike; read() blocks up to read_timeout for a NEW
    BGR frame, (False, None) on timeout or after release()."""

    def __init__(self, name_substring: str, width: int = 1920, height: int = 1440,
                 fps: float = 30.0, pixel_format: str = "bgr",
                 read_timeout: float = 2.0) -> None:  # Continuity Camera stalls up to ~0.7s (measured)
        if pixel_format != "bgr":
            raise ValueError("only pixel_format='bgr' is supported")
        import AVFoundation as AV
        import CoreMedia as CM
        import Quartz
        from libdispatch import dispatch_queue_create

        self.read_timeout = read_timeout
        self._mailbox = FrameMailbox()
        self._last_seq = 0
        self._last_pts: float = float("nan")
        self._opened = False
        self._released = False
        self._delegate = None
        self._counts = (0, 0)
        self.requested = (width, height, fps)
        self.width = width
        self.height = height
        self.fps = fps

        self._device, self._name = find_device(name_substring)

        # --- pick a format: exact match, else largest <= requested ---------
        best = None
        best_key = None
        for fmt in self._device.formats():
            dims = CM.CMVideoFormatDescriptionGetDimensions(fmt.formatDescription())
            fw, fh = int(dims.width), int(dims.height)
            if fw > width or fh > height:
                continue
            ranges = fmt.videoSupportedFrameRateRanges()
            max_fps = max((float(r.maxFrameRate()) for r in ranges), default=0.0)
            key = (fw * fh, fw == width and fh == height, max_fps >= fps, max_fps)
            if best_key is None or key > best_key:
                best, best_key = fmt, key

        locked = False
        try:
            ok, err = self._device.lockForConfiguration_(None)
            locked = bool(ok)
            if not locked:
                warnings.warn(f"lockForConfiguration failed for {self._name!r}: {err}; "
                              "using the device's current defaults")
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"lockForConfiguration raised for {self._name!r}: {exc}")

        if locked:
            try:
                if best is not None:
                    self._device.setActiveFormat_(best)
                ranges = self._device.activeFormat().videoSupportedFrameRateRanges()
                max_fps = max((float(r.maxFrameRate()) for r in ranges), default=fps)
                min_fps = min((float(r.minFrameRate()) for r in ranges), default=1.0)
                want = max(min_fps, min(fps, max_fps))
                dur = CM.CMTimeMake(1000, int(round(want * 1000)))
                self._device.setActiveVideoMinFrameDuration_(dur)
                self._device.setActiveVideoMaxFrameDuration_(dur)
            except Exception as exc:
                warnings.warn(f"could not apply format/frame rate on {self._name!r}: {exc}")
            finally:
                self._device.unlockForConfiguration()

        # Record what we actually obtained.
        try:
            af = self._device.activeFormat()
            dims = CM.CMVideoFormatDescriptionGetDimensions(af.formatDescription())
            self.width, self.height = int(dims.width), int(dims.height)
            d = self._device.activeVideoMinFrameDuration()
            secs = CM.CMTimeGetSeconds(d)
            self.fps = (1.0 / secs) if secs and secs > 0 else fps
        except Exception:
            pass

        # --- session -------------------------------------------------------
        self._session = AV.AVCaptureSession.alloc().init()
        self._session.beginConfiguration()
        self._input, err = AV.AVCaptureDeviceInput.deviceInputWithDevice_error_(self._device, None)
        if self._input is None or not self._session.canAddInput_(self._input):
            self._session.commitConfiguration()
            raise RuntimeError(f"cannot add input for {self._name!r}: {err}")
        self._session.addInput_(self._input)

        self._output = AV.AVCaptureVideoDataOutput.alloc().init()
        self._output.setVideoSettings_(
            {Quartz.kCVPixelBufferPixelFormatTypeKey: Quartz.kCVPixelFormatType_32BGRA}
        )
        self._output.setAlwaysDiscardsLateVideoFrames_(True)
        self._delegate = _make_delegate_class().alloc().initWithMailbox_(self._mailbox)
        self._queue = dispatch_queue_create(b"phase0.avsource", None)
        self._output.setSampleBufferDelegate_queue_(self._delegate, self._queue)
        if not self._session.canAddOutput_(self._output):
            self._session.commitConfiguration()
            raise RuntimeError(f"cannot add video data output for {self._name!r}")
        self._session.addOutput_(self._output)
        self._session.commitConfiguration()
        self._session.startRunning()
        self._opened = bool(self._session.isRunning())

    # -- cv2.VideoCapture surface ----------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def last_pts(self) -> float:
        """CMSampleBuffer presentation timestamp (seconds, device clock) of the
        frame most recently returned by read(); NaN before the first read."""
        return self._last_pts

    @property
    def frames_received(self) -> int:
        return self._counts[0] if self._delegate is None else self._delegate.received

    @property
    def frames_dropped(self) -> int:
        return self._counts[1] if self._delegate is None else self._delegate.dropped

    def isOpened(self) -> bool:
        return self._opened and not self._released

    def read(self, timeout: Optional[float] = None) -> tuple[bool, Optional[np.ndarray]]:
        if self._released:
            return False, None
        got = self._mailbox.wait_newer(
            self._last_seq, self.read_timeout if timeout is None else timeout
        )
        if got is None:
            return False, None
        self._last_seq, frame, self._last_pts = got
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
            return float(self._last_seq)
        return 0.0

    def set(self, prop, value) -> bool:
        # Format is fixed at construction (like cv2's AVFoundation backend,
        # which ignores CAP_PROP_FPS); report success only if already matching.
        if prop == CAP_PROP_FRAME_WIDTH:
            return int(value) == self.width
        if prop == CAP_PROP_FRAME_HEIGHT:
            return int(value) == self.height
        if prop == CAP_PROP_FPS:
            return abs(float(value) - self.fps) < 0.5
        return False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._opened = False
        self._counts = (self.frames_received, self.frames_dropped)
        self._mailbox.close()
        sess = getattr(self, "_session", None)
        if sess is None:
            return
        try:
            if sess.isRunning():
                sess.stopRunning()
        except Exception:
            pass
        try:
            sess.beginConfiguration()
            out = getattr(self, "_output", None)
            if out is not None:
                out.setSampleBufferDelegate_queue_(None, None)
                sess.removeOutput_(out)
            inp = getattr(self, "_input", None)
            if inp is not None:
                sess.removeInput_(inp)
            sess.commitConfiguration()
        except Exception:
            pass
        self._delegate = None
        self._output = None
        self._input = None
        self._session = None

    def __enter__(self) -> "AVVideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.release()
        except Exception:
            pass


# -- CLI ----------------------------------------------------------------------

def _demo(args: argparse.Namespace) -> int:
    import cv2

    src = AVVideoSource(args.camera, args.width, args.height, args.fps)
    print(f"device   : {src.name}")
    print(f"format   : {src.width}x{src.height} @ {src.fps:.1f} fps "
          f"(requested {args.width}x{args.height} @ {args.fps:.0f})")
    if not src.isOpened():
        print("ERROR: session not running", file=sys.stderr)
        src.release()
        return 2

    writer = None
    n = 0
    dup = 0
    fails = 0
    prev_gray = None
    first_t = None
    last_t = None
    first_pts = None
    last_pts = None
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
        if prev_gray is not None and np.array_equal(gray, prev_gray):
            dup += 1
        prev_gray = gray
        if args.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"avc1"),
                                         src.fps or 30.0, (w, h))
            writer.write(frame)
    src.release()
    if writer is not None:
        writer.release()

    span = (last_t - first_t) if (first_t is not None and last_t is not None) else 0.0
    fps = (n - 1) / span if span > 0 and n > 1 else 0.0
    pts_span = (last_pts - first_pts) if (first_pts is not None and last_pts is not None) else float("nan")
    shape = prev_gray.shape if prev_gray is not None else None
    print(
        f"frames   : {n} read ok, {fails} read timeouts, "
        f"{src.frames_received} delivered, {src.frames_dropped} dropped by AVF\n"
        f"fps      : {fps:.2f} measured from read() over {span:.2f}s "
        f"(pts span {pts_span:.2f}s)\n"
        f"dupes    : {dup} byte-identical consecutive gray frames\n"
        f"shape    : {shape[1]}x{shape[0]}" if shape else "shape    : (no frames)"
    )
    if args.save and n:
        print(f"wrote    : {args.save}")
    return 0 if n else 1


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m phase0.capture.avsource")
    p.add_argument("--list", action="store_true", help="list AVFoundation cameras and exit")
    p.add_argument("--demo", action="store_true", help="capture, measure fps/dupes, optionally save")
    p.add_argument("--camera", default="Desk View")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1440)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--save", default=None)
    args = p.parse_args(argv)
    if args.list:
        from phase0.capture.devices import list_cameras
        for c in list_cameras():
            print(f'{c["index"]}: {c["name"]}  ({c["model"]})')
        return 0
    if args.demo:
        return _demo(args)
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
