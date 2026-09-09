"""Map camera *names* to the indices cv2.VideoCapture expects.

Why this exists: cv2.VideoCapture only takes an integer index, and on this
machine the device list contains an OBS Virtual Camera alongside the real
ones. Silently recording the wrong device would invalidate an entire session,
so the recorder resolves a human-readable name to an index up front and prints
what it picked.

MEASURED CAVEAT (2026-09-10, macOS 26.2): OpenCV's AVFoundation backend does
NOT see every device AVFoundation reports. On this machine AVFoundation lists 4
cameras but OpenCV only accepts indices 0-2 -- it silently omits the iPhone
"Desk View" camera. The first three indices did line up, so name->index
resolution works for the addressable devices, but never assume the lists are
identical. resolve() warns when the AVFoundation index exceeds OpenCV's range.

FRAME RATE, MEASURED: Continuity Camera advertises 1920x1440 @ 60fps in its
format list, but delivers ~30fps in practice -- confirmed both through OpenCV
(which ignores CAP_PROP_FPS entirely) and through direct AVFoundation with
activeVideoMin/MaxFrameDuration forced to 1/60. 60fps over Continuity Camera is
not attainable from the Mac. Getting 120fps would require an app running ON the
iPhone. Plan for 30fps.
"""
from __future__ import annotations

import AVFoundation as _AV  # pyobjc-framework-AVFoundation


def _device_types() -> list:
    names = [
        "AVCaptureDeviceTypeBuiltInWideAngleCamera",
        "AVCaptureDeviceTypeExternal",
        "AVCaptureDeviceTypeContinuityCamera",
        "AVCaptureDeviceTypeDeskViewCamera",
    ]
    return [getattr(_AV, n) for n in names if hasattr(_AV, n)]


def list_cameras() -> list[dict]:
    """Return [{index, name, model, uid}], index being the cv2 index."""
    session = _AV.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        _device_types(), _AV.AVMediaTypeVideo, 0
    )
    out = []
    for i, dev in enumerate(session.devices()):
        out.append({
            "index": i,
            "name": str(dev.localizedName()),
            "model": str(dev.modelID()),
            "uid": str(dev.uniqueID()),
        })
    return out


def resolve(spec: str | int) -> tuple[int, str]:
    """Resolve an index or a case-insensitive name substring to (index, name).

    Raises ValueError on no match or an ambiguous match — never guesses,
    because guessing wrong costs a whole recording session.
    """
    cams = list_cameras()
    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        idx = int(spec)
        for c in cams:
            if c["index"] == idx:
                return idx, c["name"]
        return idx, f"<index {idx}, name unknown>"

    # OpenCV only accepts a prefix of AVFoundation's device list (see module
    # docstring), so flag any index it will refuse before a session is wasted.
    needle = str(spec).lower()
    hits = [c for c in cams if needle in c["name"].lower()]
    if not hits:
        avail = ", ".join(f'{c["index"]}:{c["name"]}' for c in cams)
        raise ValueError(f"no camera matching {spec!r}. Available: {avail}")
    if len(hits) > 1:
        avail = ", ".join(f'{c["index"]}:{c["name"]}' for c in hits)
        raise ValueError(f"{spec!r} is ambiguous, matches: {avail}")
    idx, name = hits[0]["index"], hits[0]["name"]
    if idx > 2:
        print(f"WARNING: {name!r} is AVFoundation index {idx}; OpenCV has "
              f"historically only accepted 0-2 on this machine and may refuse it.")
    return idx, name


def main() -> None:
    for c in list_cameras():
        print(f'{c["index"]}: {c["name"]}  ({c["model"]})')


if __name__ == "__main__":
    main()


def preflight(cap, seconds: float = 2.0) -> dict:
    """Check a capture is delivering GENUINELY NEW frames, not a frozen one.

    Why: measured 2026-09-10 on this machine, the built-in FaceTime camera with
    the lid closed happily returns True from cap.read() ~56x/second -- with
    every frame byte-identical. Frame *rate* looks great and the recording is
    worthless. Nothing else in the pipeline can detect this after the fact, so
    it has to be caught before a session starts.

    Returns a dict of measurements; caller decides whether to abort.
    """
    import time

    import cv2
    import numpy as np

    for _ in range(10):  # let exposure/AF settle
        cap.read()

    prev = None
    identical = 0
    n = 0
    diffs: list[float] = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
        if prev is not None:
            if np.array_equal(gray, prev):
                identical += 1
            diffs.append(float(np.abs(gray - prev).mean()))
        prev = gray
        n += 1
    dt = time.monotonic() - t0

    pairs = max(len(diffs), 1)
    frac_identical = identical / pairs
    return {
        "reads": n,
        "reads_per_sec": n / dt if dt else 0.0,
        "pairs": pairs,
        "identical_pairs": identical,
        "frac_identical": frac_identical,
        "mean_abs_diff": float(np.mean(diffs)) if diffs else 0.0,
        # >50% byte-identical consecutive frames means the sensor is not live.
        "frozen": frac_identical > 0.5,
    }
