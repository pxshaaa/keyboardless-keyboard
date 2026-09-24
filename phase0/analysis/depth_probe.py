"""Does iPhone LiDAR/stereo depth carry a keystroke signal? record | noise | spatial | signal."""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phase0.capture.netsource import MultipartParser


DEPTH_DTYPE = np.float32


# -- capture -------------------------------------------------------------------

@dataclass
class DepthMeta:
    seq: int
    t: float
    width: int
    height: int
    accuracy: str
    quality: str
    filtered: bool
    intrinsics: str
    intrinsics_ref: str


def _connect(host: str, path: str, timeout: float = 10.0):
    import http.client
    hostname, _, port = host.partition(":")
    conn = http.client.HTTPConnection(hostname, int(port or 8080), timeout=timeout)
    conn.request("GET", path)
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"{path} -> HTTP {resp.status}")
    return conn, resp


def record(host: str, seconds: float, out: Path, label: str) -> Path:
    import cv2

    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    rgb: list[tuple[int, float, bytes]] = []
    depth_rows: list[DepthMeta] = []
    depth_buf: list[np.ndarray] = []
    err: list[str] = []

    def pump(path: str, boundary: str, sink):
        try:
            conn, resp = _connect(host, path)
            parser = MultipartParser(boundary)
            while not stop.is_set():
                chunk = resp.read(65536)
                if not chunk:
                    break
                for headers, body in parser.feed(chunk):
                    sink(headers, body)
            conn.close()
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            err.append(f"{path}: {exc}")

    def on_rgb(h, b):
        rgb.append((int(h.get("x-seq", -1)), float(h.get("x-timestamp", 0)), b))

    def on_depth(h, b):
        w, hh = int(h["x-width"]), int(h["x-height"])
        a = np.frombuffer(b, dtype=DEPTH_DTYPE, count=w * hh).reshape(hh, w).copy()
        depth_buf.append(a)
        depth_rows.append(DepthMeta(int(h.get("x-seq", -1)), float(h.get("x-timestamp", 0)),
                                    w, hh, h.get("x-accuracy", "?"), h.get("x-quality", "?"),
                                    h.get("x-filtered", "0") == "1",
                                    h.get("x-intrinsics", ""), h.get("x-intrinsics-ref", "")))

    ts = [threading.Thread(target=pump, args=("/stream", "frame", on_rgb), daemon=True),
          threading.Thread(target=pump, args=("/depth", "depth", on_depth), daemon=True)]
    for t in ts:
        t.start()
    time.sleep(seconds)
    stop.set()
    for t in ts:
        t.join(timeout=3)

    if err:
        print("stream errors:", "; ".join(err), file=sys.stderr)
    if not rgb or not depth_buf:
        raise SystemExit(f"nothing captured (rgb={len(rgb)} depth={len(depth_buf)}): {err}")

    # video.mp4 + frames.jsonl so extract_landmarks/taps_gb can run unmodified.
    first = cv2.imdecode(np.frombuffer(rgb[0][2], np.uint8), cv2.IMREAD_COLOR)
    h, w = first.shape[:2]
    dt = np.diff([r[1] for r in rgb])
    fps = float(1.0 / np.median(dt)) if len(dt) else 30.0
    vw = cv2.VideoWriter(str(out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    with open(out / "frames.jsonl", "w") as f:
        for i, (_, t, jpg) in enumerate(rgb):
            vw.write(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR))
            f.write(json.dumps({"i": i, "t": t}) + "\n")
    vw.release()

    np.save(out / "depth.npy", np.stack(depth_buf))
    with open(out / "depth.jsonl", "w") as f:
        for r in depth_rows:
            f.write(json.dumps(r.__dict__) + "\n")
    status = json.loads(_status(host))
    (out / "meta.json").write_text(json.dumps({
        "session_id": out.name, "condition": label, "camera_name": f"WideCam@{host}",
        "width": w, "height": h, "fps_requested": fps, "fps_actual": fps,
        "t0_monotonic": rgb[0][1], "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "notes": f"depth_probe {label}; status={status}",
    }, indent=2))
    print(f"{out}: {len(rgb)} rgb @ {fps:.1f} fps, {len(depth_buf)} depth "
          f"{depth_rows[0].width}x{depth_rows[0].height} "
          f"accuracy={depth_rows[0].accuracy} quality={depth_rows[0].quality} "
          f"filtered={depth_rows[0].filtered} intrinsics={depth_rows[0].intrinsics}")
    return out


def _status(host: str) -> str:
    import http.client
    hostname, _, port = host.partition(":")
    c = http.client.HTTPConnection(hostname, int(port or 8080), timeout=5)
    c.request("GET", "/status")
    return c.getresponse().read().decode()


# -- analysis ------------------------------------------------------------------

def _load(session: Path):
    d = np.load(session / "depth.npy")
    rows = [json.loads(l) for l in open(session / "depth.jsonl")]
    return d, rows


def noise(session: Path, args) -> None:
    d, rows = _load(session)
    n, h, w = d.shape
    valid = np.isfinite(d) & (d > 0)
    frac = valid.mean()
    dm = np.where(valid, d, np.nan) * 1000.0  # mm
    med = np.nanmedian(dm)
    sd = np.nanstd(dm, axis=0)
    # Drift-free estimate: sigma from frame-to-frame differences (/sqrt2).
    diff_sd = np.nanstd(np.diff(dm, axis=0), axis=0) / np.sqrt(2)
    central = sd[h // 4:3 * h // 4, w // 4:3 * w // 4]
    print(f"session={session.name} frames={n} depth={w}x{h} valid={frac*100:.1f}% "
          f"accuracy={rows[0]['accuracy']} quality={rows[0]['quality']} filtered={rows[0]['filtered']}")
    print(f"scene median distance: {med:.1f} mm  (p5={np.nanpercentile(dm,5):.0f} p95={np.nanpercentile(dm,95):.0f})")
    for name, arr in (("per-pixel sigma (whole frame)", sd),
                      ("per-pixel sigma (central half)", central),
                      ("per-pixel sigma (frame-diff, drift-free)", diff_sd)):
        a = arr[np.isfinite(arr)]
        print(f"  {name:42s} median={np.median(a):7.2f} mm  p90={np.percentile(a,90):7.2f} mm  max={a.max():7.2f} mm")
    # A fingertip-sized patch, averaged, is what a detector would actually use.
    for k in (1, 3, 5):
        patch = _boxmean(dm, k)
        s = np.nanstd(patch, axis=0)
        s = s[np.isfinite(s)]
        print(f"  sigma of {k}x{k}-averaged patch{'':17s} median={np.median(s):7.2f} mm")


def _boxmean(dm: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return dm
    v = np.isfinite(dm)
    x = np.where(v, dm, 0.0)
    tot = np.zeros_like(x)
    cnt = np.zeros_like(x)
    p = k // 2
    for dy in range(-p, p + 1):
        for dx in range(-p, p + 1):
            tot += np.roll(np.roll(x, dy, axis=1), dx, axis=2)
            cnt += np.roll(np.roll(v, dy, axis=1), dx, axis=2)
    out = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    out[:, :p, :] = np.nan; out[:, -p:, :] = np.nan
    out[:, :, :p] = np.nan; out[:, :, -p:] = np.nan
    return out


def spatial(session: Path, args) -> None:
    d, rows = _load(session)
    h, w = d.shape[1:]
    dm = np.where(np.isfinite(d) & (d > 0), d, np.nan) * 1000.0
    dist = args.distance_mm or float(np.nanmedian(dm))
    intr = rows[0]["intrinsics"]
    if intr:
        fx, fy, cx, cy = (float(x) for x in intr.split(","))
        rw, rh = (int(x) for x in rows[0]["intrinsics_ref"].split(","))
        fx_d, fy_d = fx * w / rw, fy * h / rh
        mm_x, mm_y = dist / fx_d, dist / fy_d
        print(f"intrinsics(ref {rw}x{rh}) fx={fx:.1f} fy={fy:.1f} -> depth-map fx={fx_d:.2f} fy={fy_d:.2f}")
    else:
        fov = args.fov_deg
        fx_d = (w / 2) / np.tan(np.radians(fov / 2))
        mm_x = mm_y = dist / fx_d
        print(f"no intrinsics; using fov={fov} deg -> fx={fx_d:.2f} px")
    print(f"working distance {dist:.0f} mm, depth map {w}x{h}")
    print(f"  ground sampling: {mm_x:.2f} mm/px (x), {mm_y:.2f} mm/px (y)")
    for name, size in (("fingertip", 16.0), ("keycap", 16.0), ("finger width", 18.0)):
        print(f"  {name} ({size:.0f} mm) spans {size/mm_x:.2f} x {size/mm_y:.2f} depth px")


def signal(session: Path, args) -> None:
    import pyarrow.parquet as pq

    d, rows = _load(session)
    dm = np.where(np.isfinite(d) & (d > 0), d, np.nan) * 1000.0
    dt = np.array([r["t"] for r in rows])
    dh, dw = d.shape[1:]

    taps = [json.loads(l) for l in open(session / "taps.jsonl")]
    lm = pq.read_table(session / "landmarks.parquet").to_pandas()
    meta = json.loads((session / "meta.json").read_text())
    vw, vh = meta["width"], meta["height"]

    print(f"{len(taps)} taps, {len(dt)} depth frames, depth {dw}x{dh}, rgb {vw}x{vh}")
    if not taps:
        print("no taps detected - cannot measure signal")
        return

    noise_sd = np.nanmedian(np.nanstd(np.diff(dm, axis=0), axis=0) / np.sqrt(2))
    exc, snr = [], []
    half = args.window_s
    for tap in taps:
        # RGB pixel -> depth pixel: both cover the same FOV, so scale directly.
        x = int(round(tap["x"] / vw * dw))
        y = int(round(tap["y"] / vh * dh))
        if not (1 <= x < dw - 1 and 1 <= y < dh - 1):
            continue
        m = (dt >= tap["t"] - half) & (dt <= tap["t"] + half)
        if m.sum() < 5:
            continue
        trace = np.nanmean(dm[m, y - 1:y + 2, x - 1:x + 2], axis=(1, 2))
        if np.isnan(trace).all():
            continue
        base = np.nanmedian(trace)
        a = float(np.nanmax(np.abs(trace - base)))
        exc.append(a)
        snr.append(a / noise_sd if noise_sd else np.nan)
    if not exc:
        print("no tap fell on a valid depth pixel")
        return
    exc = np.array(exc)
    print(f"noise floor sigma (3x3 patch, frame-diff): {noise_sd:.2f} mm")
    print(f"depth excursion at {len(exc)} tap sites: median={np.median(exc):.2f} mm "
          f"p90={np.percentile(exc,90):.2f} mm max={exc.max():.2f} mm")
    print(f"SNR (excursion / sigma): median={np.median(snr):.2f} p90={np.percentile(snr,90):.2f}")
    # Control: the same statistic at random times tells us how much of that is noise.
    rng = np.random.default_rng(0)
    ctrl = []
    for _ in range(len(exc) * 5):
        i = rng.integers(0, len(dt))
        m = (dt >= dt[i] - half) & (dt <= dt[i] + half)
        y, x = rng.integers(1, dh - 1), rng.integers(1, dw - 1)
        tr = np.nanmean(dm[m, y - 1:y + 2, x - 1:x + 2], axis=(1, 2))
        if m.sum() >= 5 and not np.isnan(tr).all():
            ctrl.append(float(np.nanmax(np.abs(tr - np.nanmedian(tr)))))
    if ctrl:
        print(f"control (random time+place): median={np.median(ctrl):.2f} mm p90={np.percentile(ctrl,90):.2f} mm")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--host", default="192.168.1.23:8080")
    r.add_argument("--seconds", type=float, default=10.0)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--label", default="depth")
    for name, fn in (("noise", noise), ("spatial", spatial), ("signal", signal)):
        s = sub.add_parser(name)
        s.add_argument("session", type=Path)
        s.set_defaults(fn=fn)
        if name == "spatial":
            s.add_argument("--distance-mm", type=float, default=None)
            s.add_argument("--fov-deg", type=float, default=74.6)
        if name == "signal":
            s.add_argument("--window-s", type=float, default=0.15)
    a = ap.parse_args(argv)
    if a.cmd == "record":
        record(a.host, a.seconds, a.out, a.label)
    else:
        a.fn(a.session, a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
