"""Hi-res landmarks: track each hand, landmark a 2x-upscaled padded crop, map back to full-frame px (CONTRACT.md schema).
Usage: python -m phase0.analysis.extract_landmarks_hires data/sessions/<id> [--limit N] [--out-name landmarks_hires.parquet]"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from phase0.analysis.extract_landmarks import (
    DEFAULT_MODEL_PATH,
    N_JOINTS,
    NUM_HANDS,
    FrameAlignmentError,
    LandmarkBatchWriter,
    check_frame_alignment,
    ensure_model,
    read_frames_jsonl,
)

PALM = (0, 5, 9, 13, 17)
MIN_SIDE = 96          # px, smallest crop side in full-frame pixels
DUP_FRAC = 0.6         # two slots closer than this * knuckle width are the same hand (seqctc.normalise uses the same rule)


@dataclass
class HandObs:
    """One hand in full-frame pixel coordinates."""
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    w: np.ndarray
    label: str
    score: float
    src: str            # "crop" | "full"

    def bbox(self) -> tuple[float, float, float, float]:
        return float(self.x.min()), float(self.y.min()), float(self.x.max()), float(self.y.max())

    def palm(self) -> np.ndarray:
        return np.array([self.x[list(PALM)].mean(), self.y[list(PALM)].mean()])

    def knuckle(self) -> float:
        return float(np.hypot(self.x[5] - self.x[17], self.y[5] - self.y[17]))


@dataclass
class Stats:
    frames: int = 0
    crop_passes: int = 0
    full_passes: int = 0
    crop_fallback_full: int = 0     # hand emitted from the full-frame pass because the crop pass missed it
    lost_events: int = 0
    duplicates: int = 0
    hands: int = 0
    frames_with_hands: int = 0
    crop_sides: list = field(default_factory=list)   # sampled crop side lengths (full-frame px)


def square_crop(box, W: int, H: int, pad: float) -> tuple[int, int, int, int]:
    """Padded square around a landmark bbox, clamped to the frame -> (x0, y0, x1, y1) ints."""
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * (1.0 + pad)
    side = max(side, MIN_SIDE)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = side / 2
    ax, ay = int(np.floor(cx - half)), int(np.floor(cy - half))
    bx, by = int(np.ceil(cx + half)), int(np.ceil(cy + half))
    # shift (not shrink) into the frame where possible
    if ax < 0:
        bx, ax = bx - ax, 0
    if ay < 0:
        by, ay = by - ay, 0
    if bx > W:
        ax, bx = max(0, ax - (bx - W)), W
    if by > H:
        ay, by = max(0, ay - (by - H)), H
    return ax, ay, bx, by


def hands_from_result(result, x0: float, y0: float, sx: float, sy: float, zscale: float, src: str) -> list[HandObs]:
    """MediaPipe result on an image whose pixel (u, v) maps to full-frame (x0 + u/sx, y0 + v/sy)."""
    out = []
    wl = result.hand_world_landmarks or []
    for k, lms in enumerate(result.hand_landmarks or []):
        cats = result.handedness[k] if result.handedness else []
        label = cats[0].category_name if cats else "Unknown"
        score = float(cats[0].score) if cats else float("nan")
        xn = np.fromiter((p.x for p in lms), np.float64, len(lms))
        yn = np.fromiter((p.y for p in lms), np.float64, len(lms))
        zn = np.fromiter((p.z for p in lms), np.float64, len(lms))
        w = (np.array([[q.x, q.y, q.z] for q in wl[k]], np.float32) if k < len(wl)
             else np.full((len(lms), 3), np.nan, np.float32))
        out.append(HandObs((x0 + xn * sx).astype(np.float32), (y0 + yn * sy).astype(np.float32),
                           (zn * zscale).astype(np.float32), w, label, score, src))
    return out


class Tracker:
    """Per-slot VIDEO-mode landmarkers on tracked crops + one full-frame landmarker for (re)detection."""

    def __init__(self, model: Path, W: int, H: int, scale: float, pad: float, redetect_every: int):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

        self.mp, self.W, self.H, self.scale, self.pad, self.every = mp, W, H, scale, pad, redetect_every

        def mk(n):
            return HandLandmarker.create_from_options(HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model)), running_mode=RunningMode.VIDEO, num_hands=n))

        self.full = mk(NUM_HANDS)
        self.crop = [mk(1) for _ in range(NUM_HANDS)]
        self.ts = [-1] * (NUM_HANDS + 1)          # per-landmarker last timestamp (VIDEO mode: strictly increasing)
        self.box: list = [None] * NUM_HANDS
        self.stats = Stats()

    def close(self):
        self.full.close()
        for c in self.crop:
            c.close()

    def _ts(self, k: int, ms: int) -> int:
        self.ts[k] = max(ms, self.ts[k] + 1)
        return self.ts[k]

    def _image(self, arr):
        return self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(arr))

    def crop_pass(self, rgb, k: int, ms: int) -> HandObs | None:
        import cv2
        x0, y0, x1, y1 = square_crop(self.box[k], self.W, self.H, self.pad)
        cw, ch = x1 - x0, y1 - y0
        if cw < 8 or ch < 8:
            return None
        up = cv2.resize(rgb[y0:y1, x0:x1], None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        res = self.crop[k].detect_for_video(self._image(up), self._ts(k, ms))
        self.stats.crop_passes += 1
        if self.stats.crop_passes % 97 == 0:
            self.stats.crop_sides.append(cw)
        # landmark (u, v) in the upscaled crop -> x0 + u / scale; z is in upscaled-crop-width units -> z * up.w / scale
        hands = hands_from_result(res, x0, y0, up.shape[1] / self.scale, up.shape[0] / self.scale,
                                  up.shape[1] / self.scale, "crop")
        return hands[0] if hands else None

    def full_pass(self, rgb, ms: int) -> list[HandObs]:
        res = self.full.detect_for_video(self._image(rgb), self._ts(NUM_HANDS, ms))
        self.stats.full_passes += 1
        return hands_from_result(res, 0.0, 0.0, self.W, self.H, self.W, "full")

    @staticmethod
    def same_hand(a: HandObs, b: HandObs) -> bool:
        kw = max(1.0, (a.knuckle() + b.knuckle()) / 2)
        return float(np.linalg.norm(a.palm() - b.palm())) < DUP_FRAC * kw

    def step(self, rgb, frame_idx: int, ms: int) -> list[HandObs | None]:
        st = self.stats
        st.frames += 1
        out: list = [None] * NUM_HANDS
        need_full = False
        for k in range(NUM_HANDS):
            if self.box[k] is None:
                continue
            h = self.crop_pass(rgb, k, ms)
            if h is None:
                self.box[k] = None
                st.lost_events += 1
                need_full = True
            else:
                out[k] = h
                self.box[k] = h.bbox()
        if all(o is not None for o in out) and self.same_hand(out[0], out[1]):
            st.duplicates += 1
            drop = 0 if out[0].score < out[1].score else 1
            out[drop], self.box[drop] = None, None
            need_full = True
        if any(o is None for o in out) and (need_full or frame_idx % self.every == 0):
            for h in self.full_pass(rgb, ms):
                if any(o is not None and self.same_hand(o, h) for o in out):
                    continue
                free = [k for k in range(NUM_HANDS) if out[k] is None]
                if not free:
                    break
                k = free[0]
                self.box[k] = h.bbox()
                hc = self.crop_pass(rgb, k, ms)
                if hc is None:              # keep the full-frame estimate for this frame; re-crop next frame from its bbox
                    st.crop_fallback_full += 1
                    hc = h
                out[k] = hc
                self.box[k] = hc.bbox()
        n = sum(o is not None for o in out)
        st.hands += n
        st.frames_with_hands += n > 0
        return out


def extract(session_dir: Path, *, out_name: str = "landmarks_hires.parquet", limit: int | None = None,
            model_path: Path = DEFAULT_MODEL_PATH, scale: float = 2.0, pad: float = 0.4, redetect_every: int = 10,
            allow_frame_mismatch: bool = False, force: bool = False, progress_every: int = 500) -> Path:
    import cv2

    session_dir = Path(session_dir)
    video_path, frames_path = session_dir / "video.mp4", session_dir / "frames.jsonl"
    out_path = session_dir / out_name
    if out_name == "landmarks.parquet" and not force:
        raise SystemExit("refusing to overwrite landmarks.parquet (pass --force if you really mean it)")
    if out_path.exists() and not force:
        raise SystemExit(f"{out_path} exists; pass --force to overwrite")
    for p in (video_path, frames_path):
        if not p.exists():
            raise FileNotFoundError(p)

    frames = read_frames_jsonl(frames_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    try:
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        check_frame_alignment(n_video, frames, strict=not allow_frame_mismatch)
        n_target = len(frames) if limit is None else min(limit, len(frames))
        print(f"{session_dir.name}: {W}x{H}, {len(frames)} frames, processing {n_target} -> {out_path.name} "
              f"(scale {scale}, pad {pad}, redetect every {redetect_every})", flush=True)

        tr = Tracker(ensure_model(model_path), W, H, scale, pad, redetect_every)
        t0, started = frames[0].t, time.monotonic()
        tmp = out_path.with_suffix(".parquet.part")
        try:
            with LandmarkBatchWriter(tmp) as writer:
                for row in frames[:n_target]:
                    ok, bgr = cap.read()
                    if not ok:
                        msg = f"video ended at frame {row.i} but frames.jsonl expects {n_target} frames"
                        print(f"\n*** {msg} ***\n", file=sys.stderr)
                        if not allow_frame_mismatch:
                            raise FrameAlignmentError(msg)
                        break
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    ms = int(round((row.t - t0) * 1000.0))
                    for k, h in enumerate(tr.step(rgb, row.i, ms)):
                        if h is not None:
                            writer.add_hand(row.i, row.t, k, h.label, h.x, h.y, h.score, h.z, h.w)
                    if progress_every and tr.stats.frames % progress_every == 0:
                        el = time.monotonic() - started
                        s = tr.stats
                        print(f"  {s.frames}/{n_target} | hands {s.hands} | crop {s.crop_passes} full {s.full_passes} "
                              f"lost {s.lost_events} dup {s.duplicates} fb {s.crop_fallback_full} | "
                              f"{s.frames / max(el, 1e-9):.1f} fps", flush=True)
        finally:
            tr.close()
        tmp.replace(out_path)
        el = time.monotonic() - started
        s = tr.stats
        meta = {"session": session_dir.name, "out": str(out_path), "frames": s.frames, "hands": s.hands,
                "frames_with_hands": s.frames_with_hands, "crop_passes": s.crop_passes, "full_passes": s.full_passes,
                "lost_events": s.lost_events, "duplicates": s.duplicates, "crop_fallback_full": s.crop_fallback_full,
                "crop_side_px_median": float(np.median(s.crop_sides)) if s.crop_sides else None,
                "scale": scale, "pad": pad, "redetect_every": redetect_every, "min_side": MIN_SIDE,
                "rows": writer.rows_written, "secs": el, "fps": s.frames / max(el, 1e-9)}
        out_path.with_suffix(".stats.json").write_text(json.dumps(meta, indent=1))
        print(f"done: {s.frames} frames, {s.hands} hands, {writer.rows_written} rows -> {out_path} in {el:.0f}s "
              f"({meta['fps']:.1f} fps); crop side median {meta['crop_side_px_median']} px, "
              f"full passes {s.full_passes}, fallbacks {s.crop_fallback_full}", flush=True)
        return out_path
    finally:
        cap.release()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--out-name", default="landmarks_hires.parquet")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--pad", type=float, default=0.4, help="crop side = max(bbox side) * (1 + pad)")
    ap.add_argument("--redetect-every", type=int, default=10, help="full-frame pass cadence while a slot is empty")
    ap.add_argument("--allow-frame-mismatch", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
    a = ap.parse_args(argv)
    try:
        extract(a.session_dir, out_name=a.out_name, limit=a.limit, model_path=a.model, scale=a.scale, pad=a.pad,
                redetect_every=a.redetect_every, allow_frame_mismatch=a.allow_frame_mismatch, force=a.force)
    except FrameAlignmentError as exc:
        print(f"ABORTED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
