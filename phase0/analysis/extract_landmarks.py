"""Extract MediaPipe hand landmarks -> landmarks.parquet (phase0/CONTRACT.md schema).

CONF CAVEAT: MediaPipe Tasks NormalizedLandmark has NO per-point confidence, so `conf` holds the HAND-level handedness score replicated across all 21 joints of that hand -- it is not a per-joint quality measure. Coords are pixels (normalized*width/height), origin top-left. Usage: python -m phase0.analysis.extract_landmarks data/sessions/<id> [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# Rotation names -> cv2 constants, resolved lazily: cv2 is imported inside
# functions here, not at module scope, so a bare reference at import time fails.
ROTATION_NAMES = ("none", "cw90", "ccw90", "180")


def apply_rotation(frame, name):
    """Rotate in software so the phone's physical orientation doesn't matter.

    Applied before landmark detection. Raw footage on disk stays untouched, so
    a wrong choice here is re-runnable without re-recording.
    """
    import cv2

    codes = {
        "none": None,
        "cw90": cv2.ROTATE_90_CLOCKWISE,
        "ccw90": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "180": cv2.ROTATE_180,
    }
    code = codes.get(name)
    return frame if code is None else cv2.rotate(frame, code)


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path("models/hand_landmarker.task")

N_JOINTS = 21
NUM_HANDS = 2
BATCH_FRAMES = 500  # frames buffered before flushing a parquet row group

# Contract schema -- column order and dtypes are frozen.
SCHEMA = pa.schema(
    [
        pa.field("i", pa.int32()),
        pa.field("t", pa.float64()),
        pa.field("hand", pa.int8()),
        pa.field("handedness", pa.string()),
        pa.field("joint", pa.int8()),
        pa.field("x", pa.float32()),
        pa.field("y", pa.float32()),
        pa.field("conf", pa.float32()),
        pa.field("z", pa.float32()),   # MediaPipe image-space depth, same scale as x (px), wrist = 0
        pa.field("wx", pa.float32()),  # metric world landmarks (m), origin at hand geometric centre
        pa.field("wy", pa.float32()),
        pa.field("wz", pa.float32()),
    ]
)


class FrameAlignmentError(RuntimeError):
    """frames.jsonl and video.mp4 disagree; ``i`` cannot be trusted."""


@dataclass(frozen=True)
class FrameRow:
    i: int
    t: float


# --- inputs ---
def read_frames_jsonl(path: str | os.PathLike) -> list[FrameRow]:
    """Parse frames.jsonl -> [FrameRow]. Validates i is 0-based and contiguous."""
    rows: list[FrameRow] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append(FrameRow(int(obj["i"]), float(obj["t"])))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno}: bad frames.jsonl row: {exc}") from exc

    if not rows:
        raise ValueError(f"{path}: no frame rows")

    for pos, row in enumerate(rows):
        if row.i != pos:
            raise FrameAlignmentError(
                f"frames.jsonl is not 0-based contiguous: row #{pos} has i={row.i}. "
                "The contract requires i to match the video frame index exactly."
            )
    return rows


def check_frame_alignment(
    n_video_frames: int, frames: list[FrameRow], *, strict: bool = True
) -> None:
    """Compare OpenCV's frame count against frames.jsonl; a mismatch means `i` may
    point at a different picture than the recorder meant, so shout (and abort)."""
    n_json = len(frames)
    if n_video_frames <= 0:
        msg = (
            f"video reports {n_video_frames} frames (unknown/unseekable container); "
            f"frames.jsonl has {n_json}. Cannot cross-check alignment."
        )
    elif n_video_frames == n_json:
        return
    else:
        msg = (
            f"FRAME COUNT MISMATCH: video.mp4 has {n_video_frames} frames but "
            f"frames.jsonl has {n_json} rows (difference {n_video_frames - n_json}). "
            "Frame index `i` cannot be trusted -- landmarks would be misaligned "
            "against keystroke timestamps."
        )
    print(f"\n*** {msg} ***\n", file=sys.stderr)
    if strict:
        raise FrameAlignmentError(msg)


def ensure_model(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Download hand_landmarker.task if it isn't on disk already."""
    model_path = Path(model_path)
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading hand landmarker model -> {model_path}", flush=True)
    tmp = model_path.with_suffix(model_path.suffix + ".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)  # noqa: S310 (fixed https URL)
    tmp.replace(model_path)
    print(f"model ready ({model_path.stat().st_size} bytes)", flush=True)
    return model_path


# --- conversion ---
def normalized_to_pixels(
    landmarks, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Landmarks (.x/.y in [0,1]) -> (x_px, y_px) float32 arrays.
    Not clipped: MediaPipe extrapolates outside the frame and that is real signal."""
    if width <= 0 or height <= 0:
        raise ValueError(f"bad video resolution {width}x{height}")
    xs = np.fromiter((lm.x for lm in landmarks), dtype=np.float64, count=len(landmarks))
    ys = np.fromiter((lm.y for lm in landmarks), dtype=np.float64, count=len(landmarks))
    return (xs * width).astype(np.float32), (ys * height).astype(np.float32)


class LandmarkBatchWriter:
    """Buffer rows and flush parquet row groups incrementally; memory stays bounded
    at BATCH_FRAMES*42 rows regardless of video length (~1.5M rows for 10min@60fps)."""

    def __init__(self, out_path: Path, batch_frames: int = BATCH_FRAMES):
        self.out_path = Path(out_path)
        self.batch_rows = max(1, batch_frames) * NUM_HANDS * N_JOINTS
        self._writer: pq.ParquetWriter | None = None
        self._reset()
        self.rows_written = 0

    def _reset(self) -> None:
        self._i: list[int] = []
        self._t: list[float] = []
        self._hand: list[int] = []
        self._handedness: list[str] = []
        self._joint: list[int] = []
        self._x: list[float] = []
        self._y: list[float] = []
        self._conf: list[float] = []
        self._z: list[float] = []
        self._wx: list[float] = []
        self._wy: list[float] = []
        self._wz: list[float] = []

    def add_hand(
        self,
        i: int,
        t: float,
        hand: int,
        handedness: str,
        x_px: np.ndarray,
        y_px: np.ndarray,
        conf: float,
        z_px: np.ndarray | None = None,
        w: np.ndarray | None = None,
    ) -> None:
        n = len(x_px)
        self._i.extend([i] * n)
        self._t.extend([t] * n)
        self._hand.extend([hand] * n)
        self._handedness.extend([handedness] * n)
        self._joint.extend(range(n))
        self._x.extend(x_px.tolist())
        self._y.extend(y_px.tolist())
        # Hand-level handedness score replicated across all 21 joints; see the
        # module docstring -- MediaPipe exposes no per-landmark confidence.
        self._conf.extend([conf] * n)
        if z_px is None:
            z_px = np.full(n, np.nan, np.float32)
        if w is None:
            w = np.full((n, 3), np.nan, np.float32)
        self._z.extend(z_px.tolist())
        self._wx.extend(w[:, 0].tolist()); self._wy.extend(w[:, 1].tolist()); self._wz.extend(w[:, 2].tolist())
        if len(self._i) >= self.batch_rows:
            self.flush()

    def flush(self) -> None:
        if not self._i:
            return
        table = pa.Table.from_arrays(
            [
                pa.array(self._i, type=pa.int32()),
                pa.array(self._t, type=pa.float64()),
                pa.array(self._hand, type=pa.int8()),
                pa.array(self._handedness, type=pa.string()),
                pa.array(self._joint, type=pa.int8()),
                pa.array(self._x, type=pa.float32()),
                pa.array(self._y, type=pa.float32()),
                pa.array(self._conf, type=pa.float32()),
                pa.array(self._z, type=pa.float32()),
                pa.array(self._wx, type=pa.float32()),
                pa.array(self._wy, type=pa.float32()),
                pa.array(self._wz, type=pa.float32()),
            ],
            schema=SCHEMA,
        )
        if self._writer is None:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self.out_path, SCHEMA, compression="zstd")
        self._writer.write_table(table)
        self.rows_written += table.num_rows
        self._reset()

    def close(self) -> None:
        self.flush()
        if self._writer is None:
            # No hands detected anywhere -- still emit a valid empty file so
            # downstream stages get a schema instead of a FileNotFoundError.
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(SCHEMA.empty_table(), self.out_path, compression="zstd")
        else:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> "LandmarkBatchWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --- main extraction ---
def extract(
    session_dir: Path,
    *,
    limit: int | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    allow_frame_mismatch: bool = False,
    progress_every: int = 200,
) -> Path:
    import cv2  # imported lazily so the pure-python helpers stay importable
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )

    session_dir = Path(session_dir)
    video_path = session_dir / "video.mp4"
    frames_path = session_dir / "frames.jsonl"
    out_path = session_dir / "landmarks.parquet"
    for p in (video_path, frames_path):
        if not p.exists():
            raise FileNotFoundError(p)

    frames = read_frames_jsonl(frames_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        check_frame_alignment(n_video, frames, strict=not allow_frame_mismatch)

        n_target = len(frames) if limit is None else min(limit, len(frames))
        print(
            f"{session_dir.name}: {width}x{height}, "
            f"{len(frames)} frames in frames.jsonl, video reports {n_video}; "
            f"processing {n_target}",
            flush=True,
        )

        model = ensure_model(model_path)
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=RunningMode.VIDEO,
            num_hands=NUM_HANDS,
        )

        t0 = frames[0].t
        last_ms = -1
        n_processed = 0
        n_hands = 0
        n_frames_with_hands = 0
        started = time.monotonic()

        with HandLandmarker.create_from_options(options) as landmarker, (
            LandmarkBatchWriter(out_path)
        ) as writer:
            for row in frames[:n_target]:
                ok, frame_bgr = cap.read()
                if not ok:
                    msg = (
                        f"video ended at frame {row.i} but frames.jsonl expects "
                        f"{n_target} frames -- indices past here would be misaligned."
                    )
                    print(f"\n*** {msg} ***\n", file=sys.stderr)
                    if not allow_frame_mismatch:
                        raise FrameAlignmentError(msg)
                    break

                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # VIDEO mode demands strictly increasing integer ms timestamps.
                ts_ms = int(round((row.t - t0) * 1000.0))
                if ts_ms <= last_ms:
                    ts_ms = last_ms + 1
                last_ms = ts_ms

                result = landmarker.detect_for_video(mp_image, ts_ms)
                hands = result.hand_landmarks or []
                if hands:
                    n_frames_with_hands += 1
                for hand_idx, hand_lms in enumerate(hands[:NUM_HANDS]):
                    cats = result.handedness[hand_idx] if result.handedness else []
                    label = cats[0].category_name if cats else "Unknown"
                    score = float(cats[0].score) if cats else float("nan")
                    x_px, y_px = normalized_to_pixels(hand_lms, width, height)
                    z_px = np.array([lm.z for lm in hand_lms], dtype=np.float32) * width
                    wl = (result.hand_world_landmarks or [])
                    w = (np.array([[q.x, q.y, q.z] for q in wl[hand_idx]], dtype=np.float32)
                         if hand_idx < len(wl) else np.full((len(hand_lms), 3), np.nan, np.float32))
                    writer.add_hand(row.i, row.t, hand_idx, label, x_px, y_px, score, z_px, w)
                    n_hands += 1

                n_processed += 1
                if progress_every and n_processed % progress_every == 0:
                    el = time.monotonic() - started
                    print(
                        f"  {n_processed}/{n_target} frames | {n_hands} hands "
                        f"({n_frames_with_hands} frames w/ hands) | "
                        f"{n_processed / max(el, 1e-9):.1f} fps",
                        flush=True,
                    )

        elapsed = time.monotonic() - started
        print(
            f"done: {n_processed} frames, {n_hands} hands, "
            f"{writer.rows_written} rows -> {out_path} "
            f"in {elapsed:.1f}s ({n_processed / max(elapsed, 1e-9):.1f} fps)",
            flush=True,
        )
        return out_path
    finally:
        cap.release()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Extract hand landmarks from a Phase 0 session into landmarks.parquet"
    )
    ap.add_argument("session_dir", type=Path, help="data/sessions/<session_id>")
    ap.add_argument("--limit", type=int, default=None, help="process only first N frames")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument(
        "--allow-frame-mismatch",
        action="store_true",
        help="continue despite a frames.jsonl/video length mismatch (DANGEROUS)",
    )
    args = ap.parse_args(argv)
    try:
        extract(
            args.session_dir,
            limit=args.limit,
            model_path=args.model,
            allow_frame_mismatch=args.allow_frame_mismatch,
        )
    except FrameAlignmentError as exc:
        print(f"ABORTED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
