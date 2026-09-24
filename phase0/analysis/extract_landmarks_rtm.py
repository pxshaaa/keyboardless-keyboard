"""RTMPose (rtmlib hand5, onnxruntime CPU) -> landmarks_rtm.parquet, CONTRACT schema; joint order = MediaPipe (identity, --check verifies).
Boxes tracked from each hand's previous keypoints, seeded from landmarks.parquet (or --detector rtmdet); conf per joint; z=0; w*=NaN."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from phase0.analysis.extract_landmarks import (
    N_JOINTS, NUM_HANDS, FrameAlignmentError, LandmarkBatchWriter, check_frame_alignment, read_frames_jsonl,
)

POSE_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip")
DET_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
           "rtmdet_nano_8xb32-300e_hand-267f9c8f.zip")
POSE_INPUT = (256, 256)
BOX_PAD = 1.25          # square box side = BOX_PAD * max extent of the previous keypoints
KPT_THR = 0.3           # a keypoint counts as seen above this score
TRACK_MIN_MEAN = 0.3    # track survives while mean keypoint score >= this (0.5 dropped 21% of typing frames)
RECROP_MEAN = 0.5       # below this mean score the next crop comes from the first-pass box, if one overlaps
TRACK_MIN_SEEN = 12     # ... and at least this many keypoints above KPT_THR
SEED_IOU = 0.3          # an MP/detector box with IoU < this against every live track spawns a new one
DUP_IOU = 0.6           # two tracks closer than this are the same hand: drop the younger
OUT_NAME = "landmarks_rtm.parquet"


def square_box(x: np.ndarray, y: np.ndarray, pad: float = BOX_PAD) -> np.ndarray:
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    s = max(x.max() - x.min(), y.max() - y.min(), 32.0) * pad / 2
    return np.array([cx - s, cy - s, cx + s, cy + s], np.float32)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_mp_boxes(parquet: Path) -> dict[int, list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]]:
    """landmarks.parquet -> {frame i: [(handedness, box, x21, y21), ...]}."""
    tb = pq.read_table(parquet, columns=["i", "hand", "handedness", "joint", "x", "y"])
    i = tb.column("i").to_numpy()
    h = tb.column("hand").to_numpy().astype(int)
    j = tb.column("joint").to_numpy().astype(int)
    order = np.lexsort((j, h, i))
    i, h, j = i[order], h[order], j[order]
    x, y = tb.column("x").to_numpy()[order], tb.column("y").to_numpy()[order]
    hd = np.asarray(tb.column("handedness").to_pylist(), dtype=object)[order]
    out: dict[int, list] = {}
    n = len(i)
    k = 0
    while k + N_JOINTS <= n:
        blk = slice(k, k + N_JOINTS)
        if not (i[blk] == i[k]).all() or not (j[blk] == np.arange(N_JOINTS)).all():
            k += 1   # malformed block; skip a row and resync
            continue
        out.setdefault(int(i[k]), []).append((str(hd[k]), square_box(x[blk], y[blk]), x[blk].copy(), y[blk].copy()))
        k += N_JOINTS
    return out


class RtmWriter(LandmarkBatchWriter):
    """add_hand with a per-joint conf array; z = 0, world columns NaN."""

    def add_hand_scored(self, i, t, hand, handedness, x_px, y_px, conf):
        n = len(x_px)
        self._i.extend([i] * n); self._t.extend([t] * n); self._hand.extend([hand] * n)
        self._handedness.extend([handedness] * n); self._joint.extend(range(n))
        self._x.extend(x_px.tolist()); self._y.extend(y_px.tolist()); self._conf.extend(conf.tolist())
        self._z.extend([0.0] * n); self._wx.extend([np.nan] * n); self._wy.extend([np.nan] * n); self._wz.extend([np.nan] * n)
        if len(self._i) >= self.batch_rows:
            self.flush()


class Track:
    __slots__ = ("box", "label", "born", "kp", "sc")

    def __init__(self, box: np.ndarray, label: str, born: int):
        self.box, self.label, self.born = box, label, born
        self.kp = self.sc = None


def make_pose(threads: int):
    import onnxruntime as ort
    from rtmlib import RTMPose
    pose = RTMPose(POSE_URL, model_input_size=POSE_INPUT, backend="onnxruntime", device="cpu")
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    pose.session = ort.InferenceSession(pose.onnx_model, sess_options=so, providers=["CPUExecutionProvider"])
    return pose


def make_det(threads: int):
    import onnxruntime as ort
    from rtmlib import RTMDet
    det = RTMDet(DET_URL, model_input_size=(320, 320), backend="onnxruntime", device="cpu")
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    det.session = ort.InferenceSession(det.onnx_model, sess_options=so, providers=["CPUExecutionProvider"])
    return det


def extract(session_dir: Path, *, limit: int | None = None, detector: str = "mediapipe", threads: int = 4,
            allow_frame_mismatch: bool = False, progress_every: int = 500, out_name: str = OUT_NAME) -> dict:
    import cv2

    session_dir = Path(session_dir)
    video_path, frames_path, out_path = session_dir / "video.mp4", session_dir / "frames.jsonl", session_dir / out_name
    frames = read_frames_jsonl(frames_path)
    mp_boxes = load_mp_boxes(session_dir / "landmarks.parquet") if detector == "mediapipe" else {}
    det = make_det(threads) if detector == "rtmdet" else None
    pose = make_pose(threads)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    try:
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        check_frame_alignment(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), frames, strict=not allow_frame_mismatch)
        n_target = len(frames) if limit is None else min(limit, len(frames))
        print(f"{session_dir.name}: {width}x{height}, {len(frames)} frames, processing {n_target} (boxes: {detector})",
              flush=True)
        tracks: list[Track] = []
        n_hands = n_seed = n_frames_hands = 0
        t_pose = 0.0
        started = time.monotonic()
        with RtmWriter(out_path) as writer:
            for row in frames[:n_target]:
                ok, frame = cap.read()
                if not ok:
                    msg = f"video ended at frame {row.i} but frames.jsonl expects {n_target}"
                    print(f"\n*** {msg} ***\n", file=sys.stderr)
                    if not allow_frame_mismatch:
                        raise FrameAlignmentError(msg)
                    break
                # --- seed / re-seed tracks from the first-pass boxes of this frame
                if det is not None:
                    cands = [("Unknown", np.asarray(b[:4], np.float32)) for b in det(frame)]
                else:
                    cands = [(lab, box) for lab, box, _, _ in mp_boxes.get(row.i, [])]
                for lab, box in cands:
                    if len(tracks) >= NUM_HANDS:
                        break
                    if all(iou(box, tr.box) < SEED_IOU for tr in tracks):
                        tracks.append(Track(box.copy(), lab, row.i))
                        n_seed += 1
                if not tracks:
                    continue
                # --- pose on every live track, re-crop from its own keypoints
                t0 = time.perf_counter()
                kps, scs = pose(frame, [tr.box for tr in tracks])
                t_pose += time.perf_counter() - t0
                alive = []
                for tr, kp, sc in zip(tracks, kps, scs):
                    seen = sc >= KPT_THR
                    if sc.mean() < TRACK_MIN_MEAN or seen.sum() < TRACK_MIN_SEEN:
                        continue
                    tr.kp, tr.sc = kp.astype(np.float32), sc.astype(np.float32)
                    own = square_box(kp[seen, 0], kp[seen, 1])
                    ext = [b for _, b in cands if iou(b, own) >= SEED_IOU]
                    tr.box = ext[0] if (sc.mean() < RECROP_MEAN and ext) else own
                    alive.append(tr)
                # duplicates (two tracks on one hand): keep the older
                alive.sort(key=lambda t: t.born)
                tracks = []
                for tr in alive:
                    if all(iou(tr.box, o.box) < DUP_IOU for o in tracks):
                        tracks.append(tr)
                if tracks:
                    n_frames_hands += 1
                for hand_idx, tr in enumerate(tracks[:NUM_HANDS]):
                    writer.add_hand_scored(row.i, row.t, hand_idx, tr.label, tr.kp[:, 0], tr.kp[:, 1], tr.sc)
                    n_hands += 1
                if progress_every and (row.i + 1) % progress_every == 0:
                    el = time.monotonic() - started
                    print(f"  {row.i + 1}/{n_target} | {n_hands} hands | {(row.i + 1) / el:.1f} fps "
                          f"(pose {1000 * t_pose / max(n_hands, 1):.1f} ms/crop) | seeds {n_seed}", flush=True)
        elapsed = time.monotonic() - started
        stats = {"session": session_dir.name, "frames": n_target, "hands": n_hands, "frames_with_hands": n_frames_hands,
                 "seeds": n_seed, "secs": elapsed, "fps": n_target / max(elapsed, 1e-9),
                 "pose_ms_per_crop": 1000 * t_pose / max(n_hands, 1), "detector": detector, "threads": threads,
                 "out": str(out_path)}
        print(f"done: {json.dumps(stats)}", flush=True)
        (session_dir / (Path(out_name).stem + "_stats.json")).write_text(json.dumps(stats, indent=1))
        return stats
    finally:
        cap.release()


def check_mapping(session_dir: Path, n_frames: int = 40, threads: int = 4) -> dict:
    """Per-joint median |rtm - mediapipe| px on MediaPipe's own boxes, plus the best-matching MP joint for each RTM joint
    (identity => same order)."""
    import cv2
    session_dir = Path(session_dir)
    mp = load_mp_boxes(session_dir / "landmarks.parquet")
    pose = make_pose(threads)
    cap = cv2.VideoCapture(str(session_dir / "video.mp4"))
    keys = sorted(mp)
    pick = keys[len(keys) // 10:: max(1, len(keys) // n_frames)][:n_frames]
    D = []   # [n, 21(rtm), 21(mp)] distance matrices
    for fi in pick:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, img = cap.read()
        if not ok:
            continue
        for lab, box, x, y in mp[fi]:
            kp, sc = pose(img, [box])
            D.append(np.hypot(kp[0][:, None, 0] - x[None], kp[0][:, None, 1] - y[None]))
    D = np.array(D)
    med = np.median(D, 0)
    diag = [float(med[j, j]) for j in range(N_JOINTS)]
    best = [int(np.argmin(med[j])) for j in range(N_JOINTS)]
    res = {"session": session_dir.name, "hands_checked": len(D), "median_px_same_joint": diag,
           "nearest_mp_joint_for_each_rtm_joint": best, "identity_mapping": best == list(range(N_JOINTS))}
    print(json.dumps(res, indent=1))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--detector", choices=["mediapipe", "rtmdet"], default="mediapipe",
                    help="box source for seeding tracks: existing landmarks.parquet (default) or rtmlib RTMDet hand")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-name", default=OUT_NAME)
    ap.add_argument("--allow-frame-mismatch", action="store_true")
    ap.add_argument("--check", action="store_true", help="only verify the joint mapping against MediaPipe")
    a = ap.parse_args(argv)
    if a.out_name == "landmarks.parquet":
        ap.error("refusing to overwrite landmarks.parquet")
    try:
        if a.check:
            check_mapping(a.session_dir, threads=a.threads)
        else:
            extract(a.session_dir, limit=a.limit, detector=a.detector, threads=a.threads,
                    allow_frame_mismatch=a.allow_frame_mismatch, out_name=a.out_name)
    except FrameAlignmentError as exc:
        print(f"ABORTED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
