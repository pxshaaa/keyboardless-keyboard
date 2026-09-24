"""PressureVision++ inference helpers. Preprocessing mirrors pv2/prediction/pred_util.py exactly.
Vendored PV2 repo (MIT) lives in .cache/pressurevision/pv2, never committed."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

PV_ROOT = Path(os.environ.get("PV_ROOT", Path.home() / "cvt/.cache/pressurevision"))
PV2 = PV_ROOT / "pv2"
SITE = PV_ROOT / "site"
CKPT = PV2 / "data/model/paper_29.pth"
NET = 448
SCALE = 1.5
TIPS = (4, 8, 12, 16, 20)
FORCE_THRESHOLDS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _paths():
    for p in (str(SITE), str(PV2)):
        if p not in sys.path:
            sys.path.insert(0, p)


def load_model(device: str = "mps"):
    _paths()
    import torch

    m = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    m.eval()
    m.to(device)
    return m


def class_scalars() -> np.ndarray:
    """classes_to_scalar, vectorised: expected-pressure value of each of the 9 classes."""
    th = FORCE_THRESHOLDS
    out = np.zeros(len(th), np.float32)
    for i in range(len(th)):
        if i == 0:
            out[i] = th[0]
        elif i == len(th) - 1:
            out[i] = th[-1] + (th[-1] - th[-2]) / 2
        else:
            out[i] = (th[i] + th[i + 1]) / 2
    return out


def hand_bbox(xy: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    """xy [21,2] px -> (min_x, min_y, max_x, max_y), PV2's get_hand_bbox."""
    cx = (xy[:, 0].min() + xy[:, 0].max()) / 2
    cy = (xy[:, 1].min() + xy[:, 1].max()) / 2
    r = max(xy[:, 0].max() - cx, xy[:, 1].max() - cy) * SCALE
    return (
        int(round(max(0.0, cx - r))),
        int(round(max(0.0, cy - r))),
        int(round(min(float(w), cx + r))),
        int(round(min(float(h), cy + r))),
    )


def preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    """448x448x3 BGR uint8 -> 3x448x448 float32."""
    img = crop_bgr[:, :, ::-1].astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return np.ascontiguousarray(img.transpose(2, 0, 1))


def load_frames(sess: Path) -> tuple[np.ndarray, np.ndarray]:
    fr = [json.loads(l) for l in open(sess / "frames.jsonl")]
    return np.array([r["i"] for r in fr], np.int64), np.array([r["t"] for r in fr], np.float64)


def load_landmarks(sess: Path, n_frames: int, fname: str = "landmarks.parquet") -> np.ndarray:
    """-> P [F, 2, 21, 2] px, NaN where absent (row index = position in frames.jsonl)."""
    import pyarrow.parquet as pq

    fi, _ = load_frames(sess)
    tb = pq.read_table(sess / fname, columns=["i", "hand", "joint", "x", "y"])
    i = tb.column("i").to_numpy()
    row = np.clip(np.searchsorted(fi, i), 0, len(fi) - 1)
    ok = fi[row] == i
    h = tb.column("hand").to_numpy().astype(int)
    j = tb.column("joint").to_numpy().astype(int)
    P = np.full((n_frames, 2, 21, 2), np.nan, np.float32)
    for c, ch in enumerate("xy"):
        P[row[ok], h[ok], j[ok], c] = tb.column(ch).to_numpy()[ok]
    return P


def keydowns(sess: Path) -> np.ndarray:
    out = []
    for line in open(sess / "keys.jsonl"):
        r = json.loads(line)
        if r.get("event") == "down":
            out.append(r["t"])
    return np.array(sorted(out), np.float64)
