"""Run PressureVision++ over a session, storing per-fingertip contact/pressure per frame.
Usage: python -m phase0.analysis.pv_extract <session_id> [stride]"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from phase0.analysis import pv_common as pv

SESS = Path.home() / "cvt/data/sessions"
OUT = pv.PV_ROOT / "feats"
BATCH = 12


def disc_stats(m: np.ndarray, cx: float, cy: float, r: int) -> tuple[float, float]:
    h, w = m.shape
    x0, x1 = max(0, int(cx - r)), min(w, int(cx + r) + 1)
    y0, y1 = max(0, int(cy - r)), min(h, int(cy + r) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0
    sub = m[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    msk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    if not msk.any():
        return float(sub.max()), float(sub.mean())
    v = sub[msk]
    return float(v.max()), float(v.mean())


def main(sid: str, stride: int = 1) -> int:
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    s = SESS / sid
    fi, ft = pv.load_frames(s)
    F = len(fi)
    P = pv.load_landmarks(s, F)
    rows = np.arange(0, F, stride)
    want = set(int(x) for x in rows)

    model = pv.load_model("mps")
    scal = torch.tensor(pv.class_scalars(), device="mps")

    cp_max = np.full((F, 2, 5), np.nan, np.float32)
    cp_mean = np.full((F, 2, 5), np.nan, np.float32)
    pr_max = np.full((F, 2, 5), np.nan, np.float32)
    pr_mean = np.full((F, 2, 5), np.nan, np.float32)
    glob = np.full((F, 2, 3), np.nan, np.float32)
    bott = np.full((F, 2, 7), np.nan, np.float32)
    tip_r = np.full((F, 2), np.nan, np.float32)
    acc = np.zeros((2, pv.NET, pv.NET, 3), np.float64)
    acc_n = np.zeros(2, np.int64)

    cap = cv2.VideoCapture(str(s / "video.mp4"))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf_x, buf_m = [], []
    t0 = time.time()
    done = 0

    def flush():
        nonlocal buf_x, buf_m, done
        if not buf_x:
            return
        x = torch.from_numpy(np.stack(buf_x)).to("mps")
        with torch.no_grad():
            logits, d = model(x)
            p = torch.softmax(logits, 1)
            cpb = (1 - p[:, 0]).cpu().numpy()
            prb = (p * scal[None, :, None, None]).sum(1).cpu().numpy()
            blb = d["bottleneck_logits"].cpu().numpy()
        for b, (k, hnd, x0, y0, x1, y1) in enumerate(buf_m):
            sx, sy = pv.NET / (x1 - x0), pv.NET / (y1 - y0)
            xy = P[k, hnd]
            span = np.linalg.norm((xy[9] - xy[0]) * [sx, sy])
            r = int(max(8, round(0.25 * span)))
            tip_r[k, hnd] = r
            for ti, j in enumerate(pv.TIPS):
                cx, cy = (xy[j, 0] - x0) * sx, (xy[j, 1] - y0) * sy
                cp_max[k, hnd, ti], cp_mean[k, hnd, ti] = disc_stats(cpb[b], cx, cy, r)
                pr_max[k, hnd, ti], pr_mean[k, hnd, ti] = disc_stats(prb[b], cx, cy, r)
            glob[k, hnd] = [cpb[b].max(), cpb[b].mean(), prb[b].sum() / 1e3]
            bott[k, hnd] = blb[b]
        done += len(buf_x)
        buf_x, buf_m = [], []

    k = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if k in want:
            for hnd in (0, 1):
                xy = P[k, hnd]
                if np.isnan(xy).any():
                    continue
                x0, y0, x1, y1 = pv.hand_bbox(xy, W, H)
                if x1 - x0 < 32 or y1 - y0 < 32:
                    continue
                c = cv2.resize(frame[y0:y1, x0:x1], (pv.NET, pv.NET))
                acc[hnd] += c
                acc_n[hnd] += 1
                buf_x.append(pv.preprocess(c))
                buf_m.append((k, hnd, x0, y0, x1, y1))
            if len(buf_x) >= BATCH:
                flush()
                if done % 2400 < BATCH:
                    el = time.time() - t0
                    print(f"{k}/{F} crops={done} {done/el:.1f}/s eta={(F/stride*1.8-done)/max(done/el,1e-9)/60:.0f}min",
                          flush=True)
        k += 1
    flush()
    cap.release()

    mean_img = np.stack([acc[h] / max(acc_n[h], 1) for h in (0, 1)]).astype(np.uint8)
    x = torch.from_numpy(np.stack([pv.preprocess(mean_img[h]) for h in (0, 1)])).to("mps")
    with torch.no_grad():
        logits, d = model(x)
        p = torch.softmax(logits, 1)
        ctrl_cp = (1 - p[:, 0]).cpu().numpy()
        ctrl_pr = (p * scal[None, :, None, None]).sum(1).cpu().numpy()
    for h in (0, 1):
        cv2.imwrite(str(OUT / f"{sid}_meanimg_hand{h}.png"), mean_img[h])

    ctl_max = np.full((F, 2, 5), np.nan, np.float32)
    ctl_pr = np.full((F, 2, 5), np.nan, np.float32)
    for k in rows:
        for hnd in (0, 1):
            xy = P[k, hnd]
            if np.isnan(xy).any() or np.isnan(tip_r[k, hnd]):
                continue
            x0, y0, x1, y1 = pv.hand_bbox(xy, W, H)
            sx, sy = pv.NET / (x1 - x0), pv.NET / (y1 - y0)
            r = int(tip_r[k, hnd])
            for ti, j in enumerate(pv.TIPS):
                cx, cy = (xy[j, 0] - x0) * sx, (xy[j, 1] - y0) * sy
                ctl_max[k, hnd, ti] = disc_stats(ctrl_cp[hnd], cx, cy, r)[0]
                ctl_pr[k, hnd, ti] = disc_stats(ctrl_pr[hnd], cx, cy, r)[0]

    np.savez_compressed(OUT / f"{sid}.npz", t=ft, i=fi, cp_max=cp_max, cp_mean=cp_mean,
                        pr_max=pr_max, pr_mean=pr_mean, glob=glob, bott=bott, tip_r=tip_r,
                        ctl_max=ctl_max, ctl_pr=ctl_pr, stride=stride, crops=done)
    print("wrote", OUT / f"{sid}.npz", "crops", done, f"{time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1))
