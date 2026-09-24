"""Smoke test: crop geometry, MPS throughput, and whether PV++ fires at all on our top-down footage.
Usage: python -m phase0.analysis.pv_smoke <session_id>"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from phase0.analysis import pv_common as pv

SESS = Path.home() / "cvt/data/sessions"
OUT = pv.PV_ROOT / "smoke"


def main(sid: str, n: int = 120) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    s = SESS / sid
    fi, ft = pv.load_frames(s)
    P = pv.load_landmarks(s, len(fi))
    kt = pv.keydowns(s)

    dt = np.abs(ft[:, None] - kt[None, :]).min(1)
    both = ~np.isnan(P[:, :, 0, 0]).any(1)
    cand = np.where(both & (dt < 0.02))[0]
    far = np.where(both & (dt > 0.30))[0]
    sel = np.sort(np.concatenate([cand[:: max(1, len(cand) // (n // 2))][: n // 2],
                                  far[:: max(1, len(far) // (n // 2))][: n // 2]]))
    is_press = np.isin(sel, cand)
    print(f"frames={len(fi)} keydowns={len(kt)} both-hands={both.sum()} selected={len(sel)}")

    span = np.linalg.norm(P[:, :, 9, :] - P[:, :, 0, :], axis=2)
    bw = np.nanmax(P[:, :, :, 0], 2) - np.nanmin(P[:, :, :, 0], 2)
    bh = np.nanmax(P[:, :, :, 1], 2) - np.nanmin(P[:, :, :, 1], 2)
    side = 2 * np.fmax(bw, bh) / 2 * pv.SCALE
    print(f"median wrist-mcp span px={np.nanmedian(span):.1f}  median crop side px={np.nanmedian(side):.1f}"
          f"  upscale to 448 = {448 / np.nanmedian(side):.2f}x")

    cap = cv2.VideoCapture(str(s / "video.mp4"))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    want = set(int(x) for x in sel)
    crops, meta = [], []
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
                c = cv2.resize(frame[y0:y1, x0:x1], (pv.NET, pv.NET))
                crops.append(c)
                meta.append((k, hnd, x0, y0, x1, y1))
        k += 1
        if k > sel.max():
            break
    cap.release()
    print(f"crops={len(crops)}")

    import torch

    model = pv.load_model("mps")
    scal = torch.tensor(pv.class_scalars(), device="mps")
    B = 8
    cp, pr, bl = [], [], []
    t0 = time.time()
    for a in range(0, len(crops), B):
        x = np.stack([pv.preprocess(c) for c in crops[a:a + B]])
        with torch.no_grad():
            logits, d = model(torch.from_numpy(x).to("mps"))
            p = torch.softmax(logits, 1)
            cp.append((1 - p[:, 0]).cpu().numpy())
            pr.append((p * scal[None, :, None, None]).sum(1).cpu().numpy())
            bl.append(d["bottleneck_logits"].cpu().numpy())
    el = time.time() - t0
    print(f"inference {len(crops)} crops in {el:.1f}s = {len(crops)/el:.1f} crops/s")
    cp = np.concatenate(cp)
    pr = np.concatenate(pr)
    bl = np.concatenate(bl)
    print(f"contact prob: mean={cp.mean():.4f} max={cp.max():.4f} frac>0.5={(cp>0.5).mean():.4f}")
    print(f"pressure: mean={pr.mean():.4f} max={pr.max():.3f}")
    print(f"bottleneck logits mean per channel (thumb,index,mid,ring,pinky,palm,level): {bl.mean(0).round(3)}")

    mk = np.array([m[0] for m in meta])
    pf = np.isin(mk, cand)
    print(f"crop-level max contact prob: press={cp.reshape(len(cp),-1).max(1)[pf].mean():.4f} "
          f"far={cp.reshape(len(cp),-1).max(1)[~pf].mean():.4f}")

    for idx in (0, len(crops) // 2, len(crops) - 1):
        hm = cv2.applyColorMap((np.clip(cp[idx], 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        vis = cv2.addWeighted(crops[idx], 0.6, hm, 0.5, 0)
        k_, h_, x0, y0, x1, y1 = meta[idx]
        xy = P[k_, h_]
        sx, sy = pv.NET / (x1 - x0), pv.NET / (y1 - y0)
        for j in pv.TIPS:
            cv2.circle(vis, (int((xy[j, 0] - x0) * sx), int((xy[j, 1] - y0) * sy)), 6, (255, 255, 255), 2)
        cv2.imwrite(str(OUT / f"crop_{idx}_frame{k_}_hand{h_}.png"), np.hstack([crops[idx], vis]))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
