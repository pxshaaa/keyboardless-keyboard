"""Raking-light validation on real footage: is a fingertip's cast shadow detectable in our keyboard recordings,
and does the tip-to-shadow gap predict key-downs? Scored exactly like pv_probe (PressureVision step 1) so the
numbers are comparable: AUC vs 'key-down within +-40 ms', 10 s block bootstrap, time-shuffle and
static-average-image controls, kinematic LightGBM as the reference.
Usage: python -m phase0.analysis.shadow_probe extract <sid> | probe <sid>"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from phase0.analysis import pv_common as pv
from phase0.analysis import pv_probe as pp
from phase0.analysis import taps_gb as gb

SESS = Path("data/sessions")
CACHE = Path(".cache/shadow")
OUT = Path("results/shadow")
TIPS = (4, 8, 12, 16, 20)
FN = pp.FN
NDIR = 8
ANGLES = np.arange(NDIR) * (2 * np.pi / NDIR)          # 0 = +x (image right), pi/2 = +y (image down)
BINS = ((3, 8), (8, 14), (14, 24), (24, 36), (36, 50), (50, 80))   # px from the tip; a 25-30 deg lamp puts contact at ~7-11 px
STEP = 2.0
RMIN, RMAX = 3, 120                                      # gap search range along the darkest ray (must reach below the contact gap)
BRIGHT_SURFACE = 110                                     # mean-image gray above this = the silver deck, not keys
FEAT_NAMES = ([f"ray_d{d}_b{b}" for d in range(NDIR) for b in range(len(BINS))]
              + [f"ring_b{b}" for b in range(len(BINS))] + ["pad_mean", "contrast_b0", "contrast_b1",
                                                             "darkest_dir", "gap_px", "shadow_depth", "surface_mean"])


def _ray_offsets():
    offs = {}
    for k, th in enumerate(ANGLES):
        rs = np.arange(RMIN, RMAX + STEP, STEP)
        offs[k] = np.stack([rs * np.cos(th), rs * np.sin(th)], 1)      # [R,2]
    return offs


def tip_features(gray: np.ndarray, xy: np.ndarray, offs: dict) -> np.ndarray:
    """gray [H,W] float; xy [N,2] tip positions -> [N, len(FEAT_NAMES)] (NaN where the tip is off-image)."""
    H, W = gray.shape
    N = len(xy)
    out = np.full((N, len(FEAT_NAMES)), np.nan, np.float32)
    ok = np.isfinite(xy).all(1) & (xy[:, 0] > RMAX) & (xy[:, 0] < W - RMAX) & (xy[:, 1] > RMAX) & (xy[:, 1] < H - RMAX)
    if not ok.any():
        return out
    P = xy[ok]
    rs = np.arange(RMIN, RMAX + STEP, STEP)
    prof = np.zeros((ok.sum(), NDIR, len(rs)), np.float32)
    for k in range(NDIR):
        pts = P[:, None, :] + offs[k][None]                      # [n,R,2]
        xi = np.clip(pts[..., 0].round().astype(int), 0, W - 1)
        yi = np.clip(pts[..., 1].round().astype(int), 0, H - 1)
        prof[:, k] = gray[yi, xi]
    # pad intensity: small disc around the tip
    yy, xx = np.mgrid[-8:9, -8:9]
    disc = (xx ** 2 + yy ** 2) <= 64
    dx, dy = xx[disc], yy[disc]
    xi = np.clip(P[:, 0].round().astype(int)[:, None] + dx[None], 0, W - 1)
    yi = np.clip(P[:, 1].round().astype(int)[:, None] + dy[None], 0, H - 1)
    pad = gray[yi, xi].mean(1)
    f = []
    binm = []
    for lo, hi in BINS:
        m = (rs >= lo) & (rs < hi)
        binm.append(prof[:, :, m].mean(2))                       # [n,NDIR]
    binm = np.stack(binm, 2)                                     # [n,NDIR,nb]
    f.append(binm.reshape(len(P), -1))                           # ray_d_b
    ring = binm.mean(1)                                          # [n,nb]
    f.append(ring)
    f.append(pad[:, None])
    c0 = binm[:, :, 0].min(1) - ring[:, 0]
    c1 = binm[:, :, 1].min(1) - ring[:, 1]
    f.append(c0[:, None]); f.append(c1[:, None])
    dark = binm[:, :, :4].mean(2).argmin(1)                      # darkest direction over the bins out to 36 px
    f.append(dark[:, None].astype(np.float32))
    # gap: along the darkest ray, radius of the minimum intensity (the shadow core), in px
    pd = prof[np.arange(len(P)), dark]                           # [n,R]
    gap = rs[pd.argmin(1)]
    f.append(gap[:, None].astype(np.float32))
    depth = (ring.mean(1) - pd.min(1)) / np.maximum(ring.mean(1), 1.0)
    f.append(depth[:, None])
    f.append(ring[:, 2][:, None])                                # far ring = surface brightness proxy
    out[ok] = np.hstack(f).astype(np.float32)
    return out


def cmd_extract(a) -> int:
    import cv2
    CACHE.mkdir(parents=True, exist_ok=True)
    s = SESS / a.sid
    kfr, kt, P = gb.load_frames(s)
    n = len(kfr)
    if a.limit:
        kfr, kt, P = kfr[:a.limit], kt[:a.limit], P[:a.limit]
        n = len(kfr)
    offs = _ray_offsets()
    cap = cv2.VideoCapture(str(s / "video.mp4"))
    H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    feats = np.full((n, 2, 5, len(FEAT_NAMES)), np.nan, np.float32)
    acc = np.zeros((H, W), np.float64)
    nacc = 0
    want = {int(f): i for i, f in enumerate(kfr)}
    fi = 0
    t0 = time.time()
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if fi in want:
            i = want[fi]
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if i % 4 == 0:
                acc += g
                nacc += 1
            xy = P[i][:, list(TIPS), :2].reshape(10, 2)
            feats[i] = tip_features(g, xy, offs).reshape(2, 5, -1)
            if i % 4000 == 0:
                print(f"  {i}/{n} {time.time()-t0:.0f}s", flush=True)
        fi += 1
    cap.release()
    mean_img = (acc / max(nacc, 1)).astype(np.float32)
    # static-average-image control: same features from the mean image at the same tip positions
    ctl = np.full_like(feats, np.nan)
    for i in range(n):
        xy = P[i][:, list(TIPS), :2].reshape(10, 2)
        if np.isfinite(xy).any():
            ctl[i] = tip_features(mean_img, xy, offs).reshape(2, 5, -1)
    np.savez_compressed(CACHE / f"{a.sid}.npz", feats=feats, ctl=ctl, kfr=kfr, t=kt, mean_img=mean_img,
                        names=np.array(FEAT_NAMES))
    cv2.imwrite(str(CACHE / f"{a.sid}_meanimg.png"), mean_img.astype(np.uint8))
    print(f"wrote {CACHE / (a.sid + '.npz')} frames={n} scored={np.isfinite(feats[..., 0]).any((1, 2)).sum()} "
          f"{time.time()-t0:.0f}s", flush=True)
    return 0


def shadow_matrix(F: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """[n,2,5,NF] -> LightGBM matrix: all per-tip features + lags of the two scalars (gap, depth)."""
    n = len(F)
    X = [F.reshape(n, -1)]
    cols = [f"{FN[k]}_{nm}" for k in range(10) for nm in names]
    ig, idp, ic0 = names.index("gap_px"), names.index("shadow_depth"), names.index("contrast_b0")
    for key, idx in (("gap", ig), ("depth", idp), ("c0", ic0)):
        v = F[..., idx].reshape(n, 10)
        for lag in pp.LAGS:
            if lag == 0:
                continue
            X.append(pp._shift(v, lag)); cols += [f"{FN[k]}_{key}_l{lag}" for k in range(10)]
        X.append(v - pp._shift(v, 3)); cols += [f"{FN[k]}_{key}_d3" for k in range(10)]
    return np.hstack(X).astype(np.float32), cols


def cmd_probe(a) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(CACHE / f"{a.sid}.npz", allow_pickle=True)
    names = list(z["names"])
    F, C, kfr, t = z["feats"], z["ctl"], z["kfr"], z["t"]
    s = SESS / a.sid
    kfr2, _, P = gb.load_frames(s)
    assert (kfr2[: len(kfr)] == kfr).all()
    P = P[: len(kfr)]
    kt = pv.keydowns(s)
    y = np.abs(t[:, None] - kt[None, :]).min(1) <= a.half
    have = np.isfinite(F[..., 0]).any((1, 2))
    res = {"session": a.sid, "half_s": a.half, "frames_with_landmarks": int(len(kfr)), "frames_scored": int(have.sum()),
           "keydowns": int(len(kt)), "positives": int(y[have].sum()), "feature_names": names}
    ig, idp, ic0, idk, isf = (names.index(k) for k in ("gap_px", "shadow_depth", "contrast_b0", "darkest_dir", "surface_mean"))
    Xs, cols = shadow_matrix(F, names)
    Xc, _ = shadow_matrix(C, names)
    Xkg = gb.build_groups(P)
    Xkg = np.hstack([Xkg[g] for g in gb.GROUPS]).astype(np.float32)
    from phase0.analysis import touch_feats as tf
    per0, pool0 = tf.hand_feats(P[:, 0]); per1, pool1 = tf.hand_feats(P[:, 1])
    n0 = len(kfr)
    Xkin = np.hstack([Xkg, per0.reshape(n0, -1), per1.reshape(n0, -1), pool0.reshape(n0, -1), pool1.reshape(n0, -1)]).astype(np.float32)
    pool_tip = np.hstack([pool0[:, :, 0], pool1[:, :, 0]])

    gap = F[..., ig].reshape(n0, 10); depth = F[..., idp].reshape(n0, 10); c0 = F[..., ic0].reshape(n0, 10)
    cgap = C[..., ig].reshape(n0, 10); cdepth = C[..., idp].reshape(n0, 10)
    scores = {
        "shadow_gap_min_raw": -np.nanmin(gap, 1),                 # small gap anywhere = contact-like
        "shadow_depth_max_raw": np.nanmax(depth, 1),
        "shadow_contrast_max_raw": -np.nanmin(c0, 1),
        "kin_stopmotion_raw": np.nanmax(pool_tip, 1),
        "ctrl_staticimg_gap_raw": -np.nanmin(cgap, 1),
        "ctrl_staticimg_depth_raw": np.nanmax(cdepth, 1),
    }
    Xs, Xc, Xkin = Xs[have], Xc[have], Xkin[have]
    scores = {k: v[have] for k, v in scores.items()}
    t, y, P = t[have], y[have], P[have]
    blk = pp.block_ids(t)
    res["n_features"] = {"shadow": int(Xs.shape[1]), "kin": int(Xkin.shape[1])}
    scores["shadow_lgbm_oof"] = pp.oof_lgb(Xs, y, t)
    scores["kin_lgbm_oof"] = pp.oof_lgb(Xkin, y, t)
    scores["shadow_plus_kin_lgbm_oof"] = pp.oof_lgb(np.hstack([Xs, Xkin]), y, t)
    scores["ctrl_staticimg_lgbm_oof"] = pp.oof_lgb(Xc, y, t)
    shift = int(30 * 60)
    scores["ctrl_timeshuffle_shadow_lgbm"] = np.roll(scores["shadow_lgbm_oof"], shift)
    scores["ctrl_timeshuffle_shadow_raw"] = np.roll(scores["shadow_depth_max_raw"], shift)
    tbl = {}
    for k, v in scores.items():
        v = np.where(np.isfinite(v), v, np.nanmedian(v))
        scores[k] = v
        tbl[k] = {"auc": pp.auc(y, v), "ci95": pp.boot_auc(y, v, blk)}
        print(f"{k:32s} AUC={tbl[k]['auc']:.4f}  95% CI {tbl[k]['ci95'][0]:.4f}-{tbl[k]['ci95'][1]:.4f}", flush=True)
    res["auc"] = tbl
    res["deltas"] = {}
    for A, B in (("shadow_depth_max_raw", "kin_stopmotion_raw"), ("shadow_lgbm_oof", "kin_lgbm_oof"),
                 ("shadow_plus_kin_lgbm_oof", "kin_lgbm_oof"), ("shadow_lgbm_oof", "ctrl_staticimg_lgbm_oof")):
        d, ci = pp.boot_delta(y, scores[A], scores[B], blk)
        res["deltas"][f"{A}__minus__{B}"] = {"delta": d, "ci95": ci}
        print(f"delta {A} - {B}: {d:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]", flush=True)

    # ---- per finger, and the light-direction / surface analysis
    fk = pp.finger_of_key()
    kd = [json.loads(l) for l in open(s / "keys.jsonl")]
    kd = [(r["t"], r["key"]) for r in kd if r.get("event") == "down"]
    dt_any = np.abs(t[:, None] - kt[None, :]).min(1)
    per_f = {}
    Fh, Ch = F[have], C[have]
    for f in range(10):
        ktf = np.array([tt for tt, kk in kd if fk.get(kk) == f])
        if len(ktf) < 20:
            continue
        yf = np.abs(t[:, None] - ktf[None, :]).min(1) <= a.half
        keep = yf | (dt_any > 0.150)
        h, k = divmod(f, 5)
        sc_gap, sc_dep = -Fh[:, h, k, ig], Fh[:, h, k, idp]
        ok = keep & np.isfinite(sc_dep)
        per_f[FN[f]] = {"n_keys": int(len(ktf)), "n_pos": int(yf[ok].sum()),
                        "auc_gap": pp.auc(yf[ok], sc_gap[ok]), "auc_depth": pp.auc(yf[ok], sc_dep[ok]),
                        "auc_ctrl_depth": pp.auc(yf[ok], Ch[ok, h, k, idp]),
                        "auc_kin": pp.auc(yf[ok], pool_tip[have][ok, f]),
                        "gap_px_median_keydown": float(np.nanmedian(Fh[ok & yf, h, k, ig])) if (ok & yf).any() else None,
                        "gap_px_median_hover": float(np.nanmedian(Fh[ok & ~yf, h, k, ig])) if (ok & ~yf).any() else None,
                        "depth_median_keydown": float(np.nanmedian(Fh[ok & yf, h, k, idp])) if (ok & yf).any() else None,
                        "depth_median_hover": float(np.nanmedian(Fh[ok & ~yf, h, k, idp])) if (ok & ~yf).any() else None}
        r = per_f[FN[f]]
        print(f"{FN[f]}: keys={r['n_keys']:5d} AUC gap={r['auc_gap']:.3f} depth={r['auc_depth']:.3f} "
              f"ctrl={r['auc_ctrl_depth']:.3f} kin={r['auc_kin']:.3f} | gap px kd/hover {r['gap_px_median_keydown']}/{r['gap_px_median_hover']}",
              flush=True)
    res["per_finger"] = per_f
    # light direction: histogram of the darkest ray direction, split by surface brightness under the tip
    dd = Fh[..., idk].reshape(len(Fh), 10); sf = Ch[..., isf].reshape(len(Fh), 10)
    okd = np.isfinite(dd) & np.isfinite(sf)
    bright = okd & (sf > BRIGHT_SURFACE); dark = okd & (sf <= BRIGHT_SURFACE)
    hist = lambda m: np.bincount(dd[m].astype(int), minlength=NDIR) / max(m.sum(), 1)
    res["light"] = {"direction_labels_deg_image": [int(np.degrees(a_)) for a_ in ANGLES],
                    "note": "0 = image +x (right), 90 = image +y (down); the darkest ray points TOWARD the shadow, i.e. away from the light",
                    "darkest_dir_hist_bright_surface": hist(bright).round(3).tolist(),
                    "darkest_dir_hist_dark_surface": hist(dark).round(3).tolist(),
                    "frac_tip_frames_over_bright_surface": float(bright.sum() / max(okd.sum(), 1)),
                    "surface_gray_median_under_tip": float(np.nanmedian(sf[okd])),
                    "shadow_depth_median_bright": float(np.nanmedian(depth[have][bright])) if bright.any() else None,
                    "shadow_depth_median_dark": float(np.nanmedian(depth[have][dark])) if dark.any() else None,
                    "gap_px_median_bright_hover": float(np.nanmedian(gap[have][bright & (dt_any > 0.15)[:, None]])) if bright.any() else None,
                    "gap_px_median_bright_keydown": float(np.nanmedian(gap[have][bright & y[:, None]])) if bright.any() else None}
    # AUC restricted to tips over the bright deck (where a shadow could be seen at all)
    if bright.any():
        yb = y[:, None].repeat(10, 1)
        for nm, arr in (("depth", depth[have]), ("gap", -gap[have])):
            v = arr[bright]; yy = yb[bright]
            res["light"][f"auc_{nm}_tipframes_bright_surface"] = pp.auc(yy, v) if yy.any() and (~yy).any() else None
            v = arr[dark]; yy = yb[dark]
            res["light"][f"auc_{nm}_tipframes_dark_surface"] = pp.auc(yy, v) if yy.any() and (~yy).any() else None
    print("light:", json.dumps(res["light"], indent=None), flush=True)
    (OUT / f"step1_{a.sid}.json").write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed(CACHE / f"{a.sid}_scores.npz", t=t, y=y, **scores)
    print("wrote", OUT / f"step1_{a.sid}.json")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("extract", "probe"))
    ap.add_argument("sid")
    ap.add_argument("--half", type=float, default=0.040)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    return {"extract": cmd_extract, "probe": cmd_probe}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
