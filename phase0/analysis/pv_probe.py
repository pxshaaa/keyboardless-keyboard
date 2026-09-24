"""Step 1: does PressureVision++ contact predict a real key-down better than our kinematic features?
Usage: python -m phase0.analysis.pv_probe <session_id> [--half 0.040]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from phase0.analysis import pv_common as pv
from phase0.analysis import taps_gb as gb

SESS = Path("data/sessions")
OUT = Path("results/pressure")
FN = ["L-th", "L-ix", "L-mi", "L-ri", "L-pi", "R-th", "R-ix", "R-mi", "R-ri", "R-pi"]
LAGS = (-4, -2, 0, 2, 4)
BLOCK_S = 10.0
NBOOT = 2000


def _shift(a, k):
    out = np.full_like(a, np.nan)
    if k > 0:
        out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    else:
        out[:] = a
    return out


def auc(y, s):
    m = np.isfinite(s)
    return float(roc_auc_score(y[m], s[m])) if y[m].any() and (~y[m]).any() else float("nan")


def block_ids(t, block_s=BLOCK_S):
    return ((t - t.min()) // block_s).astype(int)


def boot_auc(y, s, blk, n=NBOOT, seed=0):
    """Block bootstrap CI for one AUC and (optionally) for paired deltas."""
    rng = np.random.default_rng(seed)
    ub = np.unique(blk)
    idx = {b: np.where(blk == b)[0] for b in ub}
    out = []
    for _ in range(n):
        pick = rng.choice(ub, len(ub), replace=True)
        r = np.concatenate([idx[b] for b in pick])
        if y[r].any() and (~y[r]).any():
            out.append(auc(y[r], s[r]))
    return np.percentile(out, [2.5, 97.5]).tolist()


def boot_delta(y, a, b, blk, n=NBOOT, seed=0):
    rng = np.random.default_rng(seed)
    ub = np.unique(blk)
    idx = {k: np.where(blk == k)[0] for k in ub}
    out = []
    for _ in range(n):
        pick = rng.choice(ub, len(ub), replace=True)
        r = np.concatenate([idx[k] for k in pick])
        if y[r].any() and (~y[r]).any():
            out.append(auc(y[r], a[r]) - auc(y[r], b[r]))
    return float(np.mean(out)), np.percentile(out, [2.5, 97.5]).tolist()


def oof_lgb(X, y, t, seed=0, folds=5, purge_s=1.0):
    """Out-of-fold probabilities over contiguous time folds, purging frames within purge_s of the fold."""
    import lightgbm as lgb

    edges = np.quantile(t, np.linspace(0, 1, folds + 1))
    p = np.full(len(y), np.nan)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    for f in range(folds):
        te = (t >= edges[f]) & (t <= edges[f + 1]) if f == folds - 1 else (t >= edges[f]) & (t < edges[f + 1])
        tr = (t < edges[f] - purge_s) | (t > edges[f + 1] + purge_s)
        m = gb.make_model("lgbm", random_state=seed, n_jobs=4)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def pv_features(z, rows):
    """-> (X [n,d], names) from the PV npz, for frames.jsonl rows `rows`."""
    cp = z["cp_max"][rows].reshape(len(rows), 10)
    cm = z["cp_mean"][rows].reshape(len(rows), 10)
    pr = z["pr_max"][rows].reshape(len(rows), 10)
    gl = z["glob"][rows].reshape(len(rows), 6)
    bo = z["bott"][rows].reshape(len(rows), 14)
    cols, names = [], []
    for k in LAGS:
        cols.append(_shift(cp, k)); names += [f"cp_max_{f}_l{k}" for f in FN]
    cols.append(cp - _shift(cp, 3)); names += [f"cp_d3_{f}" for f in FN]
    cols.append(cp - _shift(cp, -3)); names += [f"cp_dn3_{f}" for f in FN]
    cols += [cm, pr, gl, bo]
    names += [f"cp_mean_{f}" for f in FN] + [f"pr_max_{f}" for f in FN]
    names += [f"glob{i}" for i in range(6)] + [f"bott{i}" for i in range(14)]
    return np.hstack(cols).astype(np.float32), names


def finger_of_key():
    from phase0.analysis import swipe_motor as SM
    m = {}
    for c in SM.LET:
        g = SM.gt_fid(c)
        if g is not None:
            m[c] = SM.fid(*g)
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sid")
    ap.add_argument("--half", type=float, default=0.040)
    ap.add_argument("--limit", type=int, default=0, help="debug: keep only the first N landmark frames")
    ap.add_argument("--npz", default="")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(a.npz or pv.PV_ROOT / "feats" / f"{a.sid}.npz")
    s = SESS / a.sid
    fi, ft = pv.load_frames(s)

    kfr, kt_all, P = gb.load_frames(s)
    kt = pv.keydowns(s)
    if a.limit:
        kfr, P = kfr[:a.limit], P[:a.limit]
    rows = np.searchsorted(fi, kfr)
    assert (fi[rows] == kfr).all()
    t = ft[rows]
    y = np.abs(t[:, None] - kt[None, :]).min(1) <= a.half

    cp = z["cp_max"][rows].reshape(len(rows), 10)
    have = np.isfinite(cp).any(1)
    res = {"session": a.sid, "half_s": a.half, "frames_total": int(len(fi)),
           "frames_with_landmarks": int(len(rows)), "frames_scored": int(have.sum()),
           "keydowns": int(len(kt)), "positives": int(y[have].sum()),
           "tip_radius_px_448": float(np.nanmedian(z["tip_r"])),
           "keydowns_in_unscored_frames": int(((~have) & y).sum())}

    Xp, _ = pv_features(z, rows)
    ctl_all = z["ctl_max"][rows].reshape(len(rows), 10)
    Xc, _ = pv_features({k: z[k] for k in z.files} | {"cp_max": z["ctl_max"], "cp_mean": z["ctl_max"],
                                                      "pr_max": z["ctl_pr"], "glob": z["glob"] * 0,
                                                      "bott": z["bott"] * 0}, rows)
    Xkg = gb.build_groups(P)
    Xkg = np.hstack([Xkg[g] for g in gb.GROUPS]).astype(np.float32)
    from phase0.analysis import touch_feats as tf
    per0, pool0 = tf.hand_feats(P[:, 0])
    per1, pool1 = tf.hand_feats(P[:, 1])
    n0 = len(rows)
    Xt = np.hstack([per0.reshape(n0, -1), per1.reshape(n0, -1),
                    pool0.reshape(n0, -1), pool1.reshape(n0, -1)]).astype(np.float32)
    Xkin = np.hstack([Xkg, Xt])
    pool_tip = np.hstack([pool0[:, :, 0], pool1[:, :, 0]])
    pr_all = z["pr_max"][rows].reshape(n0, 10)
    bo_all = z["bott"][rows][:, :, :5].reshape(n0, 10)

    scores = {}
    scores["pv_contact_max_raw"] = np.nanmax(cp, 1)
    scores["pv_pressure_max_raw"] = np.nanmax(pr_all, 1)
    scores["pv_fingerhead_raw"] = np.nanmax(bo_all, 1)
    scores["kin_stopmotion_raw"] = np.nanmax(pool_tip, 1)
    scores["ctrl_staticimg_raw"] = np.nanmax(ctl_all, 1)
    Xp, Xc, Xkin, pool_tip = Xp[have], Xc[have], Xkin[have], pool_tip[have]
    scores = {k: v[have] for k, v in scores.items()}
    rows, t, y, cp = rows[have], t[have], y[have], cp[have]
    blk = block_ids(t)
    res["n_features"] = {"pv": int(Xp.shape[1]), "kin": int(Xkin.shape[1])}
    scores["pv_lgbm_oof"] = oof_lgb(Xp, y, t)
    scores["kin_lgbm_oof"] = oof_lgb(Xkin, y, t)
    scores["pv_plus_kin_lgbm_oof"] = oof_lgb(np.hstack([Xp, Xkin]), y, t)
    scores["ctrl_staticimg_lgbm_oof"] = oof_lgb(Xc, y, t)
    shift = int(30 * 60)
    scores["ctrl_timeshuffle_pv_lgbm"] = np.roll(scores["pv_lgbm_oof"], shift)
    scores["ctrl_timeshuffle_pv_raw"] = np.roll(scores["pv_contact_max_raw"], shift)

    tbl = {}
    for k, v in scores.items():
        tbl[k] = {"auc": auc(y, v), "ci95": boot_auc(y, v, blk)}
        print(f"{k:32s} AUC={tbl[k]['auc']:.4f}  95% CI {tbl[k]['ci95'][0]:.4f}-{tbl[k]['ci95'][1]:.4f}", flush=True)
    res["auc"] = tbl

    res["deltas"] = {}
    for A, B in (("pv_contact_max_raw", "kin_stopmotion_raw"), ("pv_lgbm_oof", "kin_lgbm_oof"),
                 ("pv_plus_kin_lgbm_oof", "kin_lgbm_oof"), ("pv_lgbm_oof", "ctrl_staticimg_lgbm_oof")):
        d, ci = boot_delta(y, scores[A], scores[B], blk)
        res["deltas"][f"{A}__minus__{B}"] = {"delta": d, "ci95": ci}
        print(f"delta {A} - {B}: {d:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]", flush=True)

    fk = finger_of_key()
    kd = [json.loads(l) for l in open(s / "keys.jsonl")]
    kd = [(r["t"], r["key"]) for r in kd if r.get("event") == "down"]
    kt_f = {f: np.array([tt for tt, kk in kd if fk.get(kk) == f]) for f in range(10)}
    dt_any = np.abs(t[:, None] - kt[None, :]).min(1)
    per_f = {}
    for f in range(10):
        if len(kt_f[f]) < 20:
            continue
        yf = np.abs(t[:, None] - kt_f[f][None, :]).min(1) <= a.half
        keep = yf | (dt_any > 0.150)
        sc = cp[:, f]
        pf = {"n_keys": int(len(kt_f[f])), "n_pos": int(yf[keep].sum()),
              "auc_pv": auc(yf[keep], sc[keep]),
              "auc_pv_ci95": boot_auc(yf[keep], sc[keep], blk[keep]),
              "auc_kin": auc(yf[keep], pool_tip[keep, f])}
        per_f[FN[f]] = pf
        print(f"{FN[f]}: keys={pf['n_keys']:5d} AUC_pv={pf['auc_pv']:.3f} "
              f"[{pf['auc_pv_ci95'][0]:.3f},{pf['auc_pv_ci95'][1]:.3f}] AUC_kin={pf['auc_kin']:.3f}", flush=True)
    sp = np.array([tt for tt, kk in kd if kk == "space"])
    if len(sp) >= 20:
        yf = np.abs(t[:, None] - sp[None, :]).min(1) <= a.half
        keep = yf | (dt_any > 0.150)
        sc = np.nanmax(cp[:, [0, 5]], 1)
        per_f["thumb(space,either)"] = {"n_keys": int(len(sp)), "n_pos": int(yf[keep].sum()),
                                        "auc_pv": auc(yf[keep], sc[keep]),
                                        "auc_pv_ci95": boot_auc(yf[keep], sc[keep], blk[keep]),
                                        "auc_kin": auc(yf[keep], np.nanmax(pool_tip[:, [0, 5]], 1)[keep])}
        print(f"thumb(space): keys={len(sp)} AUC_pv={per_f['thumb(space,either)']['auc_pv']:.3f} "
              f"AUC_kin={per_f['thumb(space,either)']['auc_kin']:.3f}", flush=True)
    res["per_finger"] = per_f

    tipxy = np.stack([P[have][:, h, j, :2] for h in (0, 1) for j in pv.TIPS], 1)
    span = np.nanmedian(np.linalg.norm(P[:, :, 9, :2] - P[:, :, 0, :2], axis=2))
    d = np.linalg.norm(tipxy[:, :, None, :] - tipxy[:, None, :, :], axis=3)
    d[:, np.arange(10), np.arange(10)] = np.inf
    occ = {}
    for f in range(10):
        near = np.nanmin(d[:, f], 1) < 0.30 * span
        occ[FN[f]] = {"frac_tip_within_0.3span_of_another_tip": float(np.nanmean(near)),
                      "frac_landmark_missing": float(np.isnan(tipxy[:, f, 0]).mean()),
                      "median_contact_prob": float(np.nanmedian(cp[:, f]))}
    res["occlusion"] = {"median_wrist_mcp_span_px": float(span), "per_tip": occ}

    (OUT / f"step1_{a.sid}.json").write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed(pv.PV_ROOT / "feats" / f"{a.sid}_scores.npz", t=t, y=y, **scores)
    print("wrote", OUT / f"step1_{a.sid}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
