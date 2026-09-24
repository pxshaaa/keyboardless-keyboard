"""Idea 3: desk self-training from each phrase's known character count (5 phrase folds).
Used via: python -m phase0.analysis.touch_runs selftrain --seeds 0 1 2"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

import lightgbm  # noqa: F401

from phase0.analysis import taps_gb as gb
from phase0.analysis import touch_common as tc

N_FOLDS = 5
GMIN, GMAX = 3, 300          # frames between consecutive taps (50 ms .. 5 s)


def iki_logprior() -> np.ndarray:
    """log P(gap = g frames) from all keyboard training sessions, smoothed, with a floor."""
    g = np.concatenate([np.diff(tc.kt(s)) * 60.0 for s in tc.KBD_TRAIN4])
    h = np.bincount(np.clip(np.round(g).astype(int), 0, GMAX), minlength=GMAX + 1).astype(float)
    from scipy.ndimage import gaussian_filter1d
    h = gaussian_filter1d(h, 2.0) + 0.5
    h[:GMIN] = 0.0
    lp = np.log(h / h.sum() + 1e-12)
    lp[:GMIN] = -1e9
    return lp


def align_count(s: np.ndarray, n: int, lp: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Best n frames in a window: max sum s[t_k] + lam * sum lp[t_k - t_{k-1}]. -> sorted indices."""
    T = len(s)
    if n <= 0 or T == 0:
        return np.array([], int)
    n = min(n, T // GMIN)
    NEG = -1e18
    best = s.copy()
    back = []
    for k in range(1, n):
        M = np.full(T, NEG)
        A = np.zeros(T, int)
        for g in range(GMIN, min(GMAX, T - 1) + 1):
            cand = np.full(T, NEG)
            cand[g:] = best[:-g] + lam * lp[g]
            better = cand > M
            M[better] = cand[better]
            A[better] = g
        best = M + s
        back.append(A)
    t = int(np.argmax(best))
    out = [t]
    for A in reversed(back):
        t = t - A[t]
        out.append(t)
    return np.array(sorted(out), int)


def emission(p: np.ndarray) -> np.ndarray:
    """logit p, but only at peak or shoulder frames: one probability bump is one tap, so the
    aligner may not buy count by taking three frames of the same bump (measured: without this it"""
    from phase0.analysis.hirecall import shoulders
    from scipy.signal import find_peaks
    s = np.log(np.clip(p, 1e-4, 1 - 1e-4)) - np.log(np.clip(1 - p, 1e-4, 1))
    cand = np.union1d(find_peaks(p)[0], shoulders(p))
    out = np.full(len(p), -30.0)
    out[cand] = s[cand]
    return out


def pseudo_taps(p: np.ndarray, lp: np.ndarray, lam: float = 1.0) -> list[np.ndarray]:
    t = tc.frames(tc.DESK)[1]
    s = emission(p)
    out = []
    for w in tc.windows():
        idx = np.where((t >= w["t0"]) & (t < w["t1"]))[0]
        out.append(idx[align_count(s[idx], w["n"], lp, lam)])
    return out


def desk_labels(taps: list[np.ndarray], phrases: list[int]):
    """-> (rows, y, yg) for the desk frames inside the given phrase windows; frames 3-5 away from a
    pseudo tap are left out (their label is the uncertain part)."""
    t = tc.frames(tc.DESK)[1]
    rows, ys, ygs = [], [], []
    for j in phrases:
        w = tc.windows()[j]
        idx = np.where((t >= w["t0"]) & (t < w["t1"]))[0]
        tt = t[taps[j]]
        d = np.abs(t[idx][:, None] - tt[None, :]).min(1) if len(tt) else np.full(len(idx), 9.0)
        keep = (d <= 2.5 / 60) | (d > 5.5 / 60)
        rows.append(idx[keep])
        ys.append(d[keep] <= 2.5 / 60)
        ygs.append(d[keep] <= gb.TYPING_HALF_WIDTH_S)
    return np.concatenate(rows), np.concatenate(ys), np.concatenate(ygs)


def fold_of(j: int) -> int:
    return j % N_FOLDS


def selftrain(feat_fn, base_p: np.ndarray, seed: int, w_desk: float = 3.0, lam: float = 0.5,
              desk_only: bool = False, verbose: bool = True) -> dict:
    lp = iki_logprior()
    taps = pseudo_taps(base_p, lp, lam)
    Xd = np.asarray(feat_fn(tc.DESK), np.float32)
    t = tc.frames(tc.DESK)[1]
    Xk = np.vstack([np.asarray(feat_fn(s), np.float32) for s in tc.KBD_TRAIN4])
    yk = np.concatenate([gb.labels(tc.frames(s)[1], tc.kt(s)) for s in tc.KBD_TRAIN4])
    ygk = np.concatenate([gb.labels(tc.frames(s)[1], tc.kt(s), gb.TYPING_HALF_WIDTH_S)
                          for s in tc.KBD_TRAIN4])
    p_new = base_p.copy()
    g_new = np.full(len(t), np.nan)
    W = tc.windows()
    models = []
    for f in range(N_FOLDS):
        t0 = time.time()
        tr = [j for j in range(len(W)) if fold_of(j) != f]
        te = [j for j in range(len(W)) if fold_of(j) == f]
        rows, yd, ygd = desk_labels(taps, tr)
        if desk_only:
            X, y, yg, sw = Xd[rows], yd, ygd, None
        else:
            X = np.vstack([Xk, Xd[rows]])
            y = np.concatenate([yk, yd])
            yg = np.concatenate([ygk, ygd])
            sw = np.concatenate([np.ones(len(yk)), np.full(len(yd), w_desk)])
        m = gb.make_model("lgbm", random_state=seed, n_jobs=4).fit(X, y, sample_weight=sw)
        g = gb.make_model("lgbm", random_state=seed, n_jobs=4).fit(X, yg, sample_weight=sw)
        for j in te:
            idx = np.where((t >= W[j]["t0"]) & (t < W[j]["t1"]))[0]
            p_new[idx] = m.predict_proba(Xd[idx])[:, 1]
            g_new[idx] = g.predict_proba(Xd[idx])[:, 1]
        models.append((m, g))
        if verbose:
            print(f"    fold{f} desk rows={len(rows)} pos={int(yd.sum())} ({time.time()-t0:.0f}s)",
                  flush=True)
    return {"p": p_new, "gate_new": g_new, "taps": taps, "models": models}


def infold_eval(seed: int, w_desk: float = 3.0) -> dict:
    """Deployment-like count error: each fold's own threshold, set on its 16 training phrases."""
    import joblib
    base = np.load(tc.CACHE / f"desk_probs_base_s{seed}.npz")
    r = selftrain(tc.base_X, base["p"], seed, w_desk=w_desk, verbose=False)
    Xd = np.asarray(tc.base_X(tc.DESK), np.float32)
    W = tc.windows()
    n = np.array([w["n"] for w in W])
    grid = np.geomspace(1e-3, 0.99, 300)
    errs, stitched_thr = np.zeros(len(W)), []
    for f, (m, g) in enumerate(r["models"]):
        p, gt = m.predict_proba(Xd)[:, 1], g.predict_proba(Xd)[:, 1]
        C = np.stack([tc.per_phrase_counts(tc.extract(p, gt, th)) for th in grid])
        tr = np.array([j for j in range(len(W)) if fold_of(j) != f])
        k = int(np.argmin(np.abs(C[:, tr].sum(1) - n[tr].sum())))
        stitched_thr.append(float(grid[k]))
        for j in range(len(W)):
            if fold_of(j) == f:
                errs[j] = abs(C[k, j] - n[j]) / n[j]
    gate = np.where(np.isfinite(r["gate_new"]), r["gate_new"], base["gate"])
    glob = tc.count_error(r["p"], gate)
    return {"seed": seed, "infold_count_err": float(errs.mean()), "infold_per": errs.tolist(),
            "stitched_count_err": glob["count_err"], "fold_thresholds": stitched_thr}


if __name__ == "__main__":
    import sys as _s
    out = [infold_eval(int(x)) for x in _s.argv[1:]]
    for o in out:
        print(f"seed{o['seed']}: in-fold-threshold count_err={o['infold_count_err']:.3f} "
              f"stitched/global={o['stitched_count_err']:.3f} fold thresholds={np.round(o['fold_thresholds'], 3)}",
              flush=True)
    Path_ = __import__("pathlib").Path("results/contact/selftrain_infold.json")
    Path_.write_text(json.dumps(out, indent=1))
