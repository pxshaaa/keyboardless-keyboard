"""Idea 2: monocular press cues (stop-motion, relative finger drop, pooled over fingertips).
Used via: python -m phase0.analysis.touch_runs touch --seeds 0 1 2"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

import lightgbm  # noqa: F401

from phase0.analysis import taps_gb as gb
from phase0.analysis import touch_common as tc

TIPS = (4, 8, 12, 16, 20)
MCPS = (2, 5, 9, 13, 17)
PIPS = (3, 6, 10, 14, 18)
PALM = (0, 5, 9, 13, 17)


def _ff(x):
    return gb._ffill(x)


def _d1(x):
    return gb._d1(x)


def _lag(x, k):
    return gb._shift(x, k)


def hand_feats(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """H [F,21,7] -> (per-finger block [F,5,k], pooled-input block [F,5,m])."""
    F = len(H)
    span = np.linalg.norm(H[:, 9, :2] - H[:, 0, :2], axis=1)
    span = np.where(span > 1e-6, span, np.nan)
    fwd = H[:, 9, :2] - H[:, 0, :2]
    fwd = uniform_filter1d(_ff(fwd / span[:, None]), 15, axis=0)
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-9
    lat = np.stack([-fwd[:, 1], fwd[:, 0]], 1)
    palm = H[:, PALM, :2].mean(1)
    tip = H[:, TIPS, :2]
    rel = (tip - palm[:, None]) / span[:, None, None]
    tf = (rel * fwd[:, None]).sum(2)                         # forward coordinate
    tl = (rel * lat[:, None]).sum(2)
    v = np.stack([_d1(tf), _d1(tl)], 2)
    sp = np.linalg.norm(v, axis=2)                           # palm-relative tip speed
    spf = _ff(sp)
    prev_max = maximum_filter1d(_lag(spf, 1), 7, axis=0, origin=3)   # max over t-7..t-1
    next_max = maximum_filter1d(_lag(spf, -1), 7, axis=0, origin=-3)
    loc_min = minimum_filter1d(spf, 7, axis=0)
    stop = (prev_max - sp) / (prev_max + 1e-3)                # 1 = came to a halt
    rebound = (next_max - sp) / (next_max + 1e-3)
    ismin = (np.abs(spf - loc_min) < 1e-9).astype(float)
    L = np.linalg.norm(H[:, TIPS, :2] - H[:, MCPS, :2], axis=2) / span[:, None]
    dL = _d1(L)
    Lrest = L - gb._roll_med(L, 61)
    # drop relative to the other four fingers (forward, MediaPipe z, world z), over 4 frames
    dfw = tf - _lag(tf, 4)
    z = (H[:, TIPS, 3] - H[:, :1, 3]) / span[:, None]
    dz = z - _lag(z, 4)
    wz = H[:, TIPS, 6] - H[:, PALM, 6].mean(1, keepdims=True)
    dwz = wz - _lag(wz, 4)

    def others(a):
        return a - (np.nansum(a, 1, keepdims=True) - a) / 4.0

    pitch = (H[:, TIPS, 3] - H[:, PIPS, 3]) / span[:, None]
    wzs = wz - gb._roll_med(wz, 61)
    per = np.stack([tf, tl, sp, stop, rebound, ismin, L, dL, Lrest, others(dfw), others(dz),
                    others(dwz), pitch, _d1(pitch), wzs], 2)
    pool = np.stack([stop * (prev_max > 0.01), others(dfw), others(dz), others(dwz), -dL,
                     rebound * stop, wzs, sp], 2)
    return per, pool


POOL_LAGS = (-6, -4, -3, -2, -1, 1, 2, 3, 4, 6)


def touch_X(P: np.ndarray) -> np.ndarray:
    F = len(P)
    pers, pools = [], []
    for s in range(2):
        per, pool = hand_feats(P[:, s].astype(np.float64))
        pers.append(per.reshape(F, -1))
        pools.append(pool)
    pool = np.concatenate(pools, 1)                          # [F,10,m]
    srt = -np.sort(-np.nan_to_num(pool, nan=-9.0), axis=1)   # descending across fingertips
    top = srt[:, :2].reshape(F, -1)
    top1 = srt[:, 0]
    ctx = np.hstack([_lag(top1, k) for k in POOL_LAGS])
    hand_max = np.hstack([np.nanmax(np.nan_to_num(p, nan=-9.0), 1) for p in pools])
    return np.hstack(pers + [top, ctx, hand_max]).astype(np.float32)


_T: dict = {}


def tX(sid: str) -> np.ndarray:
    if sid not in _T:
        f = tc.CACHE / f"T_{sid}.npy"
        if not f.exists():
            np.save(f, touch_X(tc.frames(sid)[2]))
        _T[sid] = np.load(f, mmap_mode="r")
    return _T[sid]


def feat_base_touch(sid):
    return np.hstack([np.asarray(tc.base_X(sid)), np.asarray(tX(sid))])


def feat_touch_only(sid):
    return np.asarray(tX(sid))


STATIC = (0, 1, 6, 12)   # tf, tl, L, pitch: hand-frame posture, not motion


def dyn_cols() -> np.ndarray:
    n_per = 15
    drop = [h * 5 * n_per + f * n_per + k for h in range(2) for f in range(5) for k in STATIC]
    return np.setdiff1d(np.arange(262), drop)


def feat_base_touchdyn(sid):
    return np.hstack([np.asarray(tc.base_X(sid)), np.asarray(tX(sid))[:, dyn_cols()]])
