"""Gradient-boosted per-frame keypress detector with an explicit still-gate.
Run: python -m phase0.analysis.taps_gb {train --sessions A B | predict SESSION | experiment}"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import median_filter, maximum_filter1d, uniform_filter1d
from scipy.signal import find_peaks
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss

from phase0.analysis.detect_taps import FINGERTIP_JOINTS, load_landmarks
from phase0.analysis.analyze_drift import pair_events
from phase0.analysis.eval_taps import STILL_S, WINDOW_S, keydown_times, score
from phase0.analysis.taps_ml import (
    MCP,
    PIP,
    depth_base_features,
    hand_base_features,
    _shift,
)

DEFAULT_MODEL = Path("models/taps_gb.joblib")
FPS = 60.0
SIDES = ("Left", "Right")
PALM_JOINTS = (0, 5, 9, 13, 17)
CHANNELS = ("x", "y", "conf", "z", "wx", "wy", "wz")
CTX_LAGS = (1, 2, 3, 4, 5, 6, 7, 8)
REST_W = 61  # 1 s @ 60 fps
LABEL_HALF_WIDTH_S = 0.033
TYPING_HALF_WIDTH_S = 0.300  # matches eval_taps' still-FP definition
GROUPS = ("pose", "depth", "deriv", "wflex", "palm", "rest", "still", "ctx")


# ---------------------------------------------------------------- loading
def load_frames(session: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (frame_i [F], t [F], P [F,2,21,7]) keyed by handedness; NaN if absent."""
    tb = load_landmarks(session)
    have = [ch for ch in CHANNELS if ch in tb.schema.names]
    c = {n: np.asarray(tb.column(n)) for n in ("i", "t", "hand", "handedness", "joint", *have)}
    frames, inv = np.unique(c["i"], return_inverse=True)
    t = np.zeros(len(frames))
    t[inv] = c["t"]
    P = np.full((len(frames), 2, 21, len(CHANNELS)), np.nan)
    side = (c["handedness"] != "Left").astype(int)
    # >21 rows in a (frame, handedness) group means two hands claim it; only there is `hand` better
    key = inv * 2 + side
    u, counts = np.unique(key, return_counts=True)
    collide = np.isin(key, u[counts > 21])
    side = np.where(collide, c["hand"], side)
    for k, ch in enumerate(CHANNELS):
        if ch in have:
            P[inv, side, c["joint"], k] = c[ch]
    return frames, t, P


# ---------------------------------------------------------------- helpers
def _d1(X: np.ndarray) -> np.ndarray:
    return (_shift(X, -1) - _shift(X, 1)) / 2


def _d2(X: np.ndarray) -> np.ndarray:
    return _shift(X, -1) - 2 * X + _shift(X, 1)


def _roll_med(X: np.ndarray, w: int) -> np.ndarray:
    Y = np.where(np.isfinite(X), X, np.nan)
    filled = _ffill(Y)
    return median_filter(filled, size=(w, 1), mode="nearest")


def _ffill(X: np.ndarray) -> np.ndarray:
    """Forward/backward fill NaNs along axis 0; median_filter has no NaN handling."""
    Y = X.copy()
    bad = ~np.isfinite(Y)
    if not bad.any():
        return Y
    idx = np.where(~bad, np.arange(len(Y))[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    Y = Y[idx, np.arange(Y.shape[1])[None, :]]
    Y = np.where(np.isfinite(Y), Y, 0.0)
    return Y


def _roll_mean(X: np.ndarray, w: int) -> np.ndarray:
    return uniform_filter1d(np.nan_to_num(X, nan=0.0), size=w, axis=0, mode="nearest")


def _stack_lags(X: np.ndarray) -> np.ndarray:
    return np.hstack([_shift(X, k) for k in CTX_LAGS] + [_shift(X, -k) for k in CTX_LAGS])


# ---------------------------------------------------------------- features
def _wflex(H: np.ndarray) -> np.ndarray:
    """World-coord per-finger flexion: length, 3-D PIP angle, signed palm-plane distance."""
    W = H[:, :, 4:7]
    wspan = np.linalg.norm(W[:, 9] - W[:, 0], axis=1)
    wspan = np.where(wspan > 1e-9, wspan, np.nan)[:, None]
    n = np.cross(W[:, 5] - W[:, 0], W[:, 17] - W[:, 0])
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    ln, ang, pl = [], [], []
    for tip in FINGERTIP_JOINTS:
        ln.append(np.linalg.norm(W[:, tip] - W[:, MCP[tip]], axis=1))
        a = W[:, MCP[tip]] - W[:, PIP[tip]]
        b = W[:, tip] - W[:, PIP[tip]]
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
        ang.append(np.arccos(np.clip(cos, -1, 1)))
        pl.append(((W[:, tip] - W[:, 0]) * n).sum(1))
    return np.hstack([np.stack(ln, 1) / wspan, np.stack(ang, 1), np.stack(pl, 1) / wspan])


def _palm(H: np.ndarray, span: np.ndarray) -> np.ndarray:
    """Whole-hand motion the tap model should learn to ignore."""
    C = H[:, PALM_JOINTS, :2].mean(1) / span
    Cw = H[:, PALM_JOINTS, 4:7].mean(1)
    v, a, vw = _d1(C), _d2(C), _d1(Cw)
    wr = _d1(H[:, 0, :2] / span)
    return np.hstack([
        v, a, np.linalg.norm(v, axis=1)[:, None],
        vw, np.linalg.norm(vw, axis=1)[:, None],
        wr, np.linalg.norm(wr, axis=1)[:, None],
    ])


def _rest(H: np.ndarray, span: np.ndarray) -> np.ndarray:
    """Displacement of each fingertip from its own rolling 1 s median resting pose."""
    R = ((H[:, FINGERTIP_JOINTS, :2] - H[:, :1, :2]) / span[:, :, None]).reshape(len(H), -1)
    Z = (H[:, FINGERTIP_JOINTS, 3] - H[:, :1, 3]) / span
    base = np.hstack([R, Z, H[:, 0, 1:2], H[:, 0, 6:7]])
    disp = base - _roll_med(base, REST_W)
    mag = np.linalg.norm(disp[:, :10].reshape(len(H), 5, 2), axis=2)
    return np.hstack([disp, mag])


def _still(H: np.ndarray, span: np.ndarray) -> np.ndarray:
    """Multi-scale motion energy + height above a slow resting baseline + time since motion."""
    tips = (H[:, FINGERTIP_JOINTS, :2] / span[:, :, None]).reshape(len(H), -1)
    sp = np.abs(_d1(tips)).reshape(len(H), 5, 2).sum(2)
    pc = np.linalg.norm(_d1(H[:, PALM_JOINTS, :2].mean(1) / span), axis=1)
    e = np.nan_to_num(sp.max(1) + pc, nan=0.0)
    scales = [15, 31, 61, 121]
    en = np.stack([_roll_mean(e[:, None], w)[:, 0] for w in scales], 1)
    pk = np.stack([maximum_filter1d(e, size=w, mode="nearest") for w in scales], 1)
    sd = np.stack([_roll_mean((e**2)[:, None], w)[:, 0] for w in scales], 1)
    hot = e > np.nanmedian(e) * 3
    since = _time_since(hot)
    slow = _roll_med(np.hstack([H[:, 0, 1:2], H[:, 0, 6:7], H[:, FINGERTIP_JOINTS, 1]]), 181)
    height = np.hstack([H[:, 0, 1:2], H[:, 0, 6:7], H[:, FINGERTIP_JOINTS, 1]]) - slow
    return np.hstack([en, pk, sd, e[:, None], since[:, None], height,
                      _roll_mean(np.nan_to_num(sp), 61)])


def _time_since(flag: np.ndarray) -> np.ndarray:
    out = np.full(len(flag), 1e3)
    last = -1e6
    for i in range(len(flag)):
        if flag[i]:
            last = i
        out[i] = (i - last) / FPS
    return np.minimum(out, 10.0)


def _core(H: np.ndarray, span: np.ndarray) -> np.ndarray:
    """The small signal set worth stacking at +/-1..8 frames."""
    ry = (H[:, FINGERTIP_JOINTS, 1] - H[:, :1, 1]) / span
    z = (H[:, FINGERTIP_JOINTS, 3] - H[:, :1, 3]) / span
    W = H[:, :, 4:7]
    wspan = np.linalg.norm(W[:, 9] - W[:, 0], axis=1)[:, None]
    fl = np.stack([np.linalg.norm(W[:, tip] - W[:, MCP[tip]], axis=1) for tip in FINGERTIP_JOINTS], 1)
    pc = np.linalg.norm(_d1(H[:, PALM_JOINTS, :2].mean(1) / span), axis=1)[:, None]
    return np.hstack([ry, _d1(ry), z, fl / np.where(wspan > 1e-9, wspan, np.nan), pc])


def build_groups(P: np.ndarray) -> dict[str, np.ndarray]:
    """-> {group: [F,d]} feature blocks, both hands concatenated inside each block."""
    out: dict[str, list[np.ndarray]] = {g: [] for g in GROUPS}
    for s in range(2):
        H = P[:, s]
        span = np.linalg.norm(H[:, 9, :2] - H[:, 0, :2], axis=1)[:, None]
        span = np.where(span > 1e-6, span, np.nan)
        pose = hand_base_features(H)
        dep = depth_base_features(H)
        out["pose"].append(pose)
        out["depth"].append(dep)
        pd_ = np.hstack([pose, dep])
        out["deriv"].append(np.hstack([_d1(pd_), _d2(pd_), (_shift(pd_, -3) - _shift(pd_, 3)) / 6]))
        out["wflex"].append(np.hstack([_wflex(H), _d1(_wflex(H))]))
        out["palm"].append(_palm(H, span))
        out["rest"].append(_rest(H, span))
        out["still"].append(_still(H, span))
        out["ctx"].append(_stack_lags(_core(H, span)))
    return {g: np.hstack(v) for g, v in out.items()}


def assemble(groups: dict[str, np.ndarray], use: tuple[str, ...]) -> np.ndarray:
    return np.hstack([groups[g] for g in GROUPS if g in use])


def group_names(groups: dict[str, np.ndarray], use: tuple[str, ...]) -> list[str]:
    return [f"{g}[{i}]" for g in GROUPS if g in use for i in range(groups[g].shape[1])]


def flex_velocity(P: np.ndarray) -> np.ndarray:
    """[F,2,5] per-finger flexion velocity, used only for finger attribution."""
    fv = []
    for s in range(2):
        H = P[:, s]
        span = np.linalg.norm(H[:, 9, :2] - H[:, 0, :2], axis=1)[:, None]
        f = np.stack([np.linalg.norm(H[:, tip, :2] - H[:, MCP[tip], :2], axis=1) for tip in FINGERTIP_JOINTS], 1)
        fv.append(_d1(f / np.where(span > 1e-6, span, np.nan)))
    return np.stack(fv, 1)


# ---------------------------------------------------------------- labels
def labels(t: np.ndarray, kt: np.ndarray, half: float = LABEL_HALF_WIDTH_S) -> np.ndarray:
    if len(kt) == 0:
        return np.zeros(len(t), bool)
    return np.abs(t[:, None] - kt[None, :]).min(1) <= half


# ---------------------------------------------------------------- events
def smooth(p: np.ndarray, w: int) -> np.ndarray:
    return p if w <= 1 else np.convolve(p, np.ones(w) / w, mode="same")


def pick_events(p: np.ndarray, thr: float, refractory: int = 5, n_consec: int = 1,
                run_frac: float = 0.6) -> np.ndarray:
    idx, _ = find_peaks(p, height=thr, distance=refractory)
    if n_consec > 1 and len(idx):
        run = _run_length(p >= thr * run_frac)
        idx = idx[run[idx] >= n_consec]
    return idx


def _run_length(mask: np.ndarray) -> np.ndarray:
    """Length of the contiguous True run containing each index (0 where False)."""
    out = np.zeros(len(mask), int)
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        out[i:j] = j - i
        i = j
    return out


EVENT_GRID = {
    "smooth": (1, 3, 5, 7, 9),
    "refractory": (3, 4, 5, 7, 9, 11),   # 4 frames (67ms) beats 5 on all folds; grid used to start at 5
    "n_consec": (1, 2, 3, 4, 5),
    "gate": (0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75),
    "thr": tuple(np.round(np.arange(0.05, 0.96, 0.025), 4)),
}


def prep_mask(kt: np.ndarray, t: np.ndarray, mask: np.ndarray) -> list[tuple]:
    out = []
    for lo, hi in _runs(mask):
        tlo, thi = t[lo], t[hi] + 1e-6
        out.append((lo, hi, tlo, thi, kt[(kt >= tlo) & (kt < thi)]))
    return out


def score_prep(prep: list[tuple], t: np.ndarray, ev: np.ndarray) -> dict:
    """Micro-averaged eval_taps.score over prepared runs; still-FP done vectorised."""
    keys = taps = hits = fp = 0
    for lo, hi, tlo, thi, k in prep:
        tt = t[ev[(ev >= lo) & (ev <= hi)]]
        tt = tt[(tt >= tlo) & (tt < thi)]
        keys += len(k)
        taps += len(tt)
        if not len(tt):
            continue
        if not len(k):
            fp += len(tt)
            continue
        hits += len(pair_events(k.tolist(), tt.tolist(), WINDOW_S))
        j = np.searchsorted(k, tt)
        lo_i, hi_i = np.clip(j - 1, 0, len(k) - 1), np.clip(j, 0, len(k) - 1)
        d = np.minimum(np.abs(tt - k[lo_i]), np.abs(tt - k[hi_i]))
        fp += int((d > STILL_S).sum())
    r = hits / keys if keys else 0.0
    p = hits / taps if taps else 0.0
    return {"keydowns": keys, "taps": taps, "hits": hits, "still_fp": fp,
            "recall": r, "precision": p, "f1": 2 * r * p / (r + p) if r + p else 0.0}


def score_masked(kt: np.ndarray, t: np.ndarray, mask: np.ndarray, ev: np.ndarray) -> dict:
    return score_prep(prep_mask(kt, t, mask), t, ev)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def _sweep_thr(p, t, prep, rf, nc, run_frac):
    """find_peaks' distance filter keeps the higher peak, so post-filtering by height is exact."""
    idx, props = find_peaks(p, height=EVENT_GRID["thr"][0], distance=rf)
    h = props["peak_heights"]
    best, best_f1 = None, -1.0
    for thr in EVENT_GRID["thr"]:
        cand = idx[h >= thr]
        if nc > 1 and len(cand):
            run = _run_length(p >= thr * run_frac)
            cand = cand[run[cand] >= nc]
        f1 = score_prep(prep, t, cand)["f1"]
        if f1 > best_f1:
            best_f1, best = f1, float(thr)
    return best, best_f1


def tune_events(p: np.ndarray, t: np.ndarray, kt: np.ndarray, mask: np.ndarray,
                gate: np.ndarray | None = None, run_frac: float = 0.6) -> tuple[dict, float]:
    """Two-stage: smoothing+gate first at default peak-picking, then refractory/run-length."""
    best, best_f1 = {"thr": 0.5, "smooth": 3, "refractory": 5, "n_consec": 1, "gate_thr": 0.0}, -1.0
    prep = prep_mask(kt, t, mask)
    gates = (0.0,) if gate is None else EVENT_GRID["gate"]
    for w in EVENT_GRID["smooth"]:
        ps = smooth(p, w)
        for gt in gates:
            pg = ps if gt <= 0 else np.where(gate >= gt, ps, 0.0)
            thr, f1 = _sweep_thr(pg, t, prep, 5, 1, run_frac)
            if f1 > best_f1:
                best_f1 = f1
                best = {"thr": thr, "smooth": w, "refractory": 5, "n_consec": 1, "gate_thr": gt}
    ps = smooth(p, best["smooth"])
    pg = ps if best["gate_thr"] <= 0 else np.where(gate >= best["gate_thr"], ps, 0.0)
    for rf in EVENT_GRID["refractory"]:
        for nc in EVENT_GRID["n_consec"]:
            thr, f1 = _sweep_thr(pg, t, prep, rf, nc, run_frac)
            if f1 > best_f1:
                best_f1 = f1
                best = dict(best, thr=thr, refractory=rf, n_consec=nc)
    return best, best_f1


def apply_events(p: np.ndarray, cfg: dict, gate: np.ndarray | None = None) -> np.ndarray:
    ps = smooth(p, cfg["smooth"])
    if gate is not None and cfg.get("gate_thr", 0.0) > 0:
        ps = np.where(gate >= cfg["gate_thr"], ps, 0.0)
    return pick_events(ps, cfg["thr"], cfg["refractory"], cfg["n_consec"])


# ---------------------------------------------------------------- models
def make_model(kind: str, **kw):
    if kind == "lgbm":
        import lightgbm as lgb
        params = dict(n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                      scale_pos_weight=8.0, n_jobs=-1, verbose=-1, random_state=0)
        params.update(kw)
        return lgb.LGBMClassifier(**params)
    params = dict(max_iter=400, learning_rate=0.06, max_depth=6, min_samples_leaf=40,
                  l2_regularization=1.0, class_weight="balanced", random_state=0,
                  early_stopping=False)
    params.update(kw)
    return HistGradientBoostingClassifier(**params)


# ---------------------------------------------------------------- data
class Data:
    """Per-session features (temporal diffs never cross a session boundary)."""

    def __init__(self, sessions: list[Path], groups_wanted: tuple[str, ...] = GROUPS):
        self.sessions = [Path(s) for s in sessions]
        self.g, self.t, self.kt, self.sid = [], [], [], []
        for k, sess in enumerate(self.sessions):
            _, t, P = load_frames(sess)
            self.g.append(build_groups(P))
            self.t.append(t)
            self.kt.append(keydown_times(sess))
            self.sid.append(np.full(len(t), k))
        self.t = np.concatenate(self.t)
        self.sid = np.concatenate(self.sid)
        self.kt_all = np.sort(np.concatenate(self.kt))

    def X(self, use: tuple[str, ...]) -> np.ndarray:
        return np.vstack([assemble(g, use) for g in self.g])

    def names(self, use: tuple[str, ...]) -> list[str]:
        return group_names(self.g[0], use)


def block_groups(t: np.ndarray, sid: np.ndarray, block_s: float = 20.0) -> np.ndarray:
    """Contiguous time blocks within each session; adjacent frames stay in one fold."""
    out = np.zeros(len(t), int)
    n = 0
    for s in np.unique(sid):
        m = sid == s
        b = ((t[m] - t[m].min()) // block_s).astype(int)
        out[m] = b + n
        n = out[m].max() + 1
    return out


def cv_folds(kind: str, t: np.ndarray, sid: np.ndarray, n_splits: int = 5) -> list[np.ndarray]:
    if kind == "session":
        return [sid == s for s in np.unique(sid)]
    g = block_groups(t, sid)
    u = np.unique(g)
    return [np.isin(g, u[i::n_splits]) for i in range(n_splits)]


def purge(train: np.ndarray, test: np.ndarray, t: np.ndarray, embargo: float = 0.5) -> np.ndarray:
    """Drop training frames within `embargo` s of any test frame; block edges would leak."""
    keep = train.copy()
    for lo, hi in _runs(test):
        keep &= ~((t >= t[lo] - embargo) & (t <= t[hi] + embargo))
    return keep


# ---------------------------------------------------------------- training core
def run_cv(data: Data, use: tuple[str, ...], kind: str = "lgbm", cv: str = "block",
           model_kw: dict | None = None, gate: bool = True, label_half: float = LABEL_HALF_WIDTH_S,
           frac: float = 1.0, seed: int = 0, verbose: bool = False, X=None) -> dict:
    """Grouped CV; fold k's event params are tuned on the OTHER folds' out-of-fold probs."""
    X = data.X(use) if X is None else X
    t, sid = data.t, data.sid
    kt = _subsample_keys(data.kt_all, frac, seed) if frac < 1.0 else data.kt_all
    y, yg = labels(t, kt, label_half), labels(t, kt, TYPING_HALF_WIDTH_S)
    mk = model_kw or {}
    folds = cv_folds(cv, t, sid)
    oof = np.zeros(len(t))
    oofg = np.zeros(len(t))
    for te in folds:
        tr = purge(~te, te, t)
        oof[te] = make_model(kind, **mk).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        if gate:
            oofg[te] = make_model(kind, **mk).fit(X[tr], yg[tr]).predict_proba(X[te])[:, 1]
    tot = {"keydowns": 0, "taps": 0, "hits": 0, "still_fp": 0}
    per_fold, cfgs = [], []
    for te in folds:
        cfg, _ = tune_events(oof, t, kt, ~te, oofg if gate else None)
        cfgs.append(cfg)
        s = score_masked(kt, t, te, apply_events(oof, cfg, oofg if gate else None))
        per_fold.append(s["f1"])
        for k in tot:
            tot[k] += s[k]
        if verbose:
            print(f"    fold n={te.sum()} F1={100*s['f1']:.1f}% stillFP={s['still_fp']} cfg={cfg}", flush=True)
    r = tot["hits"] / tot["keydowns"] if tot["keydowns"] else 0.0
    p = tot["hits"] / tot["taps"] if tot["taps"] else 0.0
    return {
        "f1": 2 * r * p / (r + p) if r + p else 0.0, "recall": r, "precision": p,
        "still_fp": tot["still_fp"], "taps": tot["taps"], "per_fold": per_fold,
        "ap": float(average_precision_score(y, oof)) if y.any() else float("nan"),
        "oof": oof, "oof_gate": oofg, "y": y, "cfgs": cfgs, "n_features": X.shape[1], "kt": kt,
    }


def _subsample_keys(kt: np.ndarray, frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = max(2, int(round(frac * len(kt))))
    return np.sort(rng.choice(kt, size=n, replace=False))


# ---------------------------------------------------------------- CLI: train
def cmd_train(a) -> int:
    use = tuple(a.groups.split(",")) if a.groups else GROUPS
    data = Data([Path(s) for s in a.sessions])
    X, t, kt = data.X(use), data.t, data.kt_all
    y, yg = labels(t, kt, a.label_half), labels(t, kt, TYPING_HALF_WIDTH_S)
    mk = json.loads(a.model_kw) if a.model_kw else {}
    print(f"frames={len(t)} features={X.shape[1]} pos={y.sum()} keys={len(kt)}")

    cvres = run_cv(data, use, a.kind, a.cv, mk, gate=not a.no_gate, label_half=a.label_half, verbose=True)
    print(f"[{a.cv}-CV] F1={100*cvres['f1']:.1f}% R={100*cvres['recall']:.1f}% "
          f"P={100*cvres['precision']:.1f}% stillFP={cvres['still_fp']} AP={cvres['ap']:.3f}")

    # final event params come from out-of-fold probabilities, never in-sample ones
    allm = np.ones(len(t), bool)
    cfg, f1 = tune_events(cvres["oof"], t, kt, allm, cvres["oof_gate"] if not a.no_gate else None)
    print(f"tuned on OOF probs: F1={100*f1:.1f}% cfg={cfg}")

    m = make_model(a.kind, **mk).fit(X, y)
    gm = None if a.no_gate else make_model(a.kind, **mk).fit(X, yg)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"kind": a.kind, "model": m, "gate": gm, "cfg": cfg, "groups": use}, a.out)
    print(f"saved {a.out}")
    return 0


def cmd_predict(a) -> int:
    b = joblib.load(a.model)
    frames, t, P = load_frames(a.session)
    g = build_groups(P)
    X = assemble(g, b["groups"])
    p = b["model"].predict_proba(X)[:, 1]
    gate = b["gate"].predict_proba(X)[:, 1] if b.get("gate") is not None else None
    ev = apply_events(p, b["cfg"], gate)
    fv = flex_velocity(P)
    out = a.out or a.session / "taps_gb.jsonl"
    with open(out, "w") as fh:
        for k in ev:
            m = np.abs(np.nan_to_num(fv[k], nan=-1.0))
            s, f = np.unravel_index(int(np.argmax(m)), m.shape)
            tip = FINGERTIP_JOINTS[f]
            fh.write(json.dumps({
                "t": float(t[k]), "hand": int(s), "finger": int(tip),
                "x": float(np.nan_to_num(P[k, s, tip, 0])), "y": float(np.nan_to_num(P[k, s, tip, 1])),
                "conf": float(np.nan_to_num(P[k, s, tip, 2])), "i": int(frames[k]),
            }) + "\n")
    print(f"wrote {len(ev)} events -> {out}")
    return 0


# ---------------------------------------------------------------- CLI: experiments
def cmd_experiment(a) -> int:
    data = Data([Path(s) for s in a.sessions])
    mk = json.loads(a.model_kw) if a.model_kw else {}
    t0 = time.time()
    if a.what == "ablation":
        base = tuple(a.groups.split(",")) if a.groups else GROUPS
        rows = [("ALL", base)] + [(f"-{g}", tuple(x for x in base if x != g)) for g in base]
        if a.solo:
            rows += [(f"only {g}", (g,)) for g in base]
        for name, use in rows:
            r = run_cv(data, use, a.kind, a.cv, mk, gate=not a.no_gate)
            print(f"{name:<14} d={r['n_features']:<5} F1={100*r['f1']:5.1f}%  R={100*r['recall']:5.1f}%  "
                  f"P={100*r['precision']:5.1f}%  stillFP={r['still_fp']:<4} AP={r['ap']:.3f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    elif a.what == "scaling":
        use = tuple(a.groups.split(",")) if a.groups else GROUPS
        for frac in (0.25, 0.5, 1.0):
            f1s = []
            for seed in range(3 if frac < 1.0 else 1):
                r = run_cv(data, use, a.kind, a.cv, mk, gate=not a.no_gate, frac=frac, seed=seed)
                f1s.append(r["f1"])
            print(f"frac={frac:<5} keys={int(frac*len(data.kt_all)):<5} "
                  f"F1={100*np.mean(f1s):5.1f}% +-{100*np.std(f1s):.1f}  [{time.time()-t0:.0f}s]", flush=True)
    elif a.what == "importance":
        use = tuple(a.groups.split(",")) if a.groups else GROUPS
        X, names = data.X(use), data.names(use)
        y = labels(data.t, data.kt_all, a.label_half)
        m = make_model(a.kind, **{**mk, "importance_type": "gain"} if a.kind == "lgbm" else mk).fit(X, y)
        imp = np.asarray(m.feature_importances_, dtype=float)
        tot = imp.sum() or 1.0
        per = {}
        for n, v in zip(names, imp):
            per[n.split("[")[0]] = per.get(n.split("[")[0], 0.0) + v
        print("per-group gain share:")
        for g, v in sorted(per.items(), key=lambda kv: -kv[1]):
            print(f"  {g:<8} {100*v/tot:5.1f}%  ({(np.array([n.split('[')[0] for n in names]) == g).sum()} cols)")
        print("top-30 single features (gain share):")
        for i in np.argsort(-imp)[:30]:
            print(f"  {names[i]:<16} {100*imp[i]/tot:.2f}%  {describe(names[i])}")
    elif a.what == "calib":
        use = tuple(a.groups.split(",")) if a.groups else GROUPS
        r = run_cv(data, use, a.kind, a.cv, mk, gate=not a.no_gate)
        print(reliability(r["y"], r["oof"]))
    return 0


COLMAP: dict[str, list[str]] = {}


def describe(name: str) -> str:
    """Human label for a `group[i]` column; built lazily from the same order as build_groups."""
    g, i = name.split("[")[0], int(name.split("[")[1][:-1])
    if not COLMAP:
        COLMAP.update(_colmap())
    lst = COLMAP.get(g, [])
    return lst[i] if i < len(lst) else ""


def _colmap() -> dict[str, list[str]]:
    tipn = ["thumb", "index", "middle", "ring", "pinky"]
    per = {
        "pose": [f"rel_wrist_j{j}_{c}" for j in range(1, 21) for c in "xy"]
               + [f"flex2d_{f}" for f in tipn] + [f"pipang2d_{f}" for f in tipn],
        "depth": [f"zrel_j{j}" for j in range(1, 21)]
                + [f"world_j{j}_{c}" for j in range(1, 21) for c in "xyz"],
        "wflex": [f"wlen_{f}" for f in tipn] + [f"wpipang_{f}" for f in tipn]
                + [f"wplane_{f}" for f in tipn]
                + [f"d_wlen_{f}" for f in tipn] + [f"d_wpipang_{f}" for f in tipn]
                + [f"d_wplane_{f}" for f in tipn],
        "palm": ["palmv_x", "palmv_y", "palma_x", "palma_y", "palm_speed",
                 "palmwv_x", "palmwv_y", "palmwv_z", "palmw_speed",
                 "wristv_x", "wristv_y", "wrist_speed"],
        "rest": [f"restdisp_{f}_{c}" for f in tipn for c in "xy"] + [f"restdisp_z_{f}" for f in tipn]
                + ["restdisp_wristy", "restdisp_wristwz"] + [f"restdisp_mag_{f}" for f in tipn],
        "still": [f"energy_{w}" for w in (15, 31, 61, 121)] + [f"peak_energy_{w}" for w in (15, 31, 61, 121)]
                + [f"energy2_{w}" for w in (15, 31, 61, 121)] + ["energy_now", "time_since_motion"]
                + ["height_wristy", "height_wristwz"] + [f"height_{f}" for f in tipn]
                + [f"tipenergy_{f}" for f in tipn],
    }
    core = [f"tipy_{f}" for f in tipn] + [f"d_tipy_{f}" for f in tipn] \
        + [f"tipz_{f}" for f in tipn] + [f"wlen_{f}" for f in tipn] + ["palm_speed"]
    per["ctx"] = [f"{c}@{sgn}{k}" for sgn in ("-", "+") for k in CTX_LAGS for c in core]
    base = np.hstack([np.array(per["pose"]), np.array(per["depth"])]).tolist()
    per["deriv"] = [f"{p}_{n}" for p in ("d1", "d2", "d3") for n in base]
    return {g: [f"{s}/{n}" for s in SIDES for n in v] for g, v in per.items()}


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> str:
    edges = np.linspace(0, 1, bins + 1)
    lines = [f"brier={brier_score_loss(y, p):.5f}  base_rate={y.mean():.4f}"]
    ece = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if not m.any():
            continue
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
        lines.append(f"  [{edges[i]:.1f},{edges[i+1]:.1f}) n={m.sum():<6} "
                     f"pred={p[m].mean():.3f} obs={y[m].mean():.3f}")
    lines.append(f"ECE={ece:.4f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.taps_gb", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--sessions", type=Path, nargs="+", required=True)
    tr.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    pr = sub.add_parser("predict")
    pr.add_argument("session", type=Path)
    pr.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    pr.add_argument("--out", type=Path, default=None)
    ex = sub.add_parser("experiment")
    ex.add_argument("--sessions", type=Path, nargs="+", required=True)
    ex.add_argument("--what", default="ablation", choices=("ablation", "scaling", "calib", "importance"))
    ex.add_argument("--solo", action="store_true")
    for s in (tr, ex):
        s.add_argument("--kind", default="lgbm", choices=("lgbm", "hgb"))
        s.add_argument("--cv", default="block", choices=("block", "session"))
        s.add_argument("--groups", default=None, help="comma-separated subset of feature groups")
        s.add_argument("--model-kw", default=None, help="JSON dict of model overrides")
        s.add_argument("--no-gate", action="store_true")
        s.add_argument("--label-half", type=float, default=LABEL_HALF_WIDTH_S)
    a = ap.parse_args(argv)
    return {"train": cmd_train, "predict": cmd_predict, "experiment": cmd_experiment}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
