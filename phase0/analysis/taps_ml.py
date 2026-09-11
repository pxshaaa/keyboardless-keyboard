"""Learned per-frame keypress detector; trains ONLY on the harness's first-60% slice.
Run: python -m phase0.analysis.taps_ml train <session> / predict <session> [--model M] [--out O]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import find_peaks
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from phase0.analysis.detect_taps import FINGERTIP_JOINTS, load_landmarks
from phase0.analysis.eval_taps import keydown_times, score

DEFAULT_MODEL = Path("models/taps_ml.joblib")
LABEL_HALF_WIDTH_S = 0.033
REFRACTORY_FRAMES = 5  # 80 ms @ 60 fps
SIDES = ("Left", "Right")
MCP = {4: 2, 8: 5, 12: 9, 16: 13, 20: 17}
PIP = {4: 3, 8: 6, 12: 10, 16: 14, 20: 18}


# ---------------------------------------------------------------- features
CHANNELS = ("x", "y", "conf", "z", "wx", "wy", "wz")


def load_frames(session: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (frame_i [F], t [F], P [F,2,21,7]) keyed by handedness; NaN if absent or column missing."""
    tb = load_landmarks(session)
    have = [ch for ch in CHANNELS if ch in tb.schema.names]
    c = {n: np.asarray(tb.column(n)) for n in ("i", "t", "hand", "handedness", "joint", *have)}
    frames, inv = np.unique(c["i"], return_inverse=True)
    t = np.zeros(len(frames))
    t[inv] = c["t"]
    P = np.full((len(frames), 2, 21, len(CHANNELS)), np.nan)
    side = np.array([SIDES.index(h) for h in c["handedness"]])
    # two rows claiming the same handedness in a frame: fall back to hand slot
    key = frames[inv] * 2 + side
    _, first = np.unique(key, return_index=True)
    dup = np.ones(len(key), bool)
    dup[first] = False
    side = np.where(dup, c["hand"], side)
    for k, ch in enumerate(CHANNELS):
        if ch in have:
            P[inv, side, c["joint"], k] = c[ch]
    return frames, t, P


def hand_base_features(H: np.ndarray) -> np.ndarray:
    """H [F,21,3] one hand -> [F,50]: 20 joints rel. wrist / span, 5 tip flexion, 5 PIP angles."""
    wrist = H[:, 0, :2]
    span = np.linalg.norm(H[:, 9, :2] - wrist, axis=1)[:, None]
    span = np.where(span > 1e-6, span, np.nan)
    rel = (H[:, 1:, :2] - wrist[:, None, :]) / span[:, None, :]
    flex, ang = [], []
    for tip in FINGERTIP_JOINTS:
        flex.append(np.linalg.norm(H[:, tip, :2] - H[:, MCP[tip], :2], axis=1) / span[:, 0])
        a = H[:, MCP[tip], :2] - H[:, PIP[tip], :2]
        b = H[:, tip, :2] - H[:, PIP[tip], :2]
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
        ang.append(np.arccos(np.clip(cos, -1, 1)))
    return np.hstack([rel.reshape(len(H), -1), np.stack(flex, 1), np.stack(ang, 1)])


def _shift(X: np.ndarray, k: int) -> np.ndarray:
    Y = np.full_like(X, np.nan)
    if k > 0:
        Y[k:] = X[:-k]
    elif k < 0:
        Y[:k] = X[-k:]
    else:
        Y[:] = X
    return Y


def temporal(X: np.ndarray) -> np.ndarray:
    d1 = (_shift(X, -1) - _shift(X, 1)) / 2
    d2 = _shift(X, -1) - 2 * X + _shift(X, 1)
    d3 = (_shift(X, -3) - _shift(X, 3)) / 6
    return np.hstack([X, d1, d2, d3])


def depth_base_features(H: np.ndarray) -> np.ndarray:
    """H [F,21,7] -> [F,80]: image z rel. wrist / xy-span (20) + world xyz rel. wrist / world span (60)."""
    span = np.linalg.norm(H[:, 9, :2] - H[:, 0, :2], axis=1)[:, None]
    z = (H[:, 1:, 3] - H[:, :1, 3]) / span
    W = H[:, :, 4:7]
    wspan = np.linalg.norm(W[:, 9] - W[:, 0], axis=1)[:, None, None]
    wrel = (W[:, 1:] - W[:, :1]) / np.where(wspan > 1e-6, wspan, np.nan)
    return np.hstack([z, wrel.reshape(len(H), -1)])


def build_features(P: np.ndarray, static: bool = False, depth: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """-> (X [F,D], flexvel [F,2,5]); static = keep pose itself, not only its temporal diffs."""
    per_side, fv = [], []
    for s in range(2):
        base = hand_base_features(P[:, s])
        if depth and not np.all(np.isnan(P[:, s, :, 3:])):
            base = np.hstack([base, depth_base_features(P[:, s])])
        T = temporal(base)
        if not static:
            T = T[:, base.shape[1]:]
        # absolute tip velocity / span is the baseline's signal; wrist-relative alone loses it
        span = np.linalg.norm(P[:, s, 9, :2] - P[:, s, 0, :2], axis=1)[:, None]
        tips = list(FINGERTIP_JOINTS)
        chans = [1, 3, 6] if depth else [1]
        absv = [(_shift(P[:, s, tips, ch] / span, -1) - _shift(P[:, s, tips, ch] / span, 1)) / 2 for ch in chans]
        per_side.append(np.hstack([T, *absv]))
        fv.append((_shift(base[:, 40:45], -1) - _shift(base[:, 40:45], 1)) / 2)
    return np.hstack(per_side), np.stack(fv, 1)


def labels(t: np.ndarray, kt: np.ndarray) -> np.ndarray:
    if len(kt) == 0:
        return np.zeros(len(t), bool)
    d = np.abs(t[:, None] - kt[None, :]).min(1)
    return d <= LABEL_HALF_WIDTH_S


# ---------------------------------------------------------------- events
def smooth_prob(p: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return p
    return np.convolve(p, np.ones(w) / w, mode="same")


def pick_events(p: np.ndarray, thr: float) -> np.ndarray:
    idx, _ = find_peaks(p, height=thr, distance=REFRACTORY_FRAMES)
    return idx


def tune(p: np.ndarray, t: np.ndarray, kt: np.ndarray) -> tuple[float, int, float]:
    best = (0.0, 0.3, 1)
    for w in (1, 3, 5):
        ps = smooth_prob(p, w)
        for thr in np.arange(0.05, 0.95, 0.025):
            f1 = score(kt, t[pick_events(ps, thr)])["f1"]
            if f1 > best[0]:
                best = (f1, float(thr), w)
    return best[1], best[2], best[0]


def events_to_records(idx: np.ndarray, frames, t, P, flexvel) -> list[dict]:
    out = []
    for k in idx:
        m = np.abs(np.nan_to_num(flexvel[k], nan=-1.0))
        s, f = np.unravel_index(int(np.argmax(m)), m.shape)
        tip = FINGERTIP_JOINTS[f]
        out.append({
            "t": float(t[k]), "hand": int(s), "finger": int(tip),
            "x": float(np.nan_to_num(P[k, s, tip, 0])), "y": float(np.nan_to_num(P[k, s, tip, 1])),
            "conf": float(np.nan_to_num(P[k, s, tip, 2])), "i": int(frames[k]),
        })
    return out


# ---------------------------------------------------------------- models
def make_model(kind: str):
    if kind == "lr":
        return make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0), StandardScaler(),
                             LogisticRegression(C=0.02, class_weight="balanced", max_iter=3000))
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3,
                                          min_samples_leaf=20, l2_regularization=1.0,
                                          class_weight="balanced", random_state=0)


def fit_predict(kind, Xtr, ytr, Xte):
    m = make_model(kind)
    m.fit(Xtr, ytr)
    return m, m.predict_proba(Xte)[:, 1]


def blocked_cv(kind, X, y, t, kt, n=5) -> dict:
    """Time-contiguous folds; per-fold threshold tuned on that fold's train part. Also returns OOF probs."""
    edges = np.linspace(0, len(X), n + 1).astype(int)
    aps, f1s = [], []
    oof = np.zeros(len(X))
    for f in range(n):
        te = np.zeros(len(X), bool)
        te[edges[f]:edges[f + 1]] = True
        m, ptr = fit_predict(kind, X[~te], y[~te], X[~te])
        thr, w, _ = tune(ptr, t[~te], kt[(kt >= t[~te].min()) & (kt <= t[~te].max())])
        oof[te] = m.predict_proba(X[te])[:, 1]
        pte = smooth_prob(m.predict_proba(X)[:, 1], w)[te]
        aps.append(average_precision_score(y[te], pte) if y[te].any() else np.nan)
        lo, hi = t[te].min(), t[te].max()
        f1s.append(score(kt, t[te][pick_events(pte, thr)], lo, hi + 1e-6)["f1"])
    return {"ap": aps, "f1": f1s, "oof": oof}


def load_training(sessions, static, depth):
    """Per-session features (diffs never cross a session boundary), concatenated on the shared clock."""
    Xs, ts, ks = [], [], []
    for sess in sessions:
        _, t, P = load_frames(sess)
        X, _ = build_features(P, static=static, depth=depth)
        Xs.append(X)
        ts.append(t)
        ks.append(keydown_times(sess))
    t = np.concatenate(ts)
    order = np.argsort(t, kind="stable")
    return np.vstack(Xs)[order], t[order], np.sort(np.concatenate(ks))


def cmd_train(a) -> int:
    X, t, kt = load_training(a.sessions, a.static, not a.no_depth)
    y = labels(t, kt)
    cut = kt.min() + a.split * (kt.max() - kt.min())
    use_all = a.all or len(a.sessions) > 1
    tr = t < cut if not use_all else np.ones(len(t), bool)
    ktr = kt[kt < cut] if not use_all else kt
    print(f"frames={len(t)} features={X.shape[1]} train_frames={tr.sum()} pos={y[tr].sum()} "
          f"holdout_frames={(~tr).sum()} cut={cut:.3f}")

    results = {}
    for kind in ("lr", "hgb"):
        cv = blocked_cv(kind, X[tr], y[tr], t[tr], ktr)
        m = make_model(kind).fit(X[tr], y[tr])
        # threshold from out-of-fold probs: in-sample probs are over-fit and pick a too-high thr
        thr, w, f1tr = tune(cv["oof"], t[tr], ktr)
        results[kind] = (m, thr, w, f1tr, cv)
        print(f"[{kind}] OOF-tuned F1={100*f1tr:.1f}% thr={thr:.3f} smooth={w}  "
              f"CV AP={np.nanmean(cv['ap']):.3f}±{np.nanstd(cv['ap']):.3f} "
              f"CV eventF1={np.mean(cv['f1']):.3f}±{np.std(cv['f1']):.3f} folds={np.round(cv['f1'], 2).tolist()}")

    # model selection by blocked-CV event F1 only -- hold-out never consulted
    kind = max(results, key=lambda k: np.mean(results[k][4]["f1"]))
    m, thr, w, _, _ = results[kind]
    a.model.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"kind": kind, "model": m, "thr": thr, "smooth": w, "static": a.static, "depth": not a.no_depth}
    joblib.dump(bundle, a.model)
    print(f"selected={kind}  saved {a.model}")
    a.session = a.sessions[0]
    return cmd_predict(a, bundle=bundle)


def cmd_predict(a, bundle=None) -> int:
    b = bundle or joblib.load(a.model)
    kind, m, thr, w = b["kind"], b["model"], b["thr"], b["smooth"]
    frames, t, P = load_frames(a.session)
    X, flexvel = build_features(P, static=b["static"], depth=b["depth"])
    p = smooth_prob(m.predict_proba(X)[:, 1], w)
    recs = events_to_records(pick_events(p, thr), frames, t, P, flexvel)
    out = a.out or a.session / "taps_ml.jsonl"
    with open(out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    dist = {}
    for r in recs:
        k = f"{SIDES[r['hand']]}/{r['finger']}"
        dist[k] = dist.get(k, 0) + 1
    print(f"[{kind}] wrote {len(recs)} events -> {out}")
    print("finger attribution (unlabelled, by max |flexion velocity|):", dict(sorted(dist.items())))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.taps_ml", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "predict"):
        s = sub.add_parser(name)
        if name == "train":
            s.add_argument("sessions", type=Path, nargs="+", help=">1 session implies --all")
        else:
            s.add_argument("session", type=Path)
        s.add_argument("--model", type=Path, default=DEFAULT_MODEL)
        s.add_argument("--out", type=Path, default=None)
        s.add_argument("--split", type=float, default=0.6)
        s.add_argument("--all", action="store_true", help="train on the whole session (cross-session test)")
        s.add_argument("--static", action="store_true", help="also feed raw pose, not only its diffs")
        s.add_argument("--no-depth", action="store_true", help="ignore z/world columns")
    a = ap.parse_args(argv)
    return cmd_train(a) if a.cmd == "train" else cmd_predict(a)


if __name__ == "__main__":
    sys.exit(main())
