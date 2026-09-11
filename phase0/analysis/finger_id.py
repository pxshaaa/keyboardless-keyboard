"""Tap ATTRIBUTION: which (hand, finger) pressed, from the landmark window around a tap.
Run: python -m phase0.analysis.finger_id {train --sessions A B | predict S --taps F --out F | snr --sessions A}"""

# Feed is rotated 90 deg (keyboard L->R runs along image +y); MediaPipe "Left" IS
# the physical left hand here -- verified against a decoded frame, no selfie mirror.

# German QWERTZ + macOS reports the produced CHARACTER, so 'y' is the bottom-left
# key (left pinky) and 'z' is right index. Confirmed: 'y' shows peak left flexion.

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

FINGERTIP_JOINTS = (4, 8, 12, 16, 20)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
HAND_NAMES = ("Left", "Right")
FINGER_CHAIN = {4: (1, 2, 3, 4), 8: (5, 6, 7, 8), 12: (9, 10, 11, 12),
                16: (13, 14, 15, 16), 20: (17, 18, 19, 20)}

FPS = 60.0
HALF_WINDOW = 9        # +-9 frames = +-150 ms, covers a 100-150 ms down-up excursion
SMOOTH_WINDOW = 7
SMOOTH_POLY = 2
MAX_SNAP_S = 0.050     # a keydown further than this from any frame is unusable

# character -> (hand, finger) under German QWERTZ touch typing; thumb = either hand
_ROWS = {
    ("Left", 4): "1qay", ("Left", 3): "2wsx", ("Left", 2): "3edc",
    ("Left", 1): "45rtfgvb",
    ("Right", 1): "67zuhjnm", ("Right", 2): "8ik", ("Right", 3): "9ol",
    ("Right", 4): "0p",
}
KEY_LABEL: dict[str, tuple[str, int]] = {}
for _hf, _ks in _ROWS.items():
    KEY_LABEL.update({k: _hf for k in _ks})
KEY_LABEL["space"] = ("Either", 0)

# long pinky reaches; standard mapping but a non-touch-typist often uses another finger
REACH_KEYS = {"backspace": ("Right", 4), "enter": ("Right", 4), "tab": ("Left", 4)}
DROP_KEYS = {"unknown", "shift", "ctrl", "alt", "cmd", "esc"}

CLASSES9 = [("Either", 0)] + [(h, f) for h in HAND_NAMES for f in (1, 2, 3, 4)]
CLASSES8 = [(h, f) for h in HAND_NAMES for f in (1, 2, 3, 4)]


def cls_name(c: tuple[str, int]) -> str:
    return "thumb" if c[0] == "Either" else f"{c[0][0]}-{FINGER_NAMES[c[1]]}"


# ---------------------------------------------------------------- loading

def load_hands(session: Path) -> dict[str, dict]:
    """Per handedness: t[n], P[n,21,3] image px (x,y,z), W[n,21,3] world m, conf[n,21]."""
    import pyarrow.parquet as pq

    tb = pq.read_table(session / "landmarks.parquet")
    c = {n: np.asarray(tb.column(n)) for n in tb.column_names}
    out: dict[str, dict] = {}
    for hd in HAND_NAMES:
        m = c["handedness"] == hd
        if not m.any():
            continue
        frames, inv = np.unique(c["i"][m], return_inverse=True)
        n = len(frames)
        P = np.full((n, 21, 3), np.nan)
        Wl = np.full((n, 21, 3), np.nan)
        conf = np.zeros((n, 21))
        t = np.zeros(n)
        j = c["joint"][m]
        for k, name in enumerate(("x", "y", "z")):
            P[inv, j, k] = c[name][m]
        for k, name in enumerate(("wx", "wy", "wz")):
            Wl[inv, j, k] = c[name][m]
        conf[inv, j] = c["conf"][m]
        t[inv] = c["t"][m]
        ok = ~np.isnan(P).any(axis=(1, 2))
        o = np.argsort(t[ok], kind="stable")
        out[hd] = dict(t=t[ok][o], P=P[ok][o], W=Wl[ok][o], conf=conf[ok][o])
    return out


def load_keydowns(session: Path) -> list[dict]:
    rows = [json.loads(l) for l in (session / "keys.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rows if r.get("event") == "down"]


# ---------------------------------------------------------------- signals

def _frame_axes(A: np.ndarray) -> np.ndarray:
    """Right-handed orthonormal hand frame per row of A[n,21,3]: forward/palm/normal."""
    e1 = A[:, 9] - A[:, 0]
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12
    g = A[:, 17] - A[:, 5]
    e2 = g - (g * e1).sum(1, keepdims=True) * e1
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True) + 1e-12
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=1)


def _pip_angle(A: np.ndarray, chain: tuple[int, ...]) -> np.ndarray:
    a = A[:, chain[0]] - A[:, chain[1]]
    b = A[:, chain[2]] - A[:, chain[1]]
    ca = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return np.arccos(np.clip(ca, -1, 1))


def hand_signals(h: dict) -> dict[str, np.ndarray]:
    """Named per-finger time series, each [n,5]. Prefix marks the ablation group."""
    P, Wd = h["P"], h["W"]
    n = len(P)
    span2 = np.linalg.norm(P[:, 9, :2] - P[:, 0, :2], axis=1)[:, None] + 1e-6
    span3 = np.linalg.norm(P[:, 9] - P[:, 0], axis=1)[:, None] + 1e-6
    wspan = np.linalg.norm(Wd[:, 9] - Wd[:, 0], axis=1)[:, None] + 1e-6

    # 2D hand frame from image x,y only -- the depth-free view of the same geometry
    f2 = P[:, 9, :2] - P[:, 0, :2]
    f2 = f2 / (np.linalg.norm(f2, axis=1, keepdims=True) + 1e-12)
    p2 = np.stack([-f2[:, 1], f2[:, 0]], axis=1)
    R3 = _frame_axes(P)
    RW = _frame_axes(Wd)

    sig: dict[str, np.ndarray] = {}
    tips = list(FINGERTIP_JOINTS)
    d2 = P[:, tips, :2] - P[:, :1, :2]
    sig["img2:a1"] = (d2 * f2[:, None, :]).sum(2) / span2
    sig["img2:a2"] = (d2 * p2[:, None, :]).sum(2) / span2
    sig["img2:fl"] = np.linalg.norm(d2, axis=2) / span2

    d3 = P[:, tips] - P[:, :1]
    loc = np.einsum("nkj,nfj->nfk", R3, d3) / span3[:, :, None]
    sig["imgz:a3"] = loc[:, :, 2]
    sig["imgz:fl"] = np.linalg.norm(d3, axis=2) / span3
    sig["imgz:zr"] = (P[:, tips, 2] - P[:, :1, 2]) / span3

    dw = Wd[:, tips] - Wd[:, :1]
    wl = np.einsum("nkj,nfj->nfk", RW, dw) / wspan[:, :, None]
    for k in range(3):
        sig[f"world:b{k + 1}"] = wl[:, :, k]
    sig["world:fl"] = np.linalg.norm(dw, axis=2) / wspan

    sig["img2:pip"] = np.stack(
        [_pip_angle(np.concatenate([P[:, :, :2], np.zeros((n, 21, 1))], 2), FINGER_CHAIN[j])
         for j in FINGERTIP_JOINTS], axis=1)
    sig["world:pip"] = np.stack([_pip_angle(Wd, FINGER_CHAIN[j]) for j in FINGERTIP_JOINTS], axis=1)

    w = min(SMOOTH_WINDOW, n if n % 2 else n - 1)
    if w >= SMOOTH_POLY + 2:
        sig = {k: savgol_filter(v, w, SMOOTH_POLY, axis=0) for k, v in sig.items()}
    return sig


def prep(session: Path) -> dict[str, dict]:
    hands = load_hands(session)
    for h in hands.values():
        s = hand_signals(h)
        h["sig"] = s
        h["rest"] = {k: np.median(v, axis=0) for k, v in s.items()}
        dt = np.median(np.diff(h["t"])) if len(h["t"]) > 1 else 1 / FPS
        h["vel"] = {k: np.gradient(v, axis=0) / dt for k, v in s.items()}
        h["acc"] = {k: np.gradient(v, axis=0) / dt for k, v in h["vel"].items()}
    return hands


# ---------------------------------------------------------------- features

_STATS = ("d_at", "d_min", "d_max", "rest", "v_min", "v_max", "a_ab")


def _summarize(q: np.ndarray, v: np.ndarray, a: np.ndarray, rest: np.ndarray) -> np.ndarray:
    """q,v,a are [2W+1,5] window slices; returns [len(_STATS),5]."""
    base = q[:3].mean(axis=0)
    mid = HALF_WINDOW
    return np.stack([q[mid] - base, q.min(0) - base, q.max(0) - base, q[mid] - rest,
                     v.min(0), v.max(0), np.abs(a).max(0)])


def featurize_tap(hands: dict[str, dict], t_tap: float) -> tuple[np.ndarray, list[str]] | None:
    vals: list[float] = []
    names: list[str] = []
    any_hand = False
    for hd in HAND_NAMES:
        h = hands.get(hd)
        idx = None
        if h is not None and len(h["t"]):
            i = int(np.clip(np.searchsorted(h["t"], t_tap), 0, len(h["t"]) - 1))
            if abs(h["t"][i] - t_tap) <= MAX_SNAP_S and i - HALF_WINDOW >= 0 \
                    and i + HALF_WINDOW + 1 <= len(h["t"]):
                idx = i
        present = idx is not None
        any_hand |= present
        lo, hi = (idx - HALF_WINDOW, idx + HALF_WINDOW + 1) if present else (0, 0)
        keys = sorted(h["sig"]) if h is not None else sorted(SIGNAL_KEYS)
        for sk in keys:
            if present:
                S = _summarize(h["sig"][sk][lo:hi], h["vel"][sk][lo:hi],
                               h["acc"][sk][lo:hi], h["rest"][sk])
            else:
                S = np.full((len(_STATS), 5), np.nan)
            grp = sk.split(":")[0]
            for si, st in enumerate(_STATS):
                row = S[si]
                vals.extend(row.tolist())
                names.extend(f"{grp}:{hd}:{sk}:{st}:{FINGER_NAMES[f]}" for f in range(5))
                # inter-finger contrast + which finger moved most: the rank signal
                cen = row - np.nanmean(row) if present else np.full(5, np.nan)
                vals.extend(cen.tolist())
                names.extend(f"{grp}:{hd}:{sk}:{st}:rel:{FINGER_NAMES[f]}" for f in range(5))
        # absolute image position: where on the keyboard this hand and its tips sit
        if present:
            P = h["P"][idx]
            abs_v = [P[0, 0] / 1280, P[0, 1] / 720] + \
                    [c for j in FINGERTIP_JOINTS for c in
                     (P[j, 0] / 1280, P[j, 1] / 720, P[j, 2] / 100)]
            hv = np.linalg.norm(np.diff(h["P"][lo:hi, 0], axis=0), axis=1).mean()
            abs_v += [float(h["conf"][idx].mean()), hv, 1.0]
        else:
            abs_v = [np.nan] * 19 + [0.0]
        vals.extend(abs_v)
        names.extend([f"abs:{hd}:wrist_x", f"abs:{hd}:wrist_y"] +
                     [f"abs:{hd}:tip{FINGER_NAMES[f]}_{c}" for f in range(5) for c in "xyz"] +
                     [f"abs:{hd}:conf", f"abs:{hd}:wrist_speed", f"abs:{hd}:present"])
    if not any_hand:
        return None
    return np.array(vals, dtype=float), names


SIGNAL_KEYS = ("img2:a1", "img2:a2", "img2:fl", "img2:pip", "imgz:a3", "imgz:fl",
               "imgz:zr", "world:b1", "world:b2", "world:b3", "world:fl", "world:pip")


def build_dataset(session: Path, include_reach: bool = False) -> dict:
    hands = prep(session)
    X, y, keys, times = [], [], [], []
    names: list[str] = []
    dropped = Counter()
    for k in load_keydowns(session):
        key = k["key"]
        if key in DROP_KEYS:
            dropped["not_allowlisted"] += 1
            continue
        lab = KEY_LABEL.get(key) or (REACH_KEYS.get(key) if include_reach else None)
        if lab is None:
            dropped["reach_or_unmapped"] += 1
            continue
        got = featurize_tap(hands, k["t"])
        if got is None:
            dropped["no_landmark_window"] += 1
            continue
        f, names = got
        X.append(f)
        y.append(lab)
        keys.append(key)
        times.append(k["t"])
    return dict(X=np.array(X), y=y, keys=keys, t=np.array(times),
                names=names, dropped=dropped, hands=hands)


def group_mask(names: list[str], groups: set[str]) -> np.ndarray:
    return np.array([n.split(":")[0] in groups for n in names])


ABLATIONS = {
    "all": {"img2", "imgz", "world", "abs"},
    "no_depth": {"img2", "abs2d"},          # abs2d handled below: abs minus z columns
    "depth_only": {"imgz", "world"},
    "no_abs": {"img2", "imgz", "world"},
    "img2d_only": {"img2"},
    "world_only": {"world"},
    "abs_only": {"abs"},
}


def ablation_mask(names: list[str], key: str) -> np.ndarray:
    if key == "no_depth":
        return np.array([n.split(":")[0] == "img2" or
                         (n.split(":")[0] == "abs" and not n.endswith("_z"))
                         for n in names])
    return group_mask(names, ABLATIONS[key])


# ---------------------------------------------------------------- model

def make_model(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=80, learning_rate=0.2, max_leaf_nodes=8, min_samples_leaf=15,
        l2_regularization=1.0, max_features=0.2, early_stopping=False, random_state=seed)


def _fit_predict(Xtr, ytr, Xte, seed=0):
    m = make_model(seed)
    m.fit(Xtr, ytr)
    return m, m.predict_proba(Xte), list(m.classes_)


def topk_acc(proba: np.ndarray, classes: list, y: list, k: int) -> float:
    if not len(y):
        return float("nan")
    order = np.argsort(-proba, axis=1)[:, :k]
    hit = [y[i] in [classes[j] for j in order[i]] for i in range(len(y))]
    return float(np.mean(hit))


def confusion(classes: list, y_true: list, y_pred: list) -> np.ndarray:
    ix = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)), dtype=int)
    for a, b in zip(y_true, y_pred):
        M[ix[a], ix[b]] += 1
    return M


def fmt_conf(classes: list, M: np.ndarray, label) -> str:
    w = max(9, max(len(label(c)) for c in classes) + 1)
    out = [" " * w + "".join(f"{label(c)[:8]:>9s}" for c in classes) + "   recall"]
    for i, c in enumerate(classes):
        rec = M[i, i] / M[i].sum() if M[i].sum() else float("nan")
        out.append(f"{label(c):<{w}s}" + "".join(f"{v:9d}" for v in M[i]) +
                   f"   {rec:6.3f}  (n={M[i].sum()})")
    return "\n".join(out)


# ---------------------------------------------------------------- SNR

def snr_report(sessions: list[Path]) -> str:
    """Press excursion of the labelled finger vs the tracker's own frame-to-frame jitter."""
    lines = []
    per_sig: dict[str, list[tuple[float, float, float]]] = {}
    agree: dict[str, list[tuple[float, float, int]]] = {}
    for sess in sessions:
        data = build_dataset(sess)
        hands = data["hands"]
        kd = np.array([k["t"] for k in load_keydowns(sess)])
        for hd in HAND_NAMES:
            h = hands.get(hd)
            if h is None:
                continue
            t = h["t"]
            near = np.zeros(len(t), dtype=bool)
            for tk in kd:
                lo = np.searchsorted(t, tk - 0.25)
                hi = np.searchsorted(t, tk + 0.25)
                near[lo:hi] = True
            quiet = ~near
            for sk, S in h["sig"].items():
                # jitter = residual of the smoothed signal about a 5-frame local mean, quiet only
                if quiet.sum() < 50:
                    continue
                Q = S[quiet]
                res = Q[1:-1] - 0.5 * (Q[:-2] + Q[2:])
                jit = float(np.median(np.abs(res)) * 1.4826)
                per_sig.setdefault(sk, []).append((jit, np.nan, np.nan))
        # press excursion of the labelled finger, and of the other fingers of the same hand
        for sk in SIGNAL_KEYS:
            sig_amp, oth_amp, rank_in, rank_all = [], [], [], []
            for xi, lab in enumerate(data["y"]):
                if lab[0] == "Either":
                    continue
                h = hands.get(lab[0])
                if h is None:
                    continue
                i = int(np.searchsorted(h["t"], data["t"][xi]))
                if i - HALF_WINDOW < 0 or i + HALF_WINDOW + 1 > len(h["t"]):
                    continue
                W = h["sig"][sk][i - HALF_WINDOW:i + HALF_WINDOW + 1]
                base = W[:3].mean(0)
                exc = np.abs(W - base).max(0)
                f = lab[1]
                sig_amp.append(exc[f])
                oth_amp.append(np.delete(exc, f).mean())
                rank_in.append(int(np.argmax(exc[1:]) + 1) == f)
                both = []
                for hd2 in HAND_NAMES:
                    h2 = hands.get(hd2)
                    if h2 is None:
                        both.extend([-np.inf] * 5)
                        continue
                    i2 = int(np.searchsorted(h2["t"], data["t"][xi]))
                    if i2 - HALF_WINDOW < 0 or i2 + HALF_WINDOW + 1 > len(h2["t"]):
                        both.extend([-np.inf] * 5)
                        continue
                    W2 = h2["sig"][sk][i2 - HALF_WINDOW:i2 + HALF_WINDOW + 1]
                    both.extend(np.abs(W2 - W2[:3].mean(0)).max(0).tolist())
                best = int(np.argmax(both))
                rank_all.append(HAND_NAMES[best // 5] == lab[0] and best % 5 == f)
            if sig_amp:
                agree.setdefault(sk, []).append((float(np.mean(rank_in)),
                                                 float(np.mean(rank_all)), len(rank_in)))
                per_sig.setdefault(sk, [])
                per_sig[sk].append((np.nan, float(np.median(sig_amp)), float(np.median(oth_amp))))
    lines.append(f"{'signal':<12s}{'jitter(1f)':>12s}{'press exc':>12s}{'other exc':>12s}"
                 f"{'SNR':>8s}{'contrast':>10s}")
    for sk in SIGNAL_KEYS:
        rows = per_sig.get(sk, [])
        jit = np.nanmedian([r[0] for r in rows]) if rows else np.nan
        sg = np.nanmedian([r[1] for r in rows]) if rows else np.nan
        ot = np.nanmedian([r[2] for r in rows]) if rows else np.nan
        lines.append(f"{sk:<12s}{jit:12.5f}{sg:12.5f}{ot:12.5f}{sg / jit:8.2f}"
                     f"{(sg - ot) / jit:10.2f}")
    lines.append("")
    lines.append(f"{'signal':<12s}{'argmax==label|hand':>22s}{'argmax==label|all10':>22s}{'n':>7s}")
    for sk in SIGNAL_KEYS:
        rows = agree.get(sk, [])
        if not rows:
            continue
        n = sum(r[2] for r in rows)
        a1 = sum(r[0] * r[2] for r in rows) / n
        a2 = sum(r[1] * r[2] for r in rows) / n
        lines.append(f"{sk:<12s}{a1:22.3f}{a2:22.3f}{n:7d}")
    lines.append("chance for argmax|hand (4 non-thumb fingers) = 0.250; argmax|all10 = 0.100.")
    lines.append("SNR = pressing finger's window excursion / per-frame tracker jitter.")
    lines.append("contrast = (pressing finger - mean other finger of same hand) / jitter;")
    lines.append("this is the quantity finger IDENTITY actually depends on.")
    return "\n".join(lines)


# ---------------------------------------------------------------- reporting

TAGS = ("finger9", "finger8", "hand2", "fingerL4", "fingerR4")


def _row(ab: str, r: dict) -> str:
    cells = []
    for tag in TAGS:
        if f"{tag}_top1" not in r:
            cells.append(f"{'-':>22s}")
        elif tag == "hand2":
            cells.append(f"{r[f'{tag}_top1']:>22.3f}")
        else:
            cells.append(f"{r[f'{tag}_top1']:>11.3f}{r[f'{tag}_top2']:>11.3f}")
    return f"{ab:<12s}" + "".join(cells)


def evaluate(train: list[dict], test: dict, verbose: bool = True) -> dict:
    names = train[0]["names"]
    Xtr = np.vstack([d["X"] for d in train])
    ytr = [l for d in train for l in d["y"]]
    Xte, yte = test["X"], test["y"]
    res: dict[str, dict] = {}
    for ab in ("all", "no_depth", "depth_only", "no_abs", "img2d_only", "world_only", "abs_only"):
        m = ablation_mask(names, ab)
        r: dict[str, float] = {}
        # 9-class (8 fingers + thumb/space), 8-class (space dropped), hand-only
        for tag, keep, lab in (
            ("finger9", np.ones(len(yte), bool), cls_name),
            ("finger8", np.array([l[0] != "Either" for l in yte]), cls_name),
            ("hand2", np.array([l[0] != "Either" for l in yte]), lambda l: l[0]),
            ("fingerL4", np.array([l[0] == "Left" for l in yte]), cls_name),
            ("fingerR4", np.array([l[0] == "Right" for l in yte]), cls_name),
        ):
            if tag == "finger9":
                tr_keep = np.ones(len(ytr), bool)
            elif tag in ("finger8", "hand2"):
                tr_keep = np.array([l[0] != "Either" for l in ytr])
            else:
                tr_keep = np.array([l[0] == tag[6] + ("eft" if tag[6] == "L" else "ight")
                                    for l in ytr])
            ytr_t = [lab(l) for l, k in zip(ytr, tr_keep) if k]
            yte_t = [lab(l) for l, k in zip(yte, keep) if k]
            if len(set(ytr_t)) < 2 or not yte_t:
                continue
            _, proba, classes = _fit_predict(Xtr[tr_keep][:, m], ytr_t, Xte[keep][:, m])
            r[f"{tag}_top1"] = topk_acc(proba, classes, yte_t, 1)
            r[f"{tag}_top2"] = topk_acc(proba, classes, yte_t, 2)
            r[f"{tag}_n"] = len(yte_t)
            maj = Counter(ytr_t).most_common(1)[0][0]
            r[f"{tag}_major"] = float(np.mean([y == maj for y in yte_t]))
            r[f"{tag}_chance"] = 1.0 / len(set(ytr_t))
            if ab == "all":
                pred = [classes[i] for i in proba.argmax(1)]
                res.setdefault("_conf", {})[tag] = (sorted(set(ytr_t), key=str), yte_t, pred)
        res[ab] = r
        if verbose:
            print(_row(ab, r), flush=True)
    return res


def _sess(sid: str) -> Path:
    p = Path(sid)
    return p if p.exists() else Path("data/sessions") / sid


def cmd_train(args) -> None:
    train_ids = args.sessions
    test_id = args.test
    tr = [build_dataset(_sess(s), args.include_reach) for s in train_ids]
    te = build_dataset(_sess(test_id), args.include_reach)
    print(f"train sessions: {train_ids}  n={sum(len(d['y']) for d in tr)}")
    print(f"test  session : {test_id}  n={len(te['y'])}")
    print("dropped (train):", dict(sum((d["dropped"] for d in tr), Counter())))
    print("dropped (test) :", dict(te["dropped"]))
    print(f"features       : {len(tr[0]['names'])}")
    print("\nlabel distribution (train):",
          {cls_name(c): n for c, n in Counter(l for d in tr for l in d["y"]).most_common()})
    print("label distribution (test) :",
          {cls_name(c): n for c, n in Counter(te["y"]).most_common()})

    print("\n=== ACCURACY (test session, held out) ===", flush=True)
    tags = TAGS
    print(f"{'features':<12s}" + "".join(
        f"{t:>22s}" for t in ("finger9 t1/t2", "finger8 t1/t2", "hand2 t1",
                              "L-finger t1/t2", "R-finger t1/t2")), flush=True)
    res = evaluate(tr, te)
    r = res["all"]
    print("\nbaselines (same test rows):")
    for tag in tags:
        if f"{tag}_top1" in r:
            print(f"  {tag:<9s} n={r[f'{tag}_n']:4d}  uniform-chance={r[f'{tag}_chance']:.3f}"
                  f"  majority-class={r[f'{tag}_major']:.3f}")

    for tag, (classes, yt, yp) in res.get("_conf", {}).items():
        print(f"\n=== CONFUSION ({tag}, features=all; rows=true, cols=pred) ===")
        print(fmt_conf(classes, confusion(classes, yt, yp), lambda c: c))

    if args.out:
        Xtr = np.vstack([d["X"] for d in tr])
        ytr = [l for d in tr for l in d["y"]]
        m = make_model()
        m.fit(Xtr, [cls_name(l) for l in ytr])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "wb") as fh:
            pickle.dump({"model": m, "names": tr[0]["names"], "mask": None}, fh)
        print(f"\nwrote {args.out}")


def cmd_predict(args) -> None:
    with open(args.model, "rb") as fh:
        blob = pickle.load(fh)
    hands = prep(_sess(args.session))
    taps = [json.loads(l) for l in Path(args.taps).read_text().splitlines() if l.strip()]
    out = []
    for tp in taps:
        got = featurize_tap(hands, tp["t"])
        if got is None:
            out.append({**tp, "pred_hand": None, "pred_finger": None, "pred_conf": 0.0})
            continue
        v = got[0]
        if blob.get("mask") is not None:
            v = v[blob["mask"]]
        p = blob["model"].predict_proba(v[None, :])[0]
        cs = list(blob["model"].classes_)
        j = int(p.argmax())
        out.append({**tp, "pred_class": cs[j], "pred_conf": float(p[j])})
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"wrote {len(out)} predictions to {args.out}")


def cmd_snr(args) -> None:
    print(snr_report([_sess(s) for s in args.sessions]))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="finger_id")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--sessions", nargs="+", required=True)
    t.add_argument("--test", default="20260910-015948-kbd")
    t.add_argument("--out", default=None)
    t.add_argument("--include-reach", action="store_true")
    t.set_defaults(fn=cmd_train)
    p = sub.add_parser("predict")
    p.add_argument("session")
    p.add_argument("--taps", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="models/finger_id.pkl")
    p.set_defaults(fn=cmd_predict)
    s = sub.add_parser("snr")
    s.add_argument("--sessions", nargs="+", required=True)
    s.set_defaults(fn=cmd_snr)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
