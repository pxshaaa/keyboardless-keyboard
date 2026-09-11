"""Contact-point diagnosis + correction: why the recorded tap (x, y) does not localise a key.
Run: python -m phase0.analysis.contact {decompose | frames | keyacc | apply} [--sessions ...]"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from phase0.analysis.decode import A_INDEX, NA, GaussianSpatial, labelled_taps
from phase0.analysis.finger_id import FINGERTIP_JOINTS, HAND_NAMES, KEY_LABEL
from phase0.analysis.tap_pos import Sess, load_sess, sess_path

TRAIN = ("20260910-015217-kbd", "20260910-021315-kbd", "20260910-131629-kbd")
HELDOUT = "20260910-015948-kbd"
PALM_JOINTS = (0, 5, 9, 13, 17)
MCP_OF = {4: 2, 8: 5, 12: 9, 16: 13, 20: 17}
HOME_PAIRS = (("a", "s"), ("s", "d"), ("d", "f"), ("j", "k"), ("k", "l"))
MIN_N = 12

# nominal QWERTY in key-pitch units; row stagger is the real physical offset
_ROWS = ((0.0, 0.0, "qwertyuiop"), (1.0, 0.25, "asdfghjkl"), (2.0, 0.75, "zxcvbnm"))
NOMINAL: dict[str, tuple[float, float]] = {}
for _r, _off, _ks in _ROWS:
    for _j, _c in enumerate(_ks):
        NOMINAL[_c] = (_off + _j, _r)
NOMINAL[" "] = (3.5, 3.0)


# ------------------------------------------------------------------ finger ground truth
def gt_finger(s: Sess, k: np.ndarray, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(hand[N], tip-joint[N]) a touch typist would use; -1 where undefined. Space: lower thumb."""
    hand = np.full(len(k), -1)
    tip = np.full(len(k), -1)
    for n, key in enumerate(keys):
        lab = KEY_LABEL.get("space" if key == " " else key)
        if lab is None:
            continue
        if lab[0] == "Either":
            cand = [(h, s.P[k[n], h, 4, 1]) for h in (0, 1) if np.isfinite(s.P[k[n], h, 4, :2]).all()]
            if not cand:
                continue
            hand[n], tip[n] = max(cand, key=lambda c: c[1])[0], 4
        else:
            hand[n], tip[n] = HAND_NAMES.index(lab[0]), FINGERTIP_JOINTS[lab[1]]
    return hand, tip


# ------------------------------------------------------------------ coordinate frames
def _at(s: Sess, k: np.ndarray, off: int) -> np.ndarray:
    return np.clip(k + off, 0, len(s.P) - 1)


def embed(s: Sess, k: np.ndarray, hand: np.ndarray, tip: np.ndarray, frame: str,
          off: int = 0) -> np.ndarray:
    """[N,2] the tap's contact point expressed in `frame`. NaN rows = unusable."""
    kk = _at(s, k, off)
    n = len(kk)
    out = np.full((n, 2), np.nan)
    ok = (hand >= 0) & (tip >= 0)
    if not ok.any():
        return out
    idx = np.where(ok)[0]
    kh, hh, th = kk[idx], hand[idx], tip[idx]
    P = s.P[kh, hh]                                   # [M,21,7]
    xy = P[:, :, :2]
    t_xy = xy[np.arange(len(idx)), th]
    wrist = xy[:, 0]
    palm = np.nanmean(xy[:, list(PALM_JOINTS)], axis=1)
    mcp = xy[np.arange(len(idx)), np.array([MCP_OF[int(v)] for v in th])]
    span = np.linalg.norm(xy[:, 9] - wrist, axis=1)
    span = np.where(np.isfinite(span) & (span > 1e-6), span, np.nan)
    if frame == "abs":
        v = t_xy
    elif frame == "rest":
        v = t_xy - s.anchor[hh]
    elif frame == "restn":
        v = (t_xy - s.anchor[hh]) / s.span[hh][:, None]
    elif frame == "wrist":
        v = t_xy - wrist
    elif frame == "wristn":
        v = (t_xy - wrist) / span[:, None]
    elif frame == "palm":
        v = t_xy - palm
    elif frame == "palmn":
        v = (t_xy - palm) / span[:, None]
    elif frame == "mcpn":
        v = (t_xy - mcp) / span[:, None]
    elif frame in ("local", "localpalm"):
        e1 = (xy[:, 9] - wrist) / span[:, None]
        e2 = np.stack([-e1[:, 1], e1[:, 0]], axis=1)
        d = (t_xy - (wrist if frame == "local" else palm)) / span[:, None]
        v = np.stack([(d * e1).sum(1), (d * e2).sum(1)], axis=1)
    elif frame == "localabs":                          # rotation only, keeps pixel scale
        e1 = (xy[:, 9] - wrist) / span[:, None]
        e2 = np.stack([-e1[:, 1], e1[:, 0]], axis=1)
        d = t_xy - wrist
        v = np.stack([(d * e1).sum(1), (d * e2).sum(1)], axis=1)
    else:
        raise SystemExit(f"unknown frame {frame!r}")
    out[idx] = v
    return out


FRAMES = ("abs", "rest", "restn", "wrist", "wristn", "palm", "palmn", "mcpn",
          "local", "localpalm", "localabs")


# ------------------------------------------------------------------ layout metrics
def _hand_of(c: str) -> int:
    lab = KEY_LABEL.get("space" if c == " " else c)
    return -1 if lab is None or lab[0] == "Either" else HAND_NAMES.index(lab[0])


def _affine_pitch(cent: dict[str, np.ndarray]) -> tuple[float, float]:
    """Affine nominal->centroid, fitted per hand: hand-relative frames offset the two hands
    differently, so one global fit would report that offset as layout error."""
    scales, res, w = [], [], []
    for h in (0, 1):
        ks = [c for c in cent if c in NOMINAL and _hand_of(c) == h]
        if len(ks) < 4:
            continue
        A = np.array([[*NOMINAL[c], 1.0] for c in ks])
        B = np.array([cent[c] for c in ks])
        M, *_ = np.linalg.lstsq(A, B, rcond=None)
        d = B - A @ M
        scales.append(math.sqrt(abs(np.linalg.det(M[:2].T))))
        res.append(float(np.sqrt((d ** 2).sum(1).mean())))
        w.append(len(ks))
    if not scales:
        return float("nan"), float("nan")
    return float(np.average(scales, weights=w)), float(np.average(res, weights=w))


def layout_stats(XY: np.ndarray, keys: list[str], min_n: int = MIN_N) -> dict:
    """within-key scatter, pitch from the affine layout fit, and home-row pair distances."""
    ok = np.isfinite(XY).all(1)
    by = defaultdict(list)
    for i in np.where(ok)[0]:
        by[keys[i]].append(XY[i])
    cent, scat, ns, cent5 = {}, [], [], {}
    for c, v in by.items():
        if len(v) >= 5:
            cent5[c] = np.array(v).mean(0)
        if len(v) < min_n:
            continue
        V = np.array(v)
        cent[c] = V.mean(0)
        scat.append(math.sqrt(((V - V.mean(0)) ** 2).sum(1).mean()))
        ns.append(len(v))
    pitch, resid = _affine_pitch(cent)
    pairs = {f"{a}-{b}": float(np.linalg.norm(cent5[a] - cent5[b]))
             for a, b in HOME_PAIRS if a in cent5 and b in cent5}
    # pitch measured directly off adjacent home-row keys; the affine-to-nominal fit is poor
    # because the touch-typing map piles q/a/z onto one finger, which compresses its scale
    home = float(np.median(list(pairs.values()))) if len(pairs) >= 3 else float("nan")
    med = float(np.median(scat)) if scat else float("nan")
    return {"n": int(ok.sum()), "keys": len(cent), "scatter": med, "pitch": pitch,
            "ratio": med / pitch if pitch else float("nan"),
            "resid_ratio": resid / pitch if pitch else float("nan"),
            "home_pitch": home, "home_ratio": med / home if home else float("nan"),
            "pairs": pairs, "pairs_ratio": {k: v / home for k, v in pairs.items()} if home else {},
            "centroids": cent, "median_n": float(np.median(ns)) if ns else 0.0}


def pair_spread(st: dict) -> float:
    """Home-row consistency: max/min over the five adjacent pairs. 1.0 = a perfect grid."""
    v = list(st["pairs"].values())
    return max(v) / min(v) if len(v) >= 3 and min(v) > 0 else float("nan")


# ------------------------------------------------------------------ dataset
def load(sid: str, taps_name: str = "taps.jsonl") -> dict:
    s = load_sess(sid)
    taps, keys = labelled_taps(sess_path(sid), taps_name)
    k = s.rows(taps)
    hand, tip = gt_finger(s, k, keys)
    return {"sid": sid, "sess": s, "taps": taps, "keys": keys, "k": k, "hand": hand, "tip": tip,
            "xy": np.array([[t["x"], t["y"]] for t in taps], float),
            "rhand": np.array([int(t.get("hand", 0)) for t in taps]),
            "rtip": np.array([int(t.get("finger", 0)) for t in taps]),
            "y": np.array([A_INDEX[c] for c in keys])}


def _fmt(st: dict) -> str:
    return (f"n={st['n']:5d} keys={st['keys']:2d} scatter={st['scatter']:8.3f} "
            f"homepitch={st['home_pitch']:7.2f} scat/pitch={st['home_ratio']:6.2f} "
            f"pairmax/min={pair_spread(st):5.2f}")


# ------------------------------------------------------------------ 1. decomposition
def jitter(d: dict, frame: str) -> float:
    """High-frequency tracking noise: RMS deviation of the GT tip over frames k-1..k+1."""
    V = np.stack([embed(d["sess"], d["k"], d["hand"], d["tip"], frame, o) for o in (-1, 0, 1)])
    m = V.mean(0)
    r = np.sqrt(np.nanmean(((V - m) ** 2).sum(2), axis=0))
    return float(np.nanmedian(r))


def cmd_decompose(a) -> int:
    ds = [load(s) for s in a.sessions]
    print("### scatter decomposition (median within-key RMS radius, image px)\n")
    for d in ds:
        agree = float((d["rhand"] == d["hand"])[(d["hand"] >= 0)].mean())
        both = ((d["rhand"] == d["hand"]) & (d["rtip"] == d["tip"]))[d["hand"] >= 0]
        print(f"-- {d['sid']}  labelled taps={len(d['keys'])}  "
              f"reported finger == touch-typing finger: {both.mean():.3f} (hand only {agree:.3f})")
        rows = [("reported (x,y) in taps.jsonl", layout_stats(d["xy"], d["keys"])),
                ("GT finger, tap frame, abs px", layout_stats(
                    embed(d["sess"], d["k"], d["hand"], d["tip"], "abs"), d["keys"]))]
        best = None
        for off in range(a.lo, a.hi + 1):
            st = layout_stats(embed(d["sess"], d["k"], d["hand"], d["tip"], "abs", off), d["keys"])
            if best is None or st["scatter"] < best[1]["scatter"]:
                best = (off, st)
        rows.append((f"GT finger, best offset {best[0]:+d}f, abs px", best[1]))
        bf = None
        for fr in FRAMES:
            st = layout_stats(embed(d["sess"], d["k"], d["hand"], d["tip"], fr, best[0]), d["keys"])
            if bf is None or st["home_ratio"] < bf[1]["home_ratio"]:
                bf = (fr, st)
        rows.append((f"GT finger, {best[0]:+d}f, frame={bf[0]}", bf[1]))
        for name, st in rows:
            print(f"   {name:38s} {_fmt(st)}")
        jt = jitter(d, "abs")
        print(f"   {'tracking jitter floor (+/-1 frame)':38s} abs px={jt:.2f}")
        v0, v1, v2 = (rows[0][1]["scatter"] ** 2, rows[1][1]["scatter"] ** 2,
                      rows[2][1]["scatter"] ** 2)
        if v0 > 0:
            print(f"   variance budget: wrong finger {100 * (v0 - v1) / v0:5.2f}%  "
                  f"wrong frame {100 * (v1 - v2) / v0:5.2f}%  "
                  f"tracking noise {100 * jt ** 2 / v0:5.2f}%  "
                  f"genuine within-key {100 * (v2 - jt ** 2) / v0:5.2f}%")
        print()
    return 0


# ------------------------------------------------------------------ 2. frame offset sweep
def cmd_offsets(a) -> int:
    ds = [load(s) for s in a.sessions]
    print("### within-key scatter vs frame offset (GT finger)\n")
    print(f"{'off':>4} " + " ".join(f"{d['sid'][9:13]:>12}" for d in ds) + f" {'mean ratio':>11}")
    for off in range(a.lo, a.hi + 1):
        sts = [layout_stats(embed(d["sess"], d["k"], d["hand"], d["tip"], a.frame, off), d["keys"])
               for d in ds]
        r = float(np.nanmean([s["home_ratio"] for s in sts]))
        print(f"{off:>+4d} " + " ".join(f"{s['scatter']:12.2f}" for s in sts) + f" {r:11.2f}")
    return 0


# ------------------------------------------------------------------ 3. coordinate frames
def cmd_frames(a) -> int:
    ds = [load(s) for s in a.sessions]
    keys_all = [c for d in ds for c in d["keys"]]
    print(f"### coordinate frames (GT finger, offset {a.off:+d})\n")
    for fr in FRAMES:
        sts = [layout_stats(embed(d["sess"], d["k"], d["hand"], d["tip"], fr, a.off), d["keys"])
               for d in ds]
        rr = [s["home_ratio"] for s in sts]
        ps = [pair_spread(s) for s in sts]
        pr = defaultdict(list)
        for s in sts:
            for k2, v in s["pairs_ratio"].items():
                pr[k2].append(v)
        pairs = " ".join(f"{k2}={np.mean(v):.2f}" for k2, v in pr.items())
        print(f"{fr:10s} scatter/pitch={np.nanmean(rr):5.2f} (per-sess {' '.join(f'{v:.2f}' for v in rr)})"
              f"  pairmax/min={np.nanmean(ps):4.2f}  resid={np.nanmean([s["resid_ratio"] for s in sts]):4.2f}")
        XY = np.vstack([embed(d["sess"], d["k"], d["hand"], d["tip"], fr, a.off) for d in ds])
        ps = layout_stats(XY, keys_all)
        pp = " ".join(f"{k2}={v:.2f}" for k2, v in ps["pairs_ratio"].items())
        print(f"{'':10s} per-session home pairs (pitch units): {pairs}")
        print(f"{'':10s} POOLED scat/pitch={ps['home_ratio']:.2f} resid={ps['resid_ratio']:.2f} "
              f"pairmax/min={pair_spread(ps):.2f} pairs: {pp}")
    return 0


# ------------------------------------------------------------------ finger prediction
FCLASSES = [(h, t) for h in (0, 1) for t in FINGERTIP_JOINTS]


class FingerClf:
    """pose -> which of the 10 fingertips made contact; the piece taps_gb gets at chance."""

    def __init__(self, model, classes):
        self.model, self.classes = model, classes

    @classmethod
    def fit(cls, ds: list[dict], seed: int = 0):
        import lightgbm as lgb
        from phase0.analysis.tap_pos import features
        X, y = [], []
        for d in ds:
            m = d["hand"] >= 0
            X.append(features(d["sess"], d["k"], "rel")[m])
            y += [FCLASSES.index((int(h), int(t))) for h, t in zip(d["hand"][m], d["tip"][m])]
        X, y = np.vstack(X), np.array(y)
        classes = np.unique(y)
        remap = {c: i for i, c in enumerate(classes)}
        m = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.08, num_leaves=15,
                               min_child_samples=8, colsample_bytree=0.6, verbose=-1,
                               random_state=seed)
        m.fit(X, np.array([remap[v] for v in y]))
        return cls(m, classes)

    def proba(self, d: dict) -> np.ndarray:
        """[N,10] over FCLASSES; unseen classes keep a floor so the mixture never vanishes."""
        from phase0.analysis.tap_pos import features
        p = self.model.predict_proba(features(d["sess"], d["k"], "rel"))
        out = np.full((len(d["k"]), len(FCLASSES)), 1e-4)
        out[:, self.classes] = np.maximum(p, 1e-4)
        return out / out.sum(1, keepdims=True)

    def predict(self, d: dict) -> tuple[np.ndarray, np.ndarray]:
        from phase0.analysis.tap_pos import features
        p = self.model.predict_proba(features(d["sess"], d["k"], "rel"))
        j = self.classes[p.argmax(1)]
        hand = np.array([FCLASSES[v][0] for v in j])
        tip = np.array([FCLASSES[v][1] for v in j])
        return hand, tip


# ------------------------------------------------------------------ 4. key accuracy
def _rank(ll: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    o = np.argsort(-ll, 1)
    return float((o[:, 0] == y).mean()), float(np.mean([y[i] in o[i, :5] for i in range(len(y))]))


def _gauss_topk(g: GaussianSpatial, XY: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    X = np.where(np.isfinite(XY), XY, np.nanmean(XY, axis=0))
    d = X[:, None, :] - g.mu[None, :, :]
    q = np.einsum("nkj,kjl,nkl->nk", d, g.prec, d)
    ll = -0.5 * (q + g.logdet[None, :])
    o = np.argsort(-ll, 1)
    return float((o[:, 0] == y).mean()), float(np.mean([y[i] in o[i, :5] for i in range(len(y))]))


def soft_logp(g: GaussianSpatial, d: dict, frame: str, off: int, clf) -> np.ndarray:
    """P(key) = sum_finger P(finger|pose) * N(that fingertip | key): no hard finger decision."""
    pf = clf.proba(d)
    n = len(d["k"])
    acc = np.full((n, NA), -np.inf)
    for c, (h, t) in enumerate(FCLASSES):
        XY = embed(d["sess"], d["k"], np.full(n, h), np.full(n, t), frame, off)
        XY = np.where(np.isfinite(XY), XY, np.nanmedian(XY, axis=0))
        dd = XY[:, None, :] - g.mu[None, :, :]
        q = np.einsum("nkj,kjl,nkl->nk", dd, g.prec, dd)
        ll = -0.5 * (q + g.logdet[None, :]) + np.log(pf[:, c])[:, None]
        m = np.maximum(acc, ll)
        acc = m + np.log(np.exp(acc - m) + np.exp(ll - m))
    return acc - _lse2(acc)


KEY_FINGERS: dict[int, list[int]] = {}
for _c, _i in A_INDEX.items():
    _lab = KEY_LABEL.get("space" if _c == " " else _c)
    if _lab is None:
        continue
    if _lab[0] == "Either":
        KEY_FINGERS[_i] = [FCLASSES.index((h, 4)) for h in (0, 1)]
    else:
        KEY_FINGERS[_i] = [FCLASSES.index((HAND_NAMES.index(_lab[0]), FINGERTIP_JOINTS[_lab[1]]))]


def implied_logp(g: GaussianSpatial, d: dict, frame: str, off: int, clf) -> np.ndarray:
    """Score key k only against the fingertip a touch typist uses for k, weighted by
    P(that finger | pose). No finger has to be chosen before the key is."""
    pf = clf.proba(d)
    n = len(d["k"])
    per = {}
    for c, (h, t) in enumerate(FCLASSES):
        XY = embed(d["sess"], d["k"], np.full(n, h), np.full(n, t), frame, off)
        per[c] = np.where(np.isfinite(XY), XY, np.nanmedian(XY, axis=0))
    out = np.full((n, NA), -60.0)
    for key, cs in KEY_FINGERS.items():
        best = None
        for c in cs:
            dd = per[c] - g.mu[key]
            q = np.einsum("nj,jl,nl->n", dd, g.prec[key], dd)
            ll = -0.5 * (q + g.logdet[key]) + np.log(pf[:, c])
            best = ll if best is None else np.maximum(best, ll)
        out[:, key] = best
    return out - _lse2(out)


def _lse2(a):
    m = a.max(1, keepdims=True)
    return m + np.log(np.exp(a - m).sum(1, keepdims=True))


def _xy_for(d: dict, kind: str, frame: str, off: int, clf) -> np.ndarray:
    if kind == "reported":
        return d["xy"]
    if kind == "oracle":
        h, t = d["hand"], d["tip"]
    else:
        h, t = clf.predict(d)
    return embed(d["sess"], d["k"], h, t, frame, off)


def _fit_gauss(ds: list[dict], kind: str, frame: str, off: int, clf) -> GaussianSpatial:
    XY = np.vstack([_xy_for(d, kind, frame, off, clf) for d in ds])
    labels = [c for d in ds for c in d["keys"]]
    m = np.isfinite(XY).all(1)
    return GaussianSpatial.fit(XY[m], [l for l, k in zip(labels, m) if k])


def cmd_keyacc(a) -> int:
    ds = {s: load(s) for s in list(a.sessions) + [a.heldout]}
    tr = list(a.sessions)
    kinds = ["reported", "oracle", "pred", "soft", "implied"]
    print(f"### per-key Gaussian, frame={a.frame} offset={a.off:+d}\n")
    print(f"{'split':28s} {'kind':9s} {'top1':>7} {'top5':>7} {'finger-acc':>11}")
    for held in tr:
        fit = [ds[s] for s in tr if s != held]
        clf = FingerClf.fit(fit)
        ph, pt = clf.predict(ds[held])
        m = ds[held]["hand"] >= 0
        fa = float(((ph == ds[held]["hand"]) & (pt == ds[held]["tip"]))[m].mean())
        for kind in kinds:
            g = _fit_gauss(fit, "oracle" if kind in ("soft", "implied") else kind, a.frame, a.off, clf)
            if kind in ("soft", "implied"):
                fn = soft_logp if kind == "soft" else implied_logp
                t1, t5 = _rank(fn(g, ds[held], a.frame, a.off, clf), ds[held]["y"])
            else:
                t1, t5 = _gauss_topk(g, _xy_for(ds[held], kind, a.frame, a.off, clf), ds[held]["y"])
            print(f"LOSO {held[9:13]:23s} {kind:9s} {t1:7.3f} {t5:7.3f} "
                  f"{fa if kind == 'pred' else float('nan'):11.3f}")
    fit = [ds[s] for s in tr]
    clf = FingerClf.fit(fit)
    h = ds[a.heldout]
    ph, pt = clf.predict(h)
    m = h["hand"] >= 0
    fa = float(((ph == h["hand"]) & (pt == h["tip"]))[m].mean())
    for kind in kinds:
        g = _fit_gauss(fit, "oracle" if kind in ("soft", "implied") else kind, a.frame, a.off, clf)
        if kind in ("soft", "implied"):
            fn = soft_logp if kind == "soft" else implied_logp
            t1, t5 = _rank(fn(g, h, a.frame, a.off, clf), h["y"])
        else:
            t1, t5 = _gauss_topk(g, _xy_for(h, kind, a.frame, a.off, clf), h["y"])
        print(f"HELD-OUT {a.heldout[9:13]:19s} {kind:9s} {t1:7.3f} {t5:7.3f} "
              f"{fa if kind == 'pred' else float('nan'):11.3f}")
    return 0


# ------------------------------------------------------------------ 5. write corrected taps
def cmd_apply(a) -> int:
    """Write a taps file whose (x, y) is the corrected contact point and whose key_probs come
    from the implied-finger mixture, so decode.py can consume it unmodified."""
    fit = [load(s) for s in a.sessions]
    clf = FingerClf.fit(fit)
    g = _fit_gauss(fit, "oracle", a.frame, a.off, clf)
    alpha = list("abcdefghijklmnopqrstuvwxyz ")
    for sid in a.targets:
        s = load_sess(sid)
        p = sess_path(sid)
        taps = [json.loads(l) for l in (p / a.taps_name).read_text().splitlines() if l.strip()]
        d = {"sess": s, "k": s.rows(taps), "taps": taps}
        hand, tip = clf.predict(d)
        XY = embed(s, d["k"], hand, tip, a.frame, a.off)
        XY = np.where(np.isfinite(XY), XY, np.nanmedian(XY, axis=0))
        P = np.exp(implied_logp(g, d, a.frame, a.off, clf))
        out = p / a.out
        with out.open("w") as f:
            for i, t in enumerate(taps):
                r = dict(t)
                r["x"], r["y"] = float(XY[i, 0] * a.scale), float(XY[i, 1] * a.scale)
                r["key_probs"] = {("space" if alpha[j] == " " else alpha[j]): round(float(P[i, j]), 6)
                                  for j in np.argsort(-P[i])[:10]}
                f.write(json.dumps(r) + "\n")
        print(f"{sid}: {len(taps)} taps -> {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="contact")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("decompose", cmd_decompose), ("offsets", cmd_offsets),
                     ("frames", cmd_frames), ("keyacc", cmd_keyacc), ("apply", cmd_apply)):
        p = sub.add_parser(name)
        p.add_argument("--sessions", nargs="+", default=list(TRAIN))
        p.add_argument("--frame", default="local")
        p.add_argument("--off", type=int, default=0)
        p.add_argument("--lo", type=int, default=-10)
        p.add_argument("--hi", type=int, default=10)
        p.add_argument("--heldout", default=HELDOUT)
        p.add_argument("--targets", nargs="+", default=list(TRAIN) + [HELDOUT])
        p.add_argument("--taps-name", default="taps.jsonl")
        p.add_argument("--out", default="taps_contact.jsonl")
        p.add_argument("--scale", type=float, default=1.0)
        p.add_argument("--oracle", action="store_true")
        p.set_defaults(func=fn)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
