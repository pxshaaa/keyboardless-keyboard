"""Tap POSITION + key distribution: replaces taps_gb's argmax-flexion-velocity fingertip.
Run: python -m phase0.analysis.tap_pos {fit --sessions A B | apply S --taps F --out F | eval | desk}"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from phase0.analysis.decode import (
    A_INDEX,
    ALPHABET,
    NA,
    CharLM,
    GaussianSpatial,
    WordLM,
    Weights,
    beam_decode,
    cer,
    desk_segments,
    edit_distance,
    labelled_taps,
)
from phase0.analysis.analyze_drift import read_jsonl
from phase0.analysis.finger_id import KEY_LABEL, FINGERTIP_JOINTS, HAND_NAMES

DEFAULT_MODEL = Path("models/tap_pos.pkl")
CACHE = Path(".cache/tap_pos")
TRAIN_SESSIONS = ("20260910-015217-kbd", "20260910-021315-kbd", "20260910-131629-kbd")
HELDOUT_SESSION = "20260910-015948-kbd"
DESK_SESSION = "20260910-181947-desk"
SEEDS = (0, 1, 2)
MIN_N = 12

OFFSETS = (-8, -4, -2, 0, 2, 4, 8)
JOINTS_FULL = tuple(range(21))
JOINTS_TIPS = (0, 4, 8, 12, 16, 20)
W, H = 1280.0, 720.0

# channel layout of taps_gb.load_frames: x, y, conf, z, wx, wy, wz
CH_X, CH_Y, CH_CONF, CH_Z, CH_WX = 0, 1, 2, 3, 4


CHANNELS = ("x", "y", "conf", "z", "wx", "wy", "wz")


def load_frames(session: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (frame_ids, t, P[F,2,21,7]) slotted by HANDEDNESS, whole hand at a time; taps_gb's
    version dedups per landmark row and mixes two hands in one slot (span 518 px, not 110)."""
    import pyarrow.parquet as pq

    tb = pq.read_table(session / "landmarks.parquet")
    have = [c for c in CHANNELS if c in tb.column_names]
    col = {n: np.asarray(tb.column(n)) for n in ("i", "t", "hand", "handedness", "joint", *have)}
    frames, inv = np.unique(col["i"], return_inverse=True)
    side = (col["handedness"] != "Left").astype(int)
    # both rows of a frame can claim one handedness; only there is the `hand` index better
    key, counts = np.unique(inv * 2 + side, return_counts=True)
    collide = np.isin(inv * 2 + side, key[counts > 21])
    side = np.where(collide, col["hand"], side)
    t = np.zeros(len(frames))
    t[inv] = col["t"]
    P = np.full((len(frames), 2, 21, len(CHANNELS)), np.nan)
    for k, ch in enumerate(CHANNELS):
        if ch in have:
            P[inv, side, col["joint"], k] = col[ch]
    return frames, t, P


# ------------------------------------------------------------------ sessions
class Sess:
    """One session's landmark tensor plus its own resting geometry (the desk anchor)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.frames, self.t, self.P = load_frames(self.path)
        wr = self.P[:, :, 0, :2]
        self.anchor = np.nanmedian(wr, axis=0)                      # [2,2] per-hand wrist
        span = np.linalg.norm(self.P[:, :, 9, :2] - wr, axis=2)     # [F,2]
        self.span = np.nanmedian(span, axis=0)
        self.span = np.where(np.isfinite(self.span) & (self.span > 1e-6), self.span, 100.0)
        self.anchor = np.where(np.isfinite(self.anchor), self.anchor, [[W / 2, H / 2]] * 2)

    def rows(self, taps: list[dict]) -> np.ndarray:
        k = np.searchsorted(self.frames, [tp["i"] for tp in taps])
        return np.clip(k, 0, len(self.frames) - 1)


def sess_path(sid: str) -> Path:
    p = Path(sid)
    return p if p.exists() else Path("data/sessions") / sid


_MEM: dict[str, Sess] = {}


def load_sess(sid: str) -> Sess:
    p = sess_path(sid)
    if str(p) in _MEM:
        return _MEM[str(p)]
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{p.name}.pkl"
    src = p / "landmarks.parquet"
    if cf.exists() and cf.stat().st_mtime > src.stat().st_mtime:
        s = pickle.loads(cf.read_bytes())
    else:
        s = Sess(p)
        cf.write_bytes(pickle.dumps(s, protocol=4))
    _MEM[str(p)] = s
    return s


# ------------------------------------------------------------------ features
def _gather(P: np.ndarray, k: np.ndarray, off: int) -> np.ndarray:
    return P[np.clip(k + off, 0, len(P) - 1)]


def features(s: Sess, k: np.ndarray, mode: str = "abs", joints=JOINTS_FULL,
             offsets=OFFSETS) -> np.ndarray:
    """[N,D] tap features. mode 'abs' = raw image px; 'rel' = anchored on the hands' own
    resting geometry, which is the only thing that could survive a session change."""
    J = list(joints)
    cols = []
    base = _gather(s.P, k, 0)
    for h in (0, 1):
        for o in offsets:
            Q = _gather(s.P, k, o)[:, h][:, J, :2]
            if mode == "abs":
                xy = Q / np.array([W, H])
            else:
                xy = (Q - s.anchor[h]) / s.span[h]
            cols.append(xy.reshape(len(k), -1))
            if o != 0:
                cols.append((Q - base[:, h][:, J, :2]).reshape(len(k), -1) / s.span[h])
        B = base[:, h]
        cols.append(B[:, J, CH_Z] / s.span[h])
        cols.append(B[:, J, CH_WX:CH_WX + 3].reshape(len(k), -1))
        cols.append(np.nanmean(B[:, :, CH_CONF], axis=1)[:, None])
        cols.append(np.isfinite(B[:, 0, 0]).astype(float)[:, None])
    dw = base[:, 1, 0, :2] - base[:, 0, 0, :2]
    cols.append(dw / (W if mode == "abs" else s.span.mean()))
    return np.nan_to_num(np.hstack(cols).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def flex_vel_pick(s: Sess, k: np.ndarray) -> np.ndarray:
    """[N,2] the taps_gb rule: argmax |2-D flexion velocity| over the 10 fingertips."""
    return _pick_xy(s, k, _flex_score(s))


def tip_vy_pick(s: Sess, k: np.ndarray) -> np.ndarray:
    tips = s.P[:, :, FINGERTIP_JOINTS, CH_Y]
    v = np.gradient(np.nan_to_num(tips, nan=0.0), axis=0)
    return _pick_xy(s, k, np.nan_to_num(v, nan=-1e9))


MCP_OF = {4: 2, 8: 5, 12: 9, 16: 13, 20: 17}


def _flex_score(s: Sess) -> np.ndarray:
    span = np.linalg.norm(s.P[:, :, 9, :2] - s.P[:, :, 0, :2], axis=2)[:, :, None]
    f = np.stack([np.linalg.norm(s.P[:, :, tip, :2] - s.P[:, :, MCP_OF[tip], :2], axis=2)
                  for tip in FINGERTIP_JOINTS], axis=2)
    v = np.gradient(f / np.where(span > 1e-6, span, np.nan), axis=0)
    return np.abs(np.nan_to_num(v, nan=-1.0))


def _pick_xy(s: Sess, k: np.ndarray, score: np.ndarray) -> np.ndarray:
    sc = score[k].reshape(len(k), -1)
    j = np.argmax(sc, axis=1)
    hand, fi = j // 5, j % 5
    tip = np.array(FINGERTIP_JOINTS)[fi]
    xy = s.P[k, hand, tip, :2]
    return np.nan_to_num(xy), hand, tip


def true_tip_xy(s: Sess, k: np.ndarray, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Regression target: the fingertip a touch typist would use for that key. mask=usable."""
    xy = np.full((len(k), 2), np.nan)
    for n, key in enumerate(keys):
        lab = KEY_LABEL.get(key if key != " " else "space")
        if lab is None:
            continue
        if lab[0] == "Either":  # space: whichever thumb is lower in the image
            cand = [(hh, s.P[k[n], hh, 4, :2]) for hh in (0, 1)]
            cand = [c for c in cand if np.isfinite(c[1]).all()]
            if not cand:
                continue
            xy[n] = max(cand, key=lambda c: c[1][1])[1]
        else:
            h = HAND_NAMES.index(lab[0])
            xy[n] = s.P[k[n], h, FINGERTIP_JOINTS[lab[1]], :2]
    return xy, np.isfinite(xy).all(1)


# ------------------------------------------------------------------ dataset
def dataset(sid: str, taps_name: str = "taps.jsonl") -> dict:
    s = load_sess(sid)
    taps, keys = labelled_taps(sess_path(sid), taps_name)
    k = s.rows(taps)
    return {"sid": sid, "sess": s, "taps": taps, "keys": keys, "k": k,
            "y": np.array([A_INDEX[c] for c in keys]),
            "xy": np.array([[t["x"], t["y"]] for t in taps], float)}


# ------------------------------------------------------------------ models
def _lgbm(seed: int, n_class: int):
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=250, learning_rate=0.08, num_leaves=15,
                              min_child_samples=8, subsample=0.9, subsample_freq=1,
                              colsample_bytree=0.6, verbose=-1, random_state=seed)


class KeyClf:
    """pose -> P(key). The replacement for the argmax-fingertip + per-key Gaussian chain."""

    def __init__(self, model, classes, mode, joints, offsets=OFFSETS):
        self.model, self.classes, self.mode, self.joints = model, classes, mode, joints
        self.offsets = offsets

    @classmethod
    def fit(cls, ds: list[dict], mode="abs", joints=JOINTS_FULL, seed=0, min_n=MIN_N,
            offsets=OFFSETS):
        X = np.vstack([features(d["sess"], d["k"], mode, joints, offsets) for d in ds])
        y = np.concatenate([d["y"] for d in ds])
        cnt = Counter(y.tolist())
        m = np.array([cnt[v] >= min_n for v in y])
        classes = np.unique(y[m])
        remap = {c: i for i, c in enumerate(classes)}
        mdl = _lgbm(seed, len(classes))
        mdl.fit(X[m], np.array([remap[v] for v in y[m]]))
        return cls(mdl, classes, mode, joints, offsets)

    def proba(self, sess: Sess, k: np.ndarray) -> np.ndarray:
        p = self.model.predict_proba(features(sess, k, self.mode, self.joints, self.offsets))
        out = np.full((len(k), NA), 1e-6)
        out[:, self.classes] = np.maximum(p, 1e-6)
        return out / out.sum(1, keepdims=True)


class XYReg:
    """pose -> corrected contact (x, y), then one bivariate Gaussian per key on it."""

    def __init__(self, rx, ry, gauss, mode, joints):
        self.rx, self.ry, self.gauss, self.mode, self.joints = rx, ry, gauss, mode, joints

    @classmethod
    def fit(cls, ds: list[dict], mode="abs", joints=JOINTS_FULL, seed=0, min_n=MIN_N):
        import lightgbm as lgb
        X = np.vstack([features(d["sess"], d["k"], mode, joints) for d in ds])
        tgt, msk = [], []
        for d in ds:
            a, b = true_tip_xy(d["sess"], d["k"], d["keys"])
            tgt.append(a)
            msk.append(b)
        T, M = np.vstack(tgt), np.concatenate(msk)
        kw = dict(n_estimators=300, learning_rate=0.06, num_leaves=31, min_child_samples=10,
                  colsample_bytree=0.6, verbose=-1, random_state=seed)
        rx = lgb.LGBMRegressor(**kw).fit(X[M], T[M, 0])
        ry = lgb.LGBMRegressor(**kw).fit(X[M], T[M, 1])
        pred = np.stack([rx.predict(X), ry.predict(X)], 1)
        labels = [c for d in ds for c in d["keys"]]
        return cls(rx, ry, GaussianSpatial.fit(pred, labels, min_n=min_n), mode, joints)

    def predict_xy(self, sess: Sess, k: np.ndarray) -> np.ndarray:
        X = features(sess, k, self.mode, self.joints)
        return np.stack([self.rx.predict(X), self.ry.predict(X)], 1)

    def proba(self, sess: Sess, k: np.ndarray) -> np.ndarray:
        lp = _gauss_logp(self.gauss, self.predict_xy(sess, k))
        return np.exp(lp)


def _gauss_logp(g: GaussianSpatial, xy: np.ndarray) -> np.ndarray:
    d = xy[:, None, :] - g.mu[None, :, :]
    q = np.einsum("nkj,kjl,nkl->nk", d, g.prec, d)
    ll = -0.5 * (q + g.logdet[None, :])
    return ll - _lse(ll)


def _lse(a):
    m = a.max(1, keepdims=True)
    return m + np.log(np.exp(a - m).sum(1, keepdims=True))


class GaussXY:
    """Baseline family: a per-key Gaussian over whatever (x, y) a finger-selection rule reports."""

    def __init__(self, gauss, sel):
        self.gauss, self.sel = gauss, sel

    @classmethod
    def fit(cls, ds: list[dict], kind="reported", seed=0, min_n=MIN_N):
        sel = Selector.fit(kind, ds, seed)
        xy = np.vstack([sel.pick(d)[0] for d in ds])
        labels = [c for d in ds for c in d["keys"]]
        return cls(GaussianSpatial.fit(xy, labels, min_n=min_n), sel)

    def proba(self, d: dict) -> np.ndarray:
        return np.exp(_gauss_logp(self.gauss, self.sel.pick(d)[0]))


_FCACHE: dict = {}


class Selector:
    """Picks the pressing finger, then reports that fingertip's pixel position."""

    def __init__(self, kind: str, model=None):
        self.kind, self.model = kind, model

    @classmethod
    def fit(cls, kind: str, ds: list[dict], seed: int = 0):
        if kind != "fingerid":
            return cls(kind)
        from phase0.analysis import finger_id as fid
        parts = [_fid_data(d["sid"]) for d in ds]
        X = np.vstack([p["X"] for p in parts])
        y = [fid.cls_name(l) for p in parts for l in p["y"]]
        return cls(kind, fid.make_model(seed).fit(X, y))

    def pick(self, d: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """-> (xy [N,2], is_thumb [N], hand [N], fingertip joint [N])."""
        s, k = d["sess"], d["k"]
        if self.kind in ("flexvel", "tipvy"):
            xy, hand, tip = (flex_vel_pick if self.kind == "flexvel" else tip_vy_pick)(s, k)
            return xy, tip == 4, hand, tip
        if self.kind == "reported":
            tip = np.array([int(t.get("finger", 0)) for t in d["taps"]])
            hand = np.array([int(t.get("hand", 0)) for t in d["taps"]])
            return d["xy"], tip == 4, hand, tip
        return self._fid_pick(d)

    def _fid_pick(self, d: dict):
        from phase0.analysis import finger_id as fid
        hands = _fid_hands(d["sid"])
        s, k = d["sess"], d["k"]
        xy = np.array(d["xy"], float).copy()
        thumb = np.zeros(len(k), bool)
        hand = np.array([int(t.get("hand", 0)) for t in d["taps"]])
        tip = np.array([int(t.get("finger", 0)) for t in d["taps"]])
        feats, idx = [], []
        for n, tp in enumerate(d["taps"]):
            got = fid.featurize_tap(hands, tp["t"])
            if got is not None:
                feats.append(got[0])
                idx.append(n)
        if not feats:
            return xy, thumb, hand, tip
        names = list(self.model.classes_)
        pred = self.model.predict_proba(np.vstack(feats)).argmax(1)
        for j, n in enumerate(idx):
            h, f = _decode_cls(names[int(pred[j])])
            if h is None:
                thumb[n], tip[n] = True, 4
                cand = [(hh, s.P[k[n], hh, 4, :2]) for hh in (0, 1)]
                cand = [c for c in cand if np.isfinite(c[1]).all()]
                if cand:
                    hand[n], xy[n] = max(cand, key=lambda c: c[1][1])
            else:
                hand[n], tip[n] = h, FINGERTIP_JOINTS[f]
                v = s.P[k[n], h, FINGERTIP_JOINTS[f], :2]
                if np.isfinite(v).all():
                    xy[n] = v
        return xy, thumb, hand, tip


def _fid_hands(sid: str):
    from phase0.analysis import finger_id as fid
    key = "hands:" + sid
    if key not in _FCACHE:
        _FCACHE[key] = fid.prep(sess_path(sid))
    return _FCACHE[key]


def _fid_data(sid: str) -> dict:
    from phase0.analysis import finger_id as fid
    key = "data:" + sid
    if key not in _FCACHE:
        _FCACHE[key] = fid.build_dataset(sess_path(sid))
    return _FCACHE[key]


_FNAMES = ("thumb", "index", "middle", "ring", "pinky")


def _decode_cls(name: str) -> tuple[int | None, int]:
    if name == "thumb":
        return None, 0
    side, fin = name.split("-")
    return (0 if side == "L" else 1), _FNAMES.index(fin)


# ------------------------------------------------------------------ scoring
def topk(proba: np.ndarray, y: np.ndarray, k: int) -> float:
    order = np.argsort(-proba, 1)[:, :k]
    return float(np.mean([y[i] in order[i] for i in range(len(y))]))


def space_recall(proba: np.ndarray, y: np.ndarray) -> float:
    m = y == A_INDEX[" "]
    if not m.any():
        return float("nan")
    return float((proba[m].argmax(1) == A_INDEX[" "]).mean())


def thumb_at_space(d: dict, sel: "Selector") -> tuple[float, float]:
    """P(a thumb is the selected finger) at true spaces vs elsewhere - analyze_drift's anchor."""
    m = np.array([c == " " for c in d["keys"]])
    th = sel.pick(d)[1]
    return (float(th[m].mean()) if m.any() else float("nan"),
            float(th[~m].mean()) if (~m).any() else float("nan"))


def evaluate(model, d: dict) -> dict:
    proba = model.proba(d) if isinstance(model, GaussXY) else model.proba(d["sess"], d["k"])
    keep = np.isin(d["y"], getattr(model, "classes", np.arange(NA)))
    if not keep.any():
        return {"n": 0}
    return {"n": int(keep.sum()), "top1": topk(proba[keep], d["y"][keep], 1),
            "top5": topk(proba[keep], d["y"][keep], 5),
            "space": space_recall(proba[keep], d["y"][keep])}


# ------------------------------------------------------------------ approaches
def build(name: str, ds: list[dict], seed: int):
    if name.startswith("gauss("):
        return GaussXY.fit(ds, name[6:-1], seed)
    if name == "pose->key abs tips@t0":
        return KeyClf.fit(ds, "abs", JOINTS_TIPS, seed, offsets=(0,))
    if name.startswith("pose->key "):
        return KeyClf.fit(ds, name.split()[-1], JOINTS_FULL, seed)
    if name.startswith("pose->xy->gauss "):
        return XYReg.fit(ds, name.split()[-1], JOINTS_FULL, seed)
    raise ValueError(name)


APPROACHES = ("gauss(reported)", "gauss(tipvy)", "gauss(fingerid)", "pose->key abs tips@t0",
              "pose->key abs", "pose->key rel", "pose->xy->gauss abs", "pose->xy->gauss rel")
STOCHASTIC = {"gauss(fingerid)", "pose->key abs", "pose->key abs tips@t0", "pose->key rel",
              "pose->xy->gauss abs", "pose->xy->gauss rel"}


# ------------------------------------------------------------------ CLI: eval
def cmd_eval(a) -> int:
    sids = list(a.sessions)
    ds = {s: dataset(s) for s in sids + [a.heldout]}
    for s in ds:
        print(f"{s}: {len(ds[s]['keys'])} labelled taps, {len(set(ds[s]['keys']))} keys", flush=True)
    print()
    names = list(a.only.split(",")) if a.only else list(APPROACHES)
    hdr = (f"{'approach':<24}{'LOSO top1':>18}{'LOSO top5':>18}{'LOSO space':>18}"
           f"{'held top1':>18}{'held top5':>12}{'held space':>12}")
    print(hdr)
    print("-" * len(hdr))
    rows = {}
    for name in names:
        seeds = SEEDS if name in STOCHASTIC else (0,)
        loso, held = {k: [] for k in ("top1", "top5", "space")}, {k: [] for k in ("top1", "top5", "space")}
        for seed in seeds:
            per = {k: [] for k in ("top1", "top5", "space")}
            wts = []
            for held_out in sids:
                tr = [ds[s] for s in sids if s != held_out]
                r = evaluate(build(name, tr, seed), ds[held_out])
                for k in per:
                    per[k].append(r[k])
                wts.append(r["n"])
            wts = np.array(wts, float)
            for k in per:
                loso[k].append(float(np.average(per[k], weights=wts)))
            r = evaluate(build(name, [ds[s] for s in sids], seed), ds[a.heldout])
            for k in held:
                held[k].append(r[k])
        rows[name] = {"loso": loso, "held": held}
        print(f"{name:<24}" + "".join(_cell(loso[k]) for k in ("top1", "top5", "space"))
              + f"{np.mean(held['top1']):>18.3f}{np.mean(held['top5']):>12.3f}"
              f"{np.mean(held['space']):>12.3f}", flush=True)
    print("\nfinger selector: P(thumb picked) at true space / elsewhere")
    for kind in ("reported", "flexvel", "tipvy", "fingerid"):
        cells = []
        for s in ds:
            sel = Selector.fit(kind, [ds[x] for x in sids if x != s] or [ds[sids[0]]], 0)
            sp, ns = thumb_at_space(ds[s], sel)
            cells.append(f"{s[9:15]}={sp:.3f}/{ns:.3f}")
        print(f"  {kind:<10} " + "  ".join(cells), flush=True)
    return 0


def _cell(v: list[float]) -> str:
    if len(v) == 1:
        return f"{v[0]:>18.3f}"
    return f"{np.mean(v):>13.3f}+-{np.std(v):.3f}"


# ------------------------------------------------------------------ CLI: desk
def cmd_desk(a) -> int:
    sids = list(a.sessions)
    ds = [dataset(s) for s in sids]
    desk = load_sess(a.desk)
    taps = read_jsonl(sess_path(a.desk) / a.taps_name)
    k = desk.rows(taps)
    ref_chars = Counter()
    for phrase, _ in desk_segments(sess_path(a.desk), taps):
        for c in phrase:
            if c in A_INDEX:
                ref_chars[c] += 1
    tot = sum(ref_chars.values())
    ref = np.zeros(NA)
    for c, n in ref_chars.items():
        ref[A_INDEX[c]] = n / tot
    print(f"desk: {len(taps)} taps, reference text {tot} chars over "
          f"{len(desk_segments(sess_path(a.desk), taps))} phrases\n")

    char_lm = CharLM.load(Path(a.lm))
    word_lm = WordLM.load(Path(a.lm).with_suffix(".words.json"))
    segs = desk_segments(sess_path(a.desk), taps)
    print(f"{'model':<24}{'top-class':>22}{'TV(marg,ref)':>14}{'CER':>8}")
    print("-" * 68)
    uni = np.full((len(taps), NA), 1.0 / NA)
    _desk_row("LM only (uniform obs)", uni, ref, segs, taps, char_lm, word_lm, a.beam, k)
    for name in ("pose->key abs", "pose->key rel", "pose->xy->gauss abs", "pose->xy->gauss rel",
                 "gauss(reported)"):
        dd = {"sess": desk, "k": k, "taps": taps, "sid": a.desk,
              "xy": np.array([[t["x"], t["y"]] for t in taps], float)}
        ps = []
        for seed in (SEEDS if name in STOCHASTIC else (0,)):
            m = build(name, ds, seed)
            ps.append(m.proba(dd) if isinstance(m, GaussXY) else m.proba(desk, k))
        _desk_row(name, np.mean(ps, 0), ref, segs, taps, char_lm, word_lm, a.beam, k)
    return 0


def _desk_row(name, proba, ref, segs, taps, char_lm, word_lm, beam, k):
    marg = proba.mean(0)
    top = int(marg.argmax())
    tv = 0.5 * np.abs(marg - ref).sum()
    idx = {round(t["t"], 6): i for i, t in enumerate(taps)}
    tot_c = ref_c = 0
    for text, seg in segs:
        if not seg:
            ref_c += len(text)
            tot_c += len(text)
            continue
        rows = np.array([idx[round(t["t"], 6)] for t in seg])
        hyp = beam_decode(np.log(proba[rows]), char_lm, word_lm, Weights(), beam)
        tot_c += edit_distance(text, hyp)
        ref_c += len(text)
    print(f"{name:<24}{ALPHABET[top]!r}={marg[top]:.2f}{'':>12}{tv:>14.3f}"
          f"{tot_c/max(1,ref_c):>8.3f}", flush=True)


# ------------------------------------------------------------------ CLI: control CER
def cmd_control(a) -> int:
    """Measured decode CER on a keyboard session, the budget decode.py was interpolating."""
    from phase0.analysis.decode import OracleSpatial, kbd_segments

    tr = [dataset(s) for s in a.sessions if s != a.session]
    d = dataset(a.session)
    taps = read_jsonl(sess_path(a.session) / "taps.jsonl")
    k = d["sess"].rows(taps)
    idx = {round(t["t"], 6): i for i, t in enumerate(taps)}
    segs = kbd_segments(sess_path(a.session), taps)
    char_lm = CharLM.load(Path(a.lm))
    word_lm = WordLM.load(Path(a.lm).with_suffix(".words.json"))
    dd = {"sess": d["sess"], "k": k, "taps": taps, "sid": a.session,
          "xy": np.array([[t["x"], t["y"]] for t in taps], float)}
    rows = {"LM only": np.full((len(taps), NA), 1.0 / NA)}
    for name in ("gauss(reported)", "pose->key abs"):
        m = build(name, tr, 0)
        rows[name] = m.proba(dd) if isinstance(m, GaussXY) else m.proba(d["sess"], k)
    orc = OracleSpatial(sess_path(a.session), "taps.jsonl", 1.0)
    rows["oracle keys"] = np.exp(orc.logp(sess_path(a.session), taps))
    print(f"{a.session}: {len(taps)} taps, {len(segs)} control segments")
    print(f"{'spatial':<20}{'CER':>8}")
    for name, proba in rows.items():
        tot = ref = 0
        for text, seg in segs:
            if not seg:
                tot += len(text)
                ref += len(text)
                continue
            r = np.array([idx[round(t["t"], 6)] for t in seg])
            hyp = beam_decode(np.log(np.maximum(proba[r], 1e-12)), char_lm, word_lm,
                              Weights(), a.beam)
            tot += edit_distance(text, hyp)
            ref += len(text)
        print(f"{name:<20}{tot/max(1,ref):>8.3f}", flush=True)
    return 0


# ------------------------------------------------------------------ CLI: fit/apply
def cmd_fit(a) -> int:
    ds = [dataset(s) for s in a.sessions]
    n = sum(len(d["keys"]) for d in ds)
    m = build(a.approach, ds, a.seed)
    sel = Selector.fit("fingerid", ds, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_bytes(pickle.dumps({"approach": a.approach, "model": m, "sel": sel},
                                         protocol=4))
    print(f"{a.approach} on {n} labelled taps from {len(ds)} sessions -> {a.out}")
    return 0


def cmd_apply(a) -> int:
    blob = pickle.loads(Path(a.model).read_bytes())
    m, sel = blob["model"], blob.get("sel")
    s = load_sess(a.session)
    taps = read_jsonl(Path(a.taps) if a.taps else sess_path(a.session) / "taps.jsonl")
    k = s.rows(taps)
    d = {"sess": s, "k": k, "taps": taps, "sid": a.session,
         "xy": np.array([[t["x"], t["y"]] for t in taps], float)}
    proba = m.proba(d) if isinstance(m, GaussXY) else m.proba(s, k)
    xy, hand, tip = (None, None, None)
    if sel is not None:
        xy, _, hand, tip = sel.pick(d)
    if isinstance(m, XYReg):
        xy = m.predict_xy(s, k)
    out = Path(a.out) if a.out else sess_path(a.session) / "taps_pos.jsonl"
    with open(out, "w") as fh:
        for n, tp in enumerate(taps):
            order = np.argsort(-proba[n])[:a.topk]
            row = dict(tp)
            if xy is not None:
                row["x"], row["y"] = float(xy[n, 0]), float(xy[n, 1])
            if hand is not None:
                row["hand"], row["finger"] = int(hand[n]), int(tip[n])
            row["key_probs"] = {ALPHABET[c]: round(float(proba[n, c]), 6) for c in order}
            fh.write(json.dumps(row) + "\n")
    print(f"wrote {len(taps)} taps with key_probs -> {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.tap_pos", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--sessions", nargs="+", default=list(TRAIN_SESSIONS))
    f.add_argument("--approach", default="pose->key abs")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--out", default=str(DEFAULT_MODEL))
    p = sub.add_parser("apply")
    p.add_argument("session")
    p.add_argument("--taps", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--topk", type=int, default=8)
    e = sub.add_parser("eval")
    e.add_argument("--sessions", nargs="+", default=list(TRAIN_SESSIONS))
    e.add_argument("--heldout", default=HELDOUT_SESSION)
    e.add_argument("--only", default=None)
    c = sub.add_parser("control")
    c.add_argument("--session", default=HELDOUT_SESSION)
    c.add_argument("--sessions", nargs="+", default=list(TRAIN_SESSIONS))
    c.add_argument("--lm", default="models/charlm.npz")
    c.add_argument("--beam", type=int, default=30)
    d = sub.add_parser("desk")
    d.add_argument("--sessions", nargs="+", default=list(TRAIN_SESSIONS))
    d.add_argument("--desk", default=DESK_SESSION)
    d.add_argument("--taps-name", default="taps_desk.jsonl")
    d.add_argument("--lm", default="models/charlm.npz")
    d.add_argument("--beam", type=int, default=30)
    a = ap.parse_args(argv)
    return {"fit": cmd_fit, "apply": cmd_apply, "eval": cmd_eval, "desk": cmd_desk,
            "control": cmd_control}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
