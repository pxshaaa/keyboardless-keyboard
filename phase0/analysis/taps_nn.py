"""Temporal CNN keypress detector: per-frame P(keydown) from both hands' landmarks.
Run: python -m phase0.analysis.taps_nn train --sessions A B / predict <session> [--model M]"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks

from phase0.analysis.eval_taps import keydown_times, score

DEFAULT_MODEL = Path("models/taps_nn.pt")
SIDES = ("Left", "Right")
TIPS = (4, 8, 12, 16, 20)
MCP_OF = {4: 2, 8: 5, 12: 9, 16: 13, 20: 17}
PALM = (0, 5, 9, 13, 17)
CHANNELS = ("x", "y", "conf", "z", "wx", "wy", "wz")
DT = 1.0 / 60.0

# touch-typing home assignment; ambiguous keys train no hand head at all
LEFT_KEYS = set("qwertasdfgzxcvb12345") | {"tab", "esc"}
RIGHT_KEYS = set("yuiophjklnm67890") | {"backspace", "enter"}


# ------------------------------------------------------------------ loading
def _resolve_sides(i: np.ndarray, hand: np.ndarray, handedness: np.ndarray,
                   y0: np.ndarray) -> np.ndarray:
    """Side 0/1 from the handedness LABEL (the hand slot swaps freely); on frames where both
    slots claim one label, fall back to wrist position vs per-side medians of clean frames."""
    side = np.where(handedness == "Right", 1, 0).astype(np.int8)
    pair = i.astype(np.int64) * 2 + hand.astype(np.int64)
    upair, first = np.unique(pair, return_index=True)
    pside = side[first]
    pframe = i[first]
    pslot = hand[first]
    py = y0[first]
    # a frame is ambiguous when its two slots carry the same handedness label
    order = np.argsort(pframe, kind="stable")
    bad = np.zeros(len(upair), bool)
    k = 0
    while k < len(order):
        j = k
        while j + 1 < len(order) and pframe[order[j + 1]] == pframe[order[k]]:
            j += 1
        if j > k and len(set(pside[order[k:j + 1]])) == 1:
            bad[order[k:j + 1]] = True
        k = j + 1
    med = [np.nanmedian(py[(~bad) & (pside == s)]) if ((~bad) & (pside == s)).any() else np.nan
           for s in (0, 1)]
    if np.isfinite(med).all():
        pside = np.where(bad, (np.abs(py - med[1]) < np.abs(py - med[0])).astype(np.int8), pside)
    else:
        pside = np.where(bad, pslot.astype(np.int8), pside)
    lut = dict(zip(upair.tolist(), pside.tolist()))
    return np.array([lut[p] for p in pair.tolist()], dtype=np.int8)


def load_frames(session: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (frame_i [F], t [F], P [F,2,21,7]) indexed by handedness; NaN where a hand is absent."""
    import pyarrow.parquet as pq

    tb = pq.read_table(session / "landmarks.parquet")
    have = [c for c in CHANNELS if c in tb.schema.names]
    c = {n: np.asarray(tb.column(n)) for n in ("i", "t", "hand", "handedness", "joint", *have)}
    frames, inv = np.unique(c["i"], return_inverse=True)
    t = np.zeros(len(frames))
    t[inv] = c["t"]
    wrist_y = np.zeros(len(c["i"]))
    w = c["joint"] == 0
    lut = {(a, b): v for a, b, v in zip(c["i"][w], c["hand"][w], c["y"][w])}
    wrist_y = np.array([lut.get((a, b), np.nan) for a, b in zip(c["i"], c["hand"])])
    side = _resolve_sides(c["i"], c["hand"], c["handedness"], wrist_y)
    P = np.full((len(frames), 2, 21, len(CHANNELS)), np.nan)
    for k, ch in enumerate(CHANNELS):
        if ch in have:
            P[inv, side, c["joint"], k] = c[ch]
    return frames, t, P


# ------------------------------------------------------------------ features
def _d1(A: np.ndarray) -> np.ndarray:
    D = np.full_like(A, np.nan)
    D[1:-1] = (A[2:] - A[:-2]) / 2.0
    return D


def _col(a: np.ndarray) -> np.ndarray:
    return a[:, None] if a.ndim == 1 else a


def hand_features(H: np.ndarray, depth: bool) -> np.ndarray:
    """H [F,21,7] for one hand -> [F,D] scale/translation-invariant pose + motion."""
    n = len(H)
    xy = H[:, :, :2]
    wrist = xy[:, 0]
    span = np.linalg.norm(xy[:, 9] - wrist, axis=1)
    span = np.where(span > 1e-6, span, np.nan)
    rel = (xy[:, 1:] - wrist[:, None, :]) / span[:, None, None]
    flex = []
    for tip in TIPS:
        flex.append(np.linalg.norm(xy[:, tip] - xy[:, MCP_OF[tip]], axis=1) / span)
        flex.append(np.linalg.norm(xy[:, tip] - wrist, axis=1) / span)
    base = [rel.reshape(n, -1), np.stack(flex, 1)]
    if depth:
        z = (H[:, 1:, 3] - H[:, :1, 3]) / span[:, None]
        W = H[:, :, 4:7]
        wspan = np.linalg.norm(W[:, 9] - W[:, 0], axis=1)
        wspan = np.where(wspan > 1e-9, wspan, np.nan)
        wrel = (W[:, 1:] - W[:, :1]) / wspan[:, None, None]
        wflex = []
        for tip in TIPS:
            wflex.append(np.linalg.norm(W[:, tip] - W[:, MCP_OF[tip]], axis=1) / wspan)
            wflex.append(np.linalg.norm(W[:, tip] - W[:, 0], axis=1) / wspan)
        base += [z, wrel.reshape(n, -1), np.stack(wflex, 1)]
    B = np.hstack(base)
    V = _d1(B)
    palm = xy[:, PALM, :].mean(1)
    absm = [_d1(palm) / span[:, None], _d1(xy[:, TIPS, :]).reshape(n, -1) / span[:, None]]
    if depth:
        W = H[:, :, 4:7]
        wspan = np.linalg.norm(W[:, 9] - W[:, 0], axis=1)
        wspan = np.where(wspan > 1e-9, wspan, np.nan)
        absm += [_col(_d1(H[:, PALM, 3].mean(1))) / span[:, None],
                 _d1(H[:, TIPS, 3]) / span[:, None],
                 _d1(W[:, PALM, :].mean(1)) / wspan[:, None],
                 _d1(W[:, TIPS, :]).reshape(n, -1) / wspan[:, None]]
    extra = [_col(np.log(span)), H[:, TIPS, 2], _col(np.isfinite(span).astype(float))]
    return np.hstack([B, V] + absm + extra)


def build_features(P: np.ndarray, depth: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """-> (X [F,C] both hands concatenated, flexvel [F,2,5] for finger attribution)."""
    X, fv = [], []
    for s in range(2):
        X.append(hand_features(P[:, s], depth))
        xy = P[:, s, :, :2]
        span = np.linalg.norm(xy[:, 9] - xy[:, 0], axis=1)
        span = np.where(span > 1e-6, span, np.nan)
        f = np.stack([np.linalg.norm(xy[:, tip] - xy[:, MCP_OF[tip]], axis=1) / span for tip in TIPS], 1)
        fv.append(_d1(f))
    return np.hstack(X).astype(np.float32), np.stack(fv, 1)


# ------------------------------------------------------------------ labels
def key_events(session: Path) -> tuple[np.ndarray, np.ndarray]:
    """-> (times, hand) with hand in {0 left, 1 right, -1 unknown/ambiguous}."""
    ts, hs = [], []
    for line in open(session / "keys.jsonl"):
        if not line.strip():
            continue
        k = json.loads(line)
        if k["event"] != "down":
            continue
        ts.append(k["t"])
        hs.append(0 if k["key"] in LEFT_KEYS else 1 if k["key"] in RIGHT_KEYS else -1)
    o = np.argsort(ts)
    return np.array(ts)[o], np.array(hs, dtype=np.int8)[o]


def _bump(t: np.ndarray, kt: np.ndarray, half: float, sigma: float) -> np.ndarray:
    if len(kt) == 0:
        return np.zeros(len(t), np.float32)
    d = np.abs(t[:, None] - kt[None, :]).min(1)
    return (np.exp(-0.5 * (d / sigma) ** 2) if sigma > 0 else (d <= half)).astype(np.float32)


def make_targets(t: np.ndarray, kt: np.ndarray, kh: np.ndarray, half: float, heads: int,
                 sigma: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """-> (Y [F,heads], Wt [F,heads]); channel 0 = any key, 1/2 = left/right hand."""
    Y = np.zeros((len(t), heads), np.float32)
    Wt = np.ones((len(t), heads), np.float32)
    if len(kt) == 0:
        return Y, Wt
    Y[:, 0] = _bump(t, kt, half, sigma)
    if heads >= 3:
        near_hand = np.zeros(len(t), bool)
        for h in (0, 1):
            kth = kt[kh == h]
            Y[:, 1 + h] = _bump(t, kth, half, sigma)
            near_hand |= _bump(t, kth, half, 0.0) > 0
        # a frame owned only by an unlabelled key teaches nothing about which hand moved
        amb = (_bump(t, kt[kh < 0], half, 0.0) > 0) & ~near_hand
        Wt[amb, 1:] = 0.0
    return Y, Wt


# ------------------------------------------------------------------ model
class ChannelNorm(nn.Module):
    """LayerNorm across channels at each timestep; GroupNorm would pool over time too and so
    make a 384-frame training crop and a full-session inference pass see different statistics."""

    def __init__(self, c: int):
        super().__init__()
        self.norm = nn.LayerNorm(c)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TCN(nn.Module):
    def __init__(self, cin: int, hidden: int, dilations: tuple[int, ...], nout: int, drop: float):
        super().__init__()
        self.stem = nn.Conv1d(cin, hidden, 1)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Conv1d(hidden, hidden, 3, dilation=d, padding=d),
                          ChannelNorm(hidden), nn.GELU(), nn.Dropout(drop))
            for d in dilations)
        self.head = nn.Conv1d(hidden, nout, 1)

    def forward(self, x):
        x = F.gelu(self.stem(x))
        for b in self.blocks:
            x = x + b(x)
        return self.head(x)


def receptive_field(dilations) -> int:
    return 1 + 2 * sum(dilations)


# ------------------------------------------------------------------ config
@dataclass
class Config:
    depth: bool = True
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    hidden: int = 48
    drop: float = 0.15
    heads: int = 3
    label_half: float = 0.033
    label_sigma: float = 0.0
    epochs: int = 60
    steps: int = 24
    crop: int = 384
    batch: int = 8
    lr: float = 3e-3
    wd: float = 1e-4
    pos_weight: float = 3.0
    aug: tuple[str, ...] = ("flip",)
    hand_drop: float = 0.1
    aux_weight: float = 0.4
    mc_feature: bool = False
    seed: int = 0


# ------------------------------------------------------------------ sequences
def mc_channel(session: Path, t: np.ndarray) -> np.ndarray:
    """Hand-crafted taps_mc firings rendered as two Gaussian bumps (per hand) per frame."""
    from phase0.analysis.taps_mc import Params, detect, load_hands

    taps = detect(load_hands(session), Params(nms_hand=0.040, tip_ratio=1.0))
    out = np.zeros((len(t), 2), np.float32)
    for z in taps:
        out[:, z.hand] = np.maximum(out[:, z.hand], np.exp(-0.5 * ((t - z.t) / 0.025) ** 2))
    return out


def flip_P(P: np.ndarray) -> np.ndarray:
    """Mirror across the axis separating the hands (image y here) and swap the hands."""
    Q = P[:, ::-1].copy()
    Q[..., 1] *= -1.0
    Q[..., 5] *= -1.0
    return Q


def warp_extra(E: np.ndarray, f: float) -> np.ndarray:
    n = len(E)
    src = np.clip(np.arange(n) * f, 0, n - 1)
    lo = np.floor(src).astype(int)
    a = (src - lo)[:, None]
    return E[lo] * (1 - a) + E[np.minimum(lo + 1, n - 1)] * a


def warp_P(P: np.ndarray, t: np.ndarray, factor: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(t)
    src = np.clip(np.arange(n) * factor, 0, n - 1)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    a = (src - lo)[:, None, None, None]
    return P[lo] * (1 - a) + P[hi] * a, t[0] + np.arange(n) * (t[1] - t[0])


@dataclass
class Seq:
    X: np.ndarray
    Y: np.ndarray
    W: np.ndarray
    t: np.ndarray
    sess: int = 0
    tsrc: np.ndarray | None = None


_RAW: dict = {}
_FEAT: dict = {}


def raw_session(session: Path):
    key = str(session)
    if key not in _RAW:
        frames, t, P = load_frames(session)
        kt, kh = key_events(session)
        _RAW[key] = (frames, t, P, kt, kh)
    return _RAW[key]


def _feat(session: Path, variant: str, depth: bool, mc: bool, seed: int):
    """Cached features for one variant -> (X, t, kt, kh, tsrc); tsrc is the ORIGINAL clock,
    which is what fold masking must use or a warped copy leaks the held-out slice."""
    key = (str(session), variant, depth, mc, seed)
    if key in _FEAT:
        return _FEAT[key]
    _, t, P, kt, kh = raw_session(session)
    tsrc = t
    extra = mc_channel(session, t) if mc else None
    if variant == "flip":
        P = flip_P(P)
        kh = 1 - kh if kh.size else kh
        extra = extra[:, ::-1].copy() if extra is not None else None
    elif variant.startswith("warp"):
        f = float(variant[4:])
        t0, dt = t[0], t[1] - t[0]
        tsrc = t0 + np.clip(np.arange(len(t)) * f, 0, len(t) - 1) * dt
        P, t = warp_P(P, t, f)
        kt = t0 + (kt - t0) / f
        extra = warp_extra(extra, f) if extra is not None else None
    elif variant == "jitter":
        rng = np.random.default_rng(seed + 17)
        sc = np.nanmedian(np.linalg.norm(P[:, :, 9, :2] - P[:, :, 0, :2], axis=-1))
        P = P.copy()
        P[..., :2] += rng.normal(0, 0.006 * sc, P[..., :2].shape)
        P[..., 3] += rng.normal(0, 0.006 * sc, P[..., 3].shape)
        P[..., 4:7] += rng.normal(0, 0.0006, P[..., 4:7].shape)
    X, _ = build_features(P, depth)
    if extra is not None:
        X = np.hstack([X, extra])
    _FEAT[key] = (X, t, kt, kh, tsrc)
    return _FEAT[key]


def variants(cfg: Config) -> list[str]:
    v = ["id"]
    if "flip" in cfg.aug:
        v.append("flip")
    if "warp" in cfg.aug:
        v += ["warp0.92", "warp1.09"]
    if "jitter" in cfg.aug:
        v.append("jitter")
    return v


def training_sequences(sessions: list[Path], cfg: Config) -> list[Seq]:
    seqs = []
    for si, sess in enumerate(sessions):
        for v in variants(cfg):
            X, t, kt, kh, tsrc = _feat(sess, v, cfg.depth, cfg.mc_feature, cfg.seed)
            Y, W = make_targets(t, kt, kh, cfg.label_half, cfg.heads, cfg.label_sigma)
            seqs.append(Seq(X, Y, W, t, si, tsrc))
    return seqs


def fold_ranges(sessions: list[Path], nfolds: int) -> list[list[tuple[float, float]]]:
    """Per fold, the held-out time interval of each training session."""
    out = []
    for f in range(nfolds):
        cuts = []
        for sess in sessions:
            _, t, _, _, _ = raw_session(sess)
            lo, hi = t[0], t[-1]
            cuts.append((lo + f * (hi - lo) / nfolds, lo + (f + 1) * (hi - lo) / nfolds))
        out.append(cuts)
    return out


def mask_fold(seqs: list[Seq], cuts: list[tuple[float, float]], margin: float) -> list[Seq]:
    """Zero the loss weight on held-out frames of every augmented copy, plus a leakage margin."""
    out = []
    for s in seqs:
        lo, hi = cuts[s.sess]
        ts = s.t if s.tsrc is None else s.tsrc
        keep = ~((ts >= lo - margin) & (ts <= hi + margin))
        out.append(Seq(s.X, s.Y, s.W * keep[:, None].astype(np.float32), s.t, s.sess, s.tsrc))
    return out


# ------------------------------------------------------------------ training
def normalizer(seqs: list[Seq]) -> tuple[np.ndarray, np.ndarray]:
    A = np.vstack([s.X for s in seqs])
    mu = np.nanmean(A, 0)
    sd = np.nanstd(A, 0)
    return np.nan_to_num(mu), np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)


def prep(seqs: list[Seq], mu, sd) -> list[Seq]:
    return [Seq(np.nan_to_num((s.X - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0),
                s.Y, s.W, s.t, s.sess, s.tsrc) for s in seqs]


def train_model(seqs: list[Seq], cfg: Config, verbose: bool = True) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    mu, sd = normalizer(seqs)
    S = prep(seqs, mu, sd)
    cin = S[0].X.shape[1]
    nhand = (cin - (2 if cfg.mc_feature else 0)) // 2
    model = TCN(cin, cfg.hidden, cfg.dilations, cfg.heads, cfg.drop)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=cfg.epochs * cfg.steps)
    pw = torch.tensor([cfg.pos_weight])
    lens = np.array([len(s.X) for s in S], float)
    p = lens / lens.sum()
    for ep in range(cfg.epochs):
        model.train()
        tot = 0.0
        for _ in range(cfg.steps):
            xb, yb, wb = [], [], []
            for _ in range(cfg.batch):
                k = rng.choice(len(S), p=p)
                L = min(cfg.crop, len(S[k].X))
                a = rng.integers(0, len(S[k].X) - L + 1)
                xb.append(S[k].X[a:a + L])
                yb.append(S[k].Y[a:a + L])
                wb.append(S[k].W[a:a + L])
            x = torch.from_numpy(np.stack(xb)).permute(0, 2, 1)
            y = torch.from_numpy(np.stack(yb)).permute(0, 2, 1)
            w = torch.from_numpy(np.stack(wb)).permute(0, 2, 1)
            if cfg.hand_drop > 0:
                keep = (torch.rand(x.shape[0], 2, 1) >= cfg.hand_drop).float()
                m = torch.ones(x.shape[0], cin, 1)
                m[:, :nhand] = keep[:, :1]
                m[:, nhand:2 * nhand] = keep[:, 1:]
                x = x * m
            logit = model(x)
            loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pw, reduction="none")
            hw = torch.ones(1, logit.shape[1], 1)
            hw[:, 1:] = cfg.aux_weight
            loss = (loss * w * hw).sum() / (w * hw).sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        if verbose and (ep + 1) % 20 == 0:
            print(f"  epoch {ep+1:3d}/{cfg.epochs} loss={tot/cfg.steps:.4f}")
    return model, mu, sd


@torch.no_grad()
def predict_probs(model: nn.Module, X: np.ndarray, mu, sd) -> np.ndarray:
    model.eval()
    Z = np.nan_to_num((X - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.from_numpy(Z).T[None].float()
    return torch.sigmoid(model(x))[0].T.numpy()


# ------------------------------------------------------------------ events
def smooth(p: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return p
    k = np.ones(w) / w
    return np.convolve(p, k, mode="same")


def peaks_subframe(p: np.ndarray, t: np.ndarray, thr: float, refr: int) -> tuple[np.ndarray, np.ndarray]:
    """Peak times refined by a parabola through the three samples around each peak."""
    idx, _ = find_peaks(p, height=thr, distance=max(1, refr))
    if len(idx) == 0:
        return idx, np.zeros(0)
    k = np.clip(idx, 1, len(p) - 2)
    a, b, c = p[k - 1], p[k], p[k + 1]
    den = a - 2 * b + c
    off = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / np.where(np.abs(den) > 1e-9, den, 1.0), 0.0)
    off = np.clip(off, -0.5, 0.5)
    return idx, t[k] + off * DT


def extract(prob: np.ndarray, t: np.ndarray, thr: float, w: int, refr: int,
            mode: str = "any") -> tuple[np.ndarray, np.ndarray]:
    """mode 'any' = channel 0 only; 'hands' = union of the two per-hand channels."""
    if mode == "union" and prob.shape[1] >= 3:
        ia, ta = peaks_subframe(smooth(prob[:, 0], w), t, thr, refr)
        ih, th = extract(prob, t, thr, w, refr, "hands")
        keep = [k for k in range(len(ta))
                if not len(th) or np.min(np.abs(ih - ia[k])) >= refr]
        idx = np.concatenate([ih, ia[keep]]).astype(int)
        ts = np.concatenate([th, ta[keep]])
        o = np.argsort(ts)
        return idx[o], ts[o]
    if mode == "hands" and prob.shape[1] >= 3:
        ii, tt = [], []
        for c in (1, 2):
            i2, t2 = peaks_subframe(smooth(prob[:, c], w), t, thr, refr)
            ii.append(i2)
            tt.append(t2)
        idx = np.concatenate(ii)
        ts = np.concatenate(tt)
        o = np.argsort(ts)
        return idx[o], ts[o]
    return peaks_subframe(smooth(prob[:, 0], w), t, thr, refr)


THR_GRID = np.arange(0.05, 0.96, 0.025)
SMOOTH_GRID = (1, 3, 5, 7)
REFR_GRID = (2, 3, 4, 5)
MODES = ("any", "hands", "union")


def tune_events(probs: list[np.ndarray], ts: list[np.ndarray], kts: list[np.ndarray]) -> dict:
    """Grid search threshold/smoothing/refractory/mode by pooled F1 over validation sessions."""
    best = {"f1": -1.0}
    for mode in MODES:
        if mode in ("hands", "union") and probs[0].shape[1] < 3:
            continue
        for w in SMOOTH_GRID:
            sm = [np.stack([smooth(p[:, c], w) for c in range(p.shape[1])], 1) for p in probs]
            for refr in REFR_GRID:
                for thr in THR_GRID:
                    num = den_r = den_p = 0
                    for p, t, kt in zip(sm, ts, kts):
                        _, tt = extract(p, t, thr, 1, refr, mode)
                        s = score(kt, tt)
                        num += s["hits"]
                        den_r += s["keydowns"]
                        den_p += s["taps"]
                    r = num / max(den_r, 1)
                    pr = num / max(den_p, 1)
                    f1 = 2 * r * pr / (r + pr) if (r + pr) else 0.0
                    if f1 > best["f1"]:
                        best = {"f1": f1, "thr": float(thr), "smooth": w, "refr": refr, "mode": mode}
    return best


def pr_auc(prob: np.ndarray, Y: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    y = (Y[:, 0] > 0.5).astype(int)
    return float(average_precision_score(y, prob[:, 0])) if y.any() else float("nan")


# ------------------------------------------------------------------ records
def events_to_records(idx, times, frames, P, flexvel) -> list[dict]:
    out = []
    for k, tt in zip(idx, times):
        m = np.abs(np.nan_to_num(flexvel[k], nan=-1.0))
        s, f = np.unravel_index(int(np.argmax(m)), m.shape)
        tip = TIPS[f]
        out.append({"t": float(tt), "hand": int(s), "finger": int(tip),
                    "x": float(np.nan_to_num(P[k, s, tip, 0])), "y": float(np.nan_to_num(P[k, s, tip, 1])),
                    "conf": float(np.nan_to_num(P[k, s, tip, 2])), "i": int(frames[k])})
    return out


def session_features(session: Path, cfg: Config):
    frames, t, P = load_frames(session)
    X, fv = build_features(P, cfg.depth)
    if cfg.mc_feature:
        X = np.hstack([X, mc_channel(session, t)])
    return frames, t, P, X, fv


def run_session(session: Path, members, mu, sd, cfg: Config, ev: dict):
    frames, t, P, X, fv = session_features(session, cfg)
    prob = ens_probs(members, X, mu, sd)
    idx, times = extract(prob, t, ev["thr"], ev["smooth"], ev["refr"], ev["mode"])
    return frames, t, P, prob, fv, idx, times


# ------------------------------------------------------------------ fit
def ens_probs(members: list[nn.Module], X, mu, sd) -> np.ndarray:
    return np.mean([predict_probs(m, X, mu, sd) for m in members], 0)


def _ensemble(cfg: Config, sessions: list[Path], nfolds: int, verbose: bool):
    """K time-contiguous folds per session; each member is held out of one slice of every session."""
    seqs = training_sequences(sessions, cfg)
    margin = receptive_field(cfg.dilations) * DT + 0.1
    members, mu, sd = [], None, None
    for f, cuts in enumerate(fold_ranges(sessions, nfolds)):
        model, mu, sd = train_model(mask_fold(seqs, cuts, margin), cfg, verbose=False)
        members.append(model)
        if verbose:
            print(f"  fold {f+1}/{nfolds} trained")
    return members, mu, sd, seqs[0].X.shape[1]


def fit_with_cv(cfg: Config, sessions: list[Path], nfolds: int = 4, verbose: bool = True) -> dict:
    """Threshold/refractory are tuned on a WHOLE held-out session, never on within-session folds:
    a member that saw the rest of its own session is far overconfident on it (F1 95 vs 65)."""
    calib = min(sessions, key=lambda s: len(keydown_times(s))) if len(sessions) > 1 else None
    if calib is not None:
        rest = [s for s in sessions if s != calib]
        cm, cmu, csd, _ = _ensemble(cfg, rest, nfolds, verbose)
        X, t, kt, kh, _ = _feat(calib, "id", cfg.depth, cfg.mc_feature, cfg.seed)
        cp = ens_probs(cm, X, cmu, csd)
        ev = tune_events([cp], [t], [kt])
        ap = pr_auc(cp, make_targets(t, kt, kh, cfg.label_half, cfg.heads, cfg.label_sigma)[0])
    members, mu, sd, cin = _ensemble(cfg, sessions, nfolds, verbose)
    if calib is None:
        oof_p, oof_t, oof_k, oof_y = [], [], [], []
        for f, cuts in enumerate(fold_ranges(sessions, nfolds)):
            X, t, kt, kh, _ = _feat(sessions[0], "id", cfg.depth, cfg.mc_feature, cfg.seed)
            m = (t >= cuts[0][0]) & (t <= cuts[0][1])
            oof_p.append(ens_probs([members[f]], X, mu, sd)[m])
            oof_t.append(t[m])
            oof_k.append(kt[(kt >= cuts[0][0]) & (kt <= cuts[0][1])])
            oof_y.append(make_targets(t, kt, kh, cfg.label_half, cfg.heads, cfg.label_sigma)[0][m])
        ev = tune_events(oof_p, oof_t, oof_k)
        ap = pr_auc(np.vstack(oof_p), np.vstack(oof_y))
    return {"members": members, "mu": mu, "sd": sd, "ev": ev, "cv_f1": ev["f1"], "pr_auc": ap,
            "calib": None if calib is None else calib.name, "cin": cin,
            "params": sum(q.numel() for q in members[0].parameters())}


def evaluate(fitres: dict, cfg: Config, session: Path, clip: bool = True) -> dict:
    frames, t, P, X, fv = session_features(session, cfg)
    prob = ens_probs(fitres["members"], X, fitres["mu"], fitres["sd"])
    ev = fitres["ev"]
    idx, times = extract(prob, t, ev["thr"], ev["smooth"], ev["refr"], ev["mode"])
    kt = keydown_times(session)
    if clip and len(times):
        keep = (times >= kt.min() - 0.1) & (times <= kt.max() + 0.1)
        idx, times = idx[keep], times[keep]
    out = score(kt, times)
    out["records"] = events_to_records(idx, times, frames, P, fv)
    out["prob"] = prob
    out["t"] = t
    return out


def save_bundle(path: Path, cfg: Config, r: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"states": [m.state_dict() for m in r["members"]], "cfg": asdict(cfg),
                "mu": r["mu"], "sd": r["sd"], "ev": r["ev"], "cin": r["cin"]}, path)


# ------------------------------------------------------------------ commands
def cmd_train(a) -> int:
    cfg = config_from_args(a)
    print(f"config: {asdict(cfg)}")
    rf = receptive_field(cfg.dilations)
    print(f"sessions={[str(x) for x in a.sessions]} RF={rf} frames (+-{1000*DT*rf/2:.0f} ms)")
    r = fit_with_cv(cfg, a.sessions, nfolds=a.folds)
    print(f"params/member={r['params']} members={len(r['members'])} feat={r['cin']} "
          f"CV PR-AUC={r['pr_auc']:.3f} CV event F1={100*r['cv_f1']:.1f}%")
    print(f"event params: {r['ev']}")
    save_bundle(a.out, cfg, r)
    print(f"saved {a.out}")
    return 0


def load_bundle(path: Path):
    b = torch.load(path, weights_only=False)
    cfg = Config(**{**b["cfg"], "dilations": tuple(b["cfg"]["dilations"]), "aug": tuple(b["cfg"]["aug"])})
    members = []
    for st in b["states"]:
        m = TCN(b["cin"], cfg.hidden, cfg.dilations, cfg.heads, cfg.drop)
        m.load_state_dict(st)
        members.append(m)
    return members, b["mu"], b["sd"], b["ev"], cfg


def cmd_predict(a) -> int:
    members, mu, sd, ev, cfg = load_bundle(a.model)
    frames, t, P, prob, fv, idx, times = run_session(a.session, members, mu, sd, cfg, ev)
    recs = events_to_records(idx, times, frames, P, fv)
    out = a.out or a.session / "taps_nn.jsonl"
    with open(out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(recs)} events -> {out}")
    return 0


def config_from_args(a) -> Config:
    cfg = Config()
    for k in ("hidden", "epochs", "drop", "heads", "pos_weight", "label_half", "label_sigma",
              "aux_weight", "seed"):
        v = getattr(a, k, None)
        if v is not None:
            cfg = replace(cfg, **{k: v})
    if getattr(a, "no_depth", False):
        cfg = replace(cfg, depth=False)
    if getattr(a, "dilations", None):
        cfg = replace(cfg, dilations=tuple(int(x) for x in a.dilations.split(",")))
    if getattr(a, "aug", None) is not None:
        cfg = replace(cfg, aug=tuple(x for x in a.aug.split(",") if x))
    if getattr(a, "mc_feature", False):
        cfg = replace(cfg, mc_feature=True)
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.taps_nn", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--sessions", type=Path, nargs="+", required=True)
    tr.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    tr.add_argument("--hidden", type=int, default=None)
    tr.add_argument("--epochs", type=int, default=None)
    tr.add_argument("--drop", type=float, default=None)
    tr.add_argument("--heads", type=int, default=None)
    tr.add_argument("--pos-weight", dest="pos_weight", type=float, default=None)
    tr.add_argument("--label-half", dest="label_half", type=float, default=None)
    tr.add_argument("--label-sigma", dest="label_sigma", type=float, default=None)
    tr.add_argument("--aux-weight", dest="aux_weight", type=float, default=None)
    tr.add_argument("--seed", type=int, default=None)
    tr.add_argument("--folds", type=int, default=4)
    tr.add_argument("--dilations", default=None, help="comma list, e.g. 1,2,4,8")
    tr.add_argument("--aug", default=None, help="comma list of flip,warp,jitter (empty = none)")
    tr.add_argument("--no-depth", action="store_true")
    tr.add_argument("--mc-feature", dest="mc_feature", action="store_true")
    pr = sub.add_parser("predict")
    pr.add_argument("session", type=Path)
    pr.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    pr.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    return cmd_train(a) if a.cmd == "train" else cmd_predict(a)


if __name__ == "__main__":
    sys.exit(main())
