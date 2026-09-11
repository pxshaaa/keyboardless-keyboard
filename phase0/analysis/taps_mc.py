"""Motion-compensated tap detector: fingertip velocity reversal in the palm frame.
Run: python -m phase0.analysis.taps_mc data/sessions/<session_id> [--out F] [--sweep]"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

FINGERTIPS = (4, 8, 12, 16, 20)
PIP_OF = {4: 3, 8: 6, 12: 10, 16: 14, 20: 18}
MCP_OF = {4: 2, 8: 5, 12: 9, 16: 13, 20: 17}
PALM_JOINTS = (0, 5, 9, 13, 17)
HAND_ID = {"Left": 0, "Right": 1}


@dataclass
class Params:
    frame: str = "palm"  # palm | image  (motion compensation on/off)
    axis: str = "reversal"  # reversal | pca | pca_neg | imgy
    k_desc: float = 3.0  # descent gate in MADs of this finger's velocity
    k_reb: float = 2.0  # rebound gate in MADs
    cos_max: float = -0.3  # reversal only: max cos(angle) between pre and post velocity
    tip_ratio: float = 0.0  # tip speed must exceed this * max(PIP, MCP) speed; 0 = off
    win: int = 7  # SavGol window (samples)
    vel_win: float = 0.060  # s, half-width searched for peak pre/post velocity
    refractory: float = 0.080  # s, same finger
    nms_hand: float = 0.040  # s, cross-finger within a hand; 0 = off
    refractory_global: float = 0.0  # s, across both hands; 0 = off
    pca_win: float = 1.0  # s, rolling PCA window
    min_conf: float = 0.5


@dataclass
class Tap:
    t: float
    hand: int
    finger: int
    x: float
    y: float
    conf: float
    i: int
    score: float

    def to_record(self) -> dict:
        return {"t": float(self.t), "hand": self.hand, "finger": self.finger,
                "x": float(self.x), "y": float(self.y), "conf": float(self.conf), "i": int(self.i)}


def load_hands(session: Path) -> dict[str, dict]:
    """Per handedness label: T, X[n,21], Y[n,21], C[n,21], I[n]. Keyed on handedness
    because the hand slot swaps labels for ~400 frames in this session."""
    import pyarrow.parquet as pq

    tb = pq.read_table(session / "landmarks.parquet")
    c = {n: np.asarray(tb.column(n)) for n in tb.column_names}
    out = {}
    for label in ("Left", "Right"):
        m = c["handedness"] == label
        frames = np.unique(c["i"][m])
        idx = {f: k for k, f in enumerate(frames)}
        n = len(frames)
        X = np.full((n, 21), np.nan)
        Y = np.full((n, 21), np.nan)
        C = np.zeros((n, 21))
        T = np.full(n, np.nan)
        rows = np.array([idx[i] for i in c["i"][m]])
        X[rows, c["joint"][m]] = c["x"][m]
        Y[rows, c["joint"][m]] = c["y"][m]
        C[rows, c["joint"][m]] = c["conf"][m]
        T[rows] = c["t"][m]
        out[label] = {"T": T, "X": X, "Y": Y, "C": C, "I": frames}
    return out


def palm_frame(X: np.ndarray, Y: np.ndarray):
    """Centroid, unit axis (wrist -> middle MCP) and span per frame."""
    cx = X[:, PALM_JOINTS].mean(1)
    cy = Y[:, PALM_JOINTS].mean(1)
    ax = X[:, 9] - X[:, 0]
    ay = Y[:, 9] - Y[:, 0]
    span = np.hypot(ax, ay)
    span = np.where(span > 1e-6, span, np.nan)
    return cx, cy, ax / span, ay / span, span


def compensate(X, Y, j, cx, cy, ux, uy, span, frame: str):
    """Joint j as (u, v) in span units: palm frame, or image coords scaled by median span."""
    if frame == "image":
        s = np.nanmedian(span)
        return X[:, j] / s, Y[:, j] / s
    rx, ry = X[:, j] - cx, Y[:, j] - cy
    return (rx * ux + ry * uy) / span, (-rx * uy + ry * ux) / span


def sg(v: np.ndarray, win: int, dt: float, deriv: int) -> np.ndarray:
    return savgol_filter(v, win, 2, deriv=deriv, delta=dt, mode="interp")


def rolling_pca_axis(u, v, T, win_s):
    """Dominant direction of the (u,v) trajectory in a centred window per sample,
    sign-continuous so the projection does not flip between neighbours."""
    n = len(u)
    axes = np.zeros((n, 2))
    half = win_s / 2
    prev = np.array([0.0, 1.0])
    for k in range(n):
        a = np.searchsorted(T, T[k] - half)
        b = np.searchsorted(T, T[k] + half)
        pu, pv = u[a:b] - u[a:b].mean(), v[a:b] - v[a:b].mean()
        cov = np.array([[pu @ pu, pu @ pv], [pu @ pv, pv @ pv]])
        w, e = np.linalg.eigh(cov)
        ax = e[:, np.argmax(w)]
        if ax @ prev < 0:
            ax = -ax
        axes[k] = ax
        prev = ax
    return axes


def mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) + 1e-9


def _win(T, lo, hi):
    return int(np.searchsorted(T, lo, "left")), int(np.searchsorted(T, hi, "right"))


def candidates_1d(T, s, vel, gate_d, gate_r, p: Params):
    """Positive->negative zero crossings of vel with descent/rebound gates. Returns (t, score)."""
    out = []
    for k in range(len(T) - 1):
        if not (vel[k] > 0 >= vel[k + 1]):
            continue
        frac = vel[k] / (vel[k] - vel[k + 1])
        tc = T[k] + frac * (T[k + 1] - T[k])
        lo, hi = _win(T, tc - p.vel_win, tc)
        pre = vel[min(lo, k):max(hi, k + 1)]
        desc = pre.max() if pre.size else 0.0
        lo2, hi2 = _win(T, tc, tc + p.vel_win)
        post = vel[min(lo2, k + 1):max(hi2, k + 2)]
        reb = -post.min() if post.size else 0.0
        if desc < gate_d or reb < gate_r:
            continue
        out.append((tc, k, min(desc / gate_d, reb / gate_r)))
    return out


def candidates_reversal(T, vu, vv, gate_d, gate_r, p: Params):
    """Local minima of 2D speed where pre- and post-velocity are anti-parallel and both
    strong. Direction-free: works whatever the press direction is in the image."""
    speed = np.hypot(vu, vv)
    out = []
    for k in range(1, len(T) - 1):
        if not (speed[k] <= speed[k - 1] and speed[k] < speed[k + 1]):
            continue
        tc = T[k]
        lo, hi = _win(T, tc - p.vel_win, tc)
        lo2, hi2 = _win(T, tc, tc + p.vel_win)
        pre = np.array([vu[lo:k].mean(), vv[lo:k].mean()]) if k > lo else np.zeros(2)
        post = np.array([vu[k + 1:hi2].mean(), vv[k + 1:hi2].mean()]) if hi2 > k + 1 else np.zeros(2)
        pre_mag = speed[lo:k].max() if k > lo else 0.0
        post_mag = speed[k + 1:hi2].max() if hi2 > k + 1 else 0.0
        if pre_mag < gate_d or post_mag < gate_r:
            continue
        den = np.linalg.norm(pre) * np.linalg.norm(post)
        if den == 0 or (pre @ post) / den > p.cos_max:
            continue
        out.append((tc, k, min(pre_mag / gate_d, post_mag / gate_r)))
    return out


def detect_hand(label: str, h: dict, p: Params) -> list[Tap]:
    T, X, Y, C, I = h["T"], h["X"], h["Y"], h["C"], h["I"]
    dt = float(np.median(np.diff(T)))
    cx, cy, ux, uy, span = palm_frame(X, Y)
    hand = HAND_ID[label]
    taps: list[Tap] = []
    for j in FINGERTIPS:
        u, v = compensate(X, Y, j, cx, cy, ux, uy, span, p.frame)
        ok = np.isfinite(u) & np.isfinite(v)
        # this session has no dropouts; if it had, segments would be needed here
        u = np.where(ok, u, np.nanmean(u))
        v = np.where(ok, v, np.nanmean(v))
        vu, vv = sg(u, p.win, dt, 1), sg(v, p.win, dt, 1)

        if p.axis == "reversal":
            g_d = p.k_desc * mad(np.hypot(vu, vv))
            g_r = p.k_reb * mad(np.hypot(vu, vv))
            cands = candidates_reversal(T, vu, vv, g_d, g_r, p)
        else:
            if p.axis == "imgy":
                # image-y component of the compensated tip (u,v are span units in palm frame)
                s = v if p.frame == "image" else u * uy + v * ux
            else:
                axes = rolling_pca_axis(sg(u, p.win, dt, 0), sg(v, p.win, dt, 0), T, p.pca_win)
                # sign: + = away from palm centroid (finger extension), as a press mostly is
                sign = np.sign(axes[:, 0] * u + axes[:, 1] * v + 1e-9)
                sign = np.where(np.median(sign) >= 0, 1.0, -1.0)
                s = axes[:, 0] * sg(u, p.win, dt, 0) + axes[:, 1] * sg(v, p.win, dt, 0)
                s = s * sign * (-1.0 if p.axis == "pca_neg" else 1.0)
            vel = sg(s, p.win, dt, 1)
            g_d, g_r = p.k_desc * mad(vel), p.k_reb * mad(vel)
            cands = candidates_1d(T, s, vel, g_d, g_r, p)

        if p.tip_ratio > 0:
            sp_tip = np.hypot(vu, vv)
            others = []
            for jj in (PIP_OF[j], MCP_OF[j]):
                uu, vv2 = compensate(X, Y, jj, cx, cy, ux, uy, span, p.frame)
                others.append(np.hypot(sg(uu, p.win, dt, 1), sg(vv2, p.win, dt, 1)))
            sp_other = np.max(others, axis=0)
            kept = []
            for tc, k, sc in cands:
                lo, hi = _win(T, tc - p.vel_win, tc + p.vel_win)
                if sp_tip[lo:hi].max() >= p.tip_ratio * sp_other[lo:hi].max():
                    kept.append((tc, k, sc))
            cands = kept

        xs, ys = sg(X[:, j], 5, dt, 0), sg(Y[:, j], 5, dt, 0)
        for tc, k, sc in cands:
            if C[k, j] < p.min_conf:
                continue
            taps.append(Tap(tc, hand, j, xs[k], ys[k], C[k, j], int(I[k]), sc))
    return taps


def nms(taps: list[Tap], p: Params) -> list[Tap]:
    """Greedy by score: same-finger refractory, same-hand cross-finger window, global."""
    kept: list[Tap] = []
    for z in sorted(taps, key=lambda z: (-z.score, z.t)):
        clash = False
        for a in kept:
            d = abs(z.t - a.t)
            if a.hand == z.hand and a.finger == z.finger and d < p.refractory:
                clash = True
            elif a.hand == z.hand and p.nms_hand > 0 and d < p.nms_hand:
                clash = True
            elif p.refractory_global > 0 and d < p.refractory_global:
                clash = True
            if clash:
                break
        if not clash:
            kept.append(z)
    return sorted(kept, key=lambda z: z.t)


def detect(hands: dict, p: Params) -> list[Tap]:
    taps = []
    for label, h in hands.items():
        taps.extend(detect_hand(label, h, p))
    return nms(taps, p)


def write(taps: list[Tap], out: Path) -> None:
    with out.open("w") as fh:
        for z in taps:
            fh.write(json.dumps(z.to_record()) + "\n")


# ---------------------------------------------------------------- tuning / reporting

def scores(session: Path, taps: list[Tap]) -> dict:
    from phase0.analysis.eval_taps import keydown_times, score

    kt = keydown_times(session)
    tt = np.array([z.t for z in taps])
    cut = kt.min() + 0.6 * (kt.max() - kt.min())
    return {"all": score(kt, tt), "train": score(kt, tt, None, cut), "holdout": score(kt, tt, cut, None)}


def line(name: str, s: dict) -> str:
    from phase0.analysis.eval_taps import fmt

    return fmt(name, s)


def per_finger_recall(session: Path, taps: list[Tap]) -> dict:
    """Which finger the paired tap came from, per keydown (keys carry no finger label)."""
    from phase0.analysis.analyze_drift import pair_events
    from phase0.analysis.eval_taps import keydown_times

    kt = keydown_times(session)
    taps = sorted(taps, key=lambda z: z.t)
    pairs = pair_events(kt.tolist(), [z.t for z in taps], 0.080)
    hit = {}
    for _, j in pairs:
        z = taps[j]
        hit[(z.hand, z.finger)] = hit.get((z.hand, z.finger), 0) + 1
    fired = {}
    for z in taps:
        fired[(z.hand, z.finger)] = fired.get((z.hand, z.finger), 0) + 1
    return {k: (hit.get(k, 0), fired[k]) for k in sorted(fired)}


def sweep(session: Path, hands: dict, base: Params, full: bool = False) -> Params:
    """Grid over gates on TRAIN only; pick best train F1, tie -> fewer still FPs."""
    grid = {
        "k_desc": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
        "k_reb": [1.0, 1.5, 2.0, 3.0, 4.0],
        "cos_max": [-0.2, -0.5, -0.7] if base.axis == "reversal" else [base.cos_max],
    }
    if full:
        grid["tip_ratio"] = [0.0, 1.0, 1.3]
        grid["refractory_global"] = [0.0, 0.05]
    best, best_key = None, None
    for vals in itertools.product(*grid.values()):
        p = replace(base, **dict(zip(grid.keys(), vals)))
        s = scores(session, detect(hands, p))["train"]
        key = (s["f1"], -s["still_fp"])
        if best_key is None or key > best_key:
            best, best_key = p, key
    return best


ABLATION = [
    ("A image-y, fixed-ish gates (baseline-like, MAD)", dict(frame="image", axis="imgy", nms_hand=0.0)),
    ("B + palm-frame compensation, image-y axis", dict(frame="palm", axis="imgy", nms_hand=0.0)),
    ("C palm frame, PCA axis (+)", dict(frame="palm", axis="pca", nms_hand=0.0)),
    ("D palm frame, PCA axis (-)", dict(frame="palm", axis="pca_neg", nms_hand=0.0)),
    ("E palm frame, 2D reversal", dict(frame="palm", axis="reversal", nms_hand=0.0)),
    ("F E + cross-finger NMS 40ms", dict(frame="palm", axis="reversal", nms_hand=0.040)),
    ("G F + tip-dominant gate", dict(frame="palm", axis="reversal", nms_hand=0.040, tip_ratio=1.0)),
    ("H G + global refractory 50ms", dict(frame="palm", axis="reversal", nms_hand=0.040,
                                          tip_ratio=1.0, refractory_global=0.05)),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.taps_mc", description=__doc__)
    ap.add_argument("session", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default <session>/taps_mc.jsonl")
    ap.add_argument("--sweep", action="store_true", help="tune gates on train + ablation table")
    ap.add_argument("--axis", default="reversal", choices=["reversal", "pca", "pca_neg", "imgy"])
    ap.add_argument("--frame", default="palm", choices=["palm", "image"])
    ap.add_argument("--k-desc", type=float, default=Params.k_desc)
    ap.add_argument("--k-reb", type=float, default=Params.k_reb)
    ap.add_argument("--cos-max", type=float, default=Params.cos_max)
    ap.add_argument("--tip-ratio", type=float, default=Params.tip_ratio)
    ap.add_argument("--nms-hand", type=float, default=Params.nms_hand)
    ap.add_argument("--refractory-global", type=float, default=Params.refractory_global)
    a = ap.parse_args(argv)

    hands = load_hands(a.session)
    p = Params(frame=a.frame, axis=a.axis, k_desc=a.k_desc, k_reb=a.k_reb, cos_max=a.cos_max,
               tip_ratio=a.tip_ratio, nms_hand=a.nms_hand, refractory_global=a.refractory_global)

    if a.sweep:
        print("== ablation (each row: gates re-tuned on train for that configuration) ==")
        for name, kw in ABLATION:
            fixed = {k: v for k, v in kw.items() if k in ("frame", "axis", "nms_hand")}
            q = replace(Params(**fixed), **{k: v for k, v in kw.items() if k not in fixed})
            q = sweep(a.session, hands, q)
            s = scores(a.session, detect(hands, q))
            print(f"\n{name}\n  params: k_desc={q.k_desc} k_reb={q.k_reb} cos_max={q.cos_max} "
                  f"tip_ratio={q.tip_ratio} nms_hand={q.nms_hand} refr_global={q.refractory_global}")
            for k in ("all", "train", "holdout"):
                print("  " + line(k, s[k]))
        print("\n== final: full sweep on chosen axis ==")
        p = sweep(a.session, hands, p, full=True)
        print(p)

    taps = detect(hands, p)
    out = a.out or a.session / "taps_mc.jsonl"
    write(taps, out)
    print(f"wrote {out} ({len(taps)} taps)")
    s = scores(a.session, taps)
    for k in ("all", "train", "holdout"):
        print(line(k, s[k]))
    print("per (hand, finger): hits / fired")
    for (h, f), (hit, n) in per_finger_recall(a.session, taps).items():
        print(f"  hand {h} finger {f:2d}: {hit:3d} / {n:3d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
