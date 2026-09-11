"""Taps from finger flexion relative to its own hand (whole-hand travel cancels); CONTRACT taps.jsonl.
Run: python -m phase0.analysis.taps_flexion data/sessions/<session_id> [--out ...] [--sweep]"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from phase0.analysis.detect_taps import FINGER_NAMES, FINGERTIP_JOINTS, load_landmarks
from phase0.analysis.eval_taps import fmt, keydown_times, score

# joints (base, mid, distal, tip) per fingertip id; thumb = CMC/MP/IP/tip
FINGER_CHAIN = {4: (1, 2, 3, 4), 8: (5, 6, 7, 8), 12: (9, 10, 11, 12),
                16: (13, 14, 15, 16), 20: (17, 18, 19, 20)}
PALM_JOINTS = (0, 5, 9, 13, 17)
FEATURES = ("dmcp", "dwrist", "pip", "resfwd", "resx", "rawx", "resspeed", "hand_dmcp", "hand_speed")

# nominal touch-typing finger per key, for per-finger recall against keys.jsonl
KEY_FINGER = {"space": "thumb", "esc": "pinky", "shift": "pinky", "unknown": "?"}
for _keys, _f in (("qazp", "pinky"), ("wsxol", "ring"), ("edcik", "middle"),
                  ("rfvtgbyhnujm", "index")):
    KEY_FINGER.update({k: _f for k in _keys})


@dataclass(frozen=True)
class Params:
    feature: str = "dwrist"
    smooth_window: int = 7
    min_vel: float = 1.0        # span/s (rad/s for pip) toward flexion before contact
    rebound_ratio: float = 0.6  # rebound velocity gate = rebound_ratio * min_vel
    min_amp: float = 0.02       # span units the signal must recover after contact
    vel_window: float = 0.060
    refractory: float = 0.080
    nms_window: float = 0.040
    min_conf: float = 0.5
    max_gap: float = 0.050


@dataclass
class Tap:
    t: float
    hand: int
    finger: int
    x: float
    y: float
    conf: float
    i: int
    strength: float = 0.0

    def to_record(self) -> dict:
        return {"t": float(self.t), "hand": int(self.hand), "finger": int(self.finger),
                "x": float(self.x), "y": float(self.y), "conf": float(self.conf),
                "i": int(self.i)}


def load_hands(session: Path) -> dict[int, dict]:
    """Per hand: t, i, P[n,21,2], conf[n,21] sorted by time (frames missing a joint dropped)."""
    tb = load_landmarks(session)
    c = {n: np.asarray(tb.column(n)) for n in ("i", "t", "hand", "joint", "x", "y", "conf")}
    out: dict[int, dict] = {}
    for hand in sorted({int(h) for h in c["hand"]}):
        m = c["hand"] == hand
        frames, inv = np.unique(c["i"][m], return_inverse=True)
        n = len(frames)
        P = np.full((n, 21, 2), np.nan)
        conf = np.zeros((n, 21))
        t = np.zeros(n)
        j = c["joint"][m]
        P[inv, j, 0] = c["x"][m]
        P[inv, j, 1] = c["y"][m]
        conf[inv, j] = c["conf"][m]
        t[inv] = c["t"][m]
        ok = ~np.isnan(P).any(axis=(1, 2))
        order = np.argsort(t[ok], kind="stable")
        out[hand] = {"t": t[ok][order], "i": frames[ok][order], "P": P[ok][order],
                     "conf": conf[ok][order]}
    return out


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def hand_features(h: dict) -> dict[int, dict[str, np.ndarray]]:
    """All features for one hand, keyed by fingertip id. Lower value = more flexed."""
    P, t = h["P"], h["t"]
    wrist, mmcp = P[:, 0], P[:, 9]
    span = np.linalg.norm(mmcp - wrist, axis=-1)
    # per-frame span is noisy; camera is fixed so a long smooth is safe
    span = savgol_filter(span, min(31, len(span) - (len(span) + 1) % 2), 2) if len(span) > 31 else span
    fwd = (mmcp - wrist) / (span[:, None] + 1e-9)
    centroid = P[:, PALM_JOINTS].mean(axis=1)
    feats: dict[int, dict[str, np.ndarray]] = {}
    for tip, (base, mid, dist, _) in FINGER_CHAIN.items():
        tp = P[:, tip]
        res = (tp - centroid) / span[:, None]
        res_vel = np.gradient(res, t, axis=0)
        feats[tip] = {
            "dmcp": np.linalg.norm(tp - P[:, base], axis=-1) / span,
            "dwrist": np.linalg.norm(tp - wrist, axis=-1) / span,
            "pip": _angle(P[:, base], P[:, mid], P[:, dist]),
            "resfwd": (res * fwd).sum(-1),
            "resx": res[:, 0],
            "rawx": tp[:, 0] / span,
            "resspeed": np.linalg.norm(res_vel, axis=-1),
        }
    # hand-pooled variants: per-finger flexion is near jitter level in a top-down view
    pooled_d = np.mean([feats[k]["dmcp"] for k in (8, 12, 16, 20)], axis=0)
    pooled_s = np.sum([feats[k]["resspeed"] for k in FINGER_CHAIN], axis=0)
    for k in FINGER_CHAIN:
        feats[k]["hand_dmcp"] = pooled_d
        feats[k]["hand_speed"] = pooled_s
    return feats


def _segments(t: np.ndarray, conf: np.ndarray, p: Params) -> list[tuple[int, int]]:
    good = conf >= p.min_conf
    cuts = [0]
    for k in range(1, len(t)):
        if good[k] != good[k - 1] or (t[k] - t[k - 1]) > p.max_gap:
            cuts.append(k)
    cuts.append(len(t))
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if good[a] and b - a >= 5]


def _savgol(s: np.ndarray, t: np.ndarray, w: int, deriv: int) -> np.ndarray:
    w = min(w, len(s) - (len(s) + 1) % 2)
    if w <= 2:
        return np.gradient(s, t) if deriv else s.copy()
    dt = float(np.median(np.diff(t)))
    return savgol_filter(s, w, 2, deriv=deriv, delta=dt, mode="interp")


def detect_dip(t, s, xy, conf, frame_i, hand, finger, p: Params) -> list[Tap]:
    """Local minima of s (flexion dip) with a descent gate before and a rebound gate after."""
    out: list[Tap] = []
    for a, b in _segments(t, conf, p):
        ts, ss = t[a:b], s[a:b]
        v = _savgol(ss, ts, p.smooth_window, 1)
        sm = _savgol(ss, ts, p.smooth_window, 0)
        xs = _savgol(xy[a:b, 0], ts, 5, 0)
        ys = _savgol(xy[a:b, 1], ts, 5, 0)
        for k in range(len(ts) - 1):
            if not (v[k] < 0.0 <= v[k + 1]):
                continue
            frac = float(np.clip(-v[k] / (v[k + 1] - v[k]), 0.0, 1.0))
            t_c = ts[k] + frac * (ts[k + 1] - ts[k])
            lo = np.searchsorted(ts, t_c - p.vel_window)
            hi = np.searchsorted(ts, t_c + p.vel_window, "right")
            descent = -v[lo:k + 1].min()
            rebound = v[k + 1:hi].max() if hi > k + 1 else 0.0
            amp = sm[k + 1:hi].max() - sm[k] if hi > k + 1 else 0.0
            if descent < p.min_vel or rebound < p.rebound_ratio * p.min_vel or amp < p.min_amp:
                continue
            out.append(Tap(t=float(t_c), hand=hand, finger=finger,
                           x=float(xs[k] + frac * (xs[k + 1] - xs[k])),
                           y=float(ys[k] + frac * (ys[k + 1] - ys[k])),
                           conf=float(conf[a + k]), i=int(frame_i[a + k] if frac < 0.5 else frame_i[a + k + 1]),
                           strength=min(descent, rebound)))
    return out


def detect_speed_pulse(t, s, xy, conf, frame_i, hand, finger, p: Params) -> list[Tap]:
    """Residual-speed profile: a press is two speed peaks (down, up); contact = trough between."""
    out: list[Tap] = []
    for a, b in _segments(t, conf, p):
        ts = t[a:b]
        sp = _savgol(s[a:b], ts, p.smooth_window, 0)
        peaks = [k for k in range(1, len(sp) - 1)
                 if sp[k] >= sp[k - 1] and sp[k] > sp[k + 1] and sp[k] >= p.min_vel]
        for k1, k2 in zip(peaks[:-1], peaks[1:]):
            gap = ts[k2] - ts[k1]
            if not (0.03 <= gap <= 0.20):
                continue
            k = k1 + int(np.argmin(sp[k1:k2 + 1]))
            if min(sp[k1], sp[k2]) - sp[k] < p.min_amp:
                continue
            out.append(Tap(t=float(ts[k]), hand=hand, finger=finger, x=float(xy[a + k, 0]),
                           y=float(xy[a + k, 1]), conf=float(conf[a + k]), i=int(frame_i[a + k]),
                           strength=min(sp[k1], sp[k2])))
    return out


def _assign_finger(h: dict, hf: dict, taps: list[Tap]) -> list[Tap]:
    """Pooled detection knows the hand only; label the finger with the deepest dmcp dip at t."""
    T = h["t"]
    for z in taps:
        m = (T >= z.t - 0.08) & (T <= z.t + 0.08)
        b = (T >= z.t - 0.4) & (T <= z.t + 0.4)
        dips = {tip: np.median(hf[tip]["dmcp"][b]) - hf[tip]["dmcp"][m].min() for tip in FINGERTIP_JOINTS}
        z.finger = max(dips, key=dips.get)
        k = int(np.argmin(np.abs(T - z.t)))
        z.x, z.y = float(h["P"][k, z.finger, 0]), float(h["P"][k, z.finger, 1])
    return taps


def nms(taps: list[Tap], window: float, same_finger: bool) -> list[Tap]:
    kept: list[Tap] = []
    for z in sorted(taps, key=lambda z: (-z.strength, z.t)):
        clash = (a for a in kept if a.hand == z.hand and (a.finger == z.finger or not same_finger))
        if any(abs(a.t - z.t) < window for a in clash):
            continue
        kept.append(z)
    return sorted(kept, key=lambda z: (z.t, z.hand, z.finger))


def detect(hands: dict[int, dict], feats: dict[int, dict], p: Params) -> list[Tap]:
    taps: list[Tap] = []
    fn = detect_speed_pulse if p.feature.endswith("speed") else detect_dip
    pooled = p.feature.startswith("hand_")
    for hand, h in hands.items():
        for tip in (8,) if pooled else FINGERTIP_JOINTS:
            found = fn(h["t"], feats[hand][tip][p.feature], h["P"][:, tip],
                       h["conf"][:, tip], h["i"], hand, tip, p)
            taps.extend(_assign_finger(h, feats[hand], found) if pooled else found)
    taps = nms(taps, p.refractory, same_finger=True)
    return nms(taps, p.nms_window, same_finger=False)


def evaluate(session: Path, taps: list[Tap], split: float = 0.6) -> dict:
    kt = keydown_times(session)
    tt = np.array([z.t for z in taps], dtype=float)
    cut = kt.min() + split * (kt.max() - kt.min())
    return {"all": score(kt, tt), "train": score(kt, tt, None, cut), "holdout": score(kt, tt, cut, None)}


def per_finger_recall(session: Path, taps: list[Tap]) -> list[str]:
    """Recall by the finger a touch typist would use for each key, plus which detected finger hit."""
    keys = [json.loads(l) for l in open(session / "keys.jsonl") if l.strip()]
    keys = sorted((k for k in keys if k["event"] == "down"), key=lambda k: k["t"])
    tt = np.array([z.t for z in taps])
    used: set[int] = set()
    rows: dict[str, list] = {}
    for k in keys:
        j = np.argsort(np.abs(tt - k["t"]))[:3] if len(tt) else []
        hit = next((int(x) for x in j if abs(tt[x] - k["t"]) <= 0.080 and int(x) not in used), None)
        if hit is not None:
            used.add(hit)
        rows.setdefault(KEY_FINGER.get(k["key"], "?"), []).append(
            FINGER_NAMES.get(taps[hit].finger) if hit is not None else None)
    lines = []
    for f, hits in sorted(rows.items(), key=lambda kv: -len(kv[1])):
        got = [x for x in hits if x]
        by = ",".join(f"{n}:{got.count(n)}" for n in FINGER_NAMES.values() if got.count(n))
        lines.append(f"  {f:<7} keys={len(hits):<3} recall={100 * len(got) / len(hits):5.1f}%  hit by [{by}]")
    return lines


def sweep(session: Path, hands: dict, feats: dict, features=FEATURES) -> tuple[Params, list[str]]:
    """Grid over (feature, window, min_vel, min_amp); select on TRAIN F1 only."""
    grids = {
        "pip": (np.array([2, 3, 4, 6, 8, 11, 15.0]), np.array([0.02, 0.05, 0.1, 0.15, 0.2])),
        "resspeed": (np.array([1, 1.5, 2, 3, 4, 6.0]), np.array([0.3, 0.6, 1, 1.5, 2.0])),
        "hand_speed": (np.array([3, 5, 8, 12, 18, 25.0]), np.array([1, 2, 4, 7, 10.0])),
        "hand_dmcp": (np.array([0.15, 0.25, 0.4, 0.6, 0.9, 1.3]), np.array([0.003, 0.006, 0.01, 0.02, 0.03])),
    }
    default = (np.array([0.3, 0.5, 0.8, 1.2, 1.7, 2.4, 3.2]), np.array([0.005, 0.01, 0.02, 0.035, 0.05]))
    best_by_feat: dict[str, tuple[float, Params, dict]] = {}
    for feat in features:
        vels, amps = grids.get(feat, default)
        for w, mv, ma in itertools.product((5, 7, 9), vels, amps):
            p = Params(feature=feat, smooth_window=w, min_vel=float(mv), min_amp=float(ma))
            r = evaluate(session, detect(hands, feats, p))
            if feat not in best_by_feat or r["train"]["f1"] > best_by_feat[feat][0]:
                best_by_feat[feat] = (r["train"]["f1"], p, r)
    lines = ["feature   window  min_vel  min_amp | train F1  holdout F1  holdout stillFP"]
    for feat, (_, p, r) in sorted(best_by_feat.items(), key=lambda kv: -kv[1][0]):
        lines.append(f"{feat:<9} {p.smooth_window:>6}  {p.min_vel:>7.2f}  {p.min_amp:>7.3f} | "
                     f"{100 * r['train']['f1']:7.1f}  {100 * r['holdout']['f1']:9.1f}  "
                     f"{r['holdout']['still_fp']:>6} ({100 * r['holdout']['still_fp_rate']:.0f}%)")
    winner = max(best_by_feat.values(), key=lambda v: v[0])[1]
    return winner, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.taps_flexion",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: <session>/taps_flexion.jsonl")
    ap.add_argument("--sweep", action="store_true", help="grid-search on the train slice first")
    ap.add_argument("--feature", default=Params.feature, choices=FEATURES)
    ap.add_argument("--smooth-window", type=int, default=Params.smooth_window)
    ap.add_argument("--min-vel", type=float, default=Params.min_vel)
    ap.add_argument("--min-amp", type=float, default=Params.min_amp)
    ap.add_argument("--rebound-ratio", type=float, default=Params.rebound_ratio)
    a = ap.parse_args(argv)

    hands = load_hands(a.session)
    feats = {hand: hand_features(h) for hand, h in hands.items()}
    p = Params(feature=a.feature, smooth_window=a.smooth_window, min_vel=a.min_vel,
               min_amp=a.min_amp, rebound_ratio=a.rebound_ratio)
    if a.sweep:
        p, lines = sweep(a.session, hands, feats)
        print("\n".join(lines))
        p = replace(p, rebound_ratio=a.rebound_ratio)
    taps = detect(hands, feats, p)
    out = a.out or a.session / "taps_flexion.jsonl"
    with out.open("w") as fh:
        for z in taps:
            fh.write(json.dumps(z.to_record()) + "\n")
    print(f"\nwrote {out}  ({len(taps)} taps)  params: {p}")
    if (a.session / "keys.jsonl").exists():
        for name, r in evaluate(a.session, taps).items():
            print(fmt(name, r))
        print("per-finger recall (nominal touch-typing finger of the key):")
        print("\n".join(per_finger_recall(a.session, taps)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
