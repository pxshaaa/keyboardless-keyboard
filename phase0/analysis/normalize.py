"""Camera/session-robust key identification: keyboard frames, enrolment, LM self-training.
Run: python -m phase0.analysis.normalize {diag|loso|table|enrol|selftrain|deskprep|desktable|deskcv}"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

import lightgbm  # noqa: F401  must load before torch or the two libomp copies segfault

from phase0.analysis import tap_pos as tp

for _n in ("Sess", "KeyClf", "XYReg", "GaussXY", "Selector"):
    setattr(sys.modules["__main__"], _n, getattr(tp, _n))

from phase0.analysis import keypre as K  # noqa: E402
from phase0.analysis.decode import A_INDEX, ALPHABET, NA  # noqa: E402
from phase0.analysis.finger_id import KEY_LABEL, FINGERTIP_JOINTS, HAND_NAMES  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

CACHE = Path(".cache/normalize")
RES = Path("results/normalize")
DAY1 = ("20260910-015217-kbd", "20260910-021315-kbd", "20260910-131629-kbd")
NEW = "20260911-164237-kbd"
KBD4 = (*DAY1, NEW)
HELD = "20260910-015948-kbd"
FOLDS = (*KBD4, HELD)
DESK = "20260910-202149-desk"
SEEDS = (0, 1, 2)
THREADS = int(os.environ.get("NORM_THREADS", "4"))
PIX_PROBS = Path(".cache/keypre/probs_run5")

FRAMES = ("abs", "coral", "rel", "palm", "rest_g", "rest_h", "tap_g", "tap_h")
TEMPLATE_JOINTS = (4, 5, 8, 9, 12, 13, 16, 17, 20)   # in-frame joints only; wrists get clipped
KW0 = 150.0          # canonical knuckle width, canonical units ~ pixels of a day-1 session
REST_WIN = 6         # rolling-max half window (frames) for the rest speed
REST_NEAR_S = 3.0    # rest must lie within this many seconds of a detected tap (between bursts)
REST_PCT = 20.0      # the slowest REST_PCT% of eligible frames are "at rest"


# ============================================================ small geometry
def umeyama(src: np.ndarray, dst: np.ndarray, w: np.ndarray | None = None, scale: bool = True):
    """Weighted 2-D similarity dst ~ s R src + t. -> (s, R[2,2], t[2])."""
    m = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    src, dst = src[m], dst[m]
    w = np.ones(len(src)) if w is None else np.asarray(w, float)[m]
    if len(src) < 2 or w.sum() <= 0:
        return 1.0, np.eye(2), np.zeros(2)
    w = w / w.sum()
    ms, md = w @ src, w @ dst
    a, b = src - ms, dst - md
    C = (b * w[:, None]).T @ a
    U, S, Vt = np.linalg.svd(C)
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    var = (w * (a ** 2).sum(1)).sum()
    s = float(np.trace(np.diag(S) @ D) / max(var, 1e-9)) if scale else 1.0
    return s, R, md - s * R @ ms


def apply_sim(xy: np.ndarray, T) -> np.ndarray:
    s, R, t = T
    return s * xy @ R.T + t


class View:
    """tap_pos.Sess stand-in carrying transformed landmarks (tap_pos.features reads P and span)."""

    def __init__(self, P: np.ndarray, span, anchor=None):
        self.P = P
        self.span = np.asarray(span, float)
        self.anchor = np.zeros((2, 2)) if anchor is None else anchor


def knuckle_width(P: np.ndarray) -> np.ndarray:
    return np.nanmedian(np.linalg.norm(P[:, :, 5, :2] - P[:, :, 17, :2], axis=2), axis=0)


# ============================================================ session geometry (label-free)
_SG: dict = {}


def all_tap_rows(sid: str) -> np.ndarray:
    d = K.ours(sid) if not sid.endswith("desk") else None
    if d is not None:
        return d["sess"].rows(d["all_taps"])
    return desk_tap_rows()


def desk_tap_rows() -> np.ndarray:
    """Desk taps with no labels: the keyboard-tuned detector's own stream (taps_contact)."""
    from phase0.analysis.analyze_drift import read_jsonl
    s = tp.load_sess(DESK)
    return s.rows(read_jsonl(tp.sess_path(DESK) / "taps_contact.jsonl"))


def rest_rows(sid: str) -> np.ndarray:
    """Frames where all eight non-thumb fingertips are nearly still, both hands tracked, within
    REST_NEAR_S of a detected tap: fingers parked between typing bursts. No labels used."""
    S = tp.load_sess(sid)
    P = S.P[:, :, :, :2]
    kw = knuckle_width(S.P)
    tips = P[:, :, [8, 12, 16, 20], :]
    v = np.linalg.norm(np.diff(tips, axis=0, prepend=tips[:1]), axis=-1) / kw[None, :, None]
    sp = np.nanmax(v.reshape(len(v), -1), axis=1)
    sp = np.where(np.isfinite(sp), sp, np.inf)
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(sp, REST_WIN, mode="edge")
    roll = sliding_window_view(pad, 2 * REST_WIN + 1).max(1)
    ok = np.isfinite(P[:, :, TEMPLATE_JOINTS, 0]).all((1, 2)) & np.isfinite(roll)
    tr = all_tap_rows(sid)
    tt = S.t[tr]
    near = np.zeros(len(S.t), bool)
    j = np.searchsorted(tt, S.t)
    for jj in (j - 1, j):
        jj = np.clip(jj, 0, len(tt) - 1)
        near |= np.abs(tt[jj] - S.t) <= REST_NEAR_S
    elig = np.where(ok & near)[0]
    thr = np.percentile(roll[elig], REST_PCT)
    return elig[roll[elig] <= thr]


def template(sid: str, kind: str) -> np.ndarray:
    """[2, J, 2] median positions of TEMPLATE_JOINTS per hand at rest ('rest') or over all
    detected taps ('tap')."""
    key = (sid, kind)
    if key not in _SG:
        S = tp.load_sess(sid)
        rows = rest_rows(sid) if kind == "rest" else all_tap_rows(sid)
        Q = S.P[rows][:, :, TEMPLATE_JOINTS, :2]
        _SG[key] = np.nanmedian(Q, axis=0)
    return _SG[key]


def canonical(train: tuple[str, ...], kind: str, per_hand: bool, iters: int = 10) -> np.ndarray:
    """Generalised Procrustes over the training sessions' templates, scale pinned to KW0."""
    Ts = [template(s, kind) for s in train]
    ref = Ts[-1].copy()                 # the largest session seeds the mean shape
    for _ in range(iters):
        al = [align_template(T, ref, per_hand)[1] for T in Ts]
        ref = np.nanmean(al, axis=0)
        kw = np.nanmean([np.linalg.norm(ref[h, 1] - ref[h, 7]) for h in (0, 1)])  # MCP5-MCP17
        c = np.nanmean(ref.reshape(-1, 2), axis=0)
        ref = (ref - c) * (KW0 / kw) + c
    return ref


def align_template(T: np.ndarray, ref: np.ndarray, per_hand: bool, w=None):
    """-> ([hand transforms], aligned template)."""
    if per_hand:
        Th = [umeyama(T[h], ref[h]) for h in (0, 1)]
    else:
        g = umeyama(T.reshape(-1, 2), ref.reshape(-1, 2))
        Th = [g, g]
    return Th, np.stack([apply_sim(T[h], Th[h]) for h in (0, 1)])


def transform_P(P: np.ndarray, Th) -> np.ndarray:
    Q = P.copy()
    for h in (0, 1):
        s, R, t = Th[h]
        Q[:, h, :, :2] = apply_sim(P[:, h, :, :2], Th[h])
        Q[:, h, :, tp.CH_Z] = P[:, h, :, tp.CH_Z] * s
    return Q


def palm_P(P: np.ndarray) -> np.ndarray:
    """Every frame, every hand: origin = MCP centroid, x = MCP17->MCP5, unit = knuckle width."""
    Q = P.copy()
    for h in (0, 1):
        c = np.nanmean(P[:, h, [5, 9, 13, 17], :2], axis=1)
        u = P[:, h, 5, :2] - P[:, h, 17, :2]
        n = np.linalg.norm(u, axis=1, keepdims=True)
        u = u / np.where(n > 1e-6, n, np.nan)
        v = np.stack([-u[:, 1], u[:, 0]], 1)
        d = P[:, h, :, :2] - c[:, None]
        Q[:, h, :, 0] = (d * u[:, None]).sum(-1) / n * KW0
        Q[:, h, :, 1] = (d * v[:, None]).sum(-1) / n * KW0
        Q[:, h, :, tp.CH_Z] = P[:, h, :, tp.CH_Z] / n[:, 0][:, None] * KW0
    return Q


def palm_spread(P: np.ndarray, rows: np.ndarray, Th) -> tuple[float, np.ndarray]:
    """RMS travel of the palm centroids over tap frames: keyboard extent, not hand size."""
    c = np.stack([apply_sim(np.nanmean(P[rows][:, h, [5, 9, 13, 17], :2], axis=1), Th[h]) for h in (0, 1)], 1)
    ok = np.isfinite(c).all((1, 2))
    mu = c[ok].mean(0)
    return float(np.sqrt(((c[ok] - mu) ** 2).sum(-1).mean())), mu.mean(0)


def rescale(Th, f: float, centre: np.ndarray):
    return [(f * s, R, f * t + (1 - f) * centre) for s, R, t in Th]


def ref_spread(train: tuple[str, ...]) -> float:
    ref = canonical(train, "tap", False)
    return float(np.mean([palm_spread(tp.load_sess(s).P, all_tap_rows(s),
                                      align_template(template(s, "tap"), ref, False)[0])[0] for s in train]))


def keyboard_scale(P: np.ndarray, rows: np.ndarray, Th, train: tuple[str, ...]):
    sp, c = palm_spread(P, rows, Th)
    return rescale(Th, ref_spread(train) / sp, c)


def session_transform(sid: str, frame: str, train: tuple[str, ...]):
    kind = frame.split("_")[0]
    if kind == "tapk":
        Th = align_template(template(sid, "tap"), canonical(train, "tap", False), False)[0]
        return keyboard_scale(tp.load_sess(sid).P, all_tap_rows(sid), Th, train)
    ref = canonical(train, kind, frame.endswith("_h"))
    return align_template(template(sid, kind), ref, frame.endswith("_h"))[0]


_VIEW: dict = {}


def view(sid: str, frame: str, train: tuple[str, ...], Th=None):
    """-> an object tap_pos.features can read, in `frame`. Th overrides the label-free fit."""
    key = (sid, frame, train, None if Th is None else id(Th))
    if Th is None and key in _VIEW:
        return _VIEW[key]
    S = tp.load_sess(sid)
    if Th is not None and frame in ("abs", "coral"):
        v = View(transform_P(S.P, Th), S.span * Th[0][0])
    elif frame in ("abs", "coral", "rel"):
        v = S
    elif frame == "palm":
        v = View(palm_P(S.P), [KW0, KW0])
    else:
        Th = Th if Th is not None else session_transform(sid, frame, train)
        v = View(transform_P(S.P, Th), [KW0, KW0])
    if Th is None:
        if len(_VIEW) > 12:
            _VIEW.clear()
        _VIEW[key] = v
    return v


def feats(sid: str, k: np.ndarray, frame: str, train: tuple[str, ...], Th=None) -> np.ndarray:
    frame = frame.replace("+coral", "")
    mode = "rel" if frame == "rel" else "abs"
    return tp.features(view(sid, frame, train, Th), k, mode, tp.JOINTS_FULL, tp.OFFSETS)


# ============================================================ pose model
def fit_pose(X, y, seed: int, w=None, rounds: int = 250):
    import lightgbm as lgb
    m = K.keep_frequent(y)
    p = K.lgb_params(seed)
    p["num_threads"] = THREADS
    ds = lgb.Dataset(X[m], y[m], weight=None if w is None else np.asarray(w)[m])
    return lgb.train(p, ds, rounds)


def coral_align(Xte: np.ndarray, Xtr: np.ndarray, Xref: np.ndarray | None = None) -> np.ndarray:
    """Test moments come from Xref (every detected tap of the session, label-free) when given."""
    Xref = Xte if Xref is None else Xref
    ms, ss = Xtr.mean(0), Xtr.std(0) + 1e-6
    return (Xte - Xref.mean(0)) / (Xref.std(0) + 1e-6) * ss + ms


def stream_feats(sid: str, frame: str, train: tuple[str, ...]) -> np.ndarray:
    return feats(sid, all_tap_rows(sid), frame, train)


def train_set(frame: str, train: tuple[str, ...]):
    X = np.vstack([feats(s, K.ours(s)["k"], frame, train) for s in train])
    y = np.concatenate([K.ours(s)["y"] for s in train])
    return X, y


def base_frame(frame: str) -> str:
    return "abs" if frame == "coral" else frame.replace("+coral", "")


def probs_file(frame: str, held: str, seed: int) -> Path:
    return CACHE / "probs" / f"pose_{frame}_{held}_s{seed}.npy"


def model_file(frame: str, held: str, seed: int) -> Path:
    return CACHE / "models" / f"pose_{frame}_{held}_s{seed}.txt"


def loso_fold(frame: str, held: str, seed: int) -> np.ndarray:
    out = probs_file(frame, held, seed)
    if out.exists():
        return np.load(out)
    train = tuple(s for s in KBD4 if s != held)
    Xtr, ytr = train_set(frame, train)
    Xte = feats(held, K.ours(held)["k"], frame, train)
    if frame == "coral" or frame.endswith("+coral"):
        Xte = coral_align(Xte, Xtr, stream_feats(held, frame, train))
    import lightgbm as lgb
    src = model_file(base_frame(frame), held, seed)
    b = lgb.Booster(model_file=str(src)) if src.exists() else fit_pose(Xtr, ytr, seed)
    p = K.to_na(b.predict(Xte))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, p)
    if not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        b.save_model(str(src))
    return p


def pix(held: str, seed: int) -> np.ndarray:
    return np.load(PIX_PROBS / f"pix_scratch_f1.0_{held}_s{seed}.npy")


def fused(frame: str, seed: int) -> dict:
    """Pose in `frame` fused with the cached pixel CNN; alpha chosen on the other kbd folds."""
    ys = {h: K.ours(h)["y"] for h in FOLDS}
    po = {h: loso_fold(frame, h, seed) for h in FOLDS}
    px = {h: pix(h, seed) for h in FOLDS}
    out = {}
    for h in FOLDS:
        src = [s for s in KBD4 if s != h]
        best, ba = -1, 0.5
        for a in K.ALPHAS:
            acc = np.concatenate([K.topk_hits(K.gmean(px[s], po[s], a), ys[s], 1) for s in src]).mean()
            if acc > best + 1e-12:
                best, ba = acc, a
        out[h] = (K.gmean(px[h], po[h], ba), ba)
    return out


# ============================================================ CLI: diag
def oracle_tip(sid, frame, train, Th=None):
    d = K.ours(sid)
    v = view(sid, frame, train, Th)
    xy, ok = tp.true_tip_xy(v, d["k"], d["keys"])
    return xy, ok, np.array(d["keys"])


def centroid_shift(frame: str, train=DAY1, test=NEW, min_n: int = 15) -> dict:
    """Label-using DIAGNOSTIC only: per-key oracle-finger contact centroids, day 1 vs day 2,
    in key-pitch units measured in the same frame."""
    A = [oracle_tip(s, frame, train) for s in train]
    xa = np.vstack([a[0][a[1]] for a in A])
    ka = np.concatenate([a[2][a[1]] for a in A])
    xb, okb, kb = oracle_tip(test, frame, train)
    xb, kb = xb[okb], kb[okb]
    ca = {c: xa[ka == c].mean(0) for c in set(ka) if (ka == c).sum() >= min_n}
    cb = {c: xb[kb == c].mean(0) for c in set(kb) if (kb == c).sum() >= min_n}
    pairs = (("a", "s"), ("s", "d"), ("d", "f"), ("j", "k"), ("k", "l"))
    pitch = np.median([np.linalg.norm(ca[p] - ca[q]) for p, q in pairs if p in ca and q in ca])
    common = sorted(set(ca) & set(cb) - {" "})
    sh = np.array([np.linalg.norm(ca[c] - cb[c]) for c in common]) / pitch
    # residual after removing the best single translation (what is NOT a camera shift)
    D = np.array([cb[c] - ca[c] for c in common])
    res = np.linalg.norm(D - np.median(D, 0), axis=1) / pitch
    return {"frame": frame, "keys": len(common), "pitch": float(pitch),
            "median_shift_pitch": float(np.median(sh)), "median_resid_pitch": float(np.median(res))}


def cmd_diag(a) -> int:
    for s in (*FOLDS, DESK):
        r = rest_rows(s)
        T = template(s, "rest")
        Tt = template(s, "tap") if not s.endswith("desk") else template(s, "tap")
        S = tp.load_sess(s)
        print(f"{s}: rest frames {len(r)} ({len(r)/len(S.t):.1%}); knuckle w {knuckle_width(S.P).round(0)}; "
              f"rest L index {T[0, 2].round(0)} R index {T[1, 2].round(0)}; tap-template L index "
              f"{Tt[0, 2].round(0)}", flush=True)
    # template-fit quality: residual of each session's rest template against day-1 canonical
    for fr in ("rest_g", "rest_h", "tap_g", "tap_h"):
        kind, ph = fr.split("_")[0], fr.endswith("_h")
        ref = canonical(DAY1, kind, ph)
        row = []
        for s in (*FOLDS, DESK):
            Th, al = align_template(template(s, kind), ref, ph)
            row.append(f"{s[9:15]} s={Th[0][0]:.2f}/{Th[1][0]:.2f} "
                       f"rot={np.degrees(np.arctan2(Th[0][1][1,0],Th[0][1][0,0])):+.0f}/"
                       f"{np.degrees(np.arctan2(Th[1][1][1,0],Th[1][1][0,0])):+.0f} "
                       f"res={np.nanmean(np.linalg.norm(al-ref,axis=-1)):.1f}")
        print(f"{fr}: " + " | ".join(row))
    print("\noracle-finger contact centroid shift, day-1 sessions -> 20260911-164237 (diagnostic, uses labels)")
    out = []
    for fr in ("abs", "rest_g", "rest_h", "tap_g", "tap_h"):
        r = centroid_shift(fr)
        out.append(r)
        print(f"  {fr:<7} keys={r['keys']} pitch={r['pitch']:.1f} median shift={r['median_shift_pitch']:.2f} "
              f"pitch, after best translation={r['median_resid_pitch']:.2f} pitch")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "diag_shift.json").write_text(json.dumps(out, indent=1))
    return 0


# ============================================================ CLI: loso
def cmd_loso(a) -> int:
    for s in FOLDS:
        K.ours(s)
    skip = set(os.environ.get("NORM_SKIP_FRAMES", "").split(","))
    for seed in SEEDS[:a.seeds]:
        for frame in [f for f in a.frames.split(",") if seed == 0 or f not in skip]:
            t0 = time.time()
            for h in (a.folds.split(",") if a.folds else FOLDS):
                loso_fold(frame, h, seed)
            print(f"  {frame:<7} seed{seed} ({time.time()-t0:.0f}s)", flush=True)
    return 0


def boot(x, n=10000, seed=0):
    return K.boot(np.asarray(x, float), n, seed)


def cmd_table(a) -> int:
    ys = {h: K.ours(h)["y"] for h in FOLDS}
    have = {f: [s for s in SEEDS[:a.seeds] if all(probs_file(f, h, s).exists() for h in FOLDS)]
            for f in a.frames.split(",")}
    frames = [f for f in have if have[f]]
    rows, hits = [], {}
    hdr = (f"{'frame':<8}{'model':<6}{'NEW 164237 (x-day)':>22}{'95% CI':>16}{'LOSO4 pooled':>15}"
           f"{'held 015948':>14}{'131629':>9}{'021315':>9}  alpha")
    print(f"per-key top-1, mean over seeds {SEEDS[:a.seeds]}; x-day = train on 3 day-1 sessions, "
          f"test on the re-mounted camera session")
    print(hdr + "\n" + "-" * len(hdr))
    for fr in frames:
        for kind in ("pose", "fused"):
            per = {h: [] for h in FOLDS}
            al = []
            for s in have[fr]:
                F = fused(fr, s) if kind == "fused" else None
                for h in FOLDS:
                    p = F[h][0] if F else loso_fold(fr, h, s)
                    per[h].append(K.topk_hits(p, ys[h], 1))
                if F:
                    al.append(F[NEW][1])
            m = {h: np.mean(per[h], 0) for h in FOLDS}
            hits[(fr, kind)] = m
            lo, hi = boot(m[NEW])
            loso = np.concatenate([m[h] for h in KBD4]).mean()
            seeds_new = [float(np.mean(x)) for x in per[NEW]]
            print(f"{fr+'/'+str(len(have[fr])):<8}{kind:<6}{m[NEW].mean():>12.3f} ±{np.std(seeds_new):.3f}      "
                  f"{f'[{lo:.3f},{hi:.3f}]':>16}{loso:>15.3f}{m[HELD].mean():>14.3f}"
                  f"{m['20260910-131629-kbd'].mean():>9.3f}{m['20260910-021315-kbd'].mean():>9.3f}"
                  f"  {al if al else ''}", flush=True)
            rows.append({"frame": fr, "model": kind, "new": float(m[NEW].mean()), "new_ci": [lo, hi],
                         "new_seeds": seeds_new, "loso4": float(loso), "held": float(m[HELD].mean()),
                         **{h[9:15]: float(m[h].mean()) for h in KBD4}, "alpha_new": al})
    print("\npaired deltas vs abs on NEW (x-day) [95% CI over taps]:")
    for fr in frames:
        if fr == "abs":
            continue
        for kind in ("pose", "fused"):
            d = hits[(fr, kind)][NEW] - hits[("abs", kind)][NEW]
            lo, hi = boot(d)
            dl = np.concatenate([hits[(fr, kind)][h] - hits[("abs", kind)][h] for h in KBD4])
            llo, lhi = boot(dl)
            print(f"  {fr:<8}{kind:<6} NEW {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]   LOSO4 {dl.mean():+.3f} "
                  f"[{llo:+.3f},{lhi:+.3f}]")
            for r in rows:
                if r["frame"] == fr and r["model"] == kind:
                    r["d_new_vs_abs"] = [float(d.mean()), lo, hi]
                    r["d_loso4_vs_abs"] = [float(dl.mean()), llo, lhi]
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "loso_frames.json").write_text(json.dumps(rows, indent=1))
    return 0


# ============================================================ enrolment
ENROL_N = (0, 25, 50, 100, 200)
EVAL_FROM = 200      # every enrolment size is scored on the same taps: time order >= 200
ENROL_W = 10.0       # sample weight of an enrolment tap against a day-1 tap
PRIOR_TAPS = 25.0    # label-free template correspondence counts as this many enrolment taps


def key_means(frame: str, train: tuple[str, ...]) -> dict:
    acc: dict = {}
    for s in train:
        d = K.ours(s)
        Q = view(s, frame, train).P[d["k"]][:, :, TEMPLATE_JOINTS, :2]
        for c in np.unique(d["y"]):
            acc.setdefault(int(c), []).append(Q[d["y"] == c])
    return {c: np.nanmean(np.vstack(q), 0) for c, q in acc.items()}


def enrol_transform(sid, frame, train, rows, ys):
    """Similarity mapping this session's raw joints onto the training per-key mean hand shapes,
    regularised by the label-free template fit (none for abs: identity prior)."""
    S = tp.load_sess(sid)
    ph = frame.endswith("_h")
    mu = key_means(frame, train)
    src, dst, w = [[], []], [[], []], [[], []]
    if frame not in ("abs", "coral"):
        kind = frame.split("_")[0]
        ref, T = canonical(train, kind, ph), template(sid, kind)
        for h in (0, 1):
            src[h].append(T[h]); dst[h].append(ref[h])
            w[h].append(np.full(len(TEMPLATE_JOINTS), PRIOR_TAPS))
    keep = [i for i, c in enumerate(ys) if int(c) in mu]
    for h in (0, 1):
        if keep:
            Q = S.P[rows[keep]][:, h][:, TEMPLATE_JOINTS, :2]
            M = np.stack([mu[int(ys[i])][h] for i in keep])
            src[h].append(Q.reshape(-1, 2)); dst[h].append(M.reshape(-1, 2))
            w[h].append(np.ones(Q.shape[0] * Q.shape[1]))
    cat = lambda L: np.vstack(L) if L[0].ndim == 2 else np.concatenate(L)
    if ph:
        return [umeyama(cat(src[h]), cat(dst[h]), cat(w[h])) for h in (0, 1)]
    g = umeyama(np.vstack([cat(src[h]) for h in (0, 1)]), np.vstack([cat(dst[h]) for h in (0, 1)]),
                np.concatenate([cat(w[h]) for h in (0, 1)]))
    return [g, g]


_TS: dict = {}


def train_cached(frame: str, train: tuple[str, ...]):
    if (frame, train) not in _TS:
        _TS[(frame, train)] = train_set(frame, train)
    return _TS[(frame, train)]


def enrol_probs(frame: str, method: str, n: int, seed: int) -> np.ndarray:
    import lightgbm as lgb
    out = CACHE / "enrol" / f"{frame}_{method}_n{n}_s{seed}.npy"
    if out.exists():
        return np.load(out)
    d = K.ours(NEW)
    er = d["time_order"][:n]
    Th = enrol_transform(NEW, frame, DAY1, d["k"][er], d["y"][er]) if "T" in method and n > 0 else None
    Xte = feats(NEW, d["k"], frame, DAY1, Th)
    Xtr, ytr = train_cached(frame, DAY1)
    if frame == "coral" or frame.endswith("+coral"):
        Xte = coral_align(Xte, Xtr, stream_feats(NEW, frame, DAY1) if Th is None else
                          feats(NEW, all_tap_rows(NEW), frame, DAY1, Th))
    if "ft" in method and n > 0:
        b = fit_pose(np.vstack([Xtr, Xte[er]]), np.concatenate([ytr, d["y"][er]]), seed,
                     np.r_[np.ones(len(ytr)), np.full(n, ENROL_W)])
    else:
        b = lgb.Booster(model_file=str(model_file(base_frame(frame), NEW, seed)))
    p = K.to_na(b.predict(Xte))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, p)
    return p


def cmd_enrol(a) -> int:
    d = K.ours(NEW)
    ev = d["time_order"][EVAL_FROM:]
    y = d["y"][ev]
    rows = []
    print(f"enrolment on {NEW}: first N labelled taps, scored on its {len(ev)} taps after #{EVAL_FROM}; "
          f"day-1 training; mean over seeds {SEEDS[:a.seeds]}")
    for frame in a.frames.split(","):
        base_hits = {}
        for method in a.methods.split(","):
            if "T" in method and frame in ("rel", "palm"):
                continue
            for n in ENROL_N:
                if n == 0 and method != a.methods.split(",")[0]:
                    continue
                hp, hf = [], []
                for seed in SEEDS[:a.seeds]:
                    t0 = time.time()
                    p = enrol_probs(frame, method, n, seed)
                    al = fused(frame, seed)[NEW][1]
                    hp.append(K.topk_hits(p[ev], y, 1))
                    hf.append(K.topk_hits(K.gmean(pix(NEW, seed)[ev], p[ev], al), y, 1))
                hp, hf = np.mean(hp, 0), np.mean(hf, 0)
                if n == 0:
                    base_hits = {"pose": hp, "fused": hf}
                lo, hi = boot(hf)
                dl = boot(hf - base_hits["fused"]) if n else (0.0, 0.0)
                print(f"  {frame:<7}{method:<5} N={n:>3}  pose {hp.mean():.3f}  fused {hf.mean():.3f} "
                      f"[{lo:.3f},{hi:.3f}]  d_fused vs N=0 {hf.mean()-base_hits['fused'].mean():+.3f} "
                      f"[{dl[0]:+.3f},{dl[1]:+.3f}]", flush=True)
                rows.append({"frame": frame, "method": method, "n": n, "pose": float(hp.mean()),
                             "fused": float(hf.mean()), "fused_ci": [lo, hi],
                             "d_vs_n0": [float(hf.mean() - base_hits["fused"].mean()), *dl]})
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"enrol_{a.tag}.json").write_text(json.dumps(rows, indent=1))
    return 0


# ============================================================ warm-up: unlabelled taps the frame needs
WARM_M = (10, 25, 50, 100, 200, 400)


def cmd_warmup(a) -> int:
    """Template fitted from only the first M DETECTED taps (no labels); scored on taps >= #200."""
    import lightgbm as lgb
    d = K.ours(NEW)
    S = tp.load_sess(NEW)
    ev = d["time_order"][EVAL_FROM:]
    t_all = np.array([x["t"] for x in d["all_taps"]])
    rows_all = S.rows(d["all_taps"])[np.argsort(t_all)]
    out = []
    for frame in a.frames.split(","):
        kind, ph = frame.split("_")[0], frame.endswith("_h")
        ref = canonical(DAY1, kind, ph)
        models = {sd: lgb.Booster(model_file=str(model_file(frame, NEW, sd))) for sd in SEEDS[:a.seeds]}
        for M in (*WARM_M, len(rows_all)):
            starts = np.linspace(0, len(rows_all) - M, 1 if M == len(rows_all) else 8).round().astype(int)
            accs = []
            for st in starts:
                Q = S.P[rows_all[st:st + M]][:, :, TEMPLATE_JOINTS, :2]
                Th = align_template(np.nanmedian(Q, axis=0), ref, ph)[0]
                X = feats(NEW, d["k"][ev], frame, DAY1, Th)
                hf = [K.topk_hits(K.gmean(pix(NEW, sd)[ev], K.to_na(b.predict(X)), fused(frame, sd)[NEW][1]),
                                  d["y"][ev], 1).mean() for sd, b in models.items()]
                accs.append(float(np.mean(hf)))
            print(f"  {frame:<7} {M:>4} detected taps, {len(starts)} windows: fused mean {np.mean(accs):.3f} "
                  f"min {np.min(accs):.3f} max {np.max(accs):.3f}", flush=True)
            out.append({"frame": frame, "m": M, "fused_windows": accs})
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"warmup_{a.tag}.json").write_text(json.dumps(out, indent=1))
    return 0


# ============================================================ simulated user / camera shift
PERTURB = (("none", {}), ("cam_zoom0.8", {"zoom": 0.8}), ("cam_zoom1.25", {"zoom": 1.25}),
           ("cam_rot-12", {"rot": -12.0}), ("cam_rot+12", {"rot": 12.0}), ("shift60px", {"dx": 60.0}),
           ("hand0.85", {"hand": 0.85}), ("hand1.15", {"hand": 1.15}))


def perturb_P(P: np.ndarray, centre: np.ndarray, zoom=1.0, rot=0.0, dx=0.0, hand=1.0) -> np.ndarray:
    """Camera zoom/rotation about the tap cloud centre, translation, or per-hand size about the palm."""
    Q = P.copy()
    if hand != 1.0:
        for h in (0, 1):
            c = np.nanmean(P[:, h, [5, 9, 13, 17], :2], axis=1)[:, None]
            Q[:, h, :, :2] = c + hand * (P[:, h, :, :2] - c)
            Q[:, h, :, tp.CH_Z] = P[:, h, :, tp.CH_Z] * hand
        return Q
    a = np.radians(rot)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    T = (zoom, R, centre - zoom * R @ centre + np.array([dx, 0.0]))
    return transform_P(P, [T, T])


def cmd_perturb(a) -> int:
    import lightgbm as lgb
    d = K.ours(NEW)
    S = tp.load_sess(NEW)
    rows_all = all_tap_rows(NEW)
    centre = np.nanmedian(S.P[rows_all][:, :, TEMPLATE_JOINTS, :2].reshape(-1, 2), axis=0)
    ref = canonical(DAY1, "tap", False)
    out = []
    print(f"pose-only top-1 on {NEW} (day-1 models) under synthetic geometric shift; mean over seeds {SEEDS[:a.seeds]}")
    for name, kw in PERTURB:
        Pp = perturb_P(S.P, centre, **kw)
        Th = align_template(np.nanmedian(Pp[rows_all][:, :, TEMPLATE_JOINTS, :2], axis=0), ref, False)[0]
        span = S.span * kw.get("zoom", 1.0) * kw.get("hand", 1.0)
        Xabs = tp.features(View(Pp, span), d["k"], "abs", tp.JOINTS_FULL, tp.OFFSETS)
        Xabs_all = tp.features(View(Pp, span), rows_all, "abs", tp.JOINTS_FULL, tp.OFFSETS)
        Xtap = tp.features(View(transform_P(Pp, Th), [KW0, KW0]), d["k"], "abs", tp.JOINTS_FULL, tp.OFFSETS)
        Tk = keyboard_scale(Pp, rows_all, Th, DAY1)
        Xtk = tp.features(View(transform_P(Pp, Tk), [KW0, KW0]), d["k"], "abs", tp.JOINTS_FULL, tp.OFFSETS)
        row = {"perturb": name, "tapk_scale": float(Tk[0][0] / Th[0][0])}
        for frame, X, mf in (("abs", Xabs, "abs"), ("coral", coral_align(Xabs, train_cached("abs", DAY1)[0], Xabs_all), "abs"),
                             ("tap_g", Xtap, "tap_g"), ("tapk_on_tap_g_model", Xtk, "tap_g")) + \
                ((("tapk_g", Xtk, "tapk_g"),) if model_file("tapk_g", NEW, 0).exists() else ()):
            acc = [float((K.to_na(lgb.Booster(model_file=str(model_file(mf, NEW, sd))).predict(X))
                          .argmax(1) == d["y"]).mean()) for sd in SEEDS[:a.seeds]]
            row[frame] = float(np.mean(acc))
        out.append(row)
        print(f"  {name:<13} " + "  ".join(f"{k} {v:.3f}" for k, v in row.items() if k != "perturb"), flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"perturb_{a.tag}.json").write_text(json.dumps(out, indent=1))
    return 0


HAND_AUG = (0.85, 1.15)


def aug_view_feats(sid: str, k: np.ndarray, hand: float, train: tuple[str, ...]) -> np.ndarray:
    """tap_g features of `sid` with every hand scaled about its palm; the frame is refitted on the copy."""
    S = tp.load_sess(sid)
    rows = all_tap_rows(sid)
    Pp = perturb_P(S.P, np.zeros(2), hand=hand) if hand != 1.0 else S.P
    Th = align_template(np.nanmedian(Pp[rows][:, :, TEMPLATE_JOINTS, :2], axis=0), canonical(train, "tap", False), False)[0]
    return tp.features(View(transform_P(Pp, Th), [KW0, KW0]), k, "abs", tp.JOINTS_FULL, tp.OFFSETS)


def cmd_augment(a) -> int:
    import lightgbm as lgb
    d = K.ours(NEW)
    out = []
    for seed in SEEDS[:a.seeds]:
        mf = CACHE / "models" / f"pose_tap_g_handaug_{NEW}_s{seed}.txt"
        if not mf.exists():
            X = np.vstack([aug_view_feats(s, K.ours(s)["k"], h, DAY1) for h in (1.0, *HAND_AUG) for s in DAY1])
            y = np.concatenate([K.ours(s)["y"] for h in (1.0, *HAND_AUG) for s in DAY1])
            fit_pose(X, y, seed).save_model(str(mf))
        models = {"tap_g": lgb.Booster(model_file=str(model_file("tap_g", NEW, seed))),
                  "tap_g_handaug": lgb.Booster(model_file=str(mf))}
        al = fused("tap_g", seed)[NEW][1]
        for h in (1.0, 0.85, 1.15):
            X = aug_view_feats(NEW, d["k"], h, DAY1)
            for name, b in models.items():
                p = K.to_na(b.predict(X))
                r = {"seed": seed, "hand": h, "model": name, "pose": float((p.argmax(1) == d["y"]).mean()),
                     "fused": float((K.gmean(pix(NEW, seed), p, al).argmax(1) == d["y"]).mean())}
                out.append(r)
                print(f"  s{seed} hand x{h:<4} {name:<14} pose {r['pose']:.3f} fused {r['fused']:.3f}", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"augment_{a.tag}.json").write_text(json.dumps(out, indent=1))
    return 0


# ============================================================ self-training (no keylogger)
ST_TAU = 0.7
ST_ROUNDS = 3
GAP_S = 1.5
MAX_SEG = 60
KBD_W = dict(obs=2.0, insertion=-5.0, deletion=-9.0, max_deletions=2)  # fixed a priori, never tuned


def pix_all(sid: str, seed: int) -> np.ndarray:
    """Pixel CNN fitted on the day-1 labelled crops, applied to EVERY detected tap of `sid`."""
    from phase0.analysis import masked as mk
    out = CACHE / "pix" / f"pix_all_{sid}_s{seed}.npy"
    if out.exists():
        return np.load(out)
    X = np.concatenate([K.our_rep(s, np.arange(len(K.ours(s)["y"]))) for s in DAY1])
    y = np.concatenate([K.ours(s)["y"] for s in DAY1])
    net = K.train_tipnet(lambda b: X[b], y, K.PIX_EPOCHS, K.PIX_LR, seed)
    del X
    d = K.ours(sid)
    te = mk.tip_crops(d, K.OFFSET, "none", "diff", rows=np.arange(len(d["all_taps"])))
    p = K.predict_tipnet(net, lambda b: te[b], len(te))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, p)
    return p


def gap_segments(t: np.ndarray, idx: np.ndarray) -> list[np.ndarray]:
    idx = idx[np.argsort(t[idx])]
    cuts = np.where(np.diff(t[idx]) > GAP_S)[0] + 1
    out = []
    for g in np.split(idx, cuts):
        out += [g[i:i + MAX_SEG] for i in range(0, len(g), MAX_SEG)]
    return [g for g in out if len(g) >= 3]


def pseudo_labels(p: np.ndarray, segs, w) -> tuple[np.ndarray, np.ndarray]:
    """Decode each unlabelled segment with the LM, align taps to the hypothesis, keep posteriors."""
    from phase0.analysis import adapt as ad
    from phase0.analysis.decode import beam_decode
    clm, wlm = ad.lms()
    lab, conf = np.full(len(p), -1), np.zeros(len(p))
    for idx in segs:
        lp = np.log(np.maximum(p[idx], 1e-12))
        hyp = beam_decode(lp, clm, wlm, w, 30)
        nc = sum(c in A_INDEX for c in hyp)
        if nc < 2:
            continue
        par = ad.AlignParams.from_counts(len(idx), nc, float(np.clip(1 - nc / len(idx), 0.08, 0.5)))
        ll, q, qi = ad.forward_backward(lp, hyp, par)
        if not np.isfinite(ll):
            continue
        lab[idx], conf[idx] = q.argmax(1), q.max(1) * (1 - qi)
    return lab, conf


def cer_rows(p, segs_txt, w):
    from phase0.analysis import adapt as ad
    from phase0.analysis.decode import beam_decode, edit_distance
    clm, wlm = ad.lms()
    out = []
    for text, idx in segs_txt:
        hyp = beam_decode(np.log(np.maximum(p[idx], 1e-12)), clm, wlm, w, 30) if len(idx) else ""
        out.append((text, hyp, edit_distance(text, hyp), len(text)))
    return out


def cmd_selftrain(a) -> int:
    import lightgbm as lgb
    from phase0.analysis import pipeline as pl
    from phase0.analysis.decode import Weights, kbd_segments
    w = Weights(**KBD_W)
    d = K.ours(NEW)
    taps = d["all_taps"]
    t = np.array([x["t"] for x in taps])
    tmid = float(np.median(t))
    A = np.where(t < tmid)[0]
    k_all = d["sess"].rows(taps)
    truth = np.full(len(taps), -1)
    truth[d["bank_rows"]] = d["y"]
    evB = np.array([i for i in d["bank_rows"] if t[i] >= tmid])
    pos = {round(x["t"], 6): i for i, x in enumerate(taps)}
    segB = [(txt, np.array([pos[round(x["t"], 6)] for x in ss], int))
            for txt, ss in kbd_segments(tp.sess_path(NEW), taps) if ss and min(x["t"] for x in ss) >= tmid]
    segA = gap_segments(t, A)
    print(f"{NEW}: {len(taps)} detected taps; adapt on first half ({len(A)} taps, {len(segA)} unlabelled "
          f"segments), score on second half ({len(evB)} labelled taps, {len(segB)} keylogger segments, "
          f"{sum(len(x) for x, _ in segB)} chars)", flush=True)
    res = []
    for frame in a.frames.split(","):
        Xtr, ytr = train_cached(frame, DAY1)
        Xall = feats(NEW, k_all, frame, DAY1)
        if frame == "coral" or frame.endswith("+coral"):
            Xall = coral_align(Xall, Xtr)
        for seed in ([int(x) for x in a.seed_list.split(",")] if a.seed_list else SEEDS[:a.seeds]):
            px = pix_all(NEW, seed)
            al = fused(frame, seed)[NEW][1]
            b0 = lgb.Booster(model_file=str(model_file(base_frame(frame), NEW, seed)))
            p0 = K.to_na(b0.predict(Xall))

            def score(tag, rnd, pose, lab=None, conf=None, keep=None):
                f = K.gmean(px, pose, al)
                hp = K.topk_hits(pose[evB], truth[evB], 1)
                hf = K.topk_hits(f[evB], truth[evB], 1)
                cr = cer_rows(f, segB, w)
                r = {"frame": frame, "seed": seed, "method": tag, "round": rnd,
                     "pose": float(hp.mean()), "fused": float(hf.mean()), "hits": hf.tolist(),
                     "cer": pl.pooled(cr), "per": [(x[2], x[3]) for x in cr]}
                if keep is not None:
                    m = keep & (truth >= 0)
                    r.update(n_pseudo=int(keep.sum()), pseudo_prec=float((lab[m] == truth[m]).mean()) if m.any() else float("nan"))
                res.append(r)
                print(f"  {frame:<7}s{seed} {tag:<7} r{rnd} pose {r['pose']:.3f} fused {r['fused']:.3f} "
                      f"CER {r['cer']:.3f}" + (f"  pseudo n={r['n_pseudo']} prec(labelled)={r['pseudo_prec']:.3f}"
                                               if keep is not None else ""), flush=True)

            def refit(lab, conf, keep):
                i = A[keep[A]]
                b = fit_pose(np.vstack([Xtr, Xall[i]]), np.concatenate([ytr, lab[i]]), seed,
                             np.r_[np.ones(len(ytr)), conf[i] * a.pseudo_w])
                return K.to_na(b.predict(Xall))

            score("base", 0, p0)
            m = np.zeros(len(taps), bool); m[A] = truth[A] >= 0
            score("oracle", 1, refit(truth, np.ones(len(taps)), m), truth, None, m)
            f0 = K.gmean(px, p0, al)
            lab, conf = f0.argmax(1), f0.max(1)
            keep = np.zeros(len(taps), bool); keep[A] = conf[A] >= ST_TAU
            score("nolm", 1, refit(lab, conf, keep), lab, conf, keep)
            pose = p0
            for rnd in range(1, ST_ROUNDS + 1):
                f = K.gmean(px, pose, al)
                lab, conf = pseudo_labels(f, segA, w)
                keep = np.zeros(len(taps), bool); keep[A] = (conf[A] >= ST_TAU) & (lab[A] >= 0)
                pose = refit(lab, conf, keep)
                score("lm", rnd, pose, lab, conf, keep)
    RES.mkdir(parents=True, exist_ok=True)
    prev = RES / f"selftrain_{a.tag}.json"
    if a.seed_list and prev.exists():
        done = {(r["frame"], r["seed"]) for r in res}
        res = [r for r in json.loads(prev.read_text()) if (r["frame"], r["seed"]) not in done] + res
    prev.write_text(json.dumps(res))
    print("\nsummary (mean over seeds; paired over eval taps / phrases vs base):")
    for frame in a.frames.split(","):
        base = sorted([r for r in res if r["frame"] == frame and r["method"] == "base"], key=lambda r: r["seed"])
        bh = np.mean([r["hits"] for r in base], 0)
        for tag, rnd in [("oracle", 1), ("nolm", 1)] + [("lm", r) for r in range(1, ST_ROUNDS + 1)]:
            rr = sorted([r for r in res if r["frame"] == frame and r["method"] == tag and r["round"] == rnd],
                        key=lambda r: r["seed"])
            h = np.mean([r["hits"] for r in rr], 0)
            lo, hi = boot(h - bh)
            dc = [pl.boot_delta([(0, 0, e, n) for e, n in b["per"]], [(0, 0, e, n) for e, n in r["per"]])
                  for b, r in zip(base, rr)]
            print(f"  {frame:<7}{tag:<7}r{rnd} fused {h.mean():.3f} (base {bh.mean():.3f}) d={h.mean()-bh.mean():+.3f} "
                  f"[{lo:+.3f},{hi:+.3f}]  CER {np.mean([r['cer'] for r in rr]):.3f} (base "
                  f"{np.mean([b['cer'] for b in base]):.3f}) per-seed dCER {[round(x[0], 3) for x in dc]}")
    return 0


# ============================================================ desk transfer
DESK_ALPHAS = (0.5, 0.75)
DESK_N_STREAMS = 20
DESK_OBS = (2.0, 3.0)
DESK_DELS = (-9.0, -5.0)
TAG = "base+rec"
_DV: dict = {}


def desk_sids() -> list[int]:
    from phase0.analysis import hirecall as hr
    return [i for i, _ in hr.subsample([(i, hr.ev_of(TAG, i)) for i in K.desk_streams()], DESK_N_STREAMS)]


def desk_model_file(spec: str, seed: int) -> Path:
    frame, _, model = spec.partition("@")
    frame = frame.replace("+coral", "")
    return model_file(frame, NEW, seed) if not model else CACHE / "models" / f"desk_{frame}_{model}_s{seed}.txt"


def desk_pose(sess, k: np.ndarray, spec: str, seed: int) -> np.ndarray:
    import lightgbm as lgb
    frame = spec.split("@")[0].replace("+coral", "")
    key = (spec, seed)
    mk = (spec, seed, len(k), int(np.asarray(k).sum()), int(k[len(k) // 2]) if len(k) else 0)
    if mk in _DV:
        return _DV[mk]
    if key not in _DV:
        _DV[key] = lgb.Booster(model_file=str(desk_model_file(spec, seed)))
    X = feats(DESK, k, frame, DAY1)
    if "+coral" in spec:
        X = coral_align(X, train_cached(frame, DAY1)[0])
    _DV[mk] = K.to_na(_DV[key].predict(X, num_threads=1))
    return _DV[mk]


def _desk_init(spec: str, seed: int) -> None:
    from phase0.analysis import combine as CB
    from phase0.analysis import hirecall as hr
    from phase0.analysis import pipeline as pl
    CB._shim()
    hr.use_cache()
    pl.spatial_proba = lambda sess, k, kf, st: desk_pose(sess, k, spec, seed)


def _desk_job(args):
    from phase0.analysis import combine as CB
    from phase0.analysis import hirecall as hr
    from phase0.analysis import pipeline as pl
    sid = args
    kf = pl.kbd_fit()
    s = pl.session_path(DESK)
    ev = hr.ev_of(TAG, sid)
    rows = []
    for al in DESK_ALPHAS:
        _, segs, p, _, _ = CB.stack_proba(s, ev, kf, CB.Cfg(alpha=al, em=False), "none")
        for o in DESK_OBS:
            for dl in DESK_DELS:
                c = CB.Cfg(alpha=al, obs=o, deletion=dl, em=False)
                rows.append({"sid": sid, "alpha": al, "obs": o, "deletion": dl, "n_taps": int(len(ev)),
                             "per": [(x[2], x[3]) for x in CB.decode_rows(p, segs, kf, c)]})
    return rows


def desk_table(spec: str, seed: int, procs: int) -> list[dict]:
    import multiprocessing as mp
    out = CACHE / "desk" / f"table_{spec}_s{seed}.json"
    if out.exists():
        return json.loads(out.read_text())
    sids = desk_sids()
    rows, t0 = [], time.time()
    with mp.get_context("spawn").Pool(procs, _desk_init, (spec, seed)) as pool:
        for got in pool.imap_unordered(_desk_job, sids):
            rows += got
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))
    print(f"  table {spec} s{seed}: {len(sids)} streams ({time.time()-t0:.0f}s)", flush=True)
    return rows


def _desk_cv_job(args):
    from dataclasses import replace
    from phase0.analysis import combine as CB
    from phase0.analysis import hirecall as hr
    from phase0.analysis import pipeline as pl
    j, pick, em = args
    kf = pl.kbd_fit()
    s = pl.session_path(DESK)
    ev = hr.ev_of(TAG, pick["sid"])
    c = CB.Cfg(alpha=pick["alpha"], obs=pick["obs"], deletion=pick["deletion"], em=em)
    _, segs, p, sess, k = CB.stack_proba(s, ev, kf, c, "none")
    if em:
        tr = [x for i, x in enumerate(segs) if i != j and len(x[1])]
        p = pl.weakly_supervise(sess, k, p, tr, replace(pl.FULL, deletion=pick["deletion"]))
    return j, CB.decode_rows(p, segs, kf, c, keep={j})[0]


def desk_cv(spec: str, seed: int, em: bool, procs: int) -> list[tuple]:
    import multiprocessing as mp
    from phase0.analysis import combine as CB
    out = CACHE / "desk" / f"cv_{spec}_s{seed}_em{int(em)}.json"
    if out.exists():
        return [tuple(x) for x in json.loads(out.read_text())]
    rows = desk_table(spec, seed, procs)
    n = len(rows[0]["per"])
    picks = [CB.select(rows, set(range(n)) - {j}) for j in range(n)]
    res = {}
    with mp.get_context("spawn").Pool(procs, _desk_init, (spec, seed)) as pool:
        for j, r in pool.imap_unordered(_desk_cv_job, [(j, picks[j], em) for j in range(n)]):
            res[j] = r
    r = [res[j] for j in range(n)]
    out.write_text(json.dumps(r))
    return r


def desk_selftrain_model(frame: str, seed: int, rounds: int = 2) -> str:
    """LightGBM refit on day-1 + LM pseudo-labelled desk taps of one density-chosen stream; no text."""
    from phase0.analysis import combine as CB
    from phase0.analysis import hirecall as hr
    from phase0.analysis import pipeline as pl
    from phase0.analysis.decode import Weights
    tag = "st"
    if desk_model_file(f"{frame}@{tag}", seed).exists():
        return f"{frame}@{tag}"
    CB._shim(); hr.use_cache()
    sids = desk_sids()
    nch = hr.nchars()
    sid = min(sids, key=lambda i: (abs(len(hr.ev_of(TAG, i)) / nch - 1.2), i))
    ev = hr.ev_of(TAG, sid)
    s = pl.session_path(DESK)
    d = pl.frame_probs(s)
    taps = pl.taps_from(d, ev)
    sess = tp.load_sess(DESK)
    k = sess.rows(taps)
    segs = [idx for _, idx in pl.segments(s, taps) if len(idx) >= 3]
    Xd = feats(DESK, k, frame, DAY1)
    Xtr, ytr = train_cached(frame, DAY1)
    px = CB.pix_for(s, ev, "none")
    import lightgbm as lgb
    b = lgb.Booster(model_file=str(model_file(frame, NEW, seed)))
    w = Weights(obs=3.0, deletion=-9.0, insertion=-7.0, max_deletions=3)
    for r in range(rounds):
        f = K.gmean(px, K.to_na(b.predict(Xd)), 0.5)
        lab, conf = pseudo_labels(f, segs, w)
        keep = (lab >= 0) & (conf >= ST_TAU)
        b = fit_pose(np.vstack([Xtr, Xd[keep]]), np.concatenate([ytr, lab[keep]]), seed,
                     np.r_[np.ones(len(ytr)), conf[keep]])
        print(f"  desk self-train {frame} s{seed} round {r+1}: stream {sid} ({len(ev)/nch:.2f} taps/char), "
              f"{int(keep.sum())} pseudo-labels", flush=True)
    p = desk_model_file(f"{frame}@{tag}", seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    b.save_model(str(p))
    return f"{frame}@{tag}"


def cmd_desk(a) -> int:
    from phase0.analysis import pipeline as pl
    from phase0.analysis import hirecall as hr
    specs = a.specs.split(",")
    res = {}
    for spec in specs:
        for seed in SEEDS[:a.seeds]:
            sp = spec
            if spec.endswith("@st"):
                desk_selftrain_model(spec.split("@")[0].replace("+coral", ""), seed)
            for em in (False, True):
                r = desk_cv(sp, seed, em, a.procs)
                res[(spec, seed, em)] = r
                c, lo, hi = pl.boot_ci(r)
                print(f"  {spec:<18} s{seed} EM={'text' if em else 'off '} CER {c:.3f} [{lo:.3f},{hi:.3f}]", flush=True)
    off = json.loads((Path(".cache/hirecall") / "cv_all.json").read_text())
    ref = [(None, None, e, n) for e, n in off["per"]]
    out = []
    print(f"\ndesk CER, nested leave-one-phrase-out over {len(desk_sids())} streams, grid "
          f"{DESK_ALPHAS}x{DESK_OBS}x{DESK_DELS}; official hirecall = {off['cer']:.3f}")
    base = specs[0]
    for spec in specs:
        for em in (False, True):
            per_seed = [pl.pooled(res[(spec, s, em)]) for s in SEEDS[:a.seeds]]
            dl = [pl.boot_delta(res[(base, s, em)], res[(spec, s, em)]) for s in SEEDS[:a.seeds]]
            pooled_rows = [x for s in SEEDS[:a.seeds] for x in res[(spec, s, em)]]
            c, lo, hi = pl.boot_ci(pooled_rows)
            print(f"  {spec:<18} EM={'text' if em else 'off '} CER {np.mean(per_seed):.3f} seeds "
                  f"{[round(x, 3) for x in per_seed]}  delta vs {base}: "
                  + ", ".join(f"{x[0]:+.3f} [{x[1]:+.3f},{x[2]:+.3f}]" for x in dl))
            out.append({"spec": spec, "em": em, "cer_seeds": per_seed, "delta_vs_base": dl,
                        "examples": [(x[0], x[1]) for x in res[(spec, 0, em)][:6]]})
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"desk_{a.tag}.json").write_text(json.dumps(out, indent=1))
    return 0


# ============================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.normalize", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("diag", cmd_diag), ("loso", cmd_loso), ("table", cmd_table),
                     ("enrol", cmd_enrol), ("warmup", cmd_warmup), ("perturb", cmd_perturb), ("augment", cmd_augment), ("selftrain", cmd_selftrain), ("desk", cmd_desk)):
        p = sub.add_parser(name)
        p.add_argument("--methods", default="ft,T,ftT")
        p.add_argument("--specs", default="abs+coral,tap_g")
        p.add_argument("--procs", type=int, default=3)
        p.add_argument("--pseudo-w", type=float, default=1.0)
        p.add_argument("--tag", default="main")
        p.add_argument("--seed-list", default="")
        p.add_argument("--frames", default=",".join(FRAMES))
        p.add_argument("--folds", default="")
        p.add_argument("--seeds", type=int, default=len(SEEDS))
        p.set_defaults(func=fn)
    a = ap.parse_args(argv)
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = a.func(a)
    print(f"\n[{time.time()-t0:.1f}s]", file=sys.stderr)
    return r


if __name__ == "__main__":
    sys.exit(main())
