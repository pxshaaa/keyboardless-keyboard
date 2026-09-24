"""EgoPressure -> press-vs-hover from a simulated top-down MediaPipe view, transferred to ours.
Run: python -m phase0.analysis.contact_ego {inspect | build | press_hover | transfer}"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

import lightgbm  # noqa: F401

from phase0.analysis import taps_gb as gb
from phase0.analysis import touch_common as tc

EGO = Path(".cache/contact_ego/egop")
OUT = Path(".cache/contact_ego")
RES = Path("results/contact")
PAD_X, PAD_Y = 0.120, 0.06875
TIPS = (4, 8, 12, 16, 20)
CAM = "4"                    # overhead: centre ~(-44,-87,-557) mm, optical axis (0.08,0.15,0.99)
FORCE_R = 0.012              # m: force summed within this radius of a fingertip
# our 20260910-131629-kbd medians (see REPORT conventions): wrist px, wrist->middle-MCP vector px
OURS = {"left": {"wrist": np.array([141.0, 68.0]), "dir": np.array([102.3, 17.5]), "span": 104.8,
                 "z_ratio": -2.07},
        "right": {"wrist": np.array([99.0, 698.0]), "dir": np.array([49.9, -83.5]), "span": 107.4,
                  "z_ratio": -1.43}}


def seq_dirs(pattern: str = "*") -> list[Path]:
    out = []
    for d in sorted(glob.glob(str(EGO / "data" / "p_*" / pattern))):
        d = Path(d)
        if (d / "annotation.parquet").exists() and (d / "pressure.parquet").exists():
            out.append(d)
    return out


def load_seq(d: Path) -> dict | None:
    import pyarrow.parquet as pq
    a = pq.read_table(d / "annotation.parquet", columns=["frame", "has_annotation", "hand_side",
                                                          "joint_position"])
    pr = pq.read_table(d / "pressure.parquet", columns=["frame", "force"])
    ha = a.column("has_annotation").to_numpy(zero_copy_only=False)
    if ha.sum() < 60:
        return None
    fa = a.column("frame").to_numpy()
    J = np.full((len(fa), 21, 3), np.nan)
    jp = a.column("joint_position").to_pylist()
    for i, v in enumerate(jp):
        if v is not None and ha[i]:
            J[i] = np.asarray(v, float).reshape(21, 3)
    side = a.column("hand_side").to_pylist()[0]
    fp = pr.column("frame").to_numpy()
    force = np.zeros((len(fa), 105, 185), np.float32)
    pos = {f: i for i, f in enumerate(fa)}
    fl = pr.column("force").to_pylist()
    for f, v in zip(fp, fl):
        if f in pos and v is not None:
            force[pos[f]] = np.asarray(v, np.float32).reshape(105, 185)
    cfg = json.loads((EGO / "configs" / d.parent.name / f"{d.name}.json").read_text())
    return {"name": d.name, "participant": d.parent.name, "side": side, "frame": fa, "J": J,
            "force": force, "cal": cfg["camera_calibrations"][CAM]}


def finger_force(J: np.ndarray, force: np.ndarray, r: float = FORCE_R) -> np.ndarray:
    """-> [F,5] force counts within r of each fingertip's pad projection (+ total in col 5)."""
    cx = -PAD_X + (np.arange(185) + 0.5) * (2 * PAD_X / 185)
    cy = -PAD_Y + (np.arange(105) + 0.5) * (2 * PAD_Y / 105)
    GX, GY = np.meshgrid(cx, cy)
    out = np.zeros((len(J), 6), np.float32)
    out[:, 5] = force.reshape(len(force), -1).sum(1)
    for i in range(len(J)):
        if out[i, 5] <= 0 or not np.isfinite(J[i, 0, 0]):
            continue
        for k, tip in enumerate(TIPS):
            m = (GX - J[i, tip, 0]) ** 2 + (GY - J[i, tip, 1]) ** 2 <= r * r
            out[i, k] = force[i][m].sum()
    return out


def project(J: np.ndarray, cal: dict) -> tuple[np.ndarray, np.ndarray]:
    """world metres -> (uv px [F,21,2], camera-frame mm [F,21,3]) through static camera `cal`."""
    M = np.asarray(cal["ModelViewMatrix"], float).reshape(4, 4)
    Pc = J * 1000.0 @ M[:3, :3].T + M[:3, 3]
    u = cal["fx"] * Pc[..., 0] / Pc[..., 2] + cal["cx"]
    v = cal["fy"] * Pc[..., 1] / Pc[..., 2] + cal["cy"]
    return np.stack([u, v], -1), Pc


def _parity(uv, side):
    f = np.nanmedian(uv[:, 9] - uv[:, 0], 0)
    th = np.nanmedian(uv[:, 4] - uv[:, 0], 0)
    return np.sign(f[0] * th[1] - f[1] * th[0])


OUR_PARITY = {"left": 1.0, "right": -1.0}   # sign(fwd x thumb) measured on our 131629 medians


def to_ours(uv: np.ndarray, Pc: np.ndarray, side: str, channels: str, rng=None) -> np.ndarray:
    """-> P-like [F,21,7] (x, y, conf, z, wx, wy, wz) in our image convention for this hand."""
    o = OURS[side]
    if _parity(uv, side) != OUR_PARITY[side]:
        uv = uv * np.array([1.0, -1.0])            # mirror so thumb sits on the same side
    f = np.nanmedian(uv[:, 9] - uv[:, 0], 0)
    ang = np.arctan2(o["dir"][1], o["dir"][0]) - np.arctan2(f[1], f[0])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    span = np.nanmedian(np.linalg.norm(uv[:, 9] - uv[:, 0], axis=1))
    s = o["span"] / span
    w0 = np.nanmedian(uv[:, 0], 0)
    xy = (uv - w0) @ R.T * s + o["wrist"]
    F = len(uv)
    out = np.full((F, 21, 7), np.nan, np.float32)
    out[..., :2] = xy
    out[..., 2] = 0.99
    if channels == "3d":
        dz = Pc[..., 2] - Pc[:, :1, 2]                          # mm, + = farther from camera
        zpx = dz * s * (np.nanmedian(np.abs(uv[:, 9] - uv[:, 0]).sum(1)) / 1.0) / \
            np.nanmedian(np.linalg.norm(Pc[:, 9] - Pc[:, 0], axis=1))  # mm -> px at hand scale
        k = o["z_ratio"] * o["span"] / (np.nanmedian(zpx[:, 8]) + 1e-9) if np.nanmedian(zpx[:, 8]) else 1.0
        out[..., 3] = zpx * abs(k) * np.sign(np.nanmedian(zpx[:, 8]) * o["z_ratio"] + 1e-12)
        C = np.nanmean(Pc[:, (0, 5, 9, 13, 17)], 1, keepdims=True)
        W = (Pc - C) / 1000.0
        W[..., :2] = W[..., :2] @ R.T
        out[..., 4:7] = W
    if rng is not None:                                         # MediaPipe-like jitter
        out[..., :2] += rng.normal(0, 0.8, out[..., :2].shape)
    return out


def resample60(t30: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t60 = np.arange(t30[0], t30[-1] + 1e-9, 1 / 60.0)
    flat = X.reshape(len(X), -1)
    ok = np.isfinite(flat[:, 0])
    out = np.full((len(t60), flat.shape[1]), np.nan, np.float32)
    if ok.sum() >= 2:
        for c in range(flat.shape[1]):
            out[:, c] = np.interp(t60, t30[ok], flat[ok, c], left=np.nan, right=np.nan)
        # do not bridge gaps longer than 2 frames
        gap = np.interp(t60, t30, (~ok).astype(float)) > 0.5
        out[gap] = np.nan
    return t60, out.reshape((len(t60),) + X.shape[1:])


CONTACT_THR = 150.0  # counts (~0.09 N) within FORCE_R of a tip, set by `inspect`


def onsets(ff: np.ndarray, t: np.ndarray, thr: float = CONTACT_THR, min_off: int = 3) -> np.ndarray:
    """Contact-onset times: any fingertip crossing thr upward after >= min_off frames below."""
    ev = []
    for k in range(5):
        on = ff[:, k] >= thr
        last_on = -99
        for i in range(1, len(on)):
            if on[i] and not on[i - 1] and i - last_on > min_off:
                ev.append(t[i])
            if on[i]:
                last_on = i
    ev = np.sort(np.array(ev))
    if len(ev) > 1:  # two fingers landing in the same 50 ms count once, like one keydown
        keep = np.concatenate([[True], np.diff(ev) > 0.05])
        ev = ev[keep]
    return ev


def build(channels: str = "2d", pattern: str = "*", seed: int = 0) -> dict:
    """-> per-sequence 60 Hz feature matrices + labels, cached."""
    f = OUT / f"ego_{channels}_{pattern.replace('*', 'all')}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return {k: z[k] for k in z.files}
    rng = np.random.default_rng(seed)
    Xs, ys, ygs, T, SEQ, PART, GEST, SIDE, KT, TIPH, CONT = [], [], [], [], [], [], [], [], [], [], []
    for n, d in enumerate(seq_dirs(pattern)):
        s = load_seq(d)
        if s is None:
            continue
        t30 = (s["frame"] - s["frame"][0]) / 30.0
        ff = finger_force(s["J"], s["force"])
        uv, Pc = project(s["J"], s["cal"])
        P1 = to_ours(uv, Pc, s["side"], channels, rng)
        t60, P60 = resample60(t30, P1)
        _, ff60 = resample60(t30, ff)
        _, J60 = resample60(t30, s["J"])
        slot = 0 if s["side"] == "left" else 1
        P = np.full((len(t60), 2, 21, 7), np.nan)
        P[:, slot] = P60
        X = gb.assemble(gb.build_groups(P), gb.GROUPS).astype(np.float32)
        kt = onsets(ff, t30)
        Xs.append(X)
        ys.append(gb.labels(t60, kt, gb.LABEL_HALF_WIDTH_S))
        ygs.append(gb.labels(t60, kt, gb.TYPING_HALF_WIDTH_S))
        T.append(t60 + 1000.0 * n)                   # sequences never touch in time
        KT.append(kt + 1000.0 * n)
        SEQ.append(np.full(len(t60), n))
        PART.append(np.full(len(t60), int(s["participant"][2:])))
        GEST.append(np.array([s["name"].split("_", 2)[2]] * len(t60)))
        SIDE.append(np.full(len(t60), slot))
        TIPH.append(-J60[:, list(TIPS), 2])          # height above pad, m (joint centre)
        CONT.append(np.nan_to_num(ff60[:, :5]))
        if n % 25 == 0:
            print(f"  {n} {s['name']} frames={len(t60)} onsets={len(kt)}", flush=True)
    out = {"X": np.vstack(Xs), "y": np.concatenate(ys), "yg": np.concatenate(ygs),
           "t": np.concatenate(T), "kt": np.concatenate(KT), "seq": np.concatenate(SEQ),
           "part": np.concatenate(PART), "gest": np.concatenate(GEST), "side": np.concatenate(SIDE),
           "tiph": np.vstack(TIPH), "cont": np.vstack(CONT)}
    np.savez(f, **out)
    return out


def cmd_inspect(a) -> int:
    for pat in ("p_001_index_press_high_x5_right", "p_001_index_press_no-contact_x5_right",
                "p_001_type_ipad_5x_left"):
        ds = seq_dirs(pat)
        if not ds:
            continue
        s = load_seq(ds[0])
        ff = finger_force(s["J"], s["force"])
        ok = np.isfinite(s["J"][:, 0, 0])
        h = -s["J"][:, TIPS, 2]
        print(f"{s['name']}: frames={len(ff)} annotated={ok.sum()} total force pct "
              f"{np.percentile(ff[:, 5], [50, 90, 99]).round(0)}")
        for k, tip in enumerate(TIPS):
            on = ff[:, k] > CONTACT_THR
            print(f"   tip{tip}: frames>thr={on.sum():4d} force p99={np.percentile(ff[:, k], 99):7.0f} "
                  f"height mm median(contact)={1000*np.nanmedian(h[on, k]) if on.any() else float('nan'):5.1f} "
                  f"median(no)={1000*np.nanmedian(h[~on & ok, k]):5.1f} min={1000*np.nanmin(h[ok, k]):5.1f}")
        print(f"   onsets={len(onsets(ff, np.arange(len(ff))/30.0))}  unattributed force frames="
              f"{int(((ff[:, 5] > 300) & (ff[:, :5].max(1) < CONTACT_THR)).sum())}")
        uv, Pc = project(s["J"], s["cal"])
        print(f"   cam{CAM} uv median wrist {np.nanmedian(uv[:,0],0).round(0)} span px "
              f"{np.nanmedian(np.linalg.norm(uv[:,9]-uv[:,0],axis=1)):.0f} parity {_parity(uv, s['side'])}")
    return 0



# ------------------------------------------------------------------ press vs hover on EgoPressure itself
def descent_events(D: dict, prom_mm: float = 8.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every fingertip descent bottom (true height local minimum, >= prom_mm prominence).
    -> (row index, label contact?, height mm at the bottom)."""
    from scipy.signal import find_peaks
    rows, lab, hmm = [], [], []
    for q in np.unique(D["seq"]):
        idx = np.where(D["seq"] == q)[0]
        H = D["tiph"][idx] * 1000.0
        C = D["cont"][idx]
        for k in range(5):
            h = H[:, k]
            ok = np.isfinite(h)
            if ok.sum() < 30:
                continue
            hh = np.where(ok, h, np.nanmax(h))
            pk, _ = find_peaks(-hh, prominence=prom_mm, distance=20)
            for i in pk:
                if not (0 <= hh[i] <= 60):          # outside the pad or implausibly high
                    continue
                c = C[max(0, i - 4):i + 5, k].max() >= CONTACT_THR
                rows.append(idx[i])
                lab.append(c)
                hmm.append(hh[i])
    return np.array(rows), np.array(lab), np.array(hmm)


def _group_folds(part: np.ndarray, k: int = 7) -> list[np.ndarray]:
    u = np.unique(part)
    return [np.isin(part, u[i::k]) for i in range(k)]


def cmd_press_hover(a) -> int:
    from sklearn.metrics import roc_auc_score, average_precision_score
    RES.mkdir(parents=True, exist_ok=True)
    out = {}
    for ch in ("2d", "3d"):
        E = np.load(OUT / f"ego_events_{ch}.npz", allow_pickle=True)
        X, lab, hmm, part, gest = E["X"].astype(np.float32), E["lab"], E["hmm"], E["part"], E["gest"].astype(str)
        print(f"[{ch}] descent bottoms={len(lab)} contact={int(lab.sum())} hover={int((~lab).sum())} "
              f"hover height mm median={np.median(hmm[~lab]):.1f} contact median={np.median(hmm[lab]):.1f}")
        res = {}
        ip = np.char.startswith(gest, "index_press_high") | np.char.startswith(gest, "index_press_low") | \
            np.char.startswith(gest, "index_press_no-contact")
        for name, sel in (("all", np.ones(len(lab), bool)), ("hover<15mm", lab | (hmm < 15.0)),
                          ("index_press high/low vs no-contact", ip),
                          ("index_press, hover<15mm", ip & (lab | (hmm < 15.0)))):
            aucs = []
            for seed in range(a.seeds):
                sc = np.zeros(sel.sum())
                Xs, ys, ps = X[sel], lab[sel], part[sel]
                for te in _group_folds(ps):
                    m = gb.make_model("lgbm", random_state=seed, n_jobs=4, n_estimators=300,
                                      scale_pos_weight=1.0, min_child_samples=20)
                    m.fit(Xs[~te], ys[~te])
                    sc[te] = m.predict_proba(Xs[te])[:, 1]
                aucs.append(roc_auc_score(ys, sc))
            h_auc = roc_auc_score(lab[sel], -hmm[sel])
            res[name] = {"n": int(sel.sum()), "pos": int(lab[sel].sum()), "auc": float(np.mean(aucs)),
                         "auc_seeds": [float(x) for x in aucs], "true_height_auc": float(h_auc),
                         "hover_height_mm_median": float(np.median(hmm[sel & ~lab])) if (sel & ~lab).any() else None}
            print(f"   {name:<38} n={sel.sum():5d} AUC={np.mean(aucs):.3f} "
                  f"(seeds {', '.join(f'{x:.3f}' for x in aucs)})  true-height AUC={h_auc:.3f}", flush=True)
        out[ch] = res
    T = np.load(OUT / "ego_train_2d.npz")
    X, y, w, part = T["X"].astype(np.float32), T["y"], T["w"], T["part"]
    aps = []
    for seed in range(min(a.seeds, 2)):
        oof = np.zeros(len(y))
        for te in _group_folds(part, 5):
            m = gb.make_model("lgbm", random_state=seed, n_jobs=4)
            m.fit(X[~te], y[~te], sample_weight=w[~te])
            oof[te] = m.predict_proba(X[te])[:, 1]
        aps.append(average_precision_score(y, oof, sample_weight=w))
        print(f"   [2d] frame-level contact-onset AP (participant-grouped, reweighted to the full "
              f"set) = {aps[-1]:.3f}, base rate {np.average(y, weights=w):.4f}", flush=True)
    out["onset_ap_2d"] = [float(x) for x in aps]
    out["onset_base_rate"] = float(np.average(y, weights=w))
    (RES / "ego_press_hover.json").write_text(json.dumps(out, indent=1))
    return 0


# ------------------------------------------------------------------ transfer to our landmarks
def ego_model(seed: int = 0):
    import joblib
    f = OUT / f"ego_model_2d_s{seed}.joblib"
    if f.exists():
        return joblib.load(f)
    T = np.load(OUT / "ego_train_2d.npz")
    m = gb.make_model("lgbm", random_state=seed, n_jobs=4).fit(T["X"].astype(np.float32), T["y"],
                                                                sample_weight=T["w"])
    joblib.dump(m, f)
    return m


def our_P_2d(P: np.ndarray, slot: int) -> np.ndarray:
    Q = np.array(P, np.float64, copy=True)
    Q[..., 3:] = np.nan
    Q[..., 2] = np.where(np.isfinite(Q[..., 0]), 0.99, np.nan)
    Q[:, 1 - slot] = np.nan
    return Q


def ego_scores(sid: str, seed: int = 0) -> np.ndarray:
    """-> [F,2] zero-shot EgoPressure contact-onset probability per hand on one of our sessions."""
    f = OUT / f"E_{sid}_s{seed}.npy"
    if f.exists():
        return np.load(f)
    m = ego_model(seed)
    P = tc.frames(sid)[2]
    out = np.zeros((len(P), 2))
    for slot in (0, 1):
        X = gb.assemble(gb.build_groups(our_P_2d(P, slot)), gb.GROUPS).astype(np.float32)
        out[:, slot] = m.predict_proba(X)[:, 1]
        out[~np.isfinite(P[:, slot, 0, 0]), slot] = np.nan
    np.save(f, out)
    return out


EGO_LAGS = (-4, -2, -1, 1, 2, 4)


def ego_feats(sid: str, seed: int = 0) -> np.ndarray:
    E = ego_scores(sid, seed)
    mx = np.nanmax(np.nan_to_num(E, nan=0.0), 1)
    return np.hstack([E, mx[:, None]] + [gb._shift(mx[:, None], k) for k in EGO_LAGS]).astype(np.float32)


def feat_base_ego(sid: str) -> np.ndarray:
    return np.hstack([np.asarray(tc.base_X(sid)), ego_feats(sid)])


def cmd_transfer(a) -> int:
    """Zero-shot: EgoPressure-only model's max-over-hands score as a detector on our keyboard."""
    from sklearn.metrics import average_precision_score
    RES.mkdir(parents=True, exist_ok=True)
    tt, kk, pp, off = [], [], [], 0.0
    for s in tc.KBD_TRAIN4:
        t = tc.frames(s)[1]
        e = np.nanmax(np.nan_to_num(ego_scores(s), nan=0.0), 1)
        tt.append(t - t[0] + off); kk.append(tc.kt(s) - t[0] + off); pp.append(e)
        off = tt[-1][-1] + 100.0
    T, K, Pp = map(np.concatenate, (tt, kk, pp))
    cfg, f1_tr = gb.tune_events(Pp, T, K, np.ones(len(T), bool))
    e = np.nanmax(np.nan_to_num(ego_scores(tc.KBD_TEST), nan=0.0), 1)
    r = tc.score_events(tc.KBD_TEST, e, None, cfg)
    yt = gb.labels(tc.frames(tc.KBD_TEST)[1], tc.kt(tc.KBD_TEST))
    ap = average_precision_score(yt, e)
    ce = tc.count_error(np.nanmax(np.nan_to_num(ego_scores(tc.DESK), nan=0.0), 1), None, gate_thr=0.0)
    out = {"zero_shot_kbd_heldout": r, "cfg": cfg, "train_f1": f1_tr, "frame_ap": ap,
           "frame_base_rate": float(yt.mean()), "desk_count": ce}
    print(json.dumps({k: v for k, v in out.items() if k != "desk_count"}, indent=1, default=float))
    print(f"desk count err {ce['count_err']:.3f} r={ce['count_r']:.3f}")
    (RES / "ego_zero_shot.json").write_text(json.dumps(out, indent=1, default=float))
    return 0



# ------------------------------------------------------------------ compact arrays for the Mac mini
def cmd_compact(a) -> int:
    """Full per-frame EgoPressure matrices are 3.6 GB; ship only (i) descent-bottom rows for the
    press-vs-hover test and (ii) a weighted frame subsample for the transfer model."""
    rng = np.random.default_rng(0)
    for ch in ("2d", "3d"):
        D = build(ch)
        rows, lab, hmm = descent_events(D)
        np.savez(OUT / f"ego_events_{ch}.npz", X=D["X"][rows], lab=lab, hmm=hmm,
                 part=D["part"][rows], gest=D["gest"][rows].astype(str))
        print(f"{ch}: {len(rows)} descent bottoms", flush=True)
        if ch == "2d":
            t, kt = D["t"], np.sort(D["kt"])
            j = np.clip(np.searchsorted(kt, t), 1, len(kt) - 1)
            dist = np.minimum(np.abs(t - kt[j - 1]), np.abs(kt[j] - t))
            near = dist <= 0.2
            keep = near | D["y"] | (rng.random(len(t)) < 0.12)
            w = np.where(near | D["y"], 1.0, 1.0 / 0.12)
            np.savez(OUT / "ego_train_2d.npz", X=D["X"][keep].astype(np.float16), y=D["y"][keep],
                     w=w[keep].astype(np.float32), part=D["part"][keep], seq=D["seq"][keep])
            print(f"train subsample {keep.sum()} of {len(t)} frames", flush=True)
        del D
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.contact_ego", description=__doc__)
    ap.add_argument("cmd", choices=("inspect", "build", "press_hover", "transfer", "compact"))
    ap.add_argument("--channels", default="2d")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args(argv)
    return {"inspect": cmd_inspect, "build": lambda a: (build(a.channels), 0)[1]}.get(
        a.cmd, lambda a: globals()[f"cmd_{a.cmd}"](a))(a)


if __name__ == "__main__":
    sys.exit(main())
