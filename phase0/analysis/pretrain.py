"""Public-corpus pretraining vs our labels alone (andrewt28/keystroke-typing-videos).
Run: python -m phase0.analysis.pretrain {extract|track|buildpub|dist|conditions|scaling}"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from phase0.analysis import taps_gb as G
from phase0.analysis.detect_taps import FINGERTIP_JOINTS
from phase0.analysis.eval_taps import keydown_times

PUBLIC = Path("data/public")
META = PUBLIC / "meta"
VIDEO = PUBLIC / "video"
LAND = PUBLIC / "landmarks"
SPLITS = ("train", "validation", "test")
FPS = 60.0

OUR_TRAIN = [Path("data/sessions/20260910-015217-kbd"), Path("data/sessions/20260910-021315-kbd")]
OUR_TEST = Path("data/sessions/20260910-015948-kbd")

# Public raw frames are 180-deg from a natural view; our own rig sits a further
# 90 deg over, so 90-CCW is what lines the two corpora up. Verified on frames.
ROTATION = "ccw90"
UPSCALE = 2.0
# defaults (0.5) find both hands in only 28% of public frames vs 93-99% of ours
MP_CONF = 0.15


# ---------------------------------------------------------------- metadata
def clips(split: str) -> list[dict]:
    return [json.loads(l) for l in open(META / f"{split}.jsonl") if l.strip()]


def all_clips() -> list[dict]:
    out = []
    for s in SPLITS:
        for c in clips(s):
            c["split"] = s
            out.append(c)
    return out


def land_path(split: str, file_name: str) -> Path:
    return LAND / split / (Path(file_name).stem + ".parquet")


# ---------------------------------------------------------------- extraction
def extract_clip(job: tuple[str, str, float]) -> tuple[str, int, int]:
    """One clip -> landmarks parquet. Returns (name, n_frames, n_frames_with_2_hands)."""
    import cv2
    import mediapipe as mp_
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

    from phase0.analysis.extract_landmarks import (
        DEFAULT_MODEL_PATH, LandmarkBatchWriter, NUM_HANDS, apply_rotation, normalized_to_pixels,
    )

    split, file_name, fps = job
    out = land_path(split, file_name)
    if out.exists():
        return (file_name, -1, -1)
    cap = cv2.VideoCapture(str(VIDEO / split / file_name))
    opts = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(DEFAULT_MODEL_PATH)),
                                 running_mode=RunningMode.VIDEO, num_hands=NUM_HANDS,
                                 min_hand_detection_confidence=MP_CONF,
                                 min_hand_presence_confidence=MP_CONF,
                                 min_tracking_confidence=MP_CONF)
    tmp = out.with_suffix(".part")
    n = n2 = 0
    with HandLandmarker.create_from_options(opts) as lm, LandmarkBatchWriter(tmp) as w:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = apply_rotation(frame, ROTATION)
            if UPSCALE != 1.0:
                frame = cv2.resize(frame, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
            h, wd = frame.shape[:2]
            img = mp_.Image(image_format=mp_.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = lm.detect_for_video(img, int(round(n / fps * 1000)) + n)
            hands = res.hand_landmarks or []
            n2 += len(hands) >= 2
            for k, hl in enumerate(hands[:NUM_HANDS]):
                cats = res.handedness[k] if res.handedness else []
                x, y = normalized_to_pixels(hl, wd, h)
                z = np.array([p.z for p in hl], np.float32) * wd
                wl = res.hand_world_landmarks or []
                W = (np.array([[q.x, q.y, q.z] for q in wl[k]], np.float32)
                     if k < len(wl) else np.full((len(hl), 3), np.nan, np.float32))
                w.add_hand(n, n / fps, k, cats[0].category_name if cats else "Unknown", x, y,
                           float(cats[0].score) if cats else float("nan"), z, W)
            n += 1
    cap.release()
    tmp.replace(out)
    return (file_name, n, n2)


def cmd_extract(a) -> int:
    jobs = []
    for c in all_clips():
        if not land_path(c["split"], c["file_name"]).exists():
            jobs.append((c["split"], c["file_name"], float(c["actual_fps"])))
    for s in SPLITS:
        (LAND / s).mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} clips to extract, {a.workers} workers", flush=True)
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for k, (name, n, n2) in enumerate(pool.imap_unordered(extract_clip, jobs), 1):
            if k % 20 == 0:
                print(f"  {k}/{len(jobs)} {name} n={n} 2hands={n2} [{time.time()-t0:.0f}s]", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)
    return 0


# ---------------------------------------------------------------- loading + resampling
def load_public_frames(split: str, file_name: str) -> tuple[np.ndarray, np.ndarray]:
    """-> (t [F], P [F,2,21,7]) at the clip's native rate, in our channel order."""
    tb = pq.read_table(land_path(split, file_name))
    if tb.num_rows == 0:
        return np.zeros(0), np.zeros((0, 2, 21, len(G.CHANNELS)))
    c = {n: np.asarray(tb.column(n)) for n in ("i", "t", "hand", "handedness", "joint", *G.CHANNELS)}
    frames, inv = np.unique(c["i"], return_inverse=True)
    t = np.zeros(len(frames))
    t[inv] = c["t"]
    P = np.full((len(frames), 2, 21, len(G.CHANNELS)), np.nan)
    side = np.array([G.SIDES.index(h) if h in G.SIDES else 0 for h in c["handedness"]])
    key = frames[inv] * 2 + side
    _, first = np.unique(key, return_index=True)
    dup = np.ones(len(key), bool)
    dup[first] = False
    side = np.where(dup, c["hand"], side)
    for k, ch in enumerate(G.CHANNELS):
        P[inv, side, c["joint"], k] = c[ch]
    return t, P


def resample_60(t: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """-> 60 fps grid, so taps_gb's frame-indexed temporal features span the same
    wall-clock as on our rig. NaN where the source hand was absent."""
    from scipy.interpolate import interp1d

    if len(t) < 8:
        return np.zeros(0), np.zeros((0, 2, 21, P.shape[-1]))
    tn = np.arange(t[0], t[-1], 1.0 / FPS)
    out = np.full((len(tn), *P.shape[1:]), np.nan)
    for s in range(2):
        H = P[:, s].reshape(len(t), -1)
        ok = np.isfinite(H[:, 0])
        if ok.sum() < 8:
            continue
        cols = np.isfinite(H[ok]).all(0)
        if not cols.any():
            continue
        f = interp1d(t[ok], H[ok][:, cols], kind="cubic", axis=0, bounds_error=False,
                     fill_value=np.nan, assume_sorted=True)
        Y = np.full((len(tn), H.shape[1]), np.nan)
        Y[:, cols] = f(tn)
        # a gap wider than 1.5 source frames is a tracking dropout, not interpolable
        gap = np.interp(tn, t, ok.astype(float)) > 0.99
        near = np.abs(tn[:, None] - t[ok][None, :]).min(1) <= 1.5 / 29.5
        Y[~(gap & near)] = np.nan
        out[:, s] = Y.reshape(len(tn), *P.shape[2:])
    return tn, out


# ---------------------------------------------------------------- feature datasets
def our_data(sessions: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """-> (X, t, sid, kt) for our own sessions, taps_gb's exact feature space."""
    Xs, ts, sids, kts = [], [], [], []
    for k, s in enumerate(sessions):
        _, t, P = G.load_frames(s)
        Xs.append(G.assemble(G.build_groups(P), G.GROUPS).astype(np.float32))
        ts.append(t)
        sids.append(np.full(len(t), k))
        kts.append(keydown_times(s))
    return (np.vstack(Xs), np.concatenate(ts), np.concatenate(sids), np.sort(np.concatenate(kts)))


# ---------------------------------------------------------------- tracking report
def cmd_track(a) -> int:
    n_tot = n_any = n_two = 0
    per = []
    for c in all_clips():
        p = land_path(c["split"], c["file_name"])
        if not p.exists():
            continue
        tb = pq.read_table(p, columns=["i", "hand"])
        i = np.asarray(tb.column("i"))
        exp = int(round(c["duration_sec"] * c["actual_fps"]))
        u, n = np.unique(i, return_counts=True)
        n_tot += exp
        n_any += len(u)
        n_two += int((n >= 42).sum())
        per.append(((n >= 42).sum() / max(exp, 1)))
    print(f"clips={len(per)} frames={n_tot} any-hand={n_any/n_tot:.3f} two-hands={n_two/n_tot:.3f}")
    per = np.array(per)
    print(f"per-clip two-hand rate: p10={np.percentile(per,10):.3f} med={np.median(per):.3f} "
          f"p90={np.percentile(per,90):.3f} clips<0.5={(per<0.5).mean():.3f}")
    # our own sessions, same measure
    for s in OUR_TRAIN + [OUR_TEST]:
        _, t, P = G.load_frames(s)
        ok = np.isfinite(P[:, :, 0, 0]).sum(1)
        print(f"{s.name}: frames={len(t)} any={np.mean(ok>=1):.3f} two={np.mean(ok>=2):.3f}")
    return 0


# ---------------------------------------------------------------- distributions
PROBE = {
    "pose/rel_wrist_j8_y": ("pose", 15),
    "pose/flex2d_index": ("pose", 41),
    "pose/flex2d_middle": ("pose", 42),
    "wflex/wlen_index": ("wflex", 1),
    "wflex/wpipang_index": ("wflex", 6),
    "palm/palm_speed": ("palm", 4),
    "still/energy_31": ("still", 1),
    "rest/restdisp_mag_index": ("rest", 18),
}


def cmd_dist(a) -> int:
    """Overlap of a few interpretable features between the two corpora."""
    idx = {}
    off = 0
    dims = {}
    _, _, P = G.load_frames(OUR_TEST)
    g0 = G.build_groups(P[:200])
    for grp in G.GROUPS:
        dims[grp] = g0[grp].shape[1]
    for grp in G.GROUPS:
        idx[grp] = off
        off += dims[grp]

    def cols(name):
        grp, j = PROBE[name]
        # per-group blocks are [left-hand | right-hand]; probe the left half
        return idx[grp] + j

    Xo, to, _, kto = our_data(OUR_TRAIN)
    Xp, _, yp, _, _ = load_pub("pub_train.npz")
    print(f"ours rows={len(Xo)}  public rows={len(Xp)}")
    print(f"all-NaN feature rows: ours={np.isnan(Xo[:, 0]).mean():.3f} "
          f"public={np.isnan(Xp[:, 0]).mean():.3f}\n")
    print(f"{'feature':<28} {'ours med':>10} {'pub med':>10} {'ours IQR':>18} {'pub IQR':>18} {'overlap':>8}")
    for name in PROBE:
        c = cols(name)
        a_ = Xo[:, c][np.isfinite(Xo[:, c])]
        b_ = Xp[:, c][np.isfinite(Xp[:, c])]
        if not len(a_) or not len(b_):
            continue
        lo, hi = np.percentile(np.concatenate([a_, b_]), [0.5, 99.5])
        e = np.linspace(lo, hi, 60)
        ha = np.histogram(a_, e)[0] / len(a_)
        hb = np.histogram(b_, e)[0] / len(b_)
        ov = float(np.minimum(ha, hb).sum())
        print(f"{name:<28} {np.median(a_):>10.3f} {np.median(b_):>10.3f} "
              f"{str(np.round(np.percentile(a_,[25,75]),2)):>18} "
              f"{str(np.round(np.percentile(b_,[25,75]),2)):>18} {ov:>8.2f}")
    print(f"\npositive-frame rate: ours={G.labels(to, kto).mean():.3f}  public={yp.mean():.3f}")
    return 0


# ---------------------------------------------------------------- training
def fit(X, y, init=None, w=None, **kw):
    import lightgbm as lgb
    params = dict(n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                  scale_pos_weight=8.0, n_jobs=-1, verbose=-1, random_state=0)
    params.update(kw)
    m = lgb.LGBMClassifier(**params)
    m.fit(X, y, sample_weight=w, init_model=init)
    return m


OURS_POS = 0.20


def pooled_w(yp, yo, n_pub_scale=1.0):
    """Equal total weight per corpus, with each corpus's own class balance folded in
    so a single scale_pos_weight cannot suit one and saturate the other."""
    wp = np.where(yp, spw(yp), 1.0) * (len(yo) / len(yp)) * n_pub_scale
    wo = np.where(yo, spw(yo), 1.0)
    return np.concatenate([wp, wo])


def spw(y) -> float:
    """Match our corpus's effective class balance: public frames are ~68% positive
    (10 keys/s, no rest) vs ~20% here, so a fixed scale_pos_weight=8 would saturate."""
    r = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
    return 8.0 * (OURS_POS / (1 - OURS_POS)) / (r / (1 - r))


def oof_probs(X, y, yg, t, sid, folds, fitter, inits=(None, None)) -> tuple[np.ndarray, np.ndarray]:
    p = np.zeros(len(t))
    pg = np.zeros(len(t))
    for te in folds:
        tr = G.purge(~te, te, t)
        # a short subset can yield fewer blocks than splits, leaving a fold empty
        if not te.any() or not tr.any():
            continue
        p[te] = fitter(X[tr], y[tr], inits[0]).predict_proba(X[te])[:, 1]
        pg[te] = fitter(X[tr], yg[tr], inits[1]).predict_proba(X[te])[:, 1]
    return p, pg


def predict_session(m, gm, cfg, session: Path, out: Path, pre=None) -> Path:
    frames, t, P = G.load_frames(session)
    X = G.assemble(G.build_groups(P), G.GROUPS)
    if pre is not None:
        X = pre(X)
    p = m.predict_proba(X)[:, 1]
    g = gm.predict_proba(X)[:, 1] if gm is not None else None
    return _write_taps(G.apply_events(p, cfg, g), frames, t, P, out)


def _write_taps(ev, frames, t, P, out: Path) -> Path:
    fv = G.flex_velocity(P)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for k in ev:
            mm = np.abs(np.nan_to_num(fv[k], nan=-1.0))
            s, f = np.unravel_index(int(np.argmax(mm)), mm.shape)
            tip = FINGERTIP_JOINTS[f]
            fh.write(json.dumps({"t": float(t[k]), "hand": int(s), "finger": int(tip),
                                 "x": float(np.nan_to_num(P[k, s, tip, 0])),
                                 "y": float(np.nan_to_num(P[k, s, tip, 1])),
                                 "conf": float(np.nan_to_num(P[k, s, tip, 2])),
                                 "i": int(frames[k])}) + "\n")
    return out


def eval_on_test(taps: Path) -> dict:
    from phase0.analysis.eval_taps import load_jsonl, score
    kt = keydown_times(OUR_TEST)
    tt = np.array([r["t"] for r in load_jsonl(taps)], float)
    tt = tt[(tt >= kt.min() - 0.1) & (tt <= kt.max() + 0.1)]
    return score(kt, tt)


def row(name: str, s: dict) -> str:
    e = "-" if s["median_abs_err_ms"] is None else f'{s["median_abs_err_ms"]:.0f}'
    return (f"{name:<26} R={100*s['recall']:5.1f}  P={100*s['precision']:5.1f}  "
            f"F1={100*s['f1']:5.1f}  stillFP={s['still_fp']:<4} err={e:>4}ms  taps={s['taps']}")



# ---------------------------------------------------------------- public as one pseudo-session
def public_session(splits, keep: float, gap: float = 5.0):
    """Clips concatenated with a `gap`-second hole between them, so taps_gb's
    run/peak machinery treats each clip as its own episode. -> (X, t, y, yg, kt)."""
    rng = np.random.default_rng(0)
    Xs, ts, kts, ys, ygs = [], [], [], [], []
    off = 0.0
    for c in all_clips():
        if c["split"] not in splits or not land_path(c["split"], c["file_name"]).exists():
            continue
        t, P = load_public_frames(c["split"], c["file_name"])
        if len(t) < 30:
            continue
        t, P = resample_60(t, P)
        if len(t) < 30:
            continue
        X = G.assemble(G.build_groups(P), G.GROUPS).astype(np.float32)
        kt = np.sort(np.array([k["timestamp_ms"] / 1000.0 for k in c["keystrokes"]], float))
        end = max(t[-1], kt[-1] if len(kt) else 0.0)
        if keep < 1.0:
            m = rng.random(len(t)) < keep
            X, t = X[m], t[m]
        ys.append(G.labels(t, kt))
        ygs.append(G.labels(t, kt, G.TYPING_HALF_WIDTH_S))
        Xs.append(X)
        ts.append(t + off)
        kts.append(kt + off)
        off += end + gap
    return (np.vstack(Xs), np.concatenate(ts), np.concatenate(ys), np.concatenate(ygs),
            np.sort(np.concatenate(kts)))


def cmd_buildpub(a) -> int:
    tr = public_session(("train",), a.keep)
    np.savez(PUBLIC / "pub_train.npz", X=tr[0], t=tr[1], y=tr[2], yg=tr[3], kt=tr[4])
    print(f"train rows={len(tr[1])} pos={tr[2].mean():.3f} keys={len(tr[4])}", flush=True)
    # event params are picked in FRAME units, so the tuning set must stay a full 60 fps grid
    va = public_session(("validation", "test"), 1.0)
    np.savez(PUBLIC / "pub_val.npz", X=va[0], t=va[1], y=va[2], yg=va[3], kt=va[4])
    print(f"val rows={len(va[1])} pos={va[2].mean():.3f} keys={len(va[4])}", flush=True)
    return 0


def load_pub(name):
    d = np.load(PUBLIC / name)
    return d["X"], d["t"], d["y"], d["yg"], d["kt"]


# ---------------------------------------------------------------- conditions
def tune_on(p, pg, t, kt):
    """Event params chosen on `p`; never on the held-out session."""
    return G.tune_events(p, t, kt, np.ones(len(t), bool), pg)[0]


def finish(name, m, gm, cfg, results, outdir):
    f = predict_session(m, gm, cfg, OUR_TEST, outdir / f"{name}.jsonl")
    s = eval_on_test(f)
    results[name] = s
    print(row(name, s), flush=True)


def our_oof_cfg(X, y, yg, t, sid, kt, fitter, n_splits=5, inits=(None, None)):
    folds = G.cv_folds("block", t, sid, n_splits)
    p, pg = oof_probs(X, y, yg, t, sid, folds, fitter, inits)
    return tune_on(p, pg, t, kt)


def cmd_conditions(a) -> int:
    outdir = PUBLIC / "out"
    outdir.mkdir(parents=True, exist_ok=True)
    res = {}
    Xo, to, sido, kto = our_data(OUR_TRAIN)
    yo, ygo = G.labels(to, kto), G.labels(to, kto, G.TYPING_HALF_WIDTH_S)
    print(f"ours: rows={len(to)} feat={Xo.shape[1]} keys={len(kto)} pos={yo.mean():.3f}", flush=True)

    t0 = time.time()
    if "A" in a.only:
        cfg = our_oof_cfg(Xo, yo, ygo, to, sido, kto, fit)
        print(f"A cfg={cfg} [{time.time()-t0:.0f}s]", flush=True)
        finish("A_ours_only", fit(Xo, yo), fit(Xo, ygo), cfg, res, outdir)

    Xp, tp, yp, ygp, ktp = load_pub("pub_train.npz")
    print(f"public: rows={len(tp)} keys={len(ktp)} pos={yp.mean():.3f} "
          f"nan_frac={np.isnan(Xp[:, 0]).mean():.3f} vs ours {np.isnan(Xo[:, 0]).mean():.3f}", flush=True)
    mB = fit(Xp, yp, scale_pos_weight=spw(yp))
    # +/-300 ms of a keydown covers essentially every public frame; no negative class left
    gB = None if ygp.mean() > 0.99 else fit(Xp, ygp, scale_pos_weight=spw(ygp))
    print(f"public models fitted, spw={spw(yp):.2f} gate={'yes' if gB else 'DEGENERATE'} "
          f"[{time.time()-t0:.0f}s]", flush=True)

    if "B" in a.only:
        Xv, tv, _, _, ktv = load_pub("pub_val.npz")
        gv = gB.predict_proba(Xv)[:, 1] if gB is not None else None
        cfgB = tune_on(mB.predict_proba(Xv)[:, 1], gv, tv, ktv)
        print(f"B cfg (public-val) ={cfgB}", flush=True)
        finish("B_public_only", mB, gB, cfgB, res, outdir)
        del Xv
        go = gB.predict_proba(Xo)[:, 1] if gB is not None else None
        cfgBc = tune_on(mB.predict_proba(Xo)[:, 1], go, to, kto)
        print(f"B cfg (our-train recalibrated) ={cfgBc}", flush=True)
        finish("B_public_recal", mB, gB, cfgBc, res, outdir)

    if "C" in a.only:
        folds = G.cv_folds("block", to, sido, 3)
        p = np.zeros(len(to))
        pg = np.zeros(len(to))
        for te in folds:
            tr = G.purge(~te, te, to)
            Xf = np.vstack([Xp, Xo[tr]])
            p[te] = fit(Xf, np.concatenate([yp, yo[tr]]), w=pooled_w(yp, yo[tr]),
                        scale_pos_weight=1.0).predict_proba(Xo[te])[:, 1]
            pg[te] = fit(Xf, np.concatenate([ygp, ygo[tr]]), w=pooled_w(ygp, ygo[tr]),
                         scale_pos_weight=1.0).predict_proba(Xo[te])[:, 1]
            del Xf
            print(f"  C fold done [{time.time()-t0:.0f}s]", flush=True)
        cfgC = tune_on(p, pg, to, kto)
        print(f"C cfg={cfgC}", flush=True)
        Xc = np.vstack([Xp, Xo])
        finish("C_pooled",
               fit(Xc, np.concatenate([yp, yo]), w=pooled_w(yp, yo), scale_pos_weight=1.0),
               fit(Xc, np.concatenate([ygp, ygo]), w=pooled_w(ygp, ygo), scale_pos_weight=1.0),
               cfgC, res, outdir)
        del Xc

    if "D" in a.only:
        bm, bg = mB.booster_, gB.booster_ if gB is not None else None
        ft = dict(n_estimators=a.ft_trees, learning_rate=a.ft_lr)

        def fitD(X, y, init=None):
            # no pretrained booster for this head -> train it exactly as condition A does
            return fit(X, y, init=init, **ft) if init is not None else fit(X, y)

        cfgD = our_oof_cfg(Xo, yo, ygo, to, sido, kto, fitD, inits=(bm, bg))
        print(f"D cfg={cfgD}", flush=True)
        finish("D_finetune", fitD(Xo, yo, bm), fitD(Xo, ygo, bg), cfgD, res, outdir)

        def stack(X):
            cols = [X, mB.predict_proba(X)[:, 1:2].astype(np.float32)]
            if gB is not None:
                cols.append(gB.predict_proba(X)[:, 1:2].astype(np.float32))
            return np.hstack(cols)

        Xs = stack(Xo)
        cfgS = our_oof_cfg(Xs, yo, ygo, to, sido, kto, fit)
        print(f"D-stack cfg={cfgS}", flush=True)
        f = predict_session(fit(Xs, yo), fit(Xs, ygo), cfgS, OUR_TEST,
                            outdir / "D_stack.jsonl", pre=stack)
        s = eval_on_test(f)
        res["D_stack"] = s
        print(row("D_stack", s), flush=True)

    json.dump({k: v for k, v in res.items()}, open(outdir / "conditions.json", "w"), indent=1)
    return 0


# ---------------------------------------------------------------- diagnosis
DYNAMIC = ("deriv", "palm", "rest", "still", "ctx")


def group_slice(use) -> np.ndarray:
    """Column mask selecting `use` out of the full GROUPS feature vector."""
    _, _, P = G.load_frames(OUR_TEST)
    g = G.build_groups(P[:200])
    return np.concatenate([np.full(g[k].shape[1], k in use) for k in G.GROUPS])


def cmd_diag(a) -> int:
    """Is B's failure distribution shift in static pose, or the missing rest periods?"""
    outdir = PUBLIC / "out"
    Xo, to, sido, kto = our_data(OUR_TRAIN)
    yo, ygo = G.labels(to, kto), G.labels(to, kto, G.TYPING_HALF_WIDTH_S)
    Xp, tp, yp, ygp, _ = load_pub("pub_train.npz")
    Xv, tv, _, _, ktv = load_pub("pub_val.npz")
    c = group_slice(DYNAMIC)
    print(f"dynamic-only features: {c.sum()}/{len(c)}", flush=True)
    res = {}
    for name, mask in (("full", np.ones(len(c), bool)), ("dynamic_only", c)):
        mB = fit(Xp[:, mask], yp, scale_pos_weight=spw(yp))
        cfg = tune_on(mB.predict_proba(Xv[:, mask])[:, 1], None, tv, ktv)
        f = predict_session(mB, None, cfg, OUR_TEST, outdir / f"diag_B_{name}.jsonl",
                            pre=lambda X, _m=mask: X[:, _m])
        res[f"B_{name}"] = eval_on_test(f)
        print(row(f"B_{name}", res[f"B_{name}"]), flush=True)
        if name == "full":
            og = fit(Xo, ygo)
            cfgm = tune_on(mB.predict_proba(Xo)[:, 1], og.predict_proba(Xo)[:, 1], to, kto)
            f = predict_session(mB, og, cfgm, OUR_TEST, outdir / "diag_B_ourgate.jsonl")
            res["B_ourgate"] = eval_on_test(f)
            print(row("B_ourgate", res["B_ourgate"]), flush=True)
        cfgo = our_oof_cfg(Xo[:, mask], yo, ygo, to, sido, kto, fit)
        f = predict_session(fit(Xo[:, mask], yo), fit(Xo[:, mask], ygo), cfgo, OUR_TEST,
                            outdir / f"diag_A_{name}.jsonl", pre=lambda X, _m=mask: X[:, _m])
        res[f"A_{name}"] = eval_on_test(f)
        print(row(f"A_{name}", res[f"A_{name}"]), flush=True)
    json.dump(res, open(outdir / "diag.json", "w"), indent=1)
    return 0


# ---------------------------------------------------------------- scaling
def time_subset(t, sid, frac, seed):
    """A contiguous window of each session -- the user simply recorded for less time.
    Scattered blocks would make taps_gb.purge embargo across the holes and empty a fold."""
    rng = np.random.default_rng(seed)
    m = np.zeros(len(t), bool)
    for s in np.unique(sid):
        i = np.where(sid == s)[0]
        n = max(2, int(round(frac * len(i))))
        off = 0 if n >= len(i) else int(rng.integers(0, len(i) - n + 1))
        m[i[off:off + n]] = True
    return m


def cmd_scaling(a) -> int:
    outdir = PUBLIC / "out"
    outdir.mkdir(parents=True, exist_ok=True)
    Xo, to, sido, kto = our_data(OUR_TRAIN)
    Xp, tp, yp, ygp, _ = load_pub("pub_train.npz")
    mB = fit(Xp, yp, scale_pos_weight=spw(yp))
    gB = None if ygp.mean() > 0.99 else fit(Xp, ygp, scale_pos_weight=spw(ygp))
    del Xp
    ft = dict(n_estimators=a.ft_trees, learning_rate=a.ft_lr)
    rows = []
    for frac in (0.25, 0.5, 1.0):
        for seed in range(1 if frac == 1.0 else a.seeds):
            m = time_subset(to, sido, frac, seed)
            X, t, sid = Xo[m], to[m], sido[m]
            kt = kto[np.abs(kto[:, None] - t[None, :]).min(1) <= 0.05]
            y, yg = G.labels(t, kt), G.labels(t, kt, G.TYPING_HALF_WIDTH_S)
            for cond in ("A", "D"):
                inits = ((None, None) if cond == "A"
                         else (mB.booster_, gB.booster_ if gB is not None else None))
                kw = {} if cond == "A" else ft

                def f(Xx, yy, init=None, _kw=kw):
                    return fit(Xx, yy, init=init, **_kw) if init is not None else fit(Xx, yy)

                cfg = our_oof_cfg(X, y, yg, t, sid, kt, f, 5, inits)
                nm = f"{cond}_frac{frac}_s{seed}"
                p = predict_session(f(X, y, inits[0]), f(X, yg, inits[1]), cfg, OUR_TEST,
                                    outdir / f"{nm}.jsonl")
                s = eval_on_test(p)
                rows.append({"cond": cond, "frac": frac, "seed": seed, "keys": int(len(kt)),
                             **{k: s[k] for k in ("recall", "precision", "f1", "still_fp")}})
                print(row(nm, s), flush=True)
    json.dump(rows, open(outdir / "scaling.json", "w"), indent=1)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.pretrain", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--workers", type=int, default=6)
    sub.add_parser("track")
    bp = sub.add_parser("buildpub")
    bp.add_argument("--keep", type=float, default=0.35)
    sub.add_parser("dist")
    co = sub.add_parser("conditions")
    co.add_argument("--only", default="ABCD")
    sub.add_parser("diag")
    sc = sub.add_parser("scaling")
    sc.add_argument("--seeds", type=int, default=3)
    for x in (co, sc):
        x.add_argument("--ft-trees", type=int, default=300)
        x.add_argument("--ft-lr", type=float, default=0.03)
    a = ap.parse_args(argv)
    return {"extract": cmd_extract, "track": cmd_track, "buildpub": cmd_buildpub,
            "dist": cmd_dist, "conditions": cmd_conditions, "scaling": cmd_scaling,
            "diag": cmd_diag}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
