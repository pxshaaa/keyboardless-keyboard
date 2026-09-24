"""Shared harness for the touch-detection experiments: kbd protocol and desk metrics.
Run: python -m phase0.analysis.touch_common {feats}"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import lightgbm  # noqa: F401  must load before torch

from phase0.analysis import taps_gb as gb
from phase0.analysis.eval_taps import keydown_times, score

ROOT = Path("data/sessions")
CACHE = Path(".cache/contact_ego")
KBD_TRAIN4 = ("20260910-015217-kbd", "20260910-021315-kbd", "20260910-131629-kbd",
              "20260911-164237-kbd")
KBD_TEST = "20260910-015948-kbd"
DESK = "20260910-202149-desk"
ALL = KBD_TRAIN4 + (KBD_TEST, DESK)
LGB_KW = {"n_jobs": 4}


def spath(sid: str) -> Path:
    return ROOT / sid


# ------------------------------------------------------------------ feature cache
_F: dict = {}


def frames(sid: str):
    """-> (frame_i, t, P[F,2,21,7])"""
    k = ("P", sid)
    if k not in _F:
        f = CACHE / f"P_{sid}.npz"
        if f.exists():
            z = np.load(f)
            _F[k] = (z["frames"], z["t"], z["P"])
        else:
            fr, t, P = gb.load_frames(spath(sid))
            CACHE.mkdir(parents=True, exist_ok=True)
            np.savez(f, frames=fr, t=t, P=P.astype(np.float32))
            _F[k] = (fr, t, P.astype(np.float32))
    return _F[k]


def base_X(sid: str) -> np.ndarray:
    """taps_gb's full GROUPS feature matrix, float32, cached."""
    k = ("X", sid)
    if k not in _F:
        f = CACHE / f"X_{sid}.npy"
        if f.exists():
            _F[k] = np.load(f, mmap_mode="r")
        else:
            _, _, P = frames(sid)
            X = gb.assemble(gb.build_groups(P.astype(np.float64)), gb.GROUPS).astype(np.float32)
            np.save(f, X)
            _F[k] = np.load(f, mmap_mode="r")
    return _F[k]


def kt(sid: str) -> np.ndarray:
    return keydown_times(spath(sid)) if sid != DESK else np.array([])


# ------------------------------------------------------------------ keyboard protocol
def fit_lgb(X, y, seed, **kw):
    m = gb.make_model("lgbm", random_state=seed, **dict(LGB_KW, **kw))
    return m.fit(X, y)


def kbd_protocol(feat_fn, seed: int = 0, label_half: float = gb.LABEL_HALF_WIDTH_S,
                 extra_train=None, model_kw: dict | None = None, oof: bool = True,
                 verbose: bool = True, cfg: dict | None = None) -> dict:
    """feat_fn(sid) -> X[F,d]. Returns held-out score, tuned cfg and fitted (model, gate)."""
    mk = model_kw or {}
    Xs = {s: np.asarray(feat_fn(s), np.float32) for s in KBD_TRAIN4}
    ts = {s: frames(s)[1] for s in KBD_TRAIN4}
    ys = {s: gb.labels(ts[s], kt(s), label_half) for s in KBD_TRAIN4}
    ygs = {s: gb.labels(ts[s], kt(s), gb.TYPING_HALF_WIDTH_S) for s in KBD_TRAIN4}
    t0 = time.time()
    if oof:
        P_oof, G_oof = {}, {}
        for te in KBD_TRAIN4:
            tr = [s for s in KBD_TRAIN4 if s != te]
            Xtr = np.vstack([Xs[s] for s in tr])
            m = fit_lgb(Xtr, np.concatenate([ys[s] for s in tr]), seed, **mk)
            g = fit_lgb(Xtr, np.concatenate([ygs[s] for s in tr]), seed, **mk)
            P_oof[te] = m.predict_proba(Xs[te])[:, 1]
            G_oof[te] = g.predict_proba(Xs[te])[:, 1]
        # concatenate sessions with a time offset so runs/pairing never cross a boundary
        tt, kk, pp, gg, off = [], [], [], [], 0.0
        for s in KBD_TRAIN4:
            t = ts[s] - ts[s][0] + off
            tt.append(t)
            kk.append(kt(s) - ts[s][0] + off)
            pp.append(P_oof[s])
            gg.append(G_oof[s])
            off = t[-1] + 100.0
        T, K, Pp, Gg = map(np.concatenate, (tt, kk, pp, gg))
        cfg, oof_f1 = gb.tune_events(Pp, T, K, np.ones(len(T), bool), Gg)
    else:
        cfg = cfg or {"thr": 0.5, "smooth": 1, "refractory": 4, "n_consec": 3, "gate_thr": 0.25}
        oof_f1 = float("nan")
    Xtr = np.vstack([Xs[s] for s in KBD_TRAIN4])
    y = np.concatenate([ys[s] for s in KBD_TRAIN4])
    yg = np.concatenate([ygs[s] for s in KBD_TRAIN4])
    if extra_train is not None:
        Xtr = np.vstack([Xtr, extra_train[0]])
        y = np.concatenate([y, extra_train[1]])
        yg = np.concatenate([yg, extra_train[2]])
    m = fit_lgb(Xtr, y, seed, **mk)
    g = fit_lgb(Xtr, yg, seed, **mk)
    Xte = np.asarray(feat_fn(KBD_TEST), np.float32)
    pt, gt = m.predict_proba(Xte)[:, 1], g.predict_proba(Xte)[:, 1]
    res = score_events(KBD_TEST, pt, gt, cfg)
    res["test_p"], res["test_gate"] = pt, gt
    res.update({"cfg": cfg, "oof_f1": oof_f1, "seed": seed, "secs": round(time.time() - t0, 1)})
    if verbose:
        print(f"  seed{seed} kbd held-out F1={100*res['f1']:.1f} R={100*res['recall']:.1f} "
              f"P={100*res['precision']:.1f} oofF1={100*oof_f1:.1f} ({res['secs']}s)", flush=True)
    res["model"], res["gate"] = m, g
    return res


def kbd_score(m, g, cfg, feat_fn, sid: str = KBD_TEST) -> dict:
    X = np.asarray(feat_fn(sid), np.float32)
    p, gg = m.predict_proba(X)[:, 1], g.predict_proba(X)[:, 1]
    return score_events(sid, p, gg, cfg)


def score_events(sid, p, g, cfg) -> dict:
    t = frames(sid)[1]
    k = kt(sid)
    ev = gb.apply_events(p, cfg, g)
    tt = t[ev]
    tt = tt[(tt >= k.min() - 0.1) & (tt <= k.max() + 0.1)]
    r = score(k, tt)
    return {x: r[x] for x in ("f1", "recall", "precision", "taps", "hits", "still_fp")}


def desk_probs(m, g, feat_fn) -> dict:
    X = np.asarray(feat_fn(DESK), np.float32)
    return {"p": m.predict_proba(X)[:, 1], "gate": g.predict_proba(X)[:, 1]}


# ------------------------------------------------------------------ desk ground truth
_W: list = []


def windows() -> list[dict]:
    """20 phrase windows: t0, t1, text, n (characters incl. spaces, as decode/alignment use)."""
    if not _W:
        from phase0.analysis.analyze_drift import read_jsonl
        rows = read_jsonl(spath(DESK) / "phrases.jsonl")
        shown = {r["idx"]: r for r in rows if r["event"] == "shown"}
        done = {r["idx"]: r for r in rows if r["event"] == "done"}
        for i in sorted(shown):
            if i in done:
                _W.append({"t0": shown[i]["t"], "t1": done[i]["t"], "text": shown[i]["phrase"],
                           "n": len(shown[i]["phrase"])})
    return _W


def per_phrase_counts(ev: np.ndarray) -> np.ndarray:
    t = frames(DESK)[1][ev]
    return np.array([int(((t >= w["t0"]) & (t < w["t1"])).sum()) for w in windows()])


# ------------------------------------------------------------------ extraction
def extract(p, gate, thr, rf=4, gate_thr=0.25, smooth=1, nms="peak", nc=1):
    from phase0.analysis import hirecall as hr
    ps = gb.smooth(p, smooth)
    pg = ps if gate is None or gate_thr <= 0 else np.where(gate >= gate_thr, ps, 0.0)
    return hr.pick(pg, thr, rf, nc, nms)


def _thr_for_density(p, gate, train_idx, density, **kw):
    """Bisection on threshold so taps inside the training phrases = density * their chars."""
    W = windows()
    need = density * sum(W[j]["n"] for j in train_idx)
    lo, hi = 1e-4, 0.999
    for _ in range(30):
        mid = (lo + hi) / 2
        c = per_phrase_counts(extract(p, gate, mid, **kw))[train_idx].sum()
        if c > need:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def count_error(p, gate, density: float = 1.0, **kw) -> dict:
    """(a) Leave-one-phrase-out: threshold set on 19 phrases to hit `density` taps/char, then
    the held-out phrase's |taps - density*chars| / (density*chars)."""
    W = windows()
    n = np.array([w["n"] for w in W])
    errs, counts = [], []
    # counts as a function of threshold are cached over a fine grid for speed
    grid = np.unique(np.concatenate([np.geomspace(1e-3, 0.99, 400)]))
    C = np.stack([per_phrase_counts(extract(p, gate, th, **kw)) for th in grid])  # [G,20]
    for j in range(len(W)):
        tr = np.array([i for i in range(len(W)) if i != j])
        tot = C[:, tr].sum(1)
        g = int(np.argmin(np.abs(tot - density * n[tr].sum())))
        counts.append(int(C[g, j]))
        errs.append(abs(C[g, j] - density * n[j]) / (density * n[j]))
    counts = np.array(counts)
    return {"count_err": float(np.mean(errs)), "count_err_med": float(np.median(errs)),
            "count_r": float(np.corrcoef(counts, n)[0, 1]), "per": [float(e) for e in errs]}


def boot_mean(v, n=10000, seed=0):
    v = np.asarray(v, float)
    idx = np.random.default_rng(seed).integers(0, len(v), (n, len(v)))
    m = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_paired(a, b, n=10000, seed=0):
    """mean(b - a) with a paired phrase bootstrap CI."""
    d = np.asarray(b, float) - np.asarray(a, float)
    return boot_mean(d, n, seed)



# ------------------------------------------------------------------ desk (b) oracle and (c) end-to-end
ORACLE_THR = (0.02, 0.035, 0.05, 0.07, 0.09, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6)
ORACLE_RF = (3, 4, 6)
ORACLE_DEL = (-9.0, -3.0)
DENS_BAND = (0.55, 3.0)


def _shim():
    from phase0.analysis import combine as CB
    CB._shim()


def candidate_streams(p, gate, thr=ORACLE_THR, rfs=ORACLE_RF, nms=("peak", "shoulder"),
                      gate_thrs=(0.0, 0.25), band=DENS_BAND) -> list[tuple[dict, np.ndarray]]:
    nch = sum(w["n"] for w in windows())
    seen, out = set(), []
    for gt in gate_thrs:
        for rf in rfs:
            for nm in nms:
                for th in thr:
                    ev = extract(p, gate, th, rf=rf, gate_thr=gt, nms=nm)
                    if not band[0] <= len(ev) / nch <= band[1] or ev.tobytes() in seen:
                        continue
                    seen.add(ev.tobytes())
                    out.append(({"thr": th, "rf": rf, "gate_thr": gt, "nms": nm}, ev))
    return out


def thin(streams, n):
    if len(streams) <= n:
        return streams
    order = sorted(range(len(streams)), key=lambda i: (len(streams[i][1]), i))
    keep = sorted({order[i] for i in np.linspace(0, len(order) - 1, n).round().astype(int)})
    return [streams[i] for i in keep]


def _oracle_one(args):
    _shim()
    from phase0.analysis import adapt as ad
    from phase0.analysis import pipeline as pl
    from phase0.analysis import combine as CB
    ev, dels = args
    ev = np.asarray(ev, int)
    kf = pl.kbd_fit()
    s = pl.session_path(DESK)
    c = CB.Cfg(em=False)
    _, segs, base, _, _ = CB.stack_proba(s, ev, kf, c, "none")  # alpha 0 => pose only
    live = [x for x in segs if len(x[1])]
    n = sum(len(i) for _, i in live)
    ch = sum(len(t) for t, _ in live)
    par = ad.AlignParams.from_counts(n, ch, float(np.clip(1.0 - ch / max(n, 1), 0.08, 0.5)))
    q, _, _, _ = ad.align_all(np.log(np.maximum(base, 1e-12)), live, par)
    out = []
    for dl in dels:
        r = CB.decode_rows(q, segs, kf, CB.Cfg(obs=1.0, deletion=dl, em=False))
        out.append({"deletion": dl, "per": [(x[2], x[3]) for x in r]})
    return out


def _select(rows, train):
    best, bc = None, np.inf
    for r in rows:
        e = sum(r["per"][j][0] for j in train)
        n = sum(r["per"][j][1] for j in train)
        if e / max(n, 1) < bc:
            best, bc = r, e / max(n, 1)
    return best


def lopo(rows) -> list[tuple[int, int]]:
    """Leave-one-phrase-out pick over a fold-independent table -> [(edits, chars)] per phrase."""
    n = len(rows[0]["per"])
    out = []
    for j in range(n):
        b = _select(rows, set(range(n)) - {j})
        out.append(tuple(b["per"][j]))
    return out


def cer_ci(per, n=10000, seed=0):
    e = np.array([x[0] for x in per], float)
    r = np.array([x[1] for x in per], float)
    idx = np.random.default_rng(seed).integers(0, len(e), (n, len(e)))
    v = e[idx].sum(1) / r[idx].sum(1)
    return float(e.sum() / r.sum()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def cer_delta(a, b, n=10000, seed=0):
    """paired phrase bootstrap of CER(b) - CER(a)."""
    ea = np.array([x[0] for x in a], float)
    eb = np.array([x[0] for x in b], float)
    r = np.array([x[1] for x in a], float)
    idx = np.random.default_rng(seed).integers(0, len(ea), (n, len(ea)))
    v = (eb[idx].sum(1) - ea[idx].sum(1)) / r[idx].sum(1)
    return float((eb.sum() - ea.sum()) / r.sum()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def oracle_cer(p, gate, n_streams: int = 24, procs: int = 4, dels=ORACLE_DEL) -> dict:
    """(b) tap-stream quality with key identity handed over by the true text's alignment."""
    import multiprocessing as mp
    st = thin(candidate_streams(p, gate), n_streams)
    jobs = [(ev.tolist(), dels) for _, ev in st]
    with mp.get_context("spawn").Pool(procs) as pool:
        res = pool.map(_oracle_one, jobs, chunksize=1)
    nch = sum(w["n"] for w in windows())
    rows = [dict(r, cfg=c, n_taps=int(len(ev))) for (c, ev), rr in zip(st, res) for r in rr]
    per = lopo(rows)
    c, lo, hi = cer_ci(per)
    n = len(per)
    dens = [ _select(rows, set(range(n)) - {j})["n_taps"] / nch for j in range(n)]
    return {"oracle_cer": c, "ci": [lo, hi], "per": per, "density_med": float(np.median(dens)),
            "rows": rows, "n_streams": len(st)}


# end-to-end: hirecall's decoder stack on a reduced grid around hirecall's fold medians
E2E_ALPHA = (0.5, 0.75, 1.0)
E2E_OBS = (2.0, 3.0, 4.0)
E2E_DEL = (-9.0, -5.0, -3.0)
_PIXB: dict = {}


def _pix_bank():
    if not _PIXB:
        bf = np.array(json.loads(Path(".cache/hirecall/20260910-202149-desk_frames.json")
                                 .read_text())["frames"], int)
        _PIXB["f"] = bf
        _PIXB["p"] = np.load(".cache/hirecall/pix_20260910-202149-desk_none_3x40.npy")
    return _PIXB["f"], _PIXB["p"]


def pix_snap(ev_frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """hirecall's pixel-CNN key probs at the nearest frame the crop bank holds (the bank covers
    every frame any of hirecall's 718 streams used). -> (probs, |snap distance| in frames)."""
    bf, P = _pix_bank()
    j = np.clip(np.searchsorted(bf, ev_frames), 1, len(bf) - 1)
    left, right = bf[j - 1], bf[j]
    jj = np.where(np.abs(ev_frames - left) <= np.abs(right - ev_frames), j - 1, j)
    return P[jj], np.abs(bf[jj] - ev_frames)


def _stack(ev, kf, alpha):
    from phase0.analysis import pipeline as pl
    from phase0.analysis import tap_pos as tp
    from phase0.analysis import combine as CB
    s = pl.session_path(DESK)
    d = pl.frame_probs(s)
    taps = pl.taps_from(d, ev)
    sess = tp.load_sess(str(s))
    k = sess.rows(taps)
    pose = pl.spatial_proba(sess, k, kf, replace(pl.FULL, coral=True, contact_w=0.0))
    pix, _ = pix_snap(d["frames"][ev])
    return pl.segments(s, taps), CB.fuse(pix, pose, alpha), sess, k


def _e2e_one(args):
    _shim()
    from phase0.analysis import pipeline as pl
    from phase0.analysis import combine as CB
    sid, ev = args
    ev = np.asarray(ev, int)
    kf = pl.kbd_fit()
    rows = []
    for a in E2E_ALPHA:
        segs, p, _, _ = _stack(ev, kf, a)
        for w in E2E_OBS:
            for dl in E2E_DEL:
                r = CB.decode_rows(p, segs, kf, CB.Cfg(alpha=a, obs=w, deletion=dl, em=False))
                rows.append({"sid": sid, "alpha": a, "obs": w, "deletion": dl, "n_taps": int(len(ev)),
                             "per": [(x[2], x[3]) for x in r]})
    return rows


def _e2e_fold(args):
    _shim()
    from phase0.analysis import pipeline as pl
    from phase0.analysis import combine as CB
    from phase0.analysis import adapt as ad
    from phase0.analysis import hirecall as hr
    j, ev, pick = args
    ev = np.asarray(ev, int)
    kf = pl.kbd_fit()
    segs, p, sess, k = _stack(ev, kf, pick["alpha"])
    tr = [x for i, x in enumerate(segs) if i != j and len(x[1])]
    st = replace(pl.FULL, deletion=pick["deletion"])
    p = pl.weakly_supervise(sess, k, p, tr, st)
    c = CB.Cfg(alpha=pick["alpha"], obs=pick["obs"], deletion=pick["deletion"], em=True)
    return j, CB.decode_rows(p, segs, kf, c, keep={j})[0]


def e2e_table(streams, tag: str, procs: int = 4) -> list[dict]:
    import multiprocessing as mp
    f = CACHE / f"e2e_table_{tag}.json"
    if f.exists():
        return json.loads(f.read_text())
    jobs = [(i, ev.tolist()) for i, (_, ev) in enumerate(streams)]
    out, t0 = [], time.time()
    with mp.get_context("spawn").Pool(procs) as pool:
        for n, r in enumerate(pool.imap_unordered(_e2e_one, jobs)):
            out += r
            if (n + 1) % 8 == 0:
                print(f"    [{tag}] {n+1}/{len(jobs)} streams ({time.time()-t0:.0f}s)", flush=True)
    f.write_text(json.dumps(out))
    json.dump([{"sid": i, "cfg": c, "ev": ev.tolist()} for i, (c, ev) in enumerate(streams)],
              open(CACHE / f"e2e_streams_{tag}.json", "w"))
    return out


def e2e_cv(streams, rows, procs: int = 4, em: bool = True) -> list[tuple]:
    """Nested leave-one-phrase-out: stream, alpha, obs, deletion picked on 19 phrases; EM on
    those 19 phrases' text only. Returns [(ref, hyp, edits, chars)]."""
    import multiprocessing as mp
    n = len(rows[0]["per"])
    picks = [_select(rows, set(range(n)) - {j}) for j in range(n)]
    if not em:
        return [("", "", *picks[j]["per"][j]) for j in range(n)]
    jobs = [(j, streams[picks[j]["sid"]][1].tolist(), picks[j]) for j in range(n)]
    out = {}
    with mp.get_context("spawn").Pool(procs) as pool:
        for j, r in pool.imap_unordered(_e2e_fold, jobs):
            out[j] = r
    return [out[j] for j in range(n)]


def e2e(p, gate, tag: str, n_streams: int = 30, procs: int = 4) -> dict:
    st = thin(candidate_streams(p, gate, thr=(0.02, 0.035, 0.05, 0.07, 0.09, 0.12, 0.15, 0.2,
                                               0.25, 0.3, 0.4, 0.5), band=(0.8, 3.0)), n_streams)
    sp = CACHE / f"e2e_streams_{tag}.json"
    if sp.exists():  # a cached table must be read against the streams it was built on
        st = [(r["cfg"], np.array(r["ev"], int)) for r in json.loads(sp.read_text())]
    rows = e2e_table(st, tag, procs)
    r = e2e_cv(st, rows, procs)
    per = [(x[2], x[3]) for x in r]
    c, lo, hi = cer_ci(per)
    from phase0.analysis import pipeline as pl
    fr = pl.frame_probs(pl.session_path(DESK))["frames"]
    snaps = np.concatenate([pix_snap(fr[ev])[1] for _, ev in st])
    return {"cer": c, "ci": [lo, hi], "per": per, "hyp": [x[1] for x in r],
            "snap_exact": float((snaps == 0).mean()), "snap_le1": float((snaps <= 1).mean()),
            "n_streams": len(st)}


DENSITIES = (1.0, 1.25, 1.5, 1.75, 2.0)


def oracle_density(p, gate, procs: int = 4, deletion: float = -3.0, densities=DENSITIES) -> dict:
    """Oracle CER at matched tap density: no CER-based stream choice, so far less selection noise."""
    import multiprocessing as mp
    grid = np.geomspace(0.01, 0.9, 28)
    st = [(th, extract(p, gate, th)) for th in grid]
    with mp.get_context("spawn").Pool(procs) as pool:
        res = pool.map(_oracle_one, [(ev.tolist(), (deletion,)) for _, ev in st], chunksize=1)
    per_thr = [r[0]["per"] for r in res]
    C = np.stack([per_phrase_counts(ev) for _, ev in st])
    W = windows()
    n = np.array([w["n"] for w in W])
    out = {}
    for d in densities:
        per = []
        for j in range(len(W)):
            tr = np.array([i for i in range(len(W)) if i != j])
            k = int(np.argmin(np.abs(C[:, tr].sum(1) - d * n[tr].sum())))
            per.append(tuple(per_thr[k][j]))
        c, lo, hi = cer_ci(per)
        out[str(d)] = {"cer": c, "ci": [lo, hi], "per": per}
    allper = [(sum(out[str(d)]["per"][j][0] for d in densities), sum(out[str(d)]["per"][j][1] for d in densities))
              for j in range(len(W))]
    out["mean"] = {"cer": cer_ci(allper)[0], "per": allper}
    return out


def main(argv=None) -> int:
    a = argv or sys.argv[1:]
    if a and a[0] == "feats":
        for s in ALL:
            t0 = time.time()
            X = base_X(s)
            print(s, X.shape, f"{time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
