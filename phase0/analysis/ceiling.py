"""Calibration learning curve, saturating fit, rollover ceiling and refractory sweep.
Run: python -m phase0.analysis.ceiling {curve | rollover | refractory | report | all}"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import curve_fit

from phase0.analysis.eval_taps import WINDOW_S, keydown_times, score
from phase0.analysis.taps_gb import (
    GROUPS,
    apply_events,
    assemble,
    block_groups,
    build_groups,
    labels,
    load_frames,
    make_model,
    purge,
    tune_events,
)

FPS = 60.0
SESSIONS = [
    Path("data/sessions/20260910-015217-kbd"),
    Path("data/sessions/20260910-015948-kbd"),
    Path("data/sessions/20260910-021315-kbd"),
]
OUT = Path("data/sessions")
SIZES = (50, 100, 175, 300, 500, 750, 1050)
N_SEEDS = 5
INNER_FOLDS = 3
WIN_PAD_S = 0.5
MODEL_KW = {"n_jobs": 2}


# ---------------------------------------------------------------- session cache
@lru_cache(maxsize=8)
def sess(name: str) -> dict:
    p = Path(name)
    _, t, P = load_frames(p)
    return {"t": t, "P": P, "kt": keydown_times(p), "name": p.name}


@lru_cache(maxsize=8)
def full_X(name: str) -> np.ndarray:
    return assemble(build_groups(sess(name)["P"]), GROUPS)


# ---------------------------------------------------------------- window sampling
def sample_windows(train: list[str], n_keys: int, seed: int) -> list[tuple[str, int, int]]:
    """Contiguous keydown runs (a real calibration is continuous typing), sessions in random order."""
    rng = np.random.default_rng(seed)
    order = list(train)
    rng.shuffle(order)
    out, need = [], n_keys
    for name in order:
        if need <= 0:
            break
        kt = sess(name)["kt"]
        m = min(need, len(kt))
        start = int(rng.integers(0, len(kt) - m + 1))
        out.append((name, start, m))
        need -= m
    return out


def build_train(wins: list[tuple[str, int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """-> (X, t, kt, sid) with features rebuilt from the window's own frames only (no context leak)."""
    Xs, ts, kts, sids = [], [], [], []
    for k, (name, start, m) in enumerate(wins):
        s = sess(name)
        kt = s["kt"][start:start + m]
        sel = (s["t"] >= kt[0] - WIN_PAD_S) & (s["t"] <= kt[-1] + WIN_PAD_S)
        Xs.append(assemble(build_groups(s["P"][sel]), GROUPS))
        ts.append(s["t"][sel])
        kts.append(kt)
        sids.append(np.full(int(sel.sum()), k))
    return np.vstack(Xs), np.concatenate(ts), np.sort(np.concatenate(kts)), np.concatenate(sids)


# ---------------------------------------------------------------- one calibration run
def _inner_folds(t, sid) -> list[np.ndarray]:
    """Contiguous thirds within each sub-window; interleaved blocks + 0.5 s embargo can empty a fold."""
    part = np.zeros(len(t), int)
    for s in np.unique(sid):
        m = sid == s
        r = (t[m] - t[m].min()) / max(t[m].max() - t[m].min(), 1e-9)
        part[m] = np.clip((r * INNER_FOLDS).astype(int), 0, INNER_FOLDS - 1)
    return [part == i for i in range(INNER_FOLDS)]


def _inner_oof(X, y, yg, t, sid, seed, mkw):
    folds = [f for f in _inner_folds(t, sid) if f.any()]
    oof, oofg = np.zeros(len(t)), np.zeros(len(t))
    for te in folds:
        tr = purge(~te, te, t)
        if not tr.any() or not y[tr].any():
            continue
        kw = dict(mkw, random_state=seed)
        oof[te] = make_model("lgbm", **kw).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        oofg[te] = make_model("lgbm", **kw).fit(X[tr], yg[tr]).predict_proba(X[te])[:, 1]
    return oof, oofg


def run_one(test: str, train: list[str], n_keys: int, seed: int, sweep: bool = False,
            model_kw: dict | None = None) -> dict:
    """Train on a contiguous calibration sample, tune event params on its own inner OOF, score on test."""
    t0 = time.time()
    wins = sample_windows(train, n_keys, seed)
    X, t, kt, sid = build_train(wins)
    y, yg = labels(t, kt), labels(t, kt, 0.300)
    mkw = dict(MODEL_KW, **(model_kw or {}))
    oof, oofg = _inner_oof(X, y, yg, t, sid, seed, mkw)
    cfg, _ = tune_events(oof, t, kt, np.ones(len(t), bool), oofg)
    kw = dict(mkw, random_state=seed)
    m = make_model("lgbm", **kw).fit(X, y)
    gm = make_model("lgbm", **kw).fit(X, yg)
    s = sess(test)
    Xt = full_X(test)
    p, g = m.predict_proba(Xt)[:, 1], gm.predict_proba(Xt)[:, 1]
    ocfg, _ = tune_events(p, s["t"], s["kt"], np.ones(len(s["t"]), bool), g)
    res = {"test": s["name"], "n_keys": int(len(kt)), "n_req": n_keys, "seed": seed,
           "frames": int(len(t)), "minutes": float(sum(
               sess(w[0])["kt"][w[1] + w[2] - 1] - sess(w[0])["kt"][w[1]] for w in wins) / 60.0),
           "cfg": cfg, "secs": round(time.time() - t0, 1),
           **_score_test(s, p, g, cfg),
           "f1_oracle_cfg": _score_test(s, p, g, ocfg)["f1"], "oracle_cfg": ocfg}
    if sweep:
        res["sweep"] = _refractory_sweep(s, p, g, oof, oofg, t, kt)
    return res


def _score_test(s: dict, p: np.ndarray, g: np.ndarray, cfg: dict) -> dict:
    ev = apply_events(p, cfg, g)
    tt = s["t"][ev]
    kt = s["kt"]
    tt = tt[(tt >= kt.min() - 0.1) & (tt <= kt.max() + 0.1)]  # --clip
    r = score(kt, tt)
    return {"f1": r["f1"], "recall": r["recall"], "precision": r["precision"],
            "taps": r["taps"], "hits": r["hits"], "still_fp": r["still_fp"]}


def _refractory_sweep(s, p, g, oof, oofg, t, kt) -> list[dict]:
    """Re-tune only thr/smooth/gate for each fixed (refractory, n_consec); no refits."""
    from phase0.analysis.taps_gb import EVENT_GRID, _sweep_thr, prep_mask, smooth

    prep = prep_mask(kt, t, np.ones(len(t), bool))
    rows = []
    for rf in (5, 4, 3, 2, 1):
        for nc in (1, 2, 3, 4, 5):
            best, bf1 = None, -1.0
            for w in EVENT_GRID["smooth"]:
                ps = smooth(oof, w)
                for gt in EVENT_GRID["gate"]:
                    pg = ps if gt <= 0 else np.where(oofg >= gt, ps, 0.0)
                    thr, f1 = _sweep_thr(pg, t, prep, rf, nc, 0.6)
                    if f1 > bf1:
                        bf1, best = f1, {"thr": thr, "smooth": w, "refractory": rf,
                                         "n_consec": nc, "gate_thr": gt}
            rows.append({"refractory_frames": rf, "refractory_ms": round(1000 * rf / FPS, 1),
                         "n_consec": nc, "cfg": best, "train_f1": bf1, **_score_test(s, p, g, best)})
    return rows


# ---------------------------------------------------------------- learning curve
def cmd_curve(a) -> int:
    names = [str(s) for s in SESSIONS]
    jobs = []
    for test in names:
        train = [n for n in names if n != test]
        pool = sum(len(sess(n)["kt"]) for n in train)
        sizes = sorted({min(n, pool) for n in SIZES if n <= pool} | {pool})
        for n in sizes:
            for seed in range(N_SEEDS):
                jobs.append((test, train, n, seed))
    print(f"{len(jobs)} runs", flush=True)
    res = Parallel(n_jobs=a.jobs, verbose=10)(delayed(run_one)(*j) for j in jobs)
    (OUT / "ceiling_curve.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT/'ceiling_curve.json'}")
    return 0


def cmd_smalln(a) -> int:
    """Is the small-calibration collapse a data limit or a hyperparameter artifact?"""
    names = [str(s) for s in SESSIONS]
    rows = []
    for tag, kw in (("default", {}),
                    ("small", {"min_child_samples": 5, "n_estimators": 300, "num_leaves": 15}),
                    ("small+depth", {"min_child_samples": 5, "n_estimators": 300, "num_leaves": 15,
                                     "max_depth": 4, "colsample_bytree": 0.3})):
        jobs = [(t, [n for n in names if n != t], n, seed, False, kw)
                for t in names[:2] for n in (50, 100, 175) for seed in range(3)]
        res = Parallel(n_jobs=a.jobs)(delayed(run_one)(*j) for j in jobs)
        for r in res:
            rows.append({"tag": tag, **{k: v for k, v in r.items() if k != "oracle_cfg"}})
        for n in (50, 100, 175):
            f = [r["f1"] for r in res if r["n_req"] == n]
            print(f"{tag:<12} n={n:<4} F1 mean={100*np.mean(f):5.1f}% med={100*np.median(f):5.1f}% "
                  f"min={100*min(f):5.1f}% max={100*max(f):5.1f}%", flush=True)
    (OUT / "ceiling_smalln.json").write_text(json.dumps(rows, indent=1))
    return 0


def cmd_refractory(a) -> int:
    names = [str(s) for s in SESSIONS]
    jobs = [(t, [n for n in names if n != t],
             sum(len(sess(n)["kt"]) for n in names if n != t), seed, True)
            for t in names for seed in range(3)]
    res = Parallel(n_jobs=a.jobs)(delayed(run_one)(*j) for j in jobs)
    (OUT / "ceiling_refractory.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT/'ceiling_refractory.json'}")
    return 0


# ---------------------------------------------------------------- rollover ceiling
def max_matched(kt: np.ndarray, refr_s: float, window: float = WINDOW_S,
                grid: np.ndarray | None = None) -> int:
    """Most keydowns any extractor with this refractory can hit; earliest-feasible is optimal
    because pair_events' min-|dt| 1:1 matching on a line is non-crossing (monotone)."""
    last, hit = -np.inf, 0
    for k in kt:
        e = max(k - window, last + refr_s)
        if grid is not None:
            j = int(np.searchsorted(grid, e - 1e-9))
            if j >= len(grid):
                continue
            e = grid[j]
        if e <= k + window:
            hit += 1
            last = e
    return hit


def _ceil_block(kt, grid, windows, refrs) -> dict:
    out = {}
    for w in windows:
        out[f"pair_window_{int(1000*w)}ms"] = {}
        for ms in refrs:
            mr = max_matched(kt, ms / 1000.0, w, grid) / len(kt)
            out[f"pair_window_{int(1000*w)}ms"][ms] = {
                "max_recall": round(mr, 4), "max_f1_at_perfect_precision": round(2 * mr / (1 + mr), 4)}
    return out


def cmd_rollover(a) -> int:
    windows = (0.080, 0.040, 0.020)
    refrs = (100, 80, 60, 50, 40, 30, 17)
    rows, agg = [], {}
    for s in SESSIONS:
        kt = keydown_times(s)
        grid = sess(str(s))["t"]
        d = np.diff(kt) * 1000
        span = kt[-1] - kt[0]
        rows.append({"session": s.name, "keydowns": int(len(kt)), "span_s": round(span, 1),
                     "rate_per_min": round(60 * (len(kt) - 1) / span, 1),
                     "iki_ms": {q: round(float(np.percentile(d, q)), 1)
                                for q in (1, 5, 10, 25, 50, 75, 90, 95)},
                     "frac_iki_under": {ms: round(float((d < ms).mean()), 4)
                                        for ms in (17, 30, 40, 50, 60, 80, 100)},
                     "ceiling": _ceil_block(kt, grid, windows, refrs)})
        agg[s.name] = (kt, grid)
    tot = sum(len(k) for k, _ in agg.values())
    pooled = {}
    for w in windows:
        pooled[f"pair_window_{int(1000*w)}ms"] = {}
        for ms in refrs:
            mr = sum(max_matched(k, ms / 1000, w, g) for k, g in agg.values()) / tot
            pooled[f"pair_window_{int(1000*w)}ms"][ms] = {
                "max_recall": round(mr, 4), "max_f1_at_perfect_precision": round(2 * mr / (1 + mr), 4)}
    rows.append({"session": "POOLED", "keydowns": tot, "ceiling": pooled})
    (OUT / "ceiling_rollover.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps(rows, indent=1))
    return 0


# ---------------------------------------------------------------- fit + report
def sat(n, a_, b, c):
    return a_ - b * np.power(n, -c)


def logfit(n, a_, b):
    return a_ + b * np.log(n)


def fit_curve(ns: np.ndarray, f1s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return curve_fit(sat, ns, f1s, p0=[0.85, 2.0, 0.4],
                     bounds=([0.0, 0.0, 0.01], [1.0, 1e4, 3.0]), maxfev=200000)


def _fit_with_ci(cells: dict, keys: list, n_boot: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    ns = np.array([r["n_keys"] for k in keys for r in cells[k]], float)
    f1 = np.array([r["f1"] for k in keys for r in cells[k]], float)
    popt, _ = fit_curve(ns, f1)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(n_boot):
        nn, ff = [], []
        for k in keys:
            rs = cells[k]
            for i in rng.integers(0, len(rs), len(rs)):
                nn.append(rs[i]["n_keys"])
                ff.append(rs[i]["f1"])
        try:
            boot.append(fit_curve(np.array(nn, float), np.array(ff, float))[0])
        except Exception:
            pass
    return popt, np.array(boot)


def cmd_report(a) -> int:
    res = json.loads((OUT / "ceiling_curve.json").read_text())
    cells: dict = {}
    for r in res:
        cells.setdefault((r["test"], r["n_keys"]), []).append(r)
    print("MEASURED learning curve (leave-one-session-out, contiguous windows, 5 seeds)")
    print(f"{'test-session':<22}{'n_keys':>7}{'min':>6}{'F1 mean':>9}{'med':>7}"
          f"{'IQR':>14}{'std':>6}{'R':>7}{'P':>7}{'oracle':>8}")
    for (test, n), rs in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        f = np.array([x["f1"] for x in rs])
        q1, q3 = np.percentile(f, [25, 75])
        print(f"{test:<22}{n:>7}{np.mean([x['minutes'] for x in rs]):>6.1f}"
              f"{100*f.mean():>8.1f}%{100*np.median(f):>6.1f}%"
              f"  [{100*q1:>4.1f},{100*q3:>4.1f}]{100*f.std():>6.1f}"
              f"{100*np.mean([x['recall'] for x in rs]):>6.1f}%"
              f"{100*np.mean([x['precision'] for x in rs]):>6.1f}%"
              f"{100*np.mean([x['f1_oracle_cfg'] for x in rs]):>7.1f}%")

    out = {}
    tests = sorted({k[0] for k in cells})
    print("\nsaturating fit F1 = a - b*n^-c  (EXTRAPOLATION beyond ~1,100 keys)")
    for label, keys in [("POOLED", list(cells))] + [(t, [k for k in cells if k[0] == t]) for t in tests]:
        popt, B = _fit_with_ci(cells, keys)
        lo, hi = np.percentile(B[:, 0], [2.5, 97.5])
        pred = {n: (100 * sat(n, *popt), *np.percentile([sat(n, *p) for p in B], [2.5, 97.5]))
                for n in (2000, 4000, 10000)}
        print(f"  {label:<22} a={100*popt[0]:5.1f}% CI[{100*lo:5.1f},{100*hi:6.1f}]  "
              + "  ".join(f"n={n}:{v[0]:.1f}%[{100*v[1]:.0f},{100*v[2]:.0f}]" for n, v in pred.items()))
        out[label] = {"popt": popt.tolist(), "asymptote_ci": [lo, hi],
                      "pred": {str(n): [v[0] / 100, v[1], v[2]] for n, v in pred.items()}}
        if label == "POOLED":
            pooled = (popt, B)

    print("\nminutes of calibration needed (EXTRAPOLATED, at this user's 236 keydowns/min)")
    for label in out:
        p_ = np.array(out[label]["popt"])
        for target in (0.70, 0.80, 0.85):
            if p_[0] <= target:
                print(f"  {label:<22} F1 {100*target:.0f}%: unreachable (asymptote {100*p_[0]:.1f}%)")
            else:
                n = (p_[1] / (p_[0] - target)) ** (1 / p_[2])
                print(f"  {label:<22} F1 {100*target:.0f}%: n={n:,.0f} keys = {n/236:,.1f} min")
    print("\nfold-averaged curve (sizes measured in all three folds)")
    common = sorted({k[1] for k in cells if k[0] == tests[0]}
                    & {k[1] for k in cells if k[0] == tests[1]} & {k[1] for k in cells if k[0] == tests[2]})
    for n in common:
        per = [np.mean([x["f1"] for x in cells[(t, n)]]) for t in tests]
        print(f"  n={n:<5} {n/236:.1f} min  mean-of-folds F1={100*np.mean(per):5.1f}%  "
              f"per-fold [{', '.join(f'{100*v:.1f}' for v in per)}]")

    f = np.array([r["f1"] for r in res])
    ss_tot = float(((f - f.mean()) ** 2).sum())
    for name, key in (("test session", lambda r: r["test"]), ("log2(n) bucket",
                      lambda r: int(np.log2(r["n_keys"])))):
        grp: dict = {}
        for r in res:
            grp.setdefault(key(r), []).append(r["f1"])
        ssb = sum(len(v) * (np.mean(v) - f.mean()) ** 2 for v in grp.values())
        print(f"variance in F1 explained by {name:<15}: {100*ssb/ss_tot:4.1f}%")

    ns = np.array([r["n_keys"] for r in res], float)
    f1 = np.array([r["f1"] for r in res], float)
    lp, _ = curve_fit(logfit, ns, f1)
    ss = 1 - np.sum((f1 - logfit(ns, *lp)) ** 2) / np.sum((f1 - f1.mean()) ** 2)
    print(f"\nlog fit F1 = {lp[0]:.3f} + {lp[1]:.4f}*ln(n)   R2={ss:.3f}  (no asymptote by construction)")
    for n in (2000, 4000, 10000):
        print(f"  EXTRAPOLATED log-fit F1 @ {n:>6} = {100*logfit(n, *lp):.1f}%")
    out["log_fit"] = {"popt": lp.tolist(), "r2": float(ss)}
    (OUT / "ceiling_fit.json").write_text(json.dumps(out, indent=1))
    _plot(cells, *pooled)
    return 0


def _plot(cells, popt, B):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    tests = sorted({k[0] for k in cells})
    for i, tname in enumerate(tests):
        pts = sorted([(k[1], np.array([x["f1"] for x in v])) for k, v in cells.items() if k[0] == tname])
        n = [p[0] for p in pts]
        med = [100 * np.median(p[1]) for p in pts]
        lo = [100 * np.percentile(p[1], 25) for p in pts]
        hi = [100 * np.percentile(p[1], 75) for p in pts]
        for axx in ax:
            axx.errorbar(n, med, yerr=[np.array(med) - lo, np.array(hi) - np.array(med)],
                         marker="o", capsize=3, label=f"test={tname[9:15]}", color=f"C{i}")
    xs = np.geomspace(40, 10000, 200)
    for axx in ax:
        axx.plot(xs, 100 * sat(xs, *popt), "k--", lw=1.2, label="fit a-b·n^-c (extrapolated)")
        axx.axhline(100 * popt[0], color="gray", ls=":", lw=1)
        axx.set_xscale("log")
        axx.set_xlabel("calibration keypresses (log)")
        axx.set_ylabel("F1 on held-out session (%)")
        axx.grid(alpha=0.3)
    ax[0].set_xlim(40, 1300)
    ax[0].set_title("measured (leave-one-session-out, 5 seeds, median+IQR)")
    ax[1].set_xlim(40, 12000)
    ax[1].set_ylim(0, 100)
    ax[1].set_title(f"with extrapolation — asymptote {100*popt[0]:.1f}%")
    ax[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "ceiling_curve.png", dpi=140)
    print(f"wrote {OUT/'ceiling_curve.png'}")


def cmd_all(a) -> int:
    cmd_rollover(a)
    cmd_curve(a)
    cmd_refractory(a)
    return cmd_report(a)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.ceiling", description=__doc__)
    ap.add_argument("what", choices=("curve", "rollover", "refractory", "smalln", "report", "all"))
    ap.add_argument("--jobs", type=int, default=5)
    a = ap.parse_args(argv)
    return {"curve": cmd_curve, "rollover": cmd_rollover, "refractory": cmd_refractory,
            "smalln": cmd_smalln, "report": cmd_report, "all": cmd_all}[a.what](a)


if __name__ == "__main__":
    from phase0.analysis import ceiling  # loky cannot pickle this module's caches as __main__

    sys.exit(ceiling.main())
