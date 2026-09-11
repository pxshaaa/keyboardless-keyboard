"""Post-hoc calibration of the spatial model's key_probs, selected on downstream desk CER.
Run: python -m phase0.analysis.calib {structure | metrics | sweep | cv}"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

from phase0.analysis import pipeline as pl
from phase0.analysis import tap_pos as tp
from phase0.analysis import adapt as ad
from phase0.analysis.decode import ALPHABET, NA, Weights, beam_decode, edit_distance

warnings.filterwarnings("ignore", message="Mean of empty slice")

EPS = 1e-12
FLOOR = 1e-6
# each keyboard session paired with a model that never saw it, so every fitted probability is OOF
OOF_MODELS = {
    "20260910-021315-kbd": Path("models/tap_pos_no021315.pkl"),
    "20260910-131629-kbd": Path("models/tap_pos_no131629.pkl"),
    "20260910-015948-kbd": Path("models/tap_pos.pkl"),
}
FINGERS = (4, 8, 12, 16, 20)
OBS_GRID = (0.6, 0.8, 1.0, 1.3, 1.7, 2.2)


# ------------------------------------------------------------------ evidence extraction
def motion_energy(sess: tp.Sess, k: np.ndarray, half: int = 4) -> np.ndarray:
    """Mean fingertip displacement over a +-half-frame window at each tap, both hands."""
    P = sess.P
    a = P[np.clip(k - half, 0, len(P) - 1)][:, :, tp.JOINTS_TIPS, :2]
    b = P[np.clip(k + half, 0, len(P) - 1)][:, :, tp.JOINTS_TIPS, :2]
    return np.nan_to_num(np.nanmean(np.linalg.norm(b - a, axis=-1), axis=(1, 2)))


def side_features(taps: list[dict], sess: tp.Sess, k: np.ndarray, p: np.ndarray) -> dict:
    """Everything a conditional calibrator may condition on; all of it exists at desk time."""
    e = motion_energy(sess, k)
    q = np.maximum(p, EPS)
    return {"hand": np.array([t["hand"] for t in taps], float),
            "finger": np.array([t["finger"] for t in taps], float),
            "energy": np.log1p(e),
            "x": np.array([t["x"] for t in taps], float) / tp.W,
            "y": np.array([t["y"] for t in taps], float) / tp.H,
            "maxp": q.max(1),
            "entropy": -(q * np.log(q)).sum(1)}


def feature_matrix(F: dict) -> np.ndarray:
    """[N,D] design matrix for the conditional temperature head."""
    cols = [np.ones_like(F["hand"]), F["hand"]]
    cols += [(F["finger"] == f).astype(float) for f in FINGERS[1:]]
    for n in ("energy", "maxp", "entropy"):
        v = F[n]
        cols.append((v - v.mean()) / (v.std() + 1e-9))
    return np.stack(cols, 1)


def session_probs(sid: str, model_path: Path, kf: pl.KbdFit) -> dict:
    """CORAL-standardised key_probs for a labelled keyboard session, from a model that
    never saw it. Mirrors pipeline.spatial_proba exactly."""
    m = ad.load_base_model(model_path)
    d = tp.dataset(sid)
    sess, k = d["sess"], d["k"]
    X = tp.features(sess, k, m.mode, m.joints, m.offsets)
    ms, ss = kf.src_moments
    X = (X - X.mean(0)) / (X.std(0) + 1e-6) * ss + ms
    p = m.model.predict_proba(X)
    out = np.full((len(k), NA), FLOOR)
    out[:, m.classes] = np.maximum(p, FLOOR)
    out /= out.sum(1, keepdims=True)
    return {"sid": sid, "p": out, "y": d["y"], "F": side_features(d["taps"], sess, k, out)}


_OOF: list | None = None


def oof_sessions(kf: pl.KbdFit) -> list[dict]:
    global _OOF
    if _OOF is None:
        _OOF = [session_probs(s, m, kf) for s, m in OOF_MODELS.items()]
    return _OOF


def stack(ds: list[dict]) -> dict:
    return {"p": np.vstack([d["p"] for d in ds]),
            "y": np.concatenate([d["y"] for d in ds]),
            "F": {k: np.concatenate([d["F"][k] for d in ds]) for k in ds[0]["F"]}}


# ------------------------------------------------------------------ calibrators
def _norm(lp: np.ndarray) -> np.ndarray:
    lp = lp - lp.max(1, keepdims=True)
    return lp - np.log(np.exp(lp).sum(1, keepdims=True))


def _nll(lp: np.ndarray, y: np.ndarray) -> float:
    return float(-lp[np.arange(len(y)), y].mean())


class Identity:
    name = "identity"

    def fit(self, p, y, F):
        return self

    def apply(self, p, F):
        return _norm(np.log(np.maximum(p, EPS)))


class Temperature:
    """One scalar on the log-probs. The method that has already been shown to fail."""

    name = "temp"

    def __init__(self, t: float | None = None):
        self.logt = 0.0 if t is None else float(np.log(t))
        self.fixed = t is not None

    @property
    def T(self) -> float:
        return float(np.exp(self.logt))

    def fit(self, p, y, F):
        if self.fixed:
            return self
        from scipy.optimize import minimize_scalar
        lp = np.log(np.maximum(p, EPS))
        f = lambda a: _nll(_norm(lp / np.exp(a)), y)  # noqa: E731
        self.logt = float(minimize_scalar(f, bounds=(-1.5, 1.5), method="bounded").x)
        return self

    def apply(self, p, F):
        return _norm(np.log(np.maximum(p, EPS)) / np.exp(self.logt))


class VectorScaling:
    """Per-class gain and bias on the log-probs: the cheapest fix that can be class-conditional."""

    name = "vector"

    def __init__(self, l2: float = 1e-2):
        self.l2, self.a, self.b = l2, None, None

    def fit(self, p, y, F):
        from scipy.optimize import minimize
        lp = np.log(np.maximum(p, EPS))
        n = lp.shape[1]

        def obj(v):
            a, b = v[:n], v[n:]
            z = _norm(lp * a + b)
            return _nll(z, y) + self.l2 * (np.sum((a - 1) ** 2) + np.sum(b ** 2))

        v0 = np.concatenate([np.ones(n), np.zeros(n)])
        r = minimize(obj, v0, method="L-BFGS-B", options={"maxiter": 400})
        self.a, self.b = r.x[:n], r.x[n:]
        return self

    def apply(self, p, F):
        return _norm(np.log(np.maximum(p, EPS)) * self.a + self.b)


class Dirichlet:
    """Full matrix on the log-probs with ODIR regularisation (Kull et al. 2019)."""

    name = "dirichlet"

    def __init__(self, lam: float = 0.3, mu: float = 0.3):
        self.lam, self.mu, self.W, self.b = lam, mu, None, None

    def fit(self, p, y, F):
        from scipy.optimize import minimize
        lp = np.log(np.maximum(p, EPS))
        n = lp.shape[1]
        off = ~np.eye(n, dtype=bool)

        def obj(v):
            W = v[:n * n].reshape(n, n)
            b = v[n * n:]
            z = _norm(lp @ W.T + b)
            return _nll(z, y) + self.lam * np.mean(W[off] ** 2) + self.mu * np.mean(b ** 2)

        v0 = np.concatenate([np.eye(n).ravel(), np.zeros(n)])
        r = minimize(obj, v0, method="L-BFGS-B", options={"maxiter": 600})
        self.W, self.b = r.x[:n * n].reshape(n, n), r.x[n * n:]
        return self

    def apply(self, p, F):
        return _norm(np.log(np.maximum(p, EPS)) @ self.W.T + self.b)


class IsotonicPerClass:
    """One-vs-rest isotonic per class, renormalised. Monotone, so per-class ranking survives."""

    name = "isotonic"

    def __init__(self, min_n: int = 30):
        self.min_n, self.f = min_n, {}

    def fit(self, p, y, F):
        from sklearn.isotonic import IsotonicRegression
        for c in range(p.shape[1]):
            m = p[:, c] > FLOOR * 2
            if m.sum() < self.min_n or len(np.unique(y[m] == c)) < 2:
                continue
            ir = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip")
            ir.fit(p[m, c], (y[m] == c).astype(float))
            self.f[c] = ir
        return self

    def apply(self, p, F):
        q = np.maximum(p.copy(), EPS)
        for c, ir in self.f.items():
            q[:, c] = np.maximum(ir.predict(p[:, c]), 1e-6)
        return _norm(np.log(q))


class BetaPerClass:
    """Per-class beta calibration: logistic on (log p, -log(1-p)), then renormalise."""

    name = "beta"

    def __init__(self, min_n: int = 30):
        self.min_n, self.f = min_n, {}

    def fit(self, p, y, F):
        from sklearn.linear_model import LogisticRegression
        for c in range(p.shape[1]):
            m = p[:, c] > FLOOR * 2
            t = (y[m] == c).astype(int)
            if m.sum() < self.min_n or t.sum() < 5 or t.sum() == m.sum():
                continue
            v = np.clip(p[m, c], 1e-6, 1 - 1e-6)
            X = np.stack([np.log(v), -np.log(1 - v)], 1)
            self.f[c] = LogisticRegression(C=1.0, max_iter=500).fit(X, t)
        return self

    def apply(self, p, F):
        q = np.maximum(p.copy(), EPS)
        for c, lr in self.f.items():
            v = np.clip(p[:, c], 1e-6, 1 - 1e-6)
            X = np.stack([np.log(v), -np.log(1 - v)], 1)
            q[:, c] = np.maximum(lr.predict_proba(X)[:, 1], 1e-6)
        return _norm(np.log(q))


class ConditionalTemperature:
    """T_i = exp(w . f_i): one temperature per tap, predicted from hand, finger, motion
    energy and the tap's own sharpness."""

    name = "condtemp"

    def __init__(self, l2: float = 1e-2):
        self.l2, self.w = l2, None

    def fit(self, p, y, F):
        from scipy.optimize import minimize
        lp = np.log(np.maximum(p, EPS))
        D = feature_matrix(F)

        def obj(w):
            t = np.exp(np.clip(D @ w, -1.5, 1.5))[:, None]
            return _nll(_norm(lp / t), y) + self.l2 * np.sum(w[1:] ** 2)

        r = minimize(obj, np.zeros(D.shape[1]), method="L-BFGS-B", options={"maxiter": 400})
        self.w = r.x
        return self

    def apply(self, p, F):
        t = np.exp(np.clip(feature_matrix(F) @ self.w, -1.5, 1.5))[:, None]
        return _norm(np.log(np.maximum(p, EPS)) / t)


def methods() -> dict:
    return {"identity": Identity(), "temp": Temperature(), "vector": VectorScaling(),
            "dirichlet": Dirichlet(), "isotonic": IsotonicPerClass(),
            "beta": BetaPerClass(), "condtemp": ConditionalTemperature()}


# ------------------------------------------------------------------ calibration metrics
def ece(conf: np.ndarray, ok: np.ndarray, bins: int = 10) -> float:
    b = np.clip((conf * bins).astype(int), 0, bins - 1)
    return float(sum((b == j).mean() * abs(conf[b == j].mean() - ok[b == j].mean())
                     for j in range(bins) if (b == j).any()))


def clf_metrics(lp: np.ndarray, y: np.ndarray) -> dict:
    p = np.exp(lp)
    am, conf = p.argmax(1), p.max(1)
    ok = (am == y).astype(float)
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1.0
    return {"acc": float(ok.mean()), "meanmax": float(conf.mean()), "ece": ece(conf, ok),
            "nll": _nll(lp, y), "brier": float(((p - oh) ** 2).sum(1).mean())}


def loso_metrics(kf: pl.KbdFit) -> dict:
    """Leave-one-keyboard-session-out: every calibrator is fitted without the session it scores."""
    ds = oof_sessions(kf)
    out = {}
    for name in methods():
        lps, ys = [], []
        for i, te in enumerate(ds):
            tr = stack([d for j, d in enumerate(ds) if j != i])
            m = methods()[name].fit(tr["p"], tr["y"], tr["F"])
            lps.append(m.apply(te["p"], te["F"]))
            ys.append(te["y"])
        out[name] = clf_metrics(np.vstack(lps), np.concatenate(ys))
    return out


# ------------------------------------------------------------------ desk decoding
def fold_cfgs(rows: list[dict], n: int) -> list[dict]:
    """The extractor and deletion cost pipeline.nested_cv would pick for each fold."""
    return [pl.select_cfg(rows, set(range(n)) - {j}) for j in range(n)]


def desk_probs(session: Path, cfg: dict, kf: pl.KbdFit, st: pl.Stack):
    taps, segs, base, times, sess, k = pl.run(session, cfg, st, kf)
    return taps, segs, base, times, sess, k, side_features(taps, sess, k, base)


def decode_rows(segs, lp: np.ndarray, kf: pl.KbdFit, st: pl.Stack, obs: float,
                keep=None) -> list[tuple]:
    """pipeline.score_phrases, but with the decoder's observation weight exposed."""
    w = Weights(insertion=st.insertion, deletion=st.deletion, max_deletions=st.max_del, obs=obs)
    clm, wlm = kf.lm
    out = []
    for j, (text, rows) in enumerate(segs):
        if keep is not None and j not in keep:
            continue
        hyp = "" if len(rows) == 0 else beam_decode(lp[rows], clm, wlm, w, st.beam)
        out.append((text, hyp, edit_distance(text, hyp), len(text)))
    return out


def grid_table(session: Path, kf: pl.KbdFit, rows: list[dict], names, obs_grid,
               verbose: bool = True) -> dict:
    """per-phrase (err, len) for every (fold-cfg, method, obs-weight), EM off.
    Fold-independent, so any fold's selection is a subset sum of this table."""
    ds = oof_sessions(kf)
    tr = stack(ds)
    fitted = {n: methods()[n].fit(tr["p"], tr["y"], tr["F"]) for n in names}
    n_ph = len(rows[0]["per"])
    cfgs = fold_cfgs(rows, n_ph)
    uniq = {json.dumps(c["cfg"], sort_keys=True) + f"|{c['deletion']}": c for c in cfgs}
    tab: dict = {}
    for key, pick in uniq.items():
        st = replace(pl.SWEEP, deletion=pick["deletion"])
        _, segs, base, _, _, _, F = desk_probs(session, pick["cfg"], kf, st)
        for nm in names:
            lp = fitted[nm].apply(base, F)
            for obs in obs_grid:
                r = decode_rows(segs, lp, kf, st, obs)
                tab[(key, nm, obs)] = [(x[2], x[3]) for x in r]
                if verbose:
                    print(f"  {nm:>9} obs={obs:<4} CER={sum(a for a,_ in tab[(key,nm,obs)])/max(1,sum(b for _,b in tab[(key,nm,obs)])):.3f}",
                          flush=True)
    return {"tab": tab, "keys": [json.dumps(c["cfg"], sort_keys=True) + f"|{c['deletion']}"
                                 for c in cfgs], "fitted": fitted, "cfgs": cfgs}


def arm_rows(g: dict, name: str, obs: float) -> list[tuple]:
    """Per-phrase errors for one arm, each phrase scored under its own fold's extractor."""
    return [g["tab"][(g["keys"][j], name, obs)][j] for j in range(len(g["keys"]))]


def selected_rows(g: dict, names, obs_grid) -> tuple[list[tuple], list[tuple]]:
    """Pick (method, obs) inside each fold on the other 19 phrases only."""
    out, picks = [], []
    n = len(g["keys"])
    for j in range(n):
        best, bc = None, np.inf
        for nm in names:
            for obs in obs_grid:
                per = g["tab"][(g["keys"][j], nm, obs)]
                e = sum(per[i][0] for i in range(n) if i != j)
                d = sum(per[i][1] for i in range(n) if i != j)
                if e / max(1, d) < bc:
                    best, bc = (nm, obs), e / max(1, d)
        out.append(g["tab"][(g["keys"][j], *best)][j])
        picks.append(best)
    return out, picks


def ci(per: list[tuple]) -> tuple[float, float, float]:
    return pl.boot_ci([(None, None, a, b) for a, b in per])


def delta(a: list[tuple], b: list[tuple]) -> tuple[float, float, float]:
    return pl.boot_delta([(None, None, x, y) for x, y in a], [(None, None, x, y) for x, y in b])


# ------------------------------------------------------------------ CLI
def _table(a) -> list[dict]:
    return json.loads(Path(a.table).read_text())


def cmd_structure(a) -> int:
    kf = pl.kbd_fit()
    ds = oof_sessions(kf)
    d = stack(ds)
    p, y, F = d["p"], d["y"], d["F"]
    conf, ok = p.max(1), (p.argmax(1) == y).astype(float)
    m = clf_metrics(np.log(np.maximum(p, EPS)), y)
    print(f"pooled OOF taps n={len(y)}  mean max-p={m['meanmax']:.3f}  top1={m['acc']:.3f}  "
          f"ECE={m['ece']:.3f}  NLL={m['nll']:.3f}  Brier={m['brier']:.3f}")
    print("\nreliability (10 equal-width bins)")
    print(f"{'bin':>12}{'n':>7}{'conf':>8}{'acc':>8}{'gap':>8}")
    b = np.clip((conf * 10).astype(int), 0, 9)
    for j in range(10):
        s = b == j
        if s.sum():
            print(f"{f'{j/10:.1f}-{j/10+0.1:.1f}':>12}{s.sum():>7}{conf[s].mean():>8.3f}"
                  f"{ok[s].mean():>8.3f}{conf[s].mean()-ok[s].mean():>+8.3f}")

    def grp(label, key, names=None):
        print(f"\ngap by {label}")
        print(f"{label:>12}{'n':>7}{'conf':>8}{'acc':>8}{'gap':>8}{'ECE':>8}")
        for v in sorted(set(key.tolist())):
            s = key == v
            if s.sum() < 25:
                continue
            nm = names[v] if names else v
            print(f"{nm!s:>12}{s.sum():>7}{conf[s].mean():>8.3f}{ok[s].mean():>8.3f}"
                  f"{conf[s].mean()-ok[s].mean():>+8.3f}{ece(conf[s], ok[s]):>8.3f}")

    grp("hand", F["hand"])
    grp("finger", F["finger"])
    for nm in ("energy", "x", "y"):
        q = np.digitize(F[nm], np.quantile(F[nm], [.25, .5, .75]))
        grp(nm + " quart", q)
    print("\ngap by predicted key (n>=30)")
    print(f"{'key':>12}{'n':>7}{'conf':>8}{'prec':>8}{'gap':>8}")
    am = p.argmax(1)
    for c in range(NA):
        s = am == c
        if s.sum() >= 30:
            print(f"{ALPHABET[c]!r:>12}{s.sum():>7}{conf[s].mean():>8.3f}{ok[s].mean():>8.3f}"
                  f"{conf[s].mean()-ok[s].mean():>+8.3f}")
    return 0


def cmd_metrics(a) -> int:
    kf = pl.kbd_fit()
    r = loso_metrics(kf)
    print("\nleave-one-keyboard-session-out calibration metrics (n=1998 taps)")
    print(f"{'method':>10}{'top1':>8}{'meanmax':>9}{'ECE':>8}{'NLL':>8}{'Brier':>8}")
    for n, m in r.items():
        print(f"{n:>10}{m['acc']:>8.3f}{m['meanmax']:>9.3f}{m['ece']:>8.3f}{m['nll']:>8.3f}"
              f"{m['brier']:>8.3f}")
    if a.json:
        Path(a.json).write_text(json.dumps(r))
    return 0


def cmd_sweep(a) -> int:
    kf = pl.kbd_fit()
    rows = _table(a)
    names = a.methods.split(",") if a.methods else list(methods())
    obs_grid = tuple(float(x) for x in a.obs.split(",")) if a.obs else OBS_GRID
    # identity@1.0 is the reference every delta is taken against, so it is never optional
    names = ["identity"] + [n for n in names if n != "identity"]
    obs_grid = tuple(sorted({1.0, *obs_grid}))
    g = grid_table(pl.session_path(a.session), kf, rows, names, obs_grid, verbose=a.verbose)
    base = arm_rows(g, "identity", 1.0)
    print(f"\ncalibration x obs-weight, desk CER (EM off, per-fold extractor)\n{'method':>10}"
          + "".join(f"{o:>10}" for o in obs_grid))
    for nm in names:
        line = f"{nm:>10}"
        for o in obs_grid:
            line += f"{ci(arm_rows(g, nm, o))[0]:>10.3f}"
        print(line)
    print(f"\n{'arm':>22}{'CER':>8}{'95% CI':>18}{'delta vs identity@1.0':>24}")
    best = []
    for nm in names:
        for o in obs_grid:
            r = arm_rows(g, nm, o)
            c, lo, hi = ci(r)
            best.append((c, nm, o, r))
    for c, nm, o, r in sorted(best)[:a.top]:
        d, dlo, dhi = delta(base, r)
        _, lo, hi = ci(r)
        print(f"{f'{nm}@obs={o}':>22}{c:>8.3f}{f'[{lo:.3f}, {hi:.3f}]':>18}"
              f"{f'{d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]':>24}")
    sel, picks = selected_rows(g, names, obs_grid)
    c, lo, hi = ci(sel)
    d, dlo, dhi = delta(base, sel)
    print(f"\nout-of-fold selected  CER={c:.3f} [{lo:.3f}, {hi:.3f}]  "
          f"delta={d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]")
    from collections import Counter
    print("  per-fold picks:", Counter(picks).most_common())
    if a.json:
        Path(a.json).write_text(json.dumps(
            {f"{nm}|{o}": arm_rows(g, nm, o) for nm in names for o in obs_grid}))
    return 0


def cmd_cv(a) -> int:
    """Full stack, leave-one-phrase-out with EM, for one calibrator and obs-weight."""
    kf = pl.kbd_fit()
    rows = _table(a)
    s = pl.session_path(a.session)
    ds = oof_sessions(kf)
    tr = stack(ds)
    cal = methods()[a.method].fit(tr["p"], tr["y"], tr["F"])
    obs = float(a.obs) if a.obs else 1.0
    n = len(rows[0]["per"])
    out = []
    for j in range(n):
        pick = pl.select_cfg(rows, set(range(n)) - {j})
        st = replace(pl.FULL, deletion=pick["deletion"])
        taps, segs, base, times, sess, k = pl.run(s, pick["cfg"], st, kf)
        F = side_features(taps, sess, k, base)
        p = base
        if st.em:
            p = pl.weakly_supervise(sess, k, base, [x for i, x in enumerate(segs)
                                                    if i != j and len(x[1])], st)
        lp = cal.apply(p, F)
        r = decode_rows(segs, lp, kf, st, obs, keep={j})[0]
        out.append(r)
        if a.verbose:
            print(f"  phrase{j:>2} CER={r[2]/max(1,r[3]):.3f}", flush=True)
    pl.report(f"FULL + {a.method} @ obs={obs}", out)
    if a.json:
        Path(a.json).write_text(json.dumps([(x[2], x[3]) for x in out]))
    return 0


def cmd_obscv(a) -> int:
    """Full stack with EM, obs-weight chosen inside each fold on the other 19 phrases only."""
    kf = pl.kbd_fit()
    rows = _table(a)
    s = pl.session_path(a.session)
    obs_grid = tuple(float(x) for x in a.obs.split(",")) if a.obs else OBS_GRID
    ds = oof_sessions(kf)
    tr = stack(ds)
    cal = methods()[a.method].fit(tr["p"], tr["y"], tr["F"])
    n = len(rows[0]["per"])
    per: dict = {o: [] for o in obs_grid}
    for j in range(n):
        pick = pl.select_cfg(rows, set(range(n)) - {j})
        st = replace(pl.FULL, deletion=pick["deletion"])
        taps, segs, base, _, sess, k = pl.run(s, pick["cfg"], st, kf)
        F = side_features(taps, sess, k, base)
        p = pl.weakly_supervise(sess, k, base, [x for i, x in enumerate(segs)
                                                if i != j and len(x[1])], st)
        lp = cal.apply(p, F)
        for o in obs_grid:
            per[o].append([(x[2], x[3]) for x in decode_rows(segs, lp, kf, st, o)])
        if a.verbose:
            print(f"  fold {j} done", flush=True)
    Path(a.json or "/tmp/obscv.json").write_text(json.dumps(
        {str(o): per[o] for o in obs_grid}))
    print(f"\nfull stack (EM on), {a.method} calibration, obs-weight sweep")
    print(f"{'obs':>6}{'CER(held-out)':>16}{'95% CI':>18}")
    for o in obs_grid:
        r = [per[o][j][j] for j in range(n)]
        c, lo, hi = ci(r)
        print(f"{o:>6}{c:>16.3f}{f'[{lo:.3f}, {hi:.3f}]':>18}")
    sel, picks = [], []
    for j in range(n):
        best, bc = None, np.inf
        for o in obs_grid:
            e = sum(per[o][j][i][0] for i in range(n) if i != j)
            d = sum(per[o][j][i][1] for i in range(n) if i != j)
            if e / max(1, d) < bc:
                best, bc = o, e / max(1, d)
        sel.append(per[best][j][j])
        picks.append(best)
    c, lo, hi = ci(sel)
    base_rows = [per[1.0][j][j] for j in range(n)] if 1.0 in obs_grid else sel
    d, dlo, dhi = delta(base_rows, sel)
    from collections import Counter
    print(f"\nobs chosen out-of-fold  CER={c:.3f} [{lo:.3f}, {hi:.3f}]  "
          f"delta vs obs=1.0 {d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]")
    print("  per-fold picks:", Counter(picks).most_common())
    return 0


def cmd_control(a) -> int:
    """Keyboard-session control CER: the same calibrators, scored where the labels are real.
    Each session is decoded by a model that never saw it, calibrated on the other sessions."""
    from phase0.analysis.decode import kbd_segments
    from phase0.analysis.analyze_drift import read_jsonl

    kf = pl.kbd_fit()
    ds = oof_sessions(kf)
    names = a.methods.split(",") if a.methods else list(methods())
    obs_grid = tuple(sorted({1.0, *(float(x) for x in a.obs.split(","))})) if a.obs else OBS_GRID
    for sid in (a.control or ",".join(OOF_MODELS)).split(","):
        i = list(OOF_MODELS).index(sid)
        tr = stack([d for j, d in enumerate(ds) if j != i])
        sp = pl.session_path(sid)
        m = ad.load_base_model(OOF_MODELS[sid])
        taps = read_jsonl(sp / "taps.jsonl")
        sess = tp.load_sess(sid)
        k = sess.rows(taps)
        X = tp.features(sess, k, m.mode, m.joints, m.offsets)
        ms, ss = kf.src_moments
        p = m.model.predict_proba((X - X.mean(0)) / (X.std(0) + 1e-6) * ss + ms)
        base = np.full((len(k), NA), FLOOR)
        base[:, m.classes] = np.maximum(p, FLOOR)
        base /= base.sum(1, keepdims=True)
        F = side_features(taps, sess, k, base)
        idx = {round(t["t"], 6): j for j, t in enumerate(taps)}
        segs = [(t, np.array([idx[round(x["t"], 6)] for x in s], int))
                for t, s in kbd_segments(sp, taps)]
        st = replace(pl.FULL, em=False, deletion=a.deletion)
        print(f"\n{sid}: {len(taps)} taps, {len(segs)} bursts, del={a.deletion:.0f}")
        print(f"{'method':>10}" + "".join(f"{o:>10}" for o in obs_grid))
        for nm in names:
            lp = methods()[nm].fit(tr["p"], tr["y"], tr["F"]).apply(base, F)
            line = f"{nm:>10}"
            for o in obs_grid:
                line += f"{pl.pooled(decode_rows(segs, lp, kf, st, o)):>10.3f}"
            print(line, flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.calib", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("structure", cmd_structure), ("metrics", cmd_metrics),
                     ("sweep", cmd_sweep), ("cv", cmd_cv),
                     ("control", cmd_control), ("obscv", cmd_obscv)):
        p = sub.add_parser(name)
        p.add_argument("--session", default=pl.DESK)
        p.add_argument("--table", default="data/sessions/desk_sweep.json")
        p.add_argument("--json", default=None)
        p.add_argument("--methods", default=None)
        p.add_argument("--method", default="identity")
        p.add_argument("--obs", default=None)
        p.add_argument("--top", type=int, default=10)
        p.add_argument("--verbose", action="store_true")
        p.add_argument("--control", default=None)
        p.add_argument("--deletion", type=float, default=-3.0)
        p.set_defaults(func=fn)
    a = ap.parse_args(argv)
    t0 = time.time()
    r = a.func(a)
    print(f"\n[{time.time()-t0:.1f}s]", file=sys.stderr)
    return r


if __name__ == "__main__":
    sys.exit(main())
