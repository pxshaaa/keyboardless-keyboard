"""Graded keypress targets + F-beta event tuning for the GB detector; tuned on training OOF only.
Run: python -m phase0.analysis.taps_soft {sweep | report}"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from phase0.analysis.analyze_drift import pair_events
from phase0.analysis.eval_taps import STILL_S, WINDOW_S, keydown_times
from phase0.analysis.detect_taps import FINGERTIP_JOINTS
from phase0.analysis.taps_gb import (
    EVENT_GRID,
    GROUPS,
    TYPING_HALF_WIDTH_S,
    _run_length,
    assemble,
    build_groups,
    cv_folds,
    flex_velocity,
    load_frames,
    prep_mask,
    purge,
    score_prep,
    smooth,
)
from scipy.signal import find_peaks

SESS_DIR = Path("data/sessions")
TRAIN_SESSIONS = ("20260910-015217-kbd", "20260910-021315-kbd", "20260910-131629-kbd")
HELDOUT_SESSION = "20260910-015948-kbd"
ROLLOVER_S = 0.080
SIGMA_S = 0.030
BETAS = (1.0, 1.5, 2.0)


# ---------------------------------------------------------------- targets
@dataclass(frozen=True)
class Target:
    """kind: cls = binary label; xent = continuous label under logloss; l2 = regress the bump."""
    name: str
    kind: str
    half: float = 0.033
    sigma: float = SIGMA_S
    pos_weight: float = 8.0
    weighted: bool = False


TARGETS = (
    Target("hard33", "cls", half=0.033),                                  # taps_gb baseline
    Target("gauss_l2", "l2", sigma=SIGMA_S),
    Target("gauss_xent", "xent", sigma=SIGMA_S),
    Target("wide50_w", "cls", half=0.050, sigma=0.033, weighted=True),
    Target("wide67_w", "cls", half=0.067, sigma=0.040, weighted=True),
)
GATE = Target("gate", "cls", half=TYPING_HALF_WIDTH_S)


def dist_to_key(t: np.ndarray, kt: np.ndarray) -> np.ndarray:
    if len(kt) == 0:
        return np.full(len(t), 1e3)
    return np.abs(t[:, None] - kt[None, :]).min(1)


def make_target(t: np.ndarray, kt: np.ndarray, spec: Target) -> tuple[np.ndarray, np.ndarray]:
    """-> (label, sample_weight). Hard labels stay 0/1; graded ones live in [0,1]."""
    d = dist_to_key(t, kt)
    if spec.kind == "cls":
        y = (d <= spec.half).astype(float)
        decay = np.exp(-0.5 * (d / spec.sigma) ** 2) if spec.weighted else np.ones(len(t))
        w = np.where(y > 0, spec.pos_weight * decay, 1.0)
        return y, w
    y = np.exp(-0.5 * (d / spec.sigma) ** 2)
    y = np.where(y < 1e-4, 0.0, y)
    return y, 1.0 + (spec.pos_weight - 1.0) * y


def fit_model(spec: Target, X: np.ndarray, y: np.ndarray, w: np.ndarray, seed: int):
    import lightgbm as lgb

    p = dict(n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=40,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
             n_jobs=10, verbose=-1, random_state=seed, force_col_wise=True)
    if spec.kind == "cls":
        m = lgb.LGBMClassifier(**p)
    else:
        m = lgb.LGBMRegressor(objective="cross_entropy" if spec.kind == "xent" else "regression", **p)
    m.fit(X, y, sample_weight=w)
    return m


def predict_p(m, X: np.ndarray) -> np.ndarray:
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:, 1]
    return np.clip(m.predict(X), 0.0, 1.0)


# ---------------------------------------------------------------- event tuning (F-beta)
def fbeta(s: dict, beta: float) -> float:
    r, p = s["recall"], s["precision"]
    b2 = beta * beta
    return (1 + b2) * r * p / (b2 * p + r) if (r + p) else 0.0


def _sweep_thr(p, t, prep, rf, nc, run_frac, beta):
    idx, props = find_peaks(p, height=EVENT_GRID["thr"][0], distance=rf)
    h = props["peak_heights"]
    best, best_s = None, -1.0
    for thr in EVENT_GRID["thr"]:
        cand = idx[h >= thr]
        if nc > 1 and len(cand):
            run = _run_length(p >= thr * run_frac)
            cand = cand[run[cand] >= nc]
        v = fbeta(score_prep(prep, t, cand), beta)
        if v > best_s:
            best_s, best = v, float(thr)
    return best, best_s


def tune_events(p: np.ndarray, t: np.ndarray, kt: np.ndarray, mask: np.ndarray,
                gate: np.ndarray | None = None, beta: float = 1.0,
                run_frac: float = 0.6) -> tuple[dict, float]:
    """Same two-stage grid as taps_gb.tune_events, with F-beta instead of F1 as the objective."""
    best = {"thr": 0.5, "smooth": 3, "refractory": 5, "n_consec": 1, "gate_thr": 0.0}
    best_s = -1.0
    prep = prep_mask(kt, t, mask)
    gates = (0.0,) if gate is None else EVENT_GRID["gate"]
    for w in EVENT_GRID["smooth"]:
        ps = smooth(p, w)
        for gt in gates:
            pg = ps if gt <= 0 else np.where(gate >= gt, ps, 0.0)
            thr, s = _sweep_thr(pg, t, prep, 5, 1, run_frac, beta)
            if s > best_s:
                best_s = s
                best = {"thr": thr, "smooth": w, "refractory": 5, "n_consec": 1, "gate_thr": gt}
    ps = smooth(p, best["smooth"])
    pg = ps if best["gate_thr"] <= 0 else np.where(gate >= best["gate_thr"], ps, 0.0)
    for rf in EVENT_GRID["refractory"]:
        for nc in EVENT_GRID["n_consec"]:
            thr, s = _sweep_thr(pg, t, prep, rf, nc, run_frac, beta)
            if s > best_s:
                best_s = s
                best = dict(best, thr=thr, refractory=rf, n_consec=nc)
    return best, best_s


def apply_events(p: np.ndarray, cfg: dict, gate: np.ndarray | None = None) -> np.ndarray:
    ps = smooth(p, cfg["smooth"])
    if gate is not None and cfg.get("gate_thr", 0.0) > 0:
        ps = np.where(gate >= cfg["gate_thr"], ps, 0.0)
    idx, _ = find_peaks(ps, height=cfg["thr"], distance=cfg["refractory"])
    if cfg["n_consec"] > 1 and len(idx):
        run = _run_length(ps >= cfg["thr"] * 0.6)
        idx = idx[run[idx] >= cfg["n_consec"]]
    return idx


# ---------------------------------------------------------------- scoring
def rollover_mask(kt: np.ndarray, gap: float = ROLLOVER_S) -> np.ndarray:
    """True where a keydown follows its predecessor within `gap`; the first key is never one."""
    m = np.zeros(len(kt), bool)
    if len(kt) > 1:
        m[1:] = np.diff(kt) <= gap
    return m


def breakdown(kt: np.ndarray, tt: np.ndarray) -> dict:
    """eval_taps-style scores plus recall split by rollover / non-rollover keydowns."""
    kt, tt = np.sort(kt), np.sort(tt)
    pairs = pair_events(kt.tolist(), tt.tolist(), WINDOW_S) if len(kt) and len(tt) else []
    hit_k = np.zeros(len(kt), bool)
    for i, _ in pairs:
        hit_k[i] = True
    ro = rollover_mask(kt)
    still = int(sum(np.min(np.abs(kt - x)) > STILL_S for x in tt)) if len(kt) and len(tt) else len(tt)
    r = hit_k.mean() if len(kt) else 0.0
    p = len(pairs) / len(tt) if len(tt) else 0.0
    return {
        "keydowns": int(len(kt)), "taps": int(len(tt)), "hits": int(len(pairs)),
        "recall": float(r), "precision": float(p),
        "f1": float(2 * r * p / (r + p)) if r + p else 0.0,
        "still_fp": still,
        "n_rollover": int(ro.sum()), "n_nonrollover": int((~ro).sum()),
        "recall_rollover": float(hit_k[ro].mean()) if ro.any() else float("nan"),
        "recall_nonrollover": float(hit_k[~ro].mean()) if (~ro).any() else float("nan"),
    }


# ---------------------------------------------------------------- data
class Bundle:
    """Feature matrices held once so every target/seed/beta reuses them."""

    def __init__(self, train: list[str], test: str, groups: tuple[str, ...] = GROUPS):
        self.Xs, self.ts, self.sid, self.kts = [], [], [], []
        for k, name in enumerate(train):
            _, t, P = load_frames(SESS_DIR / name)
            self.Xs.append(assemble(build_groups(P), groups))
            self.ts.append(t)
            self.sid.append(np.full(len(t), k))
            self.kts.append(keydown_times(SESS_DIR / name))
        self.X = np.vstack(self.Xs).astype(np.float32)
        self.t = np.concatenate(self.ts)
        self.sid = np.concatenate(self.sid)
        self.kt = np.sort(np.concatenate(self.kts))
        self.Xs = None
        self.frames_te, self.t_te, self.P_te = load_frames(SESS_DIR / test)
        self.X_te = assemble(build_groups(self.P_te), groups).astype(np.float32)
        self.kt_te = keydown_times(SESS_DIR / test)
        self.folds = cv_folds("block", self.t, self.sid)


def oof_probs(b: Bundle, spec: Target, seed: int) -> np.ndarray:
    y, w = make_target(b.t, b.kt, spec)
    oof = np.zeros(len(b.t))
    for te in b.folds:
        tr = purge(~te, te, b.t)
        oof[te] = predict_p(fit_model(spec, b.X[tr], y[tr], w[tr], seed), b.X[te])
    return oof


def write_taps(path: Path, ev: np.ndarray, b: Bundle) -> None:
    fv = flex_velocity(b.P_te)
    with open(path, "w") as fh:
        for k in ev:
            m = np.abs(np.nan_to_num(fv[k], nan=-1.0))
            s, f = np.unravel_index(int(np.argmax(m)), m.shape)
            tip = FINGERTIP_JOINTS[f]
            fh.write(json.dumps({
                "t": float(b.t_te[k]), "hand": int(s), "finger": int(tip),
                "x": float(np.nan_to_num(b.P_te[k, s, tip, 0])),
                "y": float(np.nan_to_num(b.P_te[k, s, tip, 1])),
                "conf": float(np.nan_to_num(b.P_te[k, s, tip, 2])), "i": int(b.frames_te[k]),
            }) + "\n")


def clip_taps(tt: np.ndarray, kt: np.ndarray) -> np.ndarray:
    return tt[(tt >= kt.min() - 0.1) & (tt <= kt.max() + 0.1)]


# ---------------------------------------------------------------- sweep
def cmd_sweep(a) -> int:
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    b = Bundle(list(a.train), a.test)
    print(f"train frames={len(b.t)} keys={len(b.kt)} feat={b.X.shape[1]} | "
          f"test frames={len(b.t_te)} keys={len(b.kt_te)}  [{time.time()-t0:.0f}s]", flush=True)
    specs = [s for s in TARGETS if not a.targets or s.name in a.targets.split(",")]
    rows = []
    gy, gw = make_target(b.t, b.kt, GATE)
    oofg = np.zeros(len(b.t))
    for te in b.folds:
        tr = purge(~te, te, b.t)
        oofg[te] = predict_p(fit_model(GATE, b.X[tr], gy[tr], gw[tr], 0), b.X[te])
    gate_te = predict_p(fit_model(GATE, b.X, gy, gw, 0), b.X_te)  # nuisance model, held fixed across seeds
    print(f"gate done [{time.time()-t0:.0f}s]", flush=True)
    for seed in range(a.seeds):
        for spec in specs:
            oof = oof_probs(b, spec, seed)
            y, w = make_target(b.t, b.kt, spec)
            p_te = predict_p(fit_model(spec, b.X, y, w, seed), b.X_te)
            allm = np.ones(len(b.t), bool)
            for beta in BETAS:
                cfg, oof_s = tune_events(oof, b.t, b.kt, allm, oofg, beta)
                oof_sc = breakdown(b.kt, b.t[apply_events(oof, cfg, oofg)])
                tt = clip_taps(b.t_te[apply_events(p_te, cfg, gate_te)], b.kt_te)
                te_sc = breakdown(b.kt_te, tt)
                tag = f"{spec.name}_b{beta}_s{seed}"
                write_taps(out_dir / f"taps_{tag}.jsonl", apply_events(p_te, cfg, gate_te), b)
                rows.append({"target": spec.name, "beta": beta, "seed": seed, "cfg": cfg,
                             "oof": oof_sc, "test": te_sc, "spec": asdict(spec)})
                print(f"  {tag:<24} OOF F1={100*oof_sc['f1']:5.1f} R={100*oof_sc['recall']:5.1f} "
                      f"P={100*oof_sc['precision']:5.1f} roll={100*oof_sc['recall_rollover']:5.1f} "
                      f"| TEST F1={100*te_sc['f1']:5.1f} R={100*te_sc['recall']:5.1f} "
                      f"P={100*te_sc['precision']:5.1f} roll={100*te_sc['recall_rollover']:5.1f} "
                      f"non={100*te_sc['recall_nonrollover']:5.1f}  [{time.time()-t0:.0f}s]", flush=True)
                (out_dir / "results.json").write_text(json.dumps(rows, indent=1))
    print(summarise(rows))
    return 0


def summarise(rows: list[dict]) -> str:
    keys = sorted({(r["target"], r["beta"]) for r in rows})
    out = [f"{'target':<12} {'beta':<5} {'OOF F1':<8} {'TEST F1':<20} {'TEST R':<20} "
           f"{'TEST P':<20} {'roll R':<8} {'non R':<8}"]
    for tgt, beta in keys:
        sel = [r for r in rows if r["target"] == tgt and r["beta"] == beta]
        def st(field, src="test"):
            v = 100 * np.array([r[src][field] for r in sel])
            return v
        f1, rec, pre = st("f1"), st("recall"), st("precision")
        out.append(
            f"{tgt:<12} {beta:<5} {np.mean(st('f1','oof')):<8.1f} "
            f"{np.mean(f1):5.1f}/{np.median(f1):5.1f}[{np.percentile(f1,25):.1f}-{np.percentile(f1,75):.1f}] "
            f"{np.mean(rec):5.1f}/{np.median(rec):5.1f}[{np.percentile(rec,25):.1f}-{np.percentile(rec,75):.1f}] "
            f"{np.mean(pre):5.1f}/{np.median(pre):5.1f}[{np.percentile(pre,25):.1f}-{np.percentile(pre,75):.1f}] "
            f"{np.mean(st('recall_rollover')):<8.1f} {np.mean(st('recall_nonrollover')):<8.1f}")
    return "\n".join(out)


CER_RE = re.compile(r"CER=([0-9.]+) WER=([0-9.]+)")


def cer_of(taps: Path, session: str, pos_model: str, spatial_model: str) -> tuple[float, float, int]:
    """tap_pos supplies the key distribution; decode scores it against the real keystrokes."""
    sess = str(SESS_DIR / session)
    pos = taps.with_suffix(".pos.jsonl")
    env = {**os.environ, "PYTHONPATH": "."}
    subprocess.run([sys.executable, "-m", "phase0.analysis.tap_pos", "apply", session,
                    "--taps", str(taps), "--model", pos_model, "--out", str(pos)],
                   check=True, capture_output=True, env=env)
    r = subprocess.run([sys.executable, "-m", "phase0.analysis.decode", "decode", sess,
                        "--taps", str(pos), "--spatial", "keyprobs", "--control",
                        "--model", spatial_model], check=True, capture_output=True, text=True, env=env)
    m = CER_RE.search(r.stdout)
    n = sum(1 for _ in open(taps))
    return (float(m.group(1)), float(m.group(2)), n) if m else (float("nan"), float("nan"), n)


def cmd_cer(a) -> int:
    rows = json.loads(Path(a.results).read_text())
    want = None if not a.targets else set(a.targets.split(","))
    out = []
    for r in rows:
        if want and r["target"] not in want:
            continue
        taps = Path(a.out_dir) / f"taps_{r['target']}_b{r['beta']}_s{r['seed']}.jsonl"
        cer, wer, n = cer_of(taps, a.test, a.pos_model, a.spatial_model)
        out.append({**{k: r[k] for k in ("target", "beta", "seed")}, "cer": cer, "wer": wer,
                    "taps": n, "recall": r["test"]["recall"], "precision": r["test"]["precision"]})
        print(f"{r['target']:<12} b={r['beta']} s={r['seed']} taps={n:<4} "
              f"R={100*r['test']['recall']:5.1f} P={100*r['test']['precision']:5.1f} "
              f"CER={cer:.3f} WER={wer:.3f}", flush=True)
        Path(a.out_dir, "cer.json").write_text(json.dumps(out, indent=1))
    for key in sorted({(r["target"], r["beta"]) for r in out}):
        v = np.array([r["cer"] for r in out if (r["target"], r["beta"]) == key])
        print(f"{key[0]:<12} beta={key[1]}  CER mean={v.mean():.3f} median={np.median(v):.3f} "
              f"IQR=[{np.percentile(v,25):.3f}-{np.percentile(v,75):.3f}]")
    return 0


def cmd_report(a) -> int:
    print(summarise(json.loads(Path(a.results).read_text())))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.taps_soft", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--train", nargs="+", default=list(TRAIN_SESSIONS))
    s.add_argument("--test", default=HELDOUT_SESSION)
    s.add_argument("--seeds", type=int, default=5)
    s.add_argument("--targets", default=None, help="comma-separated subset of target names")
    s.add_argument("--out-dir", default="/tmp/taps_soft")
    s.set_defaults(func=cmd_sweep)
    c = sub.add_parser("cer")
    c.add_argument("results")
    c.add_argument("--out-dir", default="/tmp/taps_soft")
    c.add_argument("--test", default=HELDOUT_SESSION)
    c.add_argument("--targets", default=None)
    c.add_argument("--pos-model", default="models/tap_pos_no015948.pkl")
    c.add_argument("--spatial-model", default="models/spatial_gauss_no015948.npz")
    c.set_defaults(func=cmd_cer)
    r = sub.add_parser("report")
    r.add_argument("results")
    r.set_defaults(func=cmd_report)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
