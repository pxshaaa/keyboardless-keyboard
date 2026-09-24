"""Task 3: letters per CTC word span on desk. Leave-one-set-out (old/b1/b2), label = truth word length.
Arms: CTC peaks (P3 baseline), duration only, taps_gb detections, kinematics only, kinematics+duration, all.
Saves per-word log P(m) over m=1..16 (Gaussian on OOF residuals) for fusion.
  PYTHONPATH=. .venv/bin/python -m phase0.analysis.finger_count"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from phase0.analysis import swipe_common as C

CACHE = Path(".cache/finger")
SETS = ("old", "b1", "b2")
MMAX = 16


def load():
    rows = []
    for name in SETS:
        src = C.SETS[name][2]
        gb = np.array([json.loads(l)["t"] for l in (CACHE / f"taps_gb_{src}.jsonl").read_text().splitlines()])
        for r in pickle.loads((CACHE / f"desk_{name}.pkl").read_bytes()):
            f = dict(r["count_feats"])
            t0, t1 = r["span_t"]
            f["gb_n"] = float(((gb >= t0) & (gb <= t1)).sum())
            rows.append({"set": name, "wi": r["wi"], "w": r["w"], "m": len(r["w"]), "f": f})
    return rows


ARMS = {
    "ctc_npeaks": lambda k: k == "ctc_npeaks",
    "dur": lambda k: k == "dur",
    "gb_taps": lambda k: k == "gb_n",
    "kin": lambda k: not k.startswith("ctc") and k not in ("dur", "gb_n") and "rate" not in k,
    "kin+dur": lambda k: not k.startswith("ctc") and k != "gb_n",
    "ctc+dur": lambda k: k in ("ctc_npeaks", "ctc_let_mass", "dur"),
    "all": lambda k: True,
}


def fit_predict(Xtr, ytr, Xte, kind):
    if Xtr.shape[1] <= 3 or kind == "lin":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    else:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=10,
                              colsample_bytree=0.5, n_jobs=2, verbose=-1)
    m.fit(Xtr, ytr)
    return m.predict(Xte), m.predict(Xtr)


def main():
    rows = load()
    keys = sorted(rows[0]["f"])
    X = np.array([[r["f"][k] for k in keys] for r in rows])
    y = np.array([r["m"] for r in rows], float)
    sets = np.array([r["set"] for r in rows])
    res, tables = {}, {}
    for arm, sel in ARMS.items():
        cols = [i for i, k in enumerate(keys) if sel(k)]
        for kind in (("lin", "gbm") if len(cols) > 3 else ("lin",)):
            name = f"{arm}/{kind}"
            pred = np.zeros(len(y))
            sig = np.zeros(len(y))
            for s in SETS:
                te, tr = sets == s, sets != s
                p, ptr = fit_predict(X[tr][:, cols], y[tr], X[te][:, cols], kind)
                pred[te] = p
                sig[te] = max(np.std(y[tr] - ptr), 0.5)
            rnd = np.clip(np.rint(pred), 1, MMAX)
            r = {s: {"exact": float((rnd[sets == s] == y[sets == s]).mean()), "pm1": float((np.abs(rnd - y)[sets == s] <= 1).mean())} for s in SETS}
            r["pooled"] = {"exact": float((rnd == y).mean()), "pm1": float((np.abs(rnd - y) <= 1).mean()),
                           "mae": float(np.abs(pred - y).mean())}
            res[name] = r
            mm = np.arange(1, MMAX + 1)
            lp = -0.5 * ((mm[None] - pred[:, None]) / sig[:, None]) ** 2
            lp -= np.log(np.exp(lp).sum(1, keepdims=True))
            tables[name] = lp
            print(f"{name:<16} " + "  ".join(f"{s}:{v['exact']:.2f}/{v['pm1']:.2f}" for s, v in r.items() if s != "pooled")
                  + f"  pooled exact {r['pooled']['exact']:.3f} pm1 {r['pooled']['pm1']:.3f} mae {r['pooled']['mae']:.2f}", flush=True)
    # raw detector counts, no model
    for raw in ("ctc_npeaks", "gb_n"):
        v = X[:, keys.index(raw)]
        res[f"raw_{raw}"] = {"pooled": {"exact": float((v == y).mean()), "pm1": float((np.abs(v - y) <= 1).mean())}}
        print(f"raw {raw:<12} exact {(v == y).mean():.3f} pm1 {(np.abs(v - y) <= 1).mean():.3f}")
    mean_len = {s: float(y[sets != s].mean()) for s in SETS}
    const = np.array([np.rint(mean_len[s]) for s in sets])
    res["const_mean_len"] = {"pooled": {"exact": float((const == y).mean()), "pm1": float((np.abs(const - y) <= 1).mean())}}
    print("const mean len", res["const_mean_len"])
    Path("results/finger/t3_count.json").write_text(json.dumps(res, indent=1))
    (CACHE / "count_tables.pkl").write_bytes(pickle.dumps({"rows": [(r["set"], r["wi"], r["w"]) for r in rows], "tables": tables}))


if __name__ == "__main__":
    main()
