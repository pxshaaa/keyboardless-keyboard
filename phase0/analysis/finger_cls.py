"""Task 2: label-free finger classifier (which of 10 tips pressed) from landmark kinematics around an anchor frame.
Train on kbd with finger_truth labels, anchors jittered to mimic CTC emission timing; LOSO; then desk anchors.
  PYTHONPATH=. .venv/bin/python -m phase0.analysis.finger_cls {feats|loso}"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from phase0.analysis import finger_id as FI
from phase0.analysis import finger_truth as FT

CACHE = Path(".cache/finger")
JIT = (0, -3, 3, -6, 6)   # 60 Hz rows


def feats_at(hands, times):
    X, ok, names = [], [], None
    for t in times:
        g = FI.featurize_tap(hands, float(t))
        if g is None:
            X.append(None); ok.append(False)
        else:
            X.append(g[0]); ok.append(True)
            names = g[1]
    D = len(names) if names is not None else len(np.load(CACHE / f"clsX_{FT.KBD[0]}.npz")["names"])
    return np.array([x if x is not None else np.full(D, np.nan) for x in X], np.float32), np.array(ok), names


def cmd_feats():
    for sid in FT.KBD:
        f = CACHE / f"clsX_{sid}.npz"
        if f.exists():
            continue
        z = np.load(CACHE / f"truth_{sid}.npz")
        hands = FI.prep(Path("data/sessions") / sid)
        out = {}
        for j in JIT:
            X, ok, names = feats_at(hands, z["t"] + j / 60.0)
            out[f"X{j}"], out[f"ok{j}"] = X, ok
        np.savez(f, y=z["finger"], keys=z["keys"], tok=z["ok"], names=np.array(names), **out)
        print(sid, X.shape, flush=True)


def model(seed=0):
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                              colsample_bytree=0.3, subsample=0.8, subsample_freq=1, reg_lambda=1.0,
                              n_jobs=4, random_state=seed, verbose=-1)


def load_all():
    return {sid: dict(np.load(CACHE / f"clsX_{sid}.npz")) for sid in FT.KBD}


def cmd_loso():
    D = load_all()
    names = list(D[FT.KBD[0]]["names"])
    res = {}
    P_oof = {}
    for held in FT.KBD:
        Xs, ys = [], []
        for sid, d in D.items():
            if sid == held:
                continue
            for j in JIT:
                m = d["tok"] & d[f"ok{j}"]
                Xs.append(d[f"X{j}"][m]); ys.append(d["y"][m])
        clf = model()
        clf.fit(np.concatenate(Xs), np.concatenate(ys))
        d = D[held]
        r = {}
        for j in JIT:
            m = d["tok"] & d[f"ok{j}"]
            p = np.zeros((m.sum(), 10))
            p[:, clf.classes_] = clf.predict_proba(d[f"X{j}"][m])
            y = d["y"][m]
            r[f"jit{j}"] = {"top1": float((p.argmax(1) == y).mean()), "top2": float((np.argsort(-p, 1)[:, :2] == y[:, None]).any(1).mean()),
                            "hand": float(((p.argmax(1) >= 5) == (y >= 5)).mean()), "n": int(m.sum())}
            if j == 0:
                P_oof[held] = (np.where(m)[0], p)
        res[held] = r
        print(held, json.dumps(r), flush=True)
    pooled = {}
    for j in JIT:
        n = sum(res[h][f"jit{j}"]["n"] for h in FT.KBD)
        pooled[f"jit{j}"] = {k: sum(res[h][f"jit{j}"][k] * res[h][f"jit{j}"]["n"] for h in FT.KBD) / n for k in ("top1", "top2", "hand")}
    res["pooled"] = pooled
    np.savez(CACHE / "cls_oof_jit0.npz", **{f"{h}|idx": v[0] for h, v in P_oof.items()}, **{f"{h}|p": v[1] for h, v in P_oof.items()})
    Path("results/finger/t2_kbd_loso_v0.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(pooled, indent=1))


if __name__ == "__main__":
    {"feats": cmd_feats, "loso": cmd_loso}[sys.argv[1]]()
