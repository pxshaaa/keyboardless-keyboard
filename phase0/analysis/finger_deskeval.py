"""Task 2 desk: kbd-trained finger classifier applied at desk CTC anchors. No desk finger truth -> agreement of the
predicted finger with the typist's measured majority finger for the TRUE letter (truth forced alignment anchors);
kbd OOF gives the same metric where finger truth exists. Caches per-word q tables for fusion.
  PYTHONPATH=. .venv/bin/python -m phase0.analysis.finger_deskeval [--tag v0]"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from phase0.analysis import finger_cls as FC
from phase0.analysis import finger_desk as FD
from phase0.analysis import finger_truth as FT
from phase0.analysis import swipe_motor as SM

CACHE = Path(".cache/finger")


def agree(q, letters, maj, prior):
    a = q.argmax(1)
    tgt = maj[letters]
    return {"finger_agree": float((a == tgt).mean()), "hand_agree": float(((a >= 5) == (tgt >= 5)).mean()),
            "chance_agree": float(np.sum(np.bincount(a, minlength=10) / len(a) * np.bincount(tgt, minlength=10) / len(a))),
            "nats_vs_prior": float(np.mean(np.log(np.clip(q[np.arange(len(q)), tgt], 1e-4, 1)) - np.log(prior[tgt]))),
            "n": int(len(q))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v0")
    a = ap.parse_args()
    cnt = np.load(CACHE / "finger_per_key_counts.npy")
    maj = cnt.argmax(1)
    D = FC.load_all()
    Xs, ys = [], []
    for d in D.values():
        for j in FC.JIT:
            m = d["tok"] & d[f"ok{j}"]
            Xs.append(d[f"X{j}"][m]); ys.append(d["y"][m])
    y = np.concatenate(ys)
    prior = np.bincount(y, minlength=10) / len(y)
    clf = FC.model()
    clf.fit(np.concatenate(Xs), y)
    (CACHE / f"cls_all_{a.tag}.pkl").write_bytes(pickle.dumps(clf))
    rep = {}
    O = dict(np.load(CACHE / "cls_oof_jit0.npz"))
    qk, lk = [], []
    for sid in FT.KBD:
        z = np.load(CACHE / f"truth_{sid}.npz")
        qk.append(O[f"{sid}|p"]); lk.append(z["keys"][O[f"{sid}|idx"]])
    rep["kbd_oof_jit0"] = agree(np.concatenate(qk), np.concatenate(lk), maj, prior)
    for name in ("old", "b1", "b2"):
        words = pickle.loads((CACHE / f"desk_{name}.pkl").read_bytes())
        out = []
        for o in FD.OFFS:
            qa, la = [], []
            for r in words:
                rec = {}
                for kind in ("aligned", "peaks"):
                    X = r.get(f"X_{kind}{o}")
                    if X is None or not len(X):
                        rec[kind] = np.zeros((0, 10))
                        continue
                    q = np.full((len(X), 10), 0.1)
                    ok = r[f"ok_{kind}{o}"]
                    if ok.any():
                        p = np.zeros((ok.sum(), 10))
                        p[:, clf.classes_] = clf.predict_proba(X[ok])
                        q[ok] = p
                    rec[kind] = q
                    if kind == "aligned":
                        qa.append(q[ok]); la.append(np.array([SM.LI[c] for c in r["w"]])[ok])
                if o == FD.OFFS[0]:
                    out.append({})
                out[words.index(r)][o] = rec
            rep[f"{name}/aligned{o}"] = agree(np.concatenate(qa), np.concatenate(la), maj, prior)
        (CACHE / f"deskq_{name}_{a.tag}.pkl").write_bytes(pickle.dumps(out, protocol=4))
    for k, v in rep.items():
        print(k, json.dumps({kk: round(vv, 3) for kk, vv in v.items()}), flush=True)
    Path(f"results/finger/t2_desk_agreement_{a.tag}.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
