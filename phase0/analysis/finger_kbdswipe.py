"""Kbd intermediate for task 2/4: swipe word scorer with TRUE tap positions/counts but PREDICTED finger posteriors
(LOSO OOF). Does q(f) beat a uniform-finger mixture? Channels: true finger (bound), majority-map finger (P3 'hyp'),
mix[q], mix[uniform] (control), finger-only[q].
  PYTHONPATH=. .venv/bin/python -m phase0.analysis.finger_kbdswipe {prep|eval} [--oof FILE]"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from phase0.analysis import swipe_common as C
from phase0.analysis import swipe_motor as SM
from phase0.analysis import finger_truth as FT

CACHE = Path(".cache/finger")


def cmd_prep():
    out = {}
    for sid in FT.KBD:
        z = np.load(CACHE / f"truth_{sid}.npz")
        S = FT.sess(sid)
        out[f"{sid}|tips"] = FT.tips_xy(S.P, z["rows"])
        out[f"{sid}|tmpl"] = SM.template(S.P, z["rows"])
        del S
    np.savez_compressed(CACHE / "kbd_tips.npz", **out)


def gauss_tbl(T, km):
    """T [n,10,2] -> [n,10,26] log N(tip_f ; mu_k, cov_k)."""
    d = T[:, :, None, :] - km.mu[None, None]
    q = np.einsum("nfki,kij,nfkj->nfk", d, km.icov, d)
    ll = -0.5 * q - 0.5 * km.logdet[None, None] - np.log(2 * np.pi)
    return np.where(np.isfinite(ll), ll, -50.0)


_G: dict = {}


def _init(oof):
    from phase0.analysis import seqctc2 as S2
    B = S2.blocklist()
    _G["lex"] = SM.Lexicon({w: v for w, v in C.merged_lex().items() if not S2.is_blocked(w, B)})
    _G["tips"] = dict(np.load(CACHE / "kbd_tips.npz"))
    _G["p2"] = dict(np.load(C.CACHE / "p2_feats.npz"))
    _G["oof"] = dict(np.load(oof))


def channels(G, q, km, ft=None):
    """-> dict name -> [n,26] tap log-lik tables."""
    lq = np.log(np.clip(q, 1e-4, 1))
    maj = km.logpf.argmax(1)
    n = len(G)
    ch = {"majmap": G[np.arange(n)[:, None], maj[None, :], np.arange(26)[None, :]],
          "mix[q]": logsumexp(lq[:, :, None] + km.logpf.T[None] + G, axis=1),
          "mix[unif]": logsumexp(np.log(0.1) + km.logpf.T[None] + G, axis=1),
          "fingeronly[q]": logsumexp(lq[:, :, None] + km.logpf.T[None], axis=1),
          "argmax[q]": G[np.arange(n), q.argmax(1)] + km.logpf[:, q.argmax(1)].T}
    if ft is not None:
        ch["true"] = G[np.arange(n), ft] + km.logpf[:, ft].T
    return ch


def fold(held):
    lex, TP, F, O = _G["lex"], _G["tips"], _G["p2"], _G["oof"]
    ref = SM.canonical([TP[f"{s}|tmpl"] for s in FT.KBD if s != held])
    Xs, Ks, Fs = [], [], []
    for s in FT.KBD:
        if s == held:
            continue
        z = np.load(CACHE / f"truth_{s}.npz")
        T = SM.apply_sim(TP[f"{s}|tips"].reshape(-1, 2), SM.to_frame(TP[f"{s}|tmpl"], ref)).reshape(-1, 10, 2)
        m = z["ok"]
        Xs.append(T[m, z["finger"][m]]); Ks.append(z["keys"][m]); Fs.append(z["finger"][m])
    km = SM.KeyModel(np.concatenate(Xs), np.concatenate(Ks), np.concatenate(Fs))
    z = np.load(CACHE / f"truth_{held}.npz")
    T = SM.apply_sim(TP[f"{held}|tips"].reshape(-1, 2), SM.to_frame(TP[f"{held}|tmpl"], ref)).reshape(-1, 10, 2)
    n = len(T)
    assert len(F[f"{held}|key"]) == n, (held, len(F[f"{held}|key"]), n)
    q = np.full((n, 10), 0.1)
    q[O[f"{held}|idx"]] = O[f"{held}|p"]
    hasq = np.zeros(n, bool); hasq[O[f"{held}|idx"]] = True
    ok = z["ok"] & hasq & np.isfinite(T).all((1, 2))
    CH = channels(gauss_tbl(T, km), q, km, z["finger"])
    rows = []
    for w, st, ln in zip(F[f"{held}|wtext"], F[f"{held}|wstart"], F[f"{held}|wlen"]):
        w = str(w)
        ix = np.arange(st, st + ln)
        if w not in lex.index or not ok[ix].all():
            continue
        lens = range(max(1, ln - 2), min(16, ln + 2) + 1)
        r = {"w": w, "n": int(ln)}
        for nm, ll in CH.items():
            sc = SM.score_words(ll[ix], None, lex, None, lens, p_ins=0.02, p_del=0.02, log_bg=km.log_bg)
            for wp in (0.0, 0.5):
                r[f"{nm}+{wp}prior"] = SM.rank_of(lex.index[w], lex, sc, 1.0, 0.0, wp)[0]
        rows.append(r)
    return held, rows


def cmd_eval(oof):
    with Pool(2, initializer=_init, initargs=(oof,)) as pool:
        res = dict(pool.map(fold, FT.KBD))
    R = [r for v in res.values() for r in v]
    keys = [k for k in R[0] if k not in ("w", "n")]
    summ = {"n_words": len(R)}
    base = np.array([r["mix[unif]+0.5prior"] == 1 for r in R], float)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(R), (5000, len(R)))
    for k in keys:
        t1 = np.array([r[k] == 1 for r in R], float)
        dd = (t1 - base)[idx].mean(1)
        summ[k] = {"top1": float(t1.mean()), "top5": float(np.mean([r[k] <= 5 for r in R])),
                   "d_top1_vs_mix[unif]+0.5prior": [float((t1 - base).mean()), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}
        print(f"{k:<28} top1 {summ[k]['top1']:.3f} top5 {summ[k]['top5']:.3f} d {np.round(summ[k]['d_top1_vs_mix[unif]+0.5prior'], 3)}", flush=True)
    tag = Path(oof).stem
    Path(f"results/finger/t2b_kbd_swipe_{tag}.json").write_text(json.dumps({"summary": summ, "per_word": res}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep", "eval"))
    ap.add_argument("--oof", default=str(CACHE / "cls_oof_jit0.npz"))
    a = ap.parse_args()
    cmd_prep() if a.cmd == "prep" else cmd_eval(a.oof)
