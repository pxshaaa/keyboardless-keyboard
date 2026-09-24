"""Task 4 gate: swipe_p3 realistic fusion (label-free CTC peaks) with PREDICTED finger posteriors and PREDICTED counts.
score = ctc + 0.5 prior + mu * tpl + lam * log P(len | span); weights chosen leave-one-set-out. Controls: const tap
log-lik, shuffled q (finger info destroyed), shuffled tips+q (swipe_p3 shuffled fingertips), duration-only count.
  PYTHONPATH=. .venv/bin/python -m phase0.analysis.finger_fusion [--tag v0] [--off 0]"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from phase0.analysis import swipe_common as C
from phase0.analysis import swipe_motor as SM
from phase0.analysis import swipe_p3 as P3
from phase0.analysis import finger_truth as FT
from phase0.analysis.finger_kbdswipe import gauss_tbl

CACHE = Path(".cache/finger")
SETS = ("old", "b1", "b2")
MUS = (0.0, 0.1, 0.25, 0.5, 1.0)
LAMS = (0.0, 0.5, 1.0, 2.0)
COUNTS = ("none", "kin/gbm", "kin+dur/gbm", "dur/lin", "ctc_npeaks/lin", "ctc+dur/lin", "all/lin")


def truth_keymodel(ref):
    TP = dict(np.load(CACHE / "kbd_tips.npz"))
    Xs, Ks, Fs = [], [], []
    for s in FT.KBD:
        z = np.load(CACHE / f"truth_{s}.npz")
        T = SM.apply_sim(TP[f"{s}|tips"].reshape(-1, 2), SM.to_frame(TP[f"{s}|tmpl"], ref)).reshape(-1, 10, 2)
        m = z["ok"]
        Xs.append(T[m, z["finger"][m]]); Ks.append(z["keys"][m]); Fs.append(z["finger"][m])
    return SM.KeyModel(np.concatenate(Xs), np.concatenate(Ks), np.concatenate(Fs))


def tables(tips, q, km):
    G = gauss_tbl(tips, km)
    maj = km.logpf.argmax(1)
    n = len(G)
    lq = np.log(np.clip(q, 1e-4, 1))
    mixq = logsumexp(lq[:, :, None] + km.logpf.T[None] + G, axis=1)
    return {"hyp": G[np.arange(n)[:, None], maj[None, :], np.arange(26)[None, :]],
            "mix[unif]": logsumexp(np.log(0.1) + km.logpf.T[None] + G, axis=1),
            "mix[q]": mixq, "const": np.full_like(mixq, float(np.median(mixq))),
            "fingeronly[q]": logsumexp(lq[:, :, None] + km.logpf.T[None], axis=1)}


def run_set(name, tag, off, lex, rng):
    segs, words, km_kbd = P3.prepare(name)
    mine = pickle.loads((CACHE / f"desk_{name}.pkl").read_bytes())
    assert [r["w"] for r in words] == [r["w"] for r in mine], name
    Q = pickle.loads((CACHE / f"deskq_{name}_{tag}.pkl").read_bytes())
    cache = dict(np.load(C.CACHE / f"p3_ctc_{name}.npz"))
    CT = pickle.loads((CACHE / "count_tables.pkl").read_bytes())
    cix = {(s, wi): i for i, (s, wi, _) in enumerate(CT["rows"])}
    F = dict(np.load(C.CACHE / "p2_feats.npz"))
    ref = SM.canonical([F[f"{k}|tmpl"] for k in P3.KBD])
    km_t = truth_keymodel(ref)
    maj = km_t.logpf.argmax(1)
    out = []
    for wi, r in enumerate(words):
        m = len(r["w"])
        lens = list(range(max(1, m - 2), m + 3))
        if r["w"] not in lex.index:
            continue
        lex_ix = np.concatenate([lex.by_len.get(k, np.zeros(0, int)) for k in lens])
        tgt = int(np.where(lex_ix == lex.index[r["w"]])[0][0])
        ctc = cache[f"{r['id']}|{wi}"]
        base = ctc + 0.5 * lex.prior[lex_ix]
        L = lex.len[lex_ix]
        others = [x for x in words if x["seg"] != r["seg"]]
        Xo = np.concatenate([x["tips_aligned"][np.arange(len(x["keys"])), maj[x["keys"]]] for x in others])
        Ko = np.concatenate([x["keys"] for x in others])
        km = SM.KeyModel(Xo, Ko, maj[Ko], mean_prior=km_t.mu, tau_mean=20.0)
        km.logpf = km_t.logpf
        tips = r["tips_peaks"]
        q = Q[wi][off]["peaks"]
        row = {"set": name, "wi": wi, "seg": r["seg"], "w": r["w"], "n_peaks": int(len(tips)), "scores": {}}
        variants = {}
        if len(tips):
            variants.update(tables(tips, q, km))
            same = [j for j, x in enumerate(words) if x["seg"] != r["seg"] and len(x["tips_peaks"]) == len(tips)]
            if same:
                j = same[rng.integers(len(same))]
                variants["mix[q|shuftips]"] = tables(words[j]["tips_peaks"], Q[j][off]["peaks"], km)["mix[q]"]
                variants["mix[shufq]"] = tables(tips, Q[j][off]["peaks"], km)["mix[q]"]
            else:
                pool = np.concatenate([Q[j][off]["peaks"] for j in range(len(words)) if words[j]["seg"] != r["seg"] and len(Q[j][off]["peaks"])])
                qs = pool[rng.integers(len(pool), size=len(tips))]
                variants["mix[shufq]"] = tables(tips, qs, km)["mix[q]"]
                variants["mix[q|shuftips]"] = variants["mix[shufq]"]
        for vn in ("hyp", "mix[unif]", "mix[q]", "const", "fingeronly[q]", "mix[q|shuftips]", "mix[shufq]"):
            if vn in variants:
                sc = SM.score_words(variants[vn], None, lex, None, lens, p_ins=0.1, p_del=0.1, log_bg=km.log_bg)
                tpl = np.concatenate([sc[k][0] for k in lens if k in sc])
            else:
                tpl = np.zeros(len(lex_ix))
            tpl = tpl - tpl.max()
            for cn in COUNTS:
                cnt = np.zeros(len(lex_ix)) if cn == "none" else CT["tables"][cn][cix[(name, r["wi"] if "wi" in r else wi)]][np.clip(L, 1, 16) - 1]
                for mu in MUS:
                    for lam in (LAMS if cn != "none" else (0.0,)):
                        if mu == 0 and vn != "hyp":
                            continue
                        v = base + mu * tpl + lam * cnt
                        rk = int((v > v[tgt]).sum()) + 1
                        row["scores"][f"{vn if mu else 'notpl'}|{cn}|{mu}|{lam}"] = rk
        out.append(row)
    print(name, "scored", len(out), flush=True)
    return out


def arm_of(key):
    vn, cn, mu, lam = key.rsplit("|", 3)
    return f"{vn}|{cn}"


def loso_select(rows):
    """per arm: pick (mu,lam) by top-1 on the other two sets, apply to held set -> per-word hit arrays."""
    keys = list(rows[0]["scores"])
    arms = sorted({arm_of(k) for k in keys})
    hits, chosen = {}, {}
    sets = np.array([r["set"] for r in rows])
    for arm in arms:
        ks = [k for k in keys if arm_of(k) == arm]
        H1 = np.array([[r["scores"][k] == 1 for k in ks] for r in rows], float)
        H5 = np.array([[r["scores"][k] <= 5 for k in ks] for r in rows], float)
        h1, h5 = np.zeros(len(rows)), np.zeros(len(rows))
        for s in SETS:
            tr, te = sets != s, sets == s
            b = int(np.argmax(H1[tr].mean(0) + 1e-3 * H5[tr].mean(0)))
            h1[te], h5[te] = H1[te, b], H5[te, b]
            chosen.setdefault(arm, {})[s] = ks[b]
        hits[arm] = (h1, h5)
    return hits, chosen


def boot(a, b, seg_ids, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = b - a
    wi = rng.integers(0, len(d), (n, len(d)))
    useg = np.unique(seg_ids)
    idx = {u: np.where(seg_ids == u)[0] for u in useg}
    cl = []
    for _ in range(2000):
        pick = rng.choice(useg, len(useg))
        ii = np.concatenate([idx[u] for u in pick])
        cl.append(d[ii].mean())
    return {"d": float(d.mean()), "ci_word": [float(np.percentile(d[wi].mean(1), 2.5)), float(np.percentile(d[wi].mean(1), 97.5))],
            "ci_segment": [float(np.percentile(cl, 2.5)), float(np.percentile(cl, 97.5))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--off", type=int, default=0)
    a = ap.parse_args()
    P3._init()
    lex = P3._G["lex"]
    rng = np.random.default_rng(0)
    rows = []
    for name in SETS:
        rows += run_set(name, a.tag, a.off, lex, rng)
    (CACHE / f"fusion_rows_{a.tag}_off{a.off}.pkl").write_bytes(pickle.dumps(rows))
    hits, chosen = loso_select(rows)
    base = hits["notpl|none"][0]
    seg_ids = np.array([f"{r['set']}:{r['seg']}" for r in rows])
    sets = np.array([r["set"] for r in rows])
    summ = {"n_words": len(rows), "arms": {}}
    for arm, (h1, h5) in sorted(hits.items()):
        e = {"top1": float(h1.mean()), "top5": float(h5.mean()),
             "per_set_top1": {s: float(h1[sets == s].mean()) for s in SETS},
             "vs_ctc+prior": boot(base, h1, seg_ids), "chosen": chosen[arm]}
        summ["arms"][arm] = e
        print(f"{arm:<30} top1 {e['top1']:.3f} top5 {e['top5']:.3f} d {e['vs_ctc+prior']['d']:+.3f} "
              f"word{np.round(e['vs_ctc+prior']['ci_word'], 3)} seg{np.round(e['vs_ctc+prior']['ci_segment'], 3)} "
              + " ".join(f"{s}:{v:.2f}" for s, v in e["per_set_top1"].items()), flush=True)
    # fixed-weight table (not selected), for transparency
    fixed = {}
    for k in rows[0]["scores"]:
        fixed[k] = float(np.mean([r["scores"][k] == 1 for r in rows]))
    summ["fixed_weight_top1"] = fixed
    Path(f"results/finger/t4_fusion_{a.tag}_off{a.off}.json").write_text(json.dumps({"summary": summ, "per_word": rows}, indent=1))


if __name__ == "__main__":
    main()
