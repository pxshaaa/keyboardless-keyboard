"""P3 leak control: 'aligned' observations carry the truth letter count. Re-run the fusion with
(shuf) fingertips of a random other word of the same length from another segment (keeps count, destroys position) and
(const) a constant tap log-lik table (pure count/length structure). LOPO MAP tau=20 key model, cached CTC scores.
Paired bootstrap of top-1 vs ctc+0.5prior."""
from __future__ import annotations

import json

import numpy as np

from phase0.analysis import swipe_common as C
from phase0.analysis import swipe_motor as SM
from phase0.analysis import swipe_p3 as P3


def main():
    P3._init()
    lex = P3._G["lex"]
    rng = np.random.default_rng(0)
    out = {}
    for name in ("old", "b1", "b2"):
        segs, words, km_kbd = P3.prepare(name)
        cache = dict(np.load(C.CACHE / f"p3_ctc_{name}.npz"))
        hits = {}
        for wi, r in enumerate(words):
            m = len(r["w"])
            lens = list(range(max(1, m - 2), m + 3))
            if r["w"] not in lex.index:
                continue
            lex_ix = np.concatenate([lex.by_len.get(k, np.zeros(0, int)) for k in lens])
            tgt = int(np.where(lex_ix == lex.index[r["w"]])[0][0])
            ctc = cache[f"{r['id']}|{wi}"]
            prior = lex.prior[lex_ix]
            others = [x for x in words if x["seg"] != r["seg"]]
            Xo = np.concatenate([x["tips_aligned"][np.arange(len(x["keys"])), P3.FK[x["keys"]]] for x in others])
            Ko = np.concatenate([x["keys"] for x in others])
            km = SM.KeyModel(Xo, Ko, P3.FK[Ko], mean_prior=km_kbd.mu, tau_mean=20.0)
            same = [x for x in others if len(x["w"]) == m and x["w"] != r["w"]]
            real = P3.ll_table(r["tips_aligned"], km)
            variants = {"real": real, "const": np.full_like(real, float(np.median(real)))}
            if same:
                variants["shuf"] = P3.ll_table(same[rng.integers(len(same))]["tips_aligned"], km)
            hits.setdefault("ctc+0.5prior", []).append(int((ctc + 0.5 * prior > (ctc + 0.5 * prior)[tgt]).sum()) + 1)
            for vn, ll in variants.items():
                sc = SM.score_words(ll, None, lex, None, lens, p_ins=0.1, p_del=0.1, log_bg=km.log_bg)
                loc = np.concatenate([sc[k][0] for k in lens if k in sc])
                for mu in (0.25, 1.0):
                    v = ctc + mu * loc + 0.5 * prior
                    hits.setdefault(f"ctc+{mu}tpl[{vn}]+0.5prior", []).append(int((v > v[tgt]).sum()) + 1)
                if vn == "shuf":
                    pass
            n = len(hits["ctc+0.5prior"])
            for k in list(hits):
                if len(hits[k]) < n:   # shuf missing for this word -> pad with real-less marker
                    hits[k].append(None)
        res = {}
        base = np.array([x == 1 for x in hits["ctc+0.5prior"]], float)
        for k, v in hits.items():
            ok = np.array([x is not None for x in v])
            t1 = np.array([x == 1 if x is not None else False for x in v], float)
            t5 = np.array([x is not None and x <= 5 for x in v], float)
            idx = np.random.default_rng(1).integers(0, ok.sum(), (5000, ok.sum()))
            dd = (t1[ok] - base[ok])[idx].mean(1)
            res[k] = {"top1": float(t1[ok].mean()), "top5": float(t5[ok].mean()), "n": int(ok.sum()),
                      "d_top1_vs_ctc+prior": [float((t1[ok] - base[ok]).mean()), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}
            print(name, f"{k:<32}", f"top1 {res[k]['top1']:.3f} top5 {res[k]['top5']:.3f} n={res[k]['n']} d {np.round(res[k]['d_top1_vs_ctc+prior'], 3)}", flush=True)
        out[name] = res
    (C.OUT / "p3_leak_control.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
