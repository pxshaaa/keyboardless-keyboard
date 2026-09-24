"""P1 audit: for each segment where the word-beam top-1 != truth, is it a SEARCH error (truth scores higher under the
beam's own objective: CTC + alpha*bigram + beta*words + boost + gamma per missing space) or a MODEL error?"""
from __future__ import annotations

import json

import numpy as np

from phase0.analysis import swipe_common as C
from phase0.analysis import swipe_p1 as P


def main():
    lm = P.LM()
    out = {}
    for cfg in (dict(alpha=1.0, beta=2.0, gamma=-8.0, boost=2.0), dict(alpha=2.5, beta=5.0, gamma=-4.0, boost=2.0)):
        full = dict(cfg, prune=-14.0, beam=256, topk=50, xspace=-3.0)
        rows = []
        for name in ("old", "b1", "b2"):
            segs, _ = C.load(name)
            for s in segs:
                M = C.ens28(s["lps"])
                top = P.beam(M, lm, full)[:50]
                t1 = top[0][0] if top else ""
                oov = [w for w in s["truth"].split() if w not in lm.logp]
                ll = C.ctc_ens(s["lps"], [s["truth"], t1])
                st, s1 = ll[0] + lm.sent(s["truth"], full), ll[1] + lm.sent(t1, full)
                rows.append({"set": name, "id": s["id"], "correct": t1 == s["truth"], "truth_in_top50": s["truth"] in [t for t, _ in top],
                             "oov": oov, "score_truth": float(st), "score_top1": float(s1), "ctc_truth": float(ll[0]), "ctc_top1": float(ll[1]),
                             "error": None if t1 == s["truth"] else ("oov" if oov else "search" if st > s1 else "model")})
        summ = {n: {k: sum(1 for r in rows if r["set"] == n and r["error"] == k) for k in (None, "search", "model", "oov")}
                for n in ("old", "b1", "b2")}
        summ = {n: {str(k): v for k, v in d.items()} for n, d in summ.items()}
        for n in ("old", "b1", "b2"):
            summ[n]["truth_in_top50"] = sum(1 for r in rows if r["set"] == n and r["truth_in_top50"])
        out[json.dumps(cfg)] = {"summary": summ, "rows": rows}
        print(cfg, summ, flush=True)
    (C.OUT / "p1_search_vs_model.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
