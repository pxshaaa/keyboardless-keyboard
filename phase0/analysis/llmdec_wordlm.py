"""Task 3b: Qwen-0.5B word decoder with a personal word LM interpolated into the Qwen word scores (CPU, tuning sets only).
score(w | ctx) = log( (1-a) * P_qwen(w | ctx) + a * P_personal_bigram(w | prev) ), personal bigram = abs-discount D=0.75
from .cache/llmdec/personal/vocab.json (keys source excluded, 4-gram decontaminated). Lexicon: generic or generic+personal."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np


class InterpScorer:
    def __init__(self, base, voc: dict, alpha: float):
        self.base, self.alpha = base, alpha
        self.uni = voc["unigram"]
        self.N = float(sum(self.uni.values()))
        self.bi, self.ctx = voc["bigram"], voc["ctx"]

    def p_pers(self, prev: str, w: str) -> float:
        pu = (self.uni.get(w, 0) + 0.1) / (self.N + 0.1 * 50000)
        c = self.ctx.get(prev) if prev else None
        if not c:
            return pu
        return max(self.bi.get(f"{prev} {w}", 0) - 0.75, 0) / c[0] + 0.75 * c[1] / c[0] * pu

    def score(self, pairs):
        base = self.base.score(pairs)
        out = []
        for (ws, w), b in zip(pairs, base):
            if w == "\n":
                out.append(b)
                continue
            p = (1 - self.alpha) * math.exp(b) + self.alpha * self.p_pers(ws[-1] if ws else "", w)
            out.append(math.log(max(p, 1e-30)))
        return out


def main() -> int:
    from phase0.analysis import autocorrect as AC
    from phase0.analysis import seqctc2 as S2
    from phase0.analysis.llmdec import OUT, RES, TUNE_QWEN, norm, wer_parts
    sets = sys.argv[1].split(",") if len(sys.argv) > 1 else ["old", "kbd"]
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    base = AC.NLM(".cache/decipher/qwen2.5-0.5b", device="cpu")
    voc = json.loads((OUT / "personal" / "vocab.json").read_text())
    S2._SC["qwen"] = InterpScorer(base, voc, alpha)
    cfg = json.loads(TUNE_QWEN.read_text())["best"]
    gen = S2.lexicon()
    out_f = RES / "wordlm_personal.json"
    res = json.loads(out_f.read_text()) if out_f.exists() else {}
    key = f"personal_bigram_alpha{alpha}"
    for s in sets:
        d = json.loads((OUT / "sets" / f"{s}.json").read_text())
        z = np.load(OUT / "sets" / f"{s}_lp.npz")
        if "wer" in res.get(s, {}).get(key, {}):
            continue
        S2._LEX["l"] = gen
        t0 = time.time()
        hyps = [norm(S2.run_decoder({"kind": "qwen", "cfg": cfg}, z[f"{it['id']}__zs"])) for it in d["items"]]
        we, wn, ce, cn = wer_parts(s, d, hyps)
        res.setdefault(s, {})[key] = {"wer": we / wn, "cer": ce / cn, "s_per_item": (time.time() - t0) / len(hyps), "hyps": hyps,
                                      "lexicon": "generic", "decoder_cfg": cfg}
        print(s, key, round(we / wn, 3), round(ce / cn, 3), flush=True)
        out_f.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
