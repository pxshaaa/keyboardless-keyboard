"""Task 3a: Qwen-0.5B closed-lexicon word decoder with generic vs generic+personal lexicon (CPU, tuning sets only).
Personal words = .cache/llmdec/personal/vocab.json (no 'keys' source, 4-gram decontaminated), count >= 2."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    from phase0.analysis import autocorrect as AC
    from phase0.analysis import seqctc2 as S2
    from phase0.analysis.llmdec import OUT, RES, TUNE_QWEN, norm, wer_parts
    sets = sys.argv[1].split(",") if len(sys.argv) > 1 else ["old", "kbd"]
    qpath = next(p for p in (Path(".cache/seqctc2/qwen2.5-0.5b"), Path(".cache/decipher/qwen2.5-0.5b")) if p.exists())
    S2._SC["qwen"] = AC.NLM(str(qpath), device="cpu")
    cfg = json.loads(TUNE_QWEN.read_text())["best"]
    gen = S2.lexicon()
    voc = json.loads((OUT / "personal" / "vocab.json").read_text())["unigram"]
    N = sum(voc.values())
    B = S2.blocklist()
    merged = S2.Lexicon.__new__(S2.Lexicon)
    merged.logp = dict(gen.logp)
    added = 0
    for w, c in voc.items():
        if c >= 2 and w.isalpha() and w.isascii() and w not in merged.logp and not S2.is_blocked(w, B) and len(w) <= 20:
            merged.logp[w] = math.log(0.2 * c / N)
            added += 1
    merged.n_blocked = gen.n_blocked
    merged.la = {"": max(merged.logp.values())}
    for w, lp in merged.logp.items():
        for i in range(1, len(w) + 1):
            p = w[:i]
            if lp > merged.la.get(p, -np.inf):
                merged.la[p] = lp
    out_f = RES / "wordlex_personal.json"
    res = json.loads(out_f.read_text()) if out_f.exists() else {}
    res["added_personal_words"] = added
    for s in sets:
        d = json.loads((OUT / "sets" / f"{s}.json").read_text())
        z = np.load(OUT / "sets" / f"{s}_lp.npz")
        for name, lex in (("generic", gen), ("generic+personal", merged)):
            if "wer" in res.get(s, {}).get(name, {}):
                continue
            S2._LEX["l"] = lex
            t0 = time.time()
            hyps = [norm(S2.run_decoder({"kind": "qwen", "cfg": cfg}, z[f"{it['id']}__zs"])) for it in d["items"]]
            we, wn, ce, cn = wer_parts(s, d, hyps)
            res.setdefault(s, {})[name] = {"wer": we / wn, "cer": ce / cn, "s_per_item": (time.time() - t0) / len(hyps),
                                           "hyps": hyps}
            print(s, name, round(we / wn, 3), round(ce / cn, 3), f"{(time.time() - t0) / len(hyps):.1f}s/item", flush=True)
            out_f.write_text(json.dumps(res, indent=1))
    S2._LEX["l"] = gen
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
