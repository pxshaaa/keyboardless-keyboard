"""Score llmdec2 JSONL outputs (CPU, project venv): WER / CER / words-correct per (cfg, set), + objective.
  python -m phase0.analysis.llmdec2_score .cache/llmdec/dec2/3b.jsonl [...] --out results/llmdec/dec2_3b.json"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    from phase0.analysis.llmdec import OUT, wer_parts
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    recs = defaultdict(dict)
    secs = defaultdict(list)
    for f in a.files:
        for l in Path(f).read_text().splitlines():
            r = json.loads(l)
            recs[(r["cfg"], r["set"])][r["id"]] = r["hyp"]
            secs[(r["cfg"], r["set"])].append((r["secs"], r.get("llm_s", 0.0)))
    sets = {s: json.loads((OUT / "sets" / f"{s}.json").read_text()) for s in {k[1] for k in recs}}
    res = {}
    for (cfg, s), hy in sorted(recs.items()):
        d = sets[s]
        ids = [it["id"] for it in d["items"]]
        n_done = sum(i in hy for i in ids)
        if n_done < len(ids):
            res.setdefault(cfg, {})[s] = {"incomplete": f"{n_done}/{len(ids)}"}
            continue
        we, wn, ce, cn = wer_parts(s, d, [hy[i] for i in ids])
        sc = np.array(secs[(cfg, s)])
        res.setdefault(cfg, {})[s] = {"wer": we / wn, "cer": ce / cn, "s_per_item": float(sc[:, 0].mean()),
                                      "llm_s_per_item": float(sc[:, 1].mean())}
    for cfg, v in res.items():
        if all("wer" in v.get(s, {}) for s in ("kbd", "old")):
            v["obj"] = 0.5 * (v["kbd"]["wer"] + v["old"]["wer"])
    for cfg, v in sorted(res.items(), key=lambda kv: kv[1].get("obj", 9)):
        print(f"{cfg:<60} " + "  ".join(f"{s} WER {x['wer']:.3f} CER {x['cer']:.3f} {x['s_per_item']:.1f}s" if "wer" in x
                                        else f"{s} {x['incomplete']}" for s, x in sorted(v.items()) if isinstance(x, dict))
              + (f"  obj {v['obj']:.3f}" if "obj" in v else ""))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
