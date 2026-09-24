"""P4 sentence-level: P1 word-beam top-50 lattices injected into the frozen v3 generate-and-verify pool, LM-scored with
the frozen 8B + personal-LoRA 0.5B (mini, gpulock), re-selected with the FROZEN v3 weights (no re-tuning).
Sets are copied to s_swipe-<name> so no frozen cache is touched.
  prep:   PYTHONPATH=. .venv/bin/python -m phase0.analysis.swipe_p4 prep
  select: PYTHONPATH=. .venv/bin/python -m phase0.analysis.swipe_p4 select"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from phase0.analysis import swipe_common as C

L = Path(".cache/llmdec")
CAND_FILES = [("small_pad0.5", '{"alpha": 1.0, "beta": 2.0, "gamma": -8.0, "boost": 2.0}'),
              ("big_pad0.5", '{"alpha": 2.0, "beta": 4.0, "gamma": -4.0, "boost": 2.0}')]


def new(name):
    return f"s_swipe-{name}"


def prep():
    for name in ("old", "b1", "b2"):
        src, nw = C.SETS[name][0], new(name)
        for a, b in ((f"sets/{src}.json", f"sets/{nw}.json"), (f"sets/{src}_lp.npz", f"sets/{nw}_lp.npz"),
                     (f"llm/8b/{src}.json", f"llm/8b/{nw}.json"), (f"llm/p05b/{src}.json", f"llm/p05b/{nw}.json"),
                     (f"fuzzy/{src}.json", f"fuzzy/{nw}.json")):
            if not (L / b).exists():
                shutil.copy(L / a, L / b)
        base = json.loads((L / "fuzzy" / f"{src}.json").read_text())
        items = {k: {g: list(v) for g, v in d.items()} for k, d in base["items"].items()}
        added = 0
        for tag, cfg in CAND_FILES:
            cf = json.loads((C.CACHE / f"p1_cands_{tag}.json").read_text())[name]
            for sid, bycfg in cf.items():
                ext = bycfg[cfg]
                for gk in items.setdefault(sid, {}):
                    before = len(items[sid][gk])
                    items[sid][gk] = list(dict.fromkeys(items[sid][gk] + ext))
                    added += len(items[sid][gk]) - before
        (L / "fuzzy" / f"{nw}__swipe.json").write_text(json.dumps({"cfg": base.get("cfg"), "tag": "swipe", "items": items,
                                                                  "note": "v3 fuzzy + P1 word-beam top-50 (2 configs)"}))
        print(name, nw, "added texts (all group keys)", added)


def select_one(name, tag):
    os.environ["LLMDEC_FUZZY_TAG"] = tag
    import importlib
    from phase0.analysis import llmdec as LD
    LD._FZ.clear()
    run = json.loads(Path("results/llmdec/frozen_config_v3.json").read_text())["run"]
    d, pools, _ = LD.build_pools(new(name), ("8b", "p05b"))
    hyps, unscored, npool = [], 0, []
    for p in pools:
        P = p["P"][run["groups"]]
        unscored += int(np.isnan(P["lm"]["8b"]).sum() + np.isnan(P["lm"]["p05b"]).sum())
        npool.append(len(P["texts"]))
        hyps.append(LD.select(P, run["llm"], run["prompt"], run["K"], run["lam"], run["wb"], run["cb"], run.get("mu_lex", 0.0),
                              run.get("mu_pers", 0.0), run.get("nu", 0.0), run.get("lam_p", 0.0), run.get("plm", "p05b"),
                              run.get("use_fz", 0), run.get("kappa", 0.0), run.get("mu_tech", 0.0)))
    return hyps, pools, unscored, npool


def select(names=("old", "b1", "b2")):
    from phase0.analysis.decipher import score_variant
    from phase0.analysis.decode import edit_distance
    of = C.OUT / "p4_sentence.json"
    out = json.loads(of.read_text()) if of.exists() else {}
    for name in names:
        segs, lines = C.load(name)
        res = {}
        for tag in ("", "swipe"):
            hyps, pools, uns, npool = select_one(name, tag)
            sv = score_variant(lines, hyps)
            orc = sum(min(edit_distance(s["truth"].split(), t.split()) for t in p["P"]["ens"]["texts"]) for s, p in zip(segs, pools))
            nw = sum(len(s["truth"].split()) for s in segs)
            res[tag or "frozen_pool"] = {"wer": C.boot_ci(sv["wed"], sv["wn"]), "words_correct": float(sv["wok"].sum() / sv["wn"].sum()),
                                        "hyp_by_line": sv["hyp_by_line"], "wed": sv["wed"].tolist(), "wn": sv["wn"].tolist(),
                                        "pool_oracle_wer_segwise": orc / nw, "mean_pool": float(np.mean(npool)), "unscored_lm": uns}
            print(name, tag or "frozen_pool", f"WER {res[tag or 'frozen_pool']['wer']}", f"oracle {orc / nw:.3f}", f"pool {np.mean(npool):.0f}",
                  f"unscored {uns}", flush=True)
        res["delta_swipe_minus_frozen"] = C.boot_delta(res["frozen_pool"]["wed"], res["swipe"]["wed"], res["swipe"]["wn"])
        out[name] = res
    (C.OUT / "p4_sentence.json").write_text(json.dumps(out, indent=1, default=float))
    for name, r in ((n, out[n]) for n in names):
        print(name, "delta", r["delta_swipe_minus_frozen"])
        for l1, l2 in zip(r["frozen_pool"]["hyp_by_line"], r["swipe"]["hyp_by_line"]):
            if l1 != l2:
                print("   frozen:", l1, "\n   swipe: ", l2)


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep()
    else:
        select(tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("old", "b1", "b2"))
