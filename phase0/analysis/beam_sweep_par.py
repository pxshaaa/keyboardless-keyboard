"""Parallel remainder of the beam_oracle sweep (old desk only, 3 workers); appends rows to results/beam/sweep.json."""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

from phase0.analysis import beam_oracle as B

OUT = B.OUT
NS = [10, 25, 50, 100, 250, 500]
_S = {}


def _init():
    segs, lines, setname = B.load_desk("old")
    _S["old"] = {"segs": segs}
    B.get_lm("merged")


def _job(args):
    over, lx = args
    cfg = dict(B.cfgs_from("small"), **over)
    r = B._eval_cfg(_S, cfg, NS, lx, only=("old",))
    return r


def run(stage_cands, procs=3):
    with Pool(procs, initializer=_init) as p:
        return p.map(_job, stage_cands)


def main():
    res = json.loads((OUT / "sweep.json").read_text())
    done = {json.dumps([r["cfg"], r["lex"]], sort_keys=True) for r in res["rows"] if "old" in r["sets"]}
    best = max((r for r in res["rows"] if "old" in r["sets"]),
               key=lambda r: (r["sets"]["old"]["100"]["oracle_words"], -r["sets"]["old"]["100"]["oracle_wer"]))
    base = {k: best["cfg"][k] for k in ("alpha", "beta", "gamma", "boost", "beam", "prune")}
    lexname = best["lex"]
    stages = [("alpha_beta", [(dict(base, alpha=a, beta=b), lexname) for a in (1.5, 2.0, 3.0) for b in (0.0, 2.0, 4.0, 6.0)]),
              ("gamma_boost_prune", None), ("lexicon", None)]
    for name, cand in stages:
        if name == "gamma_boost_prune":
            cand = [(dict(base, gamma=g, boost=bo, prune=pr), lexname) for g in (-8.0, -6.0, -4.0, -2.0)
                    for bo in (0.0, 2.0, 4.0) for pr in (-8.0, -12.0)]
        elif name == "lexicon":
            cand = [(dict(base), lx) for lx in ("merged", "generic")]
        todo = [(o, lx) for o, lx in cand
                if json.dumps([dict(B.cfgs_from("small"), **o), lx], sort_keys=True) not in done]
        for r in run(todo):
            res["rows"].append(r)
            done.add(json.dumps([r["cfg"], r["lex"]], sort_keys=True))
        (OUT / "sweep.json").write_text(json.dumps(res, indent=1, default=float))
        pool = [r for r in res["rows"] if "old" in r["sets"]
                and any(json.dumps([dict(B.cfgs_from("small"), **o), lx], sort_keys=True)
                        == json.dumps([r["cfg"], r["lex"]], sort_keys=True) for o, lx in cand)]
        b = max(pool, key=lambda r: (r["sets"]["old"]["100"]["oracle_words"], -r["sets"]["old"]["100"]["oracle_wer"]))
        base = {k: b["cfg"][k] for k in base}
        lexname = b["lex"]
        res["stages"][name] = {"chosen": dict(base), "lex": lexname, "old_at_100": b["sets"]["old"]["100"]}
        print("stage", name, base, lexname, round(b["sets"]["old"]["100"]["oracle_words"], 3), flush=True)
    res["chosen"] = {"cfg": base, "lex": lexname}
    (OUT / "chosen.json").write_text(json.dumps({"cfg": base, "lex": lexname,
                                                 "selected_on": "old desk only, oracle_words @N=100"}, indent=1))
    sets = B.all_sets()
    res["chosen_all_sets"] = B._eval_cfg(sets, dict(B.cfgs_from("small"), **base), NS, lexname)
    res["default_small_all_sets"] = B._eval_cfg(sets, B.cfgs_from("small"), NS, "merged")
    (OUT / "sweep.json").write_text(json.dumps(res, indent=1, default=float))
    print("CHOSEN", base, lexname)
    for s, v in res["chosen_all_sets"]["sets"].items():
        print(" ", s, {N: round(v[N]["oracle_words"], 3) for N in ("10", "50", "100", "500")},
              "secs/seg", round(res["chosen_all_sets"]["secs"][s], 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
