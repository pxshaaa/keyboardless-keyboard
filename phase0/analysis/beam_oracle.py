"""Oracle: is the 8B GENERATION stage redundant given a lexicon-constrained CTC word beam? See results/beam/SUMMARY.md.
`sweep` = word-beam N/hyperparameter sweep (tuned on old desk only); `oracle` = (a)/(b)/(c) table + current selector."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("LLMDEC_FUZZY_TAG", "")

L = Path(".cache/llmdec")
OUT = Path("results/beam")
LIVE = Path("data/live/20260915-010345")

# desk sets: the ctc_v4 leave-one-session-out posteriors used by .cache/ctc_v4/results/loso_v4.json
DESK = {
    "old": ("s_zz-v4loso-20260910-202149-desk", "zz-v4loso-20260910-202149-desk"),
    "b1": ("s_zz-v4loso-20260912-174542-desk", "zz-v4loso-20260912-174542-desk"),
    "b2": ("s_zz-v4loso-20260913-125733-desk", "zz-v4loso-20260913-125733-desk"),
}
LIVESET = "s_zz-live-20260915-010345"

# live truth: see results/beam/SUMMARY.md "live ground truth" for provenance of every line.
LIVE_TRUTH = {
    2: ("i will try this out now to talk as well", "high", "whisper 170.6-177.4s + manual note"),
    3: ("i will write to you what i want to write to you", "high", "whisper 376.3-385.2s + manual note"),
    4: ("if i talk slowly like i am doing right now i hope that this will still work", "high",
        "whisper 398.8-411.5s (manual note has an extra 'it': 'doing it right now')"),
    5: ("okay all campaigns are paused right now", "medium", "manual note only; whisper hallucinated this window"),
    6: ("did you fix it now", "low", "manual note, itself parenthesised/uncertain; no whisper support"),
    7: ("actually this is not working out", "high", "whisper 665.0-671.1s + manual note"),
}
LIVE_HIGH = [2, 3, 4, 5, 7]   # headline live set (k=6 uncertain, k=0/1 unknown)


def norm(s):
    from phase0.analysis.swipe_common import norm as n
    return n(s)


# ------------------------------------------------------------------ data
def load_desk(name):
    import phase0.analysis.swipe_common as C
    setname, sdir = DESK[name]
    d = json.loads((L / "sets" / f"{setname}.json").read_text())
    z = np.load(L / "sets" / f"{setname}_lp.npz")
    rec = json.loads(Path(f"data/sessions/{sdir}/decipher2_v3_recommended.json").read_text())
    hyps = {r["id"]: norm(r["hyp"]) for r in rec["records"]}
    lines = [norm(l) for l in Path(f"data/sessions/{sdir}/truth.txt").read_text().splitlines() if norm(l)]
    segs = [{"id": it["id"], "lps": [z[f"{it['id']}__{g}"] for g in it["c"]], "hyp": hyps[it["id"]], "c": it["c"]}
            for it in d["items"]]
    C.assign_truth(segs, lines)
    return segs, lines, setname


def load_live():
    d = json.loads((L / "sets" / f"{LIVESET}.json").read_text())
    z = np.load(L / "sets" / f"{LIVESET}_lp.npz")
    v4 = json.loads(Path("data/sessions/zz-live-20260915-010345/decipher2_v4.json").read_text())
    v4t = {s["id"]: norm(s["text"]) for s in v4["segments"]}
    segs = []
    for k, it in enumerate(d["items"]):
        if k not in LIVE_TRUTH:
            continue
        t, conf, src = LIVE_TRUTH[k]
        segs.append({"id": it["id"], "k": k, "lps": [z[f"{it['id']}__{g}"] for g in it["c"]], "c": it["c"],
                     "truth": norm(t), "conf": conf, "truth_src": src, "hyp": v4t.get(it["id"], "")})
    return segs, LIVESET


def cache_key(name):
    return name


# ------------------------------------------------------------------ candidate sources
def wb_nbest(lps, cfg, lexname="merged"):
    """full word-beam N-best (text, score), descending."""
    from phase0.analysis import wordbeam_v4 as W
    return W.beam(W.ens28(lps), get_lm(lexname), cfg)


_LMS = {}


def get_lm(lexname="merged"):
    if lexname not in _LMS:
        from phase0.analysis import wordbeam_v4 as W
        if lexname == "merged":
            _LMS[lexname] = W.get_lm()
        elif lexname == "generic":
            import json as J
            old = W.MERGED
            W.MERGED = W.GENERIC
            try:
                lm = W.LM()
            finally:
                W.MERGED = old
            _LMS[lexname] = lm
        else:
            raise SystemExit(lexname)
    return _LMS[lexname]


def cfgs_from(base="small", **over):
    from phase0.analysis import wordbeam_v4 as W
    return dict(W.SEARCH, **dict(W.CONFIGS[base], **over))


def pool_texts(setname, gkey="ens"):
    """v4 pool composition, per item: base (char n-best / greedy / qwen), 8B+0.5B rewrites per prompt, fuzzy."""
    from phase0.analysis import llmdec as LD
    LD._FZ.clear()
    d, z, LL = LD.load_all(setname, ("8b", "p05b"))
    out = {}
    for it in d["items"]:
        groups = list(it["c"])
        base = []
        for g in groups:
            base += [t for t, _ in it["c"][g]["char"]] + [it["c"][g]["greedy"], it["c"][g]["qwen"]]
        base = [norm(x) for x in base if x]
        gens = {}
        for m in LL:
            for p in LD.PROMPTS_ALL:
                v = LL[m]["items"].get(it["id"], {}).get("gens", {}).get(p, {}).get(gkey, [])
                if v:
                    gens[(m, p)] = [norm(x) for x in v if norm(x)]
        fz = [norm(x) for x in LD._fuzzy(setname).get(it["id"], {}).get(gkey, []) if norm(x)]
        out[it["id"]] = {"base": base, "gens": gens, "fuzzy": fz}
    return out


# ------------------------------------------------------------------ metrics
def ed(a, b):
    from phase0.analysis.decode import edit_distance
    return edit_distance(a, b)


def wok(truth, cand):
    from phase0.analysis.decipher import word_align
    return sum(1 for op, _, _ in word_align(truth.split(), cand.split()) if op == "ok")


def seg_oracle(truth, cands):
    """-> dict: best word-correct, min word edits, min char edits, exact-match rank (None if absent)."""
    tw, tc = truth.split(), truth
    best_ok, best_wed, best_ced = 0, len(tw), len(tc)
    rank = None
    for i, c in enumerate(cands):
        if rank is None and c == truth:
            rank = i + 1
        cw = c.split()
        best_ok = max(best_ok, wok(truth, c))
        best_wed = min(best_wed, ed(tw, cw))
        best_ced = min(best_ced, ed(list(tc), list(c)))
    return {"n_words": len(tw), "n_chars": len(tc), "ok": best_ok, "wed": best_wed, "ced": best_ced,
            "rank": rank, "n_cands": len(cands)}


def agg(rows):
    nw = sum(r["n_words"] for r in rows) or 1
    nc = sum(r["n_chars"] for r in rows) or 1
    found = [r["rank"] for r in rows if r["rank"]]
    return {"segments": len(rows), "n_words": nw,
            "oracle_words": sum(r["ok"] for r in rows) / nw,
            "oracle_wer": sum(r["wed"] for r in rows) / nw,
            "oracle_cer": sum(r["ced"] for r in rows) / nc,
            "exact_in_list": len(found) / max(len(rows), 1),
            "median_rank": float(np.median(found)) if found else None,
            "mean_cands": float(np.mean([r["n_cands"] for r in rows])),
            "ranks": found}


# ------------------------------------------------------------------ commands
def all_sets():
    sets = {}
    for n in DESK:
        segs, lines, setname = load_desk(n)
        sets[n] = {"segs": segs, "lines": lines, "set": setname}
    segs, setname = load_live()
    sets["live"] = {"segs": segs, "lines": None, "set": setname}
    return sets


def _eval_cfg(sets, cfg, Ns, lexname="merged", only=None):
    row = {"cfg": {k: v for k, v in cfg.items()}, "lex": lexname, "sets": {}, "secs": {}}
    for sname, S in sets.items():
        if only and sname not in only:
            continue
        t0 = time.time()
        nb = [wb_nbest(s["lps"], cfg, lexname) for s in S["segs"]]
        row["secs"][sname] = (time.time() - t0) / max(len(nb), 1)
        row["sets"][sname] = {str(N): agg([seg_oracle(s["truth"], [t for t, _ in b[:N]])
                                           for s, b in zip(S["segs"], nb)]) for N in Ns}
    return row


def cmd_sweep(argv):
    """staged coordinate search on OLD DESK only (dev); the chosen config is then run on all four sets."""
    OUT.mkdir(parents=True, exist_ok=True)
    sets = all_sets()
    Ns = [10, 25, 50, 100, 250, 500]
    SEL, SEL_N = "old", "100"
    prev = json.loads((OUT / "sweep.json").read_text()) if (OUT / "sweep.json").exists() else {"rows": []}
    done = {json.dumps([r["cfg"], r["lex"]], sort_keys=True): r for r in prev["rows"] if SEL in r["sets"]}
    base = dict(alpha=1.0, beta=2.0, gamma=-8.0, boost=2.0, beam=2048, prune=-8.0)
    res = {"Ns": Ns, "select_on": f"{SEL}@N={SEL_N} oracle_words (tiebreak oracle_wer)",
           "rows": list(prev["rows"]), "stages": prev.get("stages", {})}
    lexname = "merged"

    def best_of(rows):
        return max(rows, key=lambda r: (r["sets"][SEL][SEL_N]["oracle_words"], -r["sets"][SEL][SEL_N]["oracle_wer"]))

    stages = [("beam", [dict(base, beam=b) for b in (128, 256, 512, 1024, 2048)]),
              ("alpha_beta", [dict(base, alpha=a, beta=b) for a in (0.5, 1.0, 1.5, 2.0, 3.0) for b in (0.0, 2.0, 4.0, 6.0)]),
              ("gamma_boost_prune", [dict(base, gamma=g, boost=bo, prune=pr) for g in (-8.0, -6.0, -4.0, -2.0)
                                     for bo in (0.0, 2.0, 4.0) for pr in (-8.0, -12.0)]),
              ("lexicon", None)]
    for name, cand in stages:
        if name == "alpha_beta":
            cand = [dict(base, alpha=a, beta=b) for a in (0.5, 1.0, 1.5, 2.0, 3.0) for b in (0.0, 2.0, 4.0, 6.0)]
        elif name == "gamma_boost_prune":
            cand = [dict(base, gamma=g, boost=bo, prune=pr) for g in (-8.0, -6.0, -4.0, -2.0)
                    for bo in (0.0, 2.0, 4.0) for pr in (-8.0, -12.0)]
        elif name == "lexicon":
            cand = [base]
        rows = []
        for cfg_over in cand:
            for lx in (("merged", "generic") if name == "lexicon" else (lexname,)):
                cfg = dict(cfgs_from("small"), **cfg_over)
                k = json.dumps([cfg, lx], sort_keys=True)
                r = done.get(k)
                if r is None:
                    r = _eval_cfg(sets, cfg, Ns, lx, only=(SEL,))
                    res["rows"].append(r)
                    done[k] = r
                    (OUT / "sweep.json").write_text(json.dumps(res, indent=1, default=float))
                rows.append(r)
                o = r["sets"][SEL][SEL_N]
                print(f"{name:18s} {lx:7s} {cfg_over} -> old@100 words {o['oracle_words']:.3f} wer {o['oracle_wer']:.3f} "
                      f"cands {o['mean_cands']:.0f}", flush=True)
        b = best_of(rows)
        base = dict(base, **{k2: v for k2, v in b["cfg"].items() if k2 in base})
        lexname = b["lex"]
        res["stages"][name] = {"chosen": dict(base), "lex": lexname, "old_at_100": b["sets"][SEL][SEL_N]}
        print(f"  -> stage {name}: {base} lex={lexname}", flush=True)
    res["chosen"] = {"cfg": base, "lex": lexname}
    (OUT / "chosen.json").write_text(json.dumps({"cfg": base, "lex": lexname,
                                                 "selected_on": "old desk only, oracle_words @N=100"}, indent=1))
    cfg = dict(cfgs_from("small"), **base)
    res["chosen_all_sets"] = _eval_cfg(sets, cfg, Ns, lexname)
    res["default_small_all_sets"] = _eval_cfg(sets, cfgs_from("small"), Ns, "merged")
    (OUT / "sweep.json").write_text(json.dumps(res, indent=1, default=float))
    print("CHOSEN", base, lexname)
    for s2, v in res["chosen_all_sets"]["sets"].items():
        print(" ", s2, {N: round(v[N]["oracle_words"], 3) for N in ("10", "50", "100", "500")})
    return 0


def cmd_oracle(argv):
    """main oracle table: (a) word beam, (b) 8B rewrites, (c) union = v4 pool; plus what the selector picks."""
    from phase0.analysis import llmdec as LD
    OUT.mkdir(parents=True, exist_ok=True)
    best = json.loads((OUT / "chosen.json").read_text()) if (OUT / "chosen.json").exists() else {}
    wcfg = dict(cfgs_from("small"), **best.get("cfg", {}))
    sets = all_sets()
    res = {"wordbeam_cfg": wcfg, "sources": {}, "per_segment": {}}
    for sname, S in sets.items():
        pools = pool_texts(S["set"])
        segs = S["segs"]
        nb = [[t for t, _ in wb_nbest(s["lps"], wcfg)] for s in segs]
        rows = {}
        srcs = {
            "a_wordbeam_50": [b[:50] for b in nb],
            "a_wordbeam_100": [b[:100] for b in nb],
            "a_wordbeam_500": [b[:500] for b in nb],
            "b_8b_rewrites_p7": [[x for (m, p), v in pools[s["id"]]["gens"].items() if m == "8b" and p == "p7" for x in v]
                                 for s in segs],
            "b_8b_rewrites_all": [[x for (m, p), v in pools[s["id"]]["gens"].items() if m == "8b" for x in v] for s in segs],
            "base_ctc_nbest": [pools[s["id"]]["base"] for s in segs],
            "v3_pool": [list(dict.fromkeys(pools[s["id"]]["base"]
                                           + [x for v in pools[s["id"]]["gens"].values() for x in v]
                                           + pools[s["id"]]["fuzzy"])) for s in segs],
        }
        srcs["c_v4_pool_union"] = [list(dict.fromkeys(p + b[:50])) for p, b in zip(srcs["v3_pool"], nb)]
        srcs["c_v4_pool_wb500"] = [list(dict.fromkeys(p + b[:500])) for p, b in zip(srcs["v3_pool"], nb)]
        srcs["d_base_plus_wb500"] = [list(dict.fromkeys(pools[s["id"]]["base"] + b[:500])) for s, b in zip(segs, nb)]
        srcs["d_nollm_pool_wb500"] = [list(dict.fromkeys(pools[s["id"]]["base"] + pools[s["id"]]["fuzzy"] + b[:500]))
                                      for s, b in zip(segs, nb)]
        srcs["e_v3_pool_minus_8b"] = [list(dict.fromkeys(pools[s["id"]]["base"] + pools[s["id"]]["fuzzy"]
                                       + [x for (m, _), v in pools[s["id"]]["gens"].items() if m != "8b" for x in v]))
                                      for s in segs]
        for k, cl in srcs.items():
            cl = [[c for c in x if c] for x in cl]
            rows[k] = agg([seg_oracle(s["truth"], c) for s, c in zip(segs, cl)])
        # what the current selector actually picks
        sel = [s.get("hyp", "") for s in segs]
        rows["SELECTOR_current"] = agg([seg_oracle(s["truth"], [h]) for s, h in zip(segs, sel)])
        # top-1 of the word beam alone
        rows["wordbeam_top1"] = agg([seg_oracle(s["truth"], b[:1]) for s, b in zip(segs, nb)])
        res["sources"][sname] = rows
        res["per_segment"][sname] = [
            {"id": s["id"], "truth": s["truth"], "selector": s.get("hyp", ""), "wb_top1": nb[i][0] if nb[i] else "",
             "wb_rank_of_truth": next((j + 1 for j, t in enumerate(nb[i]) if t == s["truth"]), None),
             "in_8b": s["truth"] in set(srcs["b_8b_rewrites_all"][i]),
             "in_v3_pool": s["truth"] in set(srcs["v3_pool"][i]),
             "in_wb50": s["truth"] in set(nb[i][:50]),
             "conf": s.get("conf")}
            for i, s in enumerate(segs)]
        print(sname, {k: round(v["oracle_words"], 3) for k, v in rows.items()}, flush=True)
    (OUT / "oracle.json").write_text(json.dumps(res, indent=1, default=float))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv.pop(0) if argv else "oracle"
    return {"sweep": cmd_sweep, "oracle": cmd_oracle}[cmd](argv)


if __name__ == "__main__":
    raise SystemExit(main())
