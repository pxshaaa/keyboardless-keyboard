"""v4 dev evaluation (old desk CV-safe, blind-1, blind-2 = ALL dev): word-beam config choice + suggestion-bar temperature.
Pools: frozen v3 pool (tag '') vs + word-beam top-50 small / big / union. LM scores reused from the swipe_p4 copies
(s_swipe-<set>, union of the same two configs); any unscored text is reported (must be 0). Selection = frozen v3 weights.
  PYTHONPATH=. .venv/bin/python -m phase0.analysis.v4_dev prep
  PYTHONPATH=. .venv/bin/python -m phase0.analysis.v4_dev eval"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from phase0.analysis import swipe_common as C
from phase0.analysis import v4_suggest as V
from phase0.analysis import wordbeam_v4 as W

L = Path(".cache/llmdec")
OUT = Path("results/v4")
TAGS = {"": (), "v4s": ("small",), "v4b": ("big",), "v4u": ("small", "big")}
TEMPS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
FROZEN3 = Path("results/llmdec/frozen_config_v3.json")


def new(name):
    return f"s_v4dev-{name}"


def prep():
    for name in ("old", "b1", "b2"):
        src, nw = C.SETS[name][0], new(name)
        for a, b in ((f"sets/{src}.json", f"sets/{nw}.json"), (f"sets/{src}_lp.npz", f"sets/{nw}_lp.npz"),
                     (f"llm/8b/s_swipe-{name}.json", f"llm/8b/{nw}.json"), (f"llm/p05b/s_swipe-{name}.json", f"llm/p05b/{nw}.json"),
                     (f"fuzzy/{src}.json", f"fuzzy/{nw}.json")):
            shutil.copy(L / a, L / b)
        wb = W.run_set(nw, configs=("small", "big"), procs=3)
        base = json.loads((L / "fuzzy" / f"{src}.json").read_text())
        for tag, cfgs in TAGS.items():
            if not tag:
                continue
            items = {k: {g: list(v) for g, v in dd.items()} for k, dd in base["items"].items()}
            for sid, dd in items.items():
                ext = [t for c in cfgs for t in wb[sid][c]]
                for gk in dd:
                    dd[gk] = list(dict.fromkeys(dd[gk] + ext))
            (L / "fuzzy" / f"{nw}__{tag}.json").write_text(json.dumps({"cfg": base.get("cfg"), "tag": tag, "items": items,
                                                                       "wordbeam": list(cfgs)}))
        (Path(".cache/v4") / f"wb_{name}.json").write_text(json.dumps(wb))
        print("prep", name, nw, flush=True)


def pools_for(name, tag):
    os.environ["LLMDEC_FUZZY_TAG"] = tag
    from phase0.analysis import llmdec as LD
    LD._FZ.clear()
    return LD.build_pools(new(name), ("8b", "p05b"))[1]


def evaluate():
    from phase0.analysis import llmdec as LD
    from phase0.analysis.decipher import score_variant
    run = json.loads(FROZEN3.read_text())["run"]
    names = V.load_names()
    res = {"run": run, "names": names, "temps": TEMPS, "sets": {}}
    for name in ("old", "b1", "b2"):
        segs, lines = C.load(name)
        res["sets"][name] = {}
        for tag in TAGS:
            pools = pools_for(name, tag)
            hyps, rows_T, unscored, npool, cal = [], {T: [] for T in TEMPS}, 0, [], {T: [] for T in TEMPS}
            for s, p in zip(segs, pools):
                P = p["P"][run["groups"]]
                sc = V.pool_scores(P, run)
                elig = np.isfinite(sc)
                unscored += int(np.isnan(P["lm"]["8b"][elig]).sum() + np.isnan(P["lm"]["p05b"][elig]).sum())
                npool.append(len(P["texts"]))
                hyp = LD.select(P, run["llm"], run["prompt"], run["K"], run["lam"], run["wb"], run["cb"], run.get("mu_lex", 0.0),
                                run.get("mu_pers", 0.0), run.get("nu", 0.0), run.get("lam_p", 0.0), run.get("plm", "p05b"),
                                run.get("use_fz", 0), run.get("kappa", 0.0), run.get("mu_tech", 0.0))
                assert hyp == (P["texts"][int(np.argmax(sc))] if elig.any() else ""), (name, tag, p["id"])
                hyps.append(hyp)
                cs = V.cand_spans(hyp, P["texts"], sc)
                tsp = V.spans(hyp.split(), s["truth"].split())[0] if hyp else []
                for T in TEMPS:
                    post = V.slot_posteriors(hyp, P["texts"], sc, T, cached=cs)
                    m = V.seg_metrics(s["truth"], hyp, post, names)
                    m["id"] = p["id"]
                    rows_T[T].append(m)
                    cal[T] += [float(-np.log(max(post[j].get(t, 0.0), 1e-6))) for j, t in enumerate(tsp)]
            sv = score_variant(lines, hyps)
            r = {"official_wer": C.boot_ci(sv["wed"], sv["wn"]), "official_words": float(sv["wok"].sum() / sv["wn"].sum()),
                 "word_errors": float(sv["wed"].sum()), "n_words": float(sv["wn"].sum()), "hyps": hyps,
                 "hyp_by_line": sv["hyp_by_line"], "unscored_lm": unscored, "mean_pool": float(np.mean(npool)),
                 "by_T": {str(T): {**V.summarise(rows_T[T]), "truth_slot_nll": float(np.mean(cal[T])) if cal[T] else None}
                          for T in TEMPS},
                 "slots_T4": [dict(id=m["id"], slots=m["slots"]) for m in rows_T[4.0]]}
            res["sets"][name][tag or "v3"] = r
            b = r["by_T"]["4.0"]
            print(f"{name} {tag or 'v3':>4} WER {r['official_wer'][0]:.3f} words {r['official_words']:.3f} pool {r['mean_pool']:.0f} "
                  f"unscored {unscored} | T4 auto {b['auto_words']:.3f} top3 {b['top3']['words_after_le1_tap']:.3f} "
                  f"top5 {b['top5']['words_after_le1_tap']:.3f} taps5 {b['top5']['taps_per_100_words']:.1f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dev_eval.json").write_text(json.dumps(res, indent=1, default=float))
    # pooled over the three sets
    pooled = {}
    for tag in TAGS:
        k = tag or "v3"
        e = sum(res["sets"][n][k]["word_errors"] for n in res["sets"])
        nwd = sum(res["sets"][n][k]["n_words"] for n in res["sets"])
        pooled[k] = {"word_errors": e, "wer": e / nwd, "mean_pool": float(np.mean([res["sets"][n][k]["mean_pool"] for n in res["sets"]])),
                     "by_T": {}}
        for T in TEMPS:
            nn = sum(res["sets"][n][k]["by_T"][str(T)]["n_words"] for n in res["sets"])
            f = lambda key, sub: sum(res["sets"][n][k]["by_T"][str(T)][key][sub] * res["sets"][n][k]["by_T"][str(T)]["n_words"]
                                     for n in res["sets"]) / nn
            pooled[k]["by_T"][str(T)] = {"top3_words": f("top3", "words_after_le1_tap"), "top5_words": f("top5", "words_after_le1_tap"),
                                         "top3_taps": f("top3", "taps_per_100_words"),
                                         "nll": float(np.mean([res["sets"][n][k]["by_T"][str(T)]["truth_slot_nll"] for n in res["sets"]]))}
    res["pooled"] = pooled
    (OUT / "dev_eval.json").write_text(json.dumps(res, indent=1, default=float))
    for k, v in pooled.items():
        print(k, f"errors {v['word_errors']:.0f} WER {v['wer']:.3f} pool {v['mean_pool']:.0f}",
              {T: (round(x["top3_words"], 3), round(x["top5_words"], 3), round(x["nll"], 2)) for T, x in v["by_T"].items()})


def score_mini():
    """LM-score every pool text not yet scored (8B + personal-LoRA 0.5B, union tag v4u covers all options): one gpulock job."""
    from phase0.analysis import decipher2 as D2
    subs = []
    for name in ("old", "b1", "b2"):
        nw = new(name)
        subs += [f"sets/{nw}.json", f"sets/{nw}_lp.npz", f"llm/8b/{nw}.json", f"llm/p05b/{nw}.json", f"fuzzy/{nw}.json", f"fuzzy/{nw}__v4u.json"]
    D2.sh(["ssh", D2.MINI, f"mkdir -p {D2.ROOT}/.cache/llmdec/sets {D2.ROOT}/.cache/llmdec/fuzzy {D2.ROOT}/.cache/llmdec/llm/8b {D2.ROOT}/.cache/llmdec/llm/p05b"])
    D2.sh(["rsync", "-a", "--relative", *[f".cache/llmdec/./{x}" for x in subs], f"{D2.MINI}:{D2.ROOT}/.cache/llmdec/"])
    cmds = []
    for name in ("old", "b1", "b2"):
        nw = new(name)
        cmds.append(f".cache/llmdec/venv/bin/python -W ignore -m phase0.analysis.llmdec_mlx --set {nw} --llm 8b --lm-only --pool-from 8b")
        cmds.append(f".cache/llmdec/venv/bin/python -W ignore -m phase0.analysis.llmdec_mlx --set {nw} --llm p05b --base {D2.PLM_BASE} "
                    f"--adapter {D2.PLM_ADAPTER} --lm-only --pool-from 8b")
    D2.sh(["ssh", D2.MINI, f"{D2.ENV} LLMDEC_FUZZY_TAG=v4u && .venv/bin/python -m phase0.tools.gpulock -- nice -n 10 bash -c '"
                           + " && ".join(cmds) + "'"])
    for name in ("old", "b1", "b2"):
        nw = new(name)
        for m in ("8b", "p05b"):
            D2.sh(["rsync", "-a", f"{D2.MINI}:{D2.ROOT}/.cache/llmdec/llm/{m}/{nw}.json", str(L / "llm" / m / f"{nw}.json")])


if __name__ == "__main__":
    {"prep": prep, "score": score_mini, "eval": evaluate}[sys.argv[1]]()
