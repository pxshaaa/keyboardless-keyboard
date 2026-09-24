"""End-to-end: beam N-best as the candidate generator + a small local LM as a teacher-forced batched rescorer.
`nbest` (MacBook) -> `push`/`lmscore` (mini, gpulock) -> `eval`. See results/beam/SUMMARY.md."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("LLMDEC_FUZZY_TAG", "")
OUT = Path("results/beam")
MINI, ROOT = "macmini", "~/cvt"
ENV = "cd ~/cvt && export PYTHONPATH=. HF_HOME=~/cvt/.cache/llmdec/hf HF_HUB_OFFLINE=1 LLMDEC_MEM_GB=7"
MODELS = {"p05b": (".cache/llmdec/personal_llm/base", ".cache/llmdec/personal_llm/adapter_it1200"),
          "q05b": (".cache/llmdec/personal_llm/base", None),
          "q3b": ("mlx-community/Qwen2.5-3B-Instruct-4bit", None),
          "q8b": ("mlx-community/Qwen3-8B-4bit", None)}


def sh(cmd):
    import subprocess
    subprocess.run(cmd, check=True)


def cmd_nbest(argv):
    from phase0.analysis import beam_oracle as B
    from phase0.analysis import llmdec as LD
    N = int(argv[0]) if argv else 50
    cfg = dict(B.cfgs_from("small"), **json.loads((OUT / "chosen.json").read_text())["cfg"])
    if len(argv) > 1:
        cfg["beam"] = int(argv[1])
    sets = B.all_sets()
    out = {"cfg": cfg, "N": N, "sets": {}}
    for sname, S in sets.items():
        pools = B.pool_texts(S["set"])
        items = []
        for s in S["segs"]:
            t0 = time.time()
            nb = B.wb_nbest(s["lps"], cfg)[:N]
            t_beam = time.time() - t0
            texts = [t for t, _ in nb]
            base = [x for x in pools[s["id"]]["base"] if x]
            allt = list(dict.fromkeys(texts + base))
            t0 = time.time()
            ctc = np.mean([LD.ctc_ll(l, [t or " " for t in allt]) for l in s["lps"]], 0)
            items.append({"id": s["id"], "truth": s["truth"], "texts": allt, "n_beam": len(texts),
                          "beam_score": [float(sc) for _, sc in nb], "ctc": [float(x) for x in ctc],
                          "nw": [len(t.split()) for t in allt], "nc": [len(t) for t in allt],
                          "npers": [sum(LD._pers().get(w, 0) >= 2 for w in t.split()) for t in allt],
                          "plm": [LD._plm_gain(t) for t in allt],
                          "secs": {"beam": t_beam, "ctc": time.time() - t0}})
            print(sname, s["id"], len(allt), f"beam {t_beam:.2f}s ctc {items[-1]['secs']['ctc']:.2f}s", flush=True)
        out["sets"][sname] = {"set": S["set"], "items": items}
    (OUT / f"nbest{argv[2] if len(argv) > 2 else ''}.json").write_text(json.dumps(out, default=float))
    return 0


def cmd_push(argv):
    sh(["ssh", MINI, f"mkdir -p {ROOT}/results/beam {ROOT}/phase0/analysis"])
    sh(["rsync", "-a", str(OUT / "nbest.json"), f"{MINI}:{ROOT}/results/beam/"])
    sh(["rsync", "-a", "phase0/analysis/beam_e2e.py", f"{MINI}:{ROOT}/phase0/analysis/"])
    return 0


def cmd_lmscore(argv):
    """run ON THE MINI under gpulock: batched teacher-forced prefill scoring, no generation."""
    from mlx_lm import load
    from phase0.analysis import mlxsafe
    from phase0.analysis.llmdec_mlx import lm_scores
    mlxsafe.cap()
    key = argv[0]
    bs = int(argv[1]) if len(argv) > 1 else 16
    src, adapter = MODELS[key]
    if not Path(src).exists():
        from huggingface_hub import snapshot_download
        src = snapshot_download(src, local_files_only=True)
    t0 = time.time()
    model, tok = load(src, adapter_path=adapter) if adapter else load(src)
    load_s = time.time() - t0
    d = json.loads((OUT / "nbest.json").read_text())
    res = {"model": key, "bs": bs, "load_s": load_s, "sets": {}}
    for sname, S in d["sets"].items():
        rows = []
        for it in S["items"]:
            t0 = time.time()
            sc = lm_scores(model, tok, it["texts"], bs=bs)
            dt = time.time() - t0
            rows.append({"id": it["id"], "lm": [sc[t][0] for t in it["texts"]],
                         "ntok": [sc[t][1] for t in it["texts"]], "secs": dt, "n": len(it["texts"])})
            print(key, sname, it["id"], len(it["texts"]), f"{dt:.2f}s", flush=True)
        res["sets"][sname] = rows
    (OUT / f"lm_{key}.json").write_text(json.dumps(res, default=float))
    return 0


def cmd_pull(argv):
    for k in argv or list(MODELS):
        sh(["rsync", "-a", f"{MINI}:{ROOT}/results/beam/lm_{k}.json", str(OUT) + "/"])
    return 0


def _pick(it, lm, w):
    s = (np.array(it["ctc"]) + w["lam"] * np.array(lm) + w["wb"] * np.array(it["nw"], float)
         + w["cb"] * np.array(it["nc"], float) + w["mu_pers"] * np.array(it["npers"], float)
         + w["nu"] * np.array(it["plm"], float))
    if w.get("beam_only"):
        s[it["n_beam"]:] = -np.inf
    return it["texts"][int(np.argmax(s))]


def cmd_eval(argv):
    from phase0.analysis import beam_oracle as B
    d = json.loads((OUT / "nbest.json").read_text())
    models = [k for k in MODELS if (OUT / f"lm_{k}.json").exists()]
    res = {"cfg": d["cfg"], "N": d["N"], "rows": []}
    for key in models + ["none"]:
        LM = json.loads((OUT / f"lm_{key}.json").read_text()) if key != "none" else None
        for beam_only in (True, False):
            grid = [dict(lam=l, wb=-1.0, cb=3.0, mu_pers=1.0, nu=0.0, beam_only=beam_only)
                    for l in ((0.0,) if key == "none" else (0.0, 0.25, 0.5, 1.0, 2.0, 4.0))]
            scored = []
            for w in grid:
                per = {}
                for sname, S in d["sets"].items():
                    lmrows = {r["id"]: r["lm"] for r in LM["sets"][sname]} if LM else None
                    picks = [_pick(it, lmrows[it["id"]] if lmrows else np.zeros(len(it["texts"])), w) for it in S["items"]]
                    per[sname] = B.agg([B.seg_oracle(it["truth"], [p]) for it, p in zip(S["items"], picks)])
                    per[sname]["picks"] = picks
                scored.append((w, per))
            best = max(scored, key=lambda x: x[1]["old"]["oracle_words"])
            row = {"model": key, "beam_only": beam_only, "weights": best[0],
                   "sets": {k: {m: v[m] for m in ("oracle_words", "oracle_wer", "oracle_cer", "n_words")}
                            for k, v in best[1].items()},
                   "picks": {k: v["picks"] for k, v in best[1].items()},
                   "lm_secs_per_seg": float(np.mean([r["secs"] for s in LM["sets"].values() for r in s])) if LM else 0.0,
                   "lm_load_s": LM["load_s"] if LM else 0.0}
            res["rows"].append(row)
            print(f"{key:6s} beam_only={int(beam_only)} lam={best[0]['lam']:<4} "
                  + " ".join(f"{k} {v['oracle_words']:.3f}" for k, v in row["sets"].items())
                  + f"  lm {row['lm_secs_per_seg']:.2f}s/seg", flush=True)
    res["beam_secs_per_seg"] = {k: float(np.mean([it["secs"]["beam"] for it in S["items"]])) for k, S in d["sets"].items()}
    res["ctc_secs_per_seg"] = {k: float(np.mean([it["secs"]["ctc"] for it in S["items"]])) for k, S in d["sets"].items()}
    (OUT / "e2e.json").write_text(json.dumps(res, indent=1, default=float))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv.pop(0)
    return {"nbest": cmd_nbest, "push": cmd_push, "lmscore": cmd_lmscore, "pull": cmd_pull, "eval": cmd_eval}[cmd](argv)


if __name__ == "__main__":
    raise SystemExit(main())
