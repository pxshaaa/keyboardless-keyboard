"""Score blind session 1 ONCE with the frozen v2 config and its frozen report rows (no tuning here).
MacBook: PYTHONPATH=. .venv/bin/python -m phase0.analysis.llmdec2_blind1
GPU work runs on the Mac mini, one lock-guarded job at a time; selection/scoring on MacBook CPU."""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import time
from pathlib import Path

RES = Path("results/llmdec")
FROZEN = RES / "frozen_config_v2.json"
OUTF = RES / "blind1_score_v2.json"
SID = "20260912-174542-desk"
SET = f"s_{SID}"
MINI = "macmini"
ENV = "cd ~/cvt && export PYTHONPATH=. HF_HOME=~/cvt/.cache/llmdec/hf HF_HUB_OFFLINE=1 LLMDEC_MEM_GB=7"
PLM_BASE = ".cache/llmdec/personal_llm/base"
PLM_ADAPTER = ".cache/llmdec/personal_llm/adapter_it1200"



def sh(cmd: str) -> None:
    print("$", cmd[:240], flush=True)
    subprocess.run(cmd, shell=True, check=True)


def word_decoder_rows(d) -> dict:
    import math
    import numpy as np
    from phase0.analysis import autocorrect as AC
    from phase0.analysis import seqctc2 as S2
    from phase0.analysis.llmdec import OUT, TUNE_QWEN, norm
    from phase0.analysis.llmdec_wordlm import InterpScorer
    z = np.load(OUT / "sets" / f"{SET}_lp.npz")
    cfg = json.loads(TUNE_QWEN.read_text())["best"]
    base = AC.NLM(".cache/decipher/qwen2.5-0.5b", device="cpu")
    voc = json.loads((OUT / "personal" / "vocab.json").read_text())
    gen = S2.lexicon()
    N = sum(voc["unigram"].values())
    B = S2.blocklist()
    merged = S2.Lexicon.__new__(S2.Lexicon)
    merged.logp = dict(gen.logp)
    for w, c in voc["unigram"].items():
        if c >= 2 and w.isalpha() and w.isascii() and w not in merged.logp and not S2.is_blocked(w, B) and len(w) <= 20:
            merged.logp[w] = math.log(0.2 * c / N)
    merged.n_blocked = gen.n_blocked
    merged.la = {"": max(merged.logp.values())}
    for w, lp in merged.logp.items():
        for i in range(1, len(w) + 1):
            if lp > merged.la.get(w[:i], -np.inf):
                merged.la[w[:i]] = lp
    rows = {}
    for name, lex, scorer in (("Qwen-0.5B word decoder (re-run, zs)", gen, base),
                              ("Qwen-0.5B word decoder + personal lexicon (zs)", merged, base),
                              ("Qwen-0.5B word decoder + personal word LM a=0.2 (zs)", gen, InterpScorer(base, voc, 0.2))):
        S2._LEX["l"], S2._SC["qwen"] = lex, scorer
        rows[name] = [norm(S2.run_decoder({"kind": "qwen", "cfg": cfg}, z[f"{it['id']}__zs"])) for it in d["items"]]
    S2._LEX["l"] = gen
    return rows


def main() -> int:
    from phase0.analysis import llmdec as LD
    from phase0.analysis.decipher import norm, score_variant
    from phase0.analysis.seqctc import boot_ratio
    if OUTF.exists():
        raise SystemExit(f"{OUTF} exists: blind-1 is scored once")
    import sys
    score_only = "--score-only" in sys.argv   # GPU outputs already on the mini: only pull + score (still scored once)
    cfg = json.loads(FROZEN.read_text())
    Path(".cache/llmdec/dec2").mkdir(parents=True, exist_ok=True)
    code = " ".join(f"phase0/analysis/{f}" for f in ("llmdec.py", "llmdec2.py", "llmdec_mlx.py", "mlxsafe.py"))
    score_only or sh(f"rsync -a {code} {MINI}:cvt/phase0/analysis/ && rsync -a --relative {FROZEN} {MINI}:cvt/")
    score_only or sh(f"rsync -a data/sessions/{SID}/decipher.json {MINI}:cvt/data/sessions/{SID}/")
    score_only or sh(f"ssh {MINI} '{ENV} && nice -n 10 .venv/bin/python -W ignore -m phase0.analysis.llmdec cands {SET}'")
    rows = cfg["report_rows"]
    gv_prompts: dict = {}
    for r in rows.values():
        if r["run"]["method"] == "gv":
            gv_prompts.setdefault(r["run"]["llm"], set()).add(r["run"]["prompt"])
    for llm, prs in gv_prompts.items():
        flags = "--ctx-prompts" if prs & {"p5", "p6", "p7"} else ""
        score_only or sh(f"ssh {MINI} '{ENV} && .venv/bin/python -m phase0.tools.gpulock -- nice -n 10 .cache/llmdec/venv/bin/python -W ignore "
           f"-m phase0.analysis.llmdec_mlx --set {SET} --llm {llm} {flags} --prompts {','.join(sorted(prs))}'")
    plms = sorted({r["run"]["plm"] for r in rows.values() if r["run"].get("lam_p")})
    for plm in plms:
        score_only or sh(f"ssh {MINI} '{ENV} && .venv/bin/python -m phase0.tools.gpulock -- nice -n 10 .cache/llmdec/venv/bin/python -W ignore "
           f"-m phase0.analysis.llmdec_mlx --set {SET} --llm {plm} --base {PLM_BASE} --adapter {PLM_ADAPTER} --lm-only --pool-from 8b'")
    for fam, r in rows.items():
        if r["run"]["method"] == "dec2":
            out = f".cache/llmdec/dec2/blind1_{fam.replace(':', '_')}.jsonl"
            score_only or sh(f"ssh {MINI} '{ENV} && rm -f {out} && .venv/bin/python -m phase0.tools.gpulock -- nice -n 10 "
               f".cache/llmdec/venv/bin/python -W ignore -m phase0.analysis.llmdec2 run --llm {r['run']['llm']} --sets {SET} "
               f"--grid \"{r['run']['grid']}\" --out {out}'")
            r["out"] = out
    sh(f"rsync -a {MINI}:cvt/.cache/llmdec/sets/{SET}.json {MINI}:cvt/.cache/llmdec/sets/{SET}_lp.npz .cache/llmdec/sets/")
    for llm in list(gv_prompts) + plms:
        Path(f".cache/llmdec/llm/{llm}").mkdir(parents=True, exist_ok=True)
        sh(f"rsync -a {MINI}:cvt/.cache/llmdec/llm/{llm}/{SET}.json .cache/llmdec/llm/{llm}/")
    lines = [norm(l) for l in (Path("data/sessions") / SID / "truth.txt").read_text().splitlines() if norm(l)]
    d = json.loads((LD.OUT / "sets" / f"{SET}.json").read_text())
    variants = {}
    segs = json.loads((Path("data/sessions") / SID / "decipher.json").read_text())["segments"]
    variants["baseline: char n-gram (zs)"] = [s["zs"]["char"] for s in segs]
    variants["baseline: Qwen-0.5B word decoder (zs)"] = [s["zs"]["qwen"] for s in segs]
    variants["baseline: Qwen-0.5B word decoder (desk ft)"] = [s["desk"]["qwen"] for s in segs]
    if gv_prompts:
        d_, pools, _ = LD.build_pools(SET, tuple(gv_prompts) + tuple(plms))
        for fam, r in rows.items():
            if r["run"]["method"] == "gv":
                rr = r["run"]
                for g in ("zs", "ens"):
                    variants[f"{r['label']} [{g}]"] = [LD.select(p["P"][g], rr["llm"], rr["prompt"], rr["K"], rr["lam"], rr["wb"],
                                                                 rr["cb"], rr.get("mu_lex", 0.0), rr.get("mu_pers", 0.0),
                                                                 rr.get("nu", 0.0), rr.get("lam_p", 0.0), rr.get("plm", "p05b"))
                                                       for p in pools]
    for fam, r in rows.items():
        if r["run"]["method"] == "dec2":
            local = Path(".cache/llmdec/dec2") / Path(r["out"]).name
            sh(f"rsync -a {MINI}:cvt/{r['out']} {local}")
            recs = {json.loads(l)["id"]: json.loads(l)["hyp"] for l in local.read_text().splitlines()}
            variants[f"{r['label']} [zs]"] = [recs.get(it["id"], "") for it in d["items"]]
    # Qwen-0.5B closed-lexicon word decoder rows (CPU): current, + personal lexicon, + personal word LM (alpha tuned on kbd = 0.2)
    variants.update(word_decoder_rows(d))
    out = {"scored_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "frozen_config_md5": hashlib.md5(FROZEN.read_bytes()).hexdigest(), "chosen": cfg["chosen"],
           "disclosure": cfg["blind1_disclosure"], "lines": lines, "variants": {}}
    for k, hyps in variants.items():
        sv = score_variant(lines, hyps)
        out["variants"][k] = {"wer": boot_ratio(sv["wed"], sv["wn"]), "cer": boot_ratio(sv["ced"], sv["cn"]),
                              "words_correct": boot_ratio(sv["wok"], sv["wn"]), "hyps": hyps, "hyp_by_line": sv["hyp_by_line"]}
        v = out["variants"][k]
        print(f"{k:<70} WER {v['wer'][0]:.3f} CER {v['cer'][0]:.3f} words {100 * v['words_correct'][0]:.0f}%", flush=True)
    OUTF.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", OUTF)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
