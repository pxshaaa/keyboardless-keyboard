"""Blind-1 for v3 (NOT blind: the fuzzy generator was designed after seeing blind-1 errors). Scored once into
results/llmdec/blind1_score_v3.json. Reuses the blind-1 candidates and 8B p3/p6/p7 outputs already on the mini; adds p8 rewrites,
fuzzy candidates, LM-only scores (8B + personal LoRA), then selects with frozen v3 personal (+ v3 without fuzzy, v2 personal)."""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

RES = Path("results/llmdec")
F3 = RES / "frozen_config_v3.json"
OUTF = RES / "blind1_score_v3.json"
SID = "20260912-174542-desk"
SET = f"s_{SID}"
MINI = "macmini"
ENV = "cd ~/cvt && export PYTHONPATH=. HF_HOME=~/cvt/.cache/llmdec/hf HF_HUB_OFFLINE=1"
PLM = "--base .cache/llmdec/personal_llm/base --adapter .cache/llmdec/personal_llm/adapter_it1200"


def sh(cmd: str) -> None:
    print("$", cmd[:220], flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main() -> int:
    from phase0.analysis import llmdec as LD
    from phase0.analysis.decipher import norm, score_variant
    from phase0.analysis.seqctc import boot_ratio
    if OUTF.exists():
        raise SystemExit(f"{OUTF} exists: scored once")
    cfg = json.loads(F3.read_text())
    code = " ".join(f"phase0/analysis/{f}" for f in ("llmdec.py", "llmdec_mlx.py", "mlxsafe.py"))
    if "--score-only" not in sys.argv:
        sh(f"rsync -a {code} {MINI}:cvt/phase0/analysis/ && rsync -a --relative {F3} {MINI}:cvt/")
        sh(f"ssh {MINI} '{ENV} LLMDEC_MEM_GB=7 && .venv/bin/python -m phase0.tools.gpulock -- nice -n 10 .cache/llmdec/venv/bin/python "
           f"-W ignore -m phase0.analysis.llmdec_mlx --set {SET} --llm 8b --ctx-prompts --prompts p8'")
        for sub in (f"sets/{SET}.json", f"sets/{SET}_lp.npz", f"llm/8b/{SET}.json"):
            sh(f"rsync -a {MINI}:cvt/.cache/llmdec/{sub} .cache/llmdec/{sub}")
        sh(f"{sys.executable} -m phase0.analysis.llmdec_fuzzy {SET}")
        sh(f"rsync -a .cache/llmdec/fuzzy/{SET}.json {MINI}:cvt/.cache/llmdec/fuzzy/")
        sh(f"ssh {MINI} '{ENV} LLMDEC_MEM_GB=7 && .venv/bin/python -m phase0.tools.gpulock -- nice -n 10 .cache/llmdec/venv/bin/python "
           f"-W ignore -m phase0.analysis.llmdec_mlx --set {SET} --llm 8b --lm-only --pool-from 8b'")
        sh(f"ssh {MINI} '{ENV} LLMDEC_MEM_GB=4 && .venv/bin/python -m phase0.tools.gpulock -- nice -n 10 .cache/llmdec/venv/bin/python "
           f"-W ignore -m phase0.analysis.llmdec_mlx --set {SET} --llm p05b {PLM} --lm-only --pool-from 8b'")
    for m in ("8b", "p05b"):
        sh(f"rsync -a {MINI}:cvt/.cache/llmdec/llm/{m}/{SET}.json .cache/llmdec/llm/{m}/")
    lines = [norm(l) for l in (Path("data/sessions") / SID / "truth.txt").read_text().splitlines() if norm(l)]
    d, pools, _ = LD.build_pools(SET, ("8b", "p05b"))
    rows = {"FROZEN personal_v3": cfg["personal_v3"]["run"], "v2 personal": cfg["v2_personal"]["run"]}
    nf = cfg["personal_v3_no_fuzzy"]
    rows["v3 best without fuzzy (kbd-tuned)"] = {**cfg["personal_v3"]["run"], **{k: nf[k] for k in ("prompt", "lam", "cb", "lam_p", "mu_pers")},
                                                "use_fz": 0, "kappa": 0.0}
    out = {"scored_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "NOT_BLIND": cfg["disclosure"],
           "frozen_v3_md5": hashlib.md5(F3.read_bytes()).hexdigest(), "lines": lines, "variants": {}}
    for name, rr in rows.items():
        for g in ("zs", "ens"):
            hyps = [LD.select(p["P"][g], "8b", rr["prompt"], rr["K"], rr["lam"], rr["wb"], rr["cb"], rr.get("mu_lex", 0.0),
                              rr.get("mu_pers", 0.0), rr.get("nu", 0.0), rr.get("lam_p", 0.0), "p05b", rr.get("use_fz", 0),
                              rr.get("kappa", 0.0)) for p in pools]
            sv = score_variant(lines, hyps)
            out["variants"][f"{name} [{g}]"] = {"wer": boot_ratio(sv["wed"], sv["wn"]), "cer": boot_ratio(sv["ced"], sv["cn"]),
                                                "words_correct": boot_ratio(sv["wok"], sv["wn"]), "hyps": hyps,
                                                "hyp_by_line": sv["hyp_by_line"], "run": rr}
    from phase0.analysis.decode import edit_distance
    sv = score_variant(lines, out["variants"]["FROZEN personal_v3 [ens]"]["hyps"])
    for g in ("zs", "ens"):
        for nm, keep_all in (("pool_v2", False), ("pool_v3", True)):
            hy = []
            for k, p in enumerate(pools):
                ref = " ".join(lines[i] for i in sv["seg_lines"][k])
                tx = [t for t, fz in zip(p["P"][g]["texts"], p["P"][g]["isfz"]) if keep_all or not fz]
                hy.append(min(tx, key=lambda t: (edit_distance(ref.split(), t.split()), edit_distance(ref, t))) if tx else "")
            o = score_variant(lines, hy)
            out.setdefault("oracle", {})[f"{nm} [{g}]"] = float(o["wed"].sum() / o["wn"].sum())
    OUTF.write_text(json.dumps(out, indent=1, default=float))
    for k, v in out["variants"].items():
        print(f"{k:<45} WER {v['wer'][0]:.3f} CER {v['cer'][0]:.3f} words {100 * v['words_correct'][0]:.0f}%")
    print("oracle", out["oracle"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
