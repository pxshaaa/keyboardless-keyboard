"""One-command blind desk decode with the FROZEN llmdec config (results/llmdec/frozen_config_v2.json).
MacBook:  python -m phase0.analysis.decipher2 data/sessions/<id> [--score truth.txt]
Steps: landmarks -> Mac mini CTC posteriors + segments + Qwen-0.5B (decipher.py) -> char n-best (mini CPU)
-> frozen LLM decoder on the mini GPU (gpulock, one job) -> phrases printed (+ scored if --score)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from phase0.analysis import decipher as D

FROZEN = Path("results/llmdec/frozen_config_v2.json")
FROZEN3 = Path("results/llmdec/frozen_config_v3.json")
FROZEN31 = Path("results/llmdec/frozen_config_v3_1.json")
MINI, ROOT = D.MINI, D.MINI_ROOT
ENV = "cd ~/cvt && export PYTHONPATH=. HF_HOME=~/cvt/.cache/llmdec/hf HF_HUB_OFFLINE=1 LLMDEC_MEM_GB=7"
PLM_BASE = ".cache/llmdec/personal_llm/base"
PLM_ADAPTER = ".cache/llmdec/personal_llm/adapter_it1200"



def sh(cmd):
    D.sh(cmd)


def run_dec2(setname: str, run: dict) -> list[dict]:
    """frozen LLM-guided constrained decoder (one GPU job, lock-guarded)"""
    out = f".cache/llmdec/dec2/{setname}.jsonl"
    sh(["ssh", MINI, f"{ENV} && rm -f {out} && .venv/bin/python -m phase0.tools.gpulock -- nice -n 10 "
                     f".cache/llmdec/venv/bin/python -W ignore -m phase0.analysis.llmdec2 run --llm {run['llm']} --sets {setname} "
                     f"--grid '{run['grid']}' --out {out}"])
    local = Path(".cache/llmdec/dec2") / f"{setname}.jsonl"
    local.parent.mkdir(parents=True, exist_ok=True)
    sh(["rsync", "-a", f"{MINI}:{ROOT}/{out}", str(local)])
    recs = [json.loads(l) for l in local.read_text().splitlines()]
    return sorted(recs, key=lambda r: int(r["id"][3:]))


def run_gv(setname: str, run: dict) -> list[dict]:
    """frozen generate-and-verify: LLM rewrites (mini GPU, lock-guarded) -> CTC ll + lam*LLM + bonuses (MacBook CPU)"""
    import time
    from phase0.analysis import llmdec as LD
    flags = "--ctx-prompts" if run["prompt"] in ("p5", "p6", "p7", "p8") else ""
    sh(["ssh", MINI, f"{ENV} && rm -f .cache/llmdec/llm/{run['llm']}/{setname}.json && .venv/bin/python -m phase0.tools.gpulock -- "
                     f"nice -n 10 .cache/llmdec/venv/bin/python -W ignore -m phase0.analysis.llmdec_mlx --set {setname} "
                     f"--llm {run['llm']} {flags} --prompts {run['prompt']} --gks {run['groups']}"])
    llms = [run["llm"]]
    if run.get("use_fz"):
        for sub in (f"sets/{setname}.json", f"sets/{setname}_lp.npz", f"llm/{run['llm']}/{setname}.json"):
            (Path(".cache/llmdec") / sub).parent.mkdir(parents=True, exist_ok=True)
            sh(["rsync", "-a", f"{MINI}:{ROOT}/.cache/llmdec/{sub}", f".cache/llmdec/{sub}"])
        tag = os.environ.get("LLMDEC_FUZZY_TAG", "")
        sh([sys.executable, "-m", "phase0.analysis.llmdec_fuzzy", setname])
        sh(["rsync", "-a", f".cache/llmdec/fuzzy/{setname}{'__' + tag if tag else ''}.json", f"{MINI}:{ROOT}/.cache/llmdec/fuzzy/"])
        if tag:
            sh(["rsync", "-a", ".cache/llmdec/techvocab/tech_vocab.json", f"{MINI}:{ROOT}/.cache/llmdec/techvocab/"])
        sh(["ssh", MINI, f"{ENV} LLMDEC_FUZZY_TAG={tag} && .venv/bin/python -m phase0.tools.gpulock -- nice -n 10 .cache/llmdec/venv/bin/python "
                         f"-W ignore -m phase0.analysis.llmdec_mlx --set {setname} --llm {run['llm']} --lm-only --pool-from {run['llm']}"])
    if run.get("lam_p"):
        sh(["ssh", MINI, f"{ENV} LLMDEC_FUZZY_TAG={os.environ.get('LLMDEC_FUZZY_TAG', '')} && rm -f .cache/llmdec/llm/{run['plm']}/{setname}.json && .venv/bin/python -m phase0.tools.gpulock -- "
                         f"nice -n 10 .cache/llmdec/venv/bin/python -W ignore -m phase0.analysis.llmdec_mlx --set {setname} "
                         f"--llm {run['plm']} --base {PLM_BASE} --adapter {PLM_ADAPTER} --lm-only --pool-from {run['llm']}"])
        llms.append(run["plm"])
    for sub in [f"sets/{setname}.json", f"sets/{setname}_lp.npz"] + [f"llm/{m}/{setname}.json" for m in llms]:
        (Path(".cache/llmdec") / sub).parent.mkdir(parents=True, exist_ok=True)
        sh(["rsync", "-a", f"{MINI}:{ROOT}/.cache/llmdec/{sub}", f".cache/llmdec/{sub}"])
    t0 = time.time()
    d, pools, _ = LD.build_pools(setname, tuple(llms))
    g = run["groups"]
    recs = []
    L = json.loads((Path(".cache/llmdec/llm") / run["llm"] / f"{setname}.json").read_text())["items"]
    for p in pools:
        hyp = LD.select(p["P"][g], run["llm"], run["prompt"], run["K"], run["lam"], run["wb"], run["cb"],
                        run.get("mu_lex", 0.0), run.get("mu_pers", 0.0), run.get("nu", 0.0), run.get("lam_p", 0.0),
                        run.get("plm", "p05b"), run.get("use_fz", 0), run.get("kappa", 0.0), run.get("mu_tech", 0.0))
        it = L[p["id"]]
        gen_s = sum(v for x in it["secs"]["gen"].values() for v in x.values())
        recs.append({"id": p["id"], "hyp": hyp, "secs": gen_s + it["secs"]["lm"]})
    sel_s = (time.time() - t0) / max(len(recs), 1)
    for r in recs:
        r["secs"] += sel_s
    return recs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session_dir", type=Path)
    p.add_argument("--score", type=Path, default=None)
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--config", default="recommended", choices=("recommended", "generic", "personal"))
    p.add_argument("--version", default="v3", choices=("v2", "v3", "v3.1"), help="frozen config: v2, v3 (fuzzy) or v3.1 (+ tech vocab)")
    a = p.parse_args(argv)
    frozen = {"v3": FROZEN3, "v3.1": FROZEN31}.get(a.version, FROZEN)
    tag = "v31" if a.version == "v3.1" else ""
    os.environ["LLMDEC_FUZZY_TAG"] = tag
    cfg = json.loads(frozen.read_text())
    sdir = a.session_dir
    sid = sdir.name
    setname = f"s_{sid}"
    T = {}
    t_all = time.time()
    # 1) landmarks (MacBook) -> CTC posteriors, segmentation, baseline decoders (mini, lock-guarded GPU job)
    if a.rerun or not (sdir / "decipher.json").exists():
        t0 = time.time()
        if not (sdir / "landmarks.parquet").exists():
            sh([sys.executable, "-m", "phase0.analysis.extract_landmarks", str(sdir)])
        T["landmarks_macbook"] = time.time() - t0
        t0 = time.time()
        files = [str(sdir / f) for f in ("frames.jsonl", "landmarks.parquet", "meta.json") if (sdir / f).exists()]
        sh(["ssh", MINI, f"mkdir -p {ROOT}/data/sessions/{sid}"])
        sh(["rsync", "-a", *files, f"{MINI}:{ROOT}/data/sessions/{sid}/"])
        sh(["rsync", "-a", "phase0/analysis/decipher.py", "phase0/analysis/seqctc2.py", f"{MINI}:{ROOT}/phase0/analysis/"])
        sh(["ssh", MINI, f"cd {ROOT} && .venv/bin/python -m phase0.tools.gpulock -- .cache/decipher/s2.sh phase0.analysis.decipher "
                         f"remote {sid} --gap 2.0 --pad 0.5 --thr 0.5 --seeds 0,1"])
        sh(["rsync", "-a", f"{MINI}:{ROOT}/.cache/decipher/out/{sid}.json", str(sdir / "decipher.json")])
        T["ctc_remote"] = time.time() - t0
    sh(["rsync", "-a", str(sdir / "decipher.json"), f"{MINI}:{ROOT}/data/sessions/{sid}/"])
    code = [f"phase0/analysis/{f}" for f in ("llmdec.py", "llmdec2.py", "mlxsafe.py", "decipher.py", "seqctc2.py")]
    sh(["rsync", "-a", *code, f"{MINI}:{ROOT}/phase0/analysis/"])
    sh(["rsync", "-a", "--relative", str(FROZEN), str(frozen), f"{MINI}:{ROOT}/"])
    # 2) candidates (CPU)
    t0 = time.time()
    sh(["ssh", MINI, f"{ENV} && nice -n 10 .venv/bin/python -W ignore -m phase0.analysis.llmdec cands {setname}"])
    T["candidates"] = time.time() - t0
    t0 = time.time()
    if a.version in ("v3", "v3.1"):
        pk = "personal_v3_1" if a.version == "v3.1" else "personal_v3"
        run = cfg[pk]["run"] if a.config == "personal" else cfg["generic"]["run"] if a.config == "generic" else cfg["run"]
    else:
        run = cfg[a.config]["run"] if a.config in ("generic", "personal") else cfg["run"]
    if run["method"] == "gv":
        recs = run_gv(setname, run)
    else:
        recs = run_dec2(setname, run)
    T["llm_decoder"] = time.time() - t0
    T["total"] = time.time() - t_all
    res = json.loads((sdir / "decipher.json").read_text())
    print(f"\nSession {sid}: {len(recs)} segments, frozen config {FROZEN} ({cfg['frozen_at']}), llm {run['llm']}")
    for r, seg in zip(recs, res["segments"]):
        print(f"{r['id'][3:]:>2}. [{seg['t0']:7.1f}s - {seg['t1']:7.1f}s] {r['hyp']}")
        print(f"      (baseline Qwen-0.5B: {seg.get('zs', {}).get('qwen', '')})")
    print(f"runtime: {sum(r['secs'] for r in recs) / max(len(recs), 1):.1f}s LLM-decoder per phrase; stages "
          f"{ {k: round(v) for k, v in T.items()} } s; video {res['video_s']:.0f}s")
    out_j = {"session": sid, "config": a.config, "frozen_config": cfg, "phrases": [r["hyp"] for r in recs], "records": recs,
             "timing_s": T}
    if a.score is not None:
        from phase0.analysis.seqctc import boot_ratio
        lines = [D.norm(l) for l in a.score.read_text().splitlines() if D.norm(l)]
        for name, hyps in (("frozen_llm_decoder", [r["hyp"] for r in recs]),
                           ("baseline_zs_qwen05", [s["zs"]["qwen"] for s in res["segments"]])):
            sv = D.score_variant(lines, hyps)
            w, c, k = boot_ratio(sv["wed"], sv["wn"]), boot_ratio(sv["ced"], sv["cn"]), boot_ratio(sv["wok"], sv["wn"])
            out_j[name] = {"wer": w, "cer": c, "words_correct": k, "hyp_by_line": sv["hyp_by_line"]}
            print(f"{name:<22} WER {w[0]:.3f} [{w[1]:.3f},{w[2]:.3f}]  CER {c[0]:.3f}  words {100 * k[0]:.0f}%")
            for i, l in enumerate(lines):
                print(f"    {i + 1}. truth: {l}\n       hyp:   {sv['hyp_by_line'][i]}")
    (sdir / f"decipher2_{a.version}_{a.config}.json").write_text(json.dumps(out_j, indent=1, default=float))
    print(f"wrote {sdir / f'decipher2_{a.version}_{a.config}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
