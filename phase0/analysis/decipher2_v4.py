"""Frozen v4 one-command desk decode = frozen v3 generate-and-verify + M1 lexicon word-beam candidates + M7 suggestion bar.
  PYTHONPATH=. .venv/bin/python -m phase0.analysis.decipher2_ctc3 data/sessions/<id> --version v4 [--manifest <path>] [--score truth.txt]
Steps: landmarks (MacBook, gpulock) -> CTC posteriors from the deploy manifest (CPU; default .cache/ctc_v3/deploy/manifest.json)
-> twin session + segments -> mini: char n-best, 8B rewrites (gpulock) -> MacBook: fuzzy + word beam (top-50 per config, CPU)
-> mini: 8B + personal-LoRA 0.5B LM scores of the whole pool (one gpulock job) -> MacBook: frozen v3 selection, per-word
alternatives, word times -> decipher2_v4.json + decipher2_v4.txt in the twin session dir."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from phase0.analysis import decipher as D
from phase0.analysis import decipher2 as D2

FROZEN4 = Path("results/llmdec/frozen_config_v4.json")
V3MAN = Path(".cache/ctc_v3/deploy/manifest.json")
MINI, ROOT, ENV = D2.MINI, D2.ROOT, D2.ENV
TAG = "v4"


def md5(p: Path) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def check_frozen(cfg: dict) -> None:
    bad = {f: (h, md5(Path(f)) if Path(f).exists() else None) for f, h in cfg["md5"].items()
           if not Path(f).exists() or md5(Path(f)) != h}
    if bad:
        sys.exit(f"[v4] frozen md5 mismatch (refusing to run a non-frozen v4): {bad}")


def lgpu(cmd: list[str]) -> None:
    """local (MacBook) GPU-capable job under the machine's GPU lock"""
    D2.sh([sys.executable, "-m", "phase0.tools.gpulock", "--", *cmd])


# ---------------------------------------------------------------- 1) posteriors + twin session
def prepare(src: Path, manifest: Path, rerun: bool, T: dict) -> Path:
    default = manifest.resolve() == V3MAN.resolve()
    dj = src / "decipher.json"
    if dj.exists() and json.loads(dj.read_text()).get("ctc") == "v3" and default:   # CV/dev twin already built
        return src
    man = json.loads(manifest.read_text())
    tag = "ctc3" if default else f"ctcm{md5(manifest)[:8]}"
    tgt = D.SESS / f"zz-{tag}-{src.name}"
    if not rerun and (tgt / "decipher.json").exists():
        return tgt
    t0 = time.time()
    if not (src / "landmarks.parquet").exists():
        D2.sh(["nice", "-n", "10", sys.executable, "-m", "phase0.analysis.extract_landmarks", str(src)])   # mediapipe CPU delegate: no GPU lock
    T["landmarks"] = time.time() - t0
    t0 = time.time()
    if default:
        D2.sh([sys.executable, "-W", "ignore", ".cache/ctc_v3/predict.py", str(src)])
        post = Path(".cache/ctc_v3/out") / src.name / "posteriors.npz"
    else:
        out = Path(".cache/v4/ctc") / tag / src.name
        D2.sh([sys.executable, "-W", "ignore", "-m", "phase0.analysis.predict_manifest", str(src), "--manifest", str(manifest),
               "--out-dir", str(out)])
        post = out / "posteriors.npz"
    groups = list(man["groups"])
    mp = man.get("decipher_map") or ("zs=zeroshot,desk=desk" if {"zeroshot", "desk"} <= set(groups) else
                                     f"zs={next(g for g in groups if g != man.get('primary', groups[0]))},desk={man.get('primary', groups[0])}")
    cmd = [sys.executable, "-W", "ignore", "-m", "phase0.analysis.ctcv3_eval", "mkdecipher", "--lpfile", str(post),
           "--src", src.name, "--out-sid", tgt.name, "--map", mp]
    if (src / "truth.txt").exists():
        cmd += ["--truth", str(src / "truth.txt")]
    D2.sh(cmd)
    T["ctc_posteriors"] = time.time() - t0
    return tgt


# ---------------------------------------------------------------- 2) candidates, LLM, word beam
def pull(subs: list[str]) -> None:
    for sub in subs:
        (Path(".cache/llmdec") / sub).parent.mkdir(parents=True, exist_ok=True)
        D2.sh(["rsync", "-a", f"{MINI}:{ROOT}/.cache/llmdec/{sub}", f".cache/llmdec/{sub}"])


def build_pool(tgt: Path, cfg: dict, T: dict) -> str:
    from phase0.analysis import wordbeam_v4 as W
    sid, setname, run = tgt.name, f"s_{tgt.name}", cfg["run"]
    D2.sh(["ssh", MINI, f"mkdir -p {ROOT}/data/sessions/{sid} {ROOT}/.cache/decipher/out {ROOT}/.cache/llmdec/fuzzy"])
    D2.sh(["rsync", "-a", f".cache/decipher/out/{sid}_lp.npz", f"{MINI}:{ROOT}/.cache/decipher/out/"])
    D2.sh(["rsync", "-a", str(tgt / "decipher.json"), f"{MINI}:{ROOT}/data/sessions/{sid}/"])
    code = [f"phase0/analysis/{f}" for f in ("llmdec.py", "llmdec2.py", "llmdec_mlx.py", "mlxsafe.py", "decipher.py", "seqctc2.py")]
    D2.sh(["rsync", "-a", *code, f"{MINI}:{ROOT}/phase0/analysis/"])
    D2.sh(["rsync", "-a", "--relative", str(D2.FROZEN), str(D2.FROZEN3), str(FROZEN4), f"{MINI}:{ROOT}/"])
    t0 = time.time()
    D2.sh(["ssh", MINI, f"{ENV} && nice -n 10 .venv/bin/python -W ignore -m phase0.analysis.llmdec cands {setname}"])
    T["candidates"] = time.time() - t0
    t0 = time.time()
    flags = "--ctx-prompts" if run["prompt"] in ("p5", "p6", "p7", "p8") else ""
    D2.sh(["ssh", MINI, f"{ENV} && rm -f .cache/llmdec/llm/{run['llm']}/{setname}.json && .venv/bin/python -m phase0.tools.gpulock -- "
                        f"nice -n 10 .cache/llmdec/venv/bin/python -W ignore -m phase0.analysis.llmdec_mlx --set {setname} "
                        f"--llm {run['llm']} {flags} --prompts {run['prompt']} --gks {run['groups']}"])
    pull([f"sets/{setname}.json", f"sets/{setname}_lp.npz", f"llm/{run['llm']}/{setname}.json"])
    T["llm_rewrites"] = time.time() - t0
    t0 = time.time()
    env = dict(os.environ, LLMDEC_FUZZY_TAG="")   # v3 fuzzy generator, untagged
    subprocess.run([sys.executable, "-m", "phase0.analysis.llmdec_fuzzy", setname], check=True, env=env)
    T["fuzzy"] = time.time() - t0
    t0 = time.time()
    wb = W.run_set(setname, configs=tuple(cfg["wordbeam"]["configs"]), procs=3)
    base = json.loads((Path(".cache/llmdec/fuzzy") / f"{setname}.json").read_text())
    items = {k: {g: list(v) for g, v in dd.items()} for k, dd in base["items"].items()}
    for iid, dd in items.items():
        ext = [t for c in cfg["wordbeam"]["configs"] for t in wb[iid][c]]
        for gk in dd:
            dd[gk] = list(dict.fromkeys(dd[gk] + ext))
    fz = Path(".cache/llmdec/fuzzy") / f"{setname}__{TAG}.json"
    fz.write_text(json.dumps({"cfg": base.get("cfg"), "tag": TAG, "items": items, "wordbeam": wb}))
    T["wordbeam"] = time.time() - t0
    t0 = time.time()
    D2.sh(["rsync", "-a", str(fz), f"{MINI}:{ROOT}/.cache/llmdec/fuzzy/"])
    lm8 = (f".cache/llmdec/venv/bin/python -W ignore -m phase0.analysis.llmdec_mlx --set {setname} --llm {run['llm']} "
           f"--lm-only --pool-from {run['llm']}")
    lmp = (f"rm -f .cache/llmdec/llm/{run['plm']}/{setname}.json && .cache/llmdec/venv/bin/python -W ignore -m phase0.analysis.llmdec_mlx "
           f"--set {setname} --llm {run['plm']} --base {D2.PLM_BASE} --adapter {D2.PLM_ADAPTER} --lm-only --pool-from {run['llm']}")
    D2.sh(["ssh", MINI, f"{ENV} LLMDEC_FUZZY_TAG={TAG} && .venv/bin/python -m phase0.tools.gpulock -- nice -n 10 "
                        f"bash -c '{lm8} && {lmp}'"])
    pull([f"llm/{run['llm']}/{setname}.json", f"llm/{run['plm']}/{setname}.json"])
    T["llm_scores"] = time.time() - t0
    return setname


# ---------------------------------------------------------------- 3) selection + suggestion bar
def select_and_suggest(tgt: Path, setname: str, cfg: dict) -> tuple[list[dict], dict]:
    os.environ["LLMDEC_FUZZY_TAG"] = TAG
    from phase0.analysis import llmdec as LD
    from phase0.analysis import v4_suggest as V
    from phase0.analysis import wordbeam_v4 as W
    LD._FZ.clear()
    run, Tmp = cfg["run"], cfg["suggest"]["T"]
    d, pools, _ = LD.build_pools(setname, (run["llm"], run["plm"]))
    res = json.loads((tgt / "decipher.json").read_text())
    times = np.load(Path(".cache/decipher/out") / f"{tgt.name}_lp.npz")["times"]
    z = np.load(Path(".cache/llmdec/sets") / f"{setname}_lp.npz")
    segs, n_unscored = [], 0
    for p, row in zip(pools, res["segments"]):
        P = p["P"][run["groups"]]
        sc = V.pool_scores(P, run)
        n_unscored += int(np.isnan(P["lm"][run["llm"]][np.isfinite(sc)]).sum())
        hyp = LD.select(P, run["llm"], run["prompt"], run["K"], run["lam"], run["wb"], run["cb"], run.get("mu_lex", 0.0),
                        run.get("mu_pers", 0.0), run.get("nu", 0.0), run.get("lam_p", 0.0), run.get("plm", "p05b"),
                        run.get("use_fz", 0), run.get("kappa", 0.0), run.get("mu_tech", 0.0))
        post = V.slot_posteriors(hyp, P["texts"], sc, Tmp)
        s0 = row["frames30"][0]
        M = W.ens28([z[f"{p['id']}__{g}"] for g in P["groups"]])
        wt = [(row["t0"] + float(times[s0 + a] - times[s0]), row["t0"] + float(times[s0 + b] - times[s0]))
              for a, b in V.word_frames(M, hyp)] if hyp else []
        segs.append({"i": row["i"], "id": p["id"], "t0": row["t0"], "t1": row["t1"], "text": hyp,
                     "words": V.word_entries(hyp, post, wt, k=cfg["suggest"]["k"]), "_post": post, "pool_size": len(P["texts"]),
                     "wordbeam_in_pool": int(sum(1 for t in P["texts"] if t in set(sum(json.loads(
                         (Path(".cache/llmdec/fuzzy") / f"{setname}__{TAG}.json").read_text())["wordbeam"][p["id"]].values(), []))))})
    return segs, {"unscored_lm": n_unscored}


def score(segs: list[dict], truth: Path, names: list[str]) -> dict:
    from phase0.analysis import swipe_common as C
    from phase0.analysis import v4_suggest as V
    from phase0.analysis.seqctc import boot_ratio
    lines = [D.norm(l) for l in truth.read_text().splitlines() if D.norm(l)]
    sv = D.score_variant(lines, [s["text"] for s in segs])
    tmp = [{"hyp": s["text"], "lps": [np.zeros((1, 29))]} for s in segs]
    out = {"wer": boot_ratio(sv["wed"], sv["wn"]), "cer": boot_ratio(sv["ced"], sv["cn"]), "words_correct": boot_ratio(sv["wok"], sv["wn"]),
           "hyp_by_line": sv["hyp_by_line"], "lines": lines}
    try:   # per-segment truth (alignment + CTC boundary refinement needs the posteriors)
        for s, t in zip(tmp, segs):
            s["lps"] = t["_lps"]
        C.assign_truth(tmp, lines)
        rows = [V.seg_metrics(a["truth"], s["text"], s["_post"], names) for a, s in zip(tmp, segs)]
        out["suggestion_bar"] = V.summarise(rows)
        out["slots"] = [{"i": s["i"], "truth": a["truth"], "slots": r["slots"]} for a, s, r in zip(tmp, segs, rows)]
    except Exception as e:   # scoring must never break decoding
        out["suggestion_bar_error"] = repr(e)
    return out


def report(sid: str, segs: list[dict], names: list[str], sc: dict | None, T: dict, video_s: float) -> str:
    L = [f"Session {sid} - decoder v4 (frozen v3 gen-verify + word beam + suggestion bar)", f"names quick list: {' | '.join(names)}", ""]
    for s in segs:
        L.append(f"{s['i']:>2}. [{s['t0']:7.1f}s - {s['t1']:7.1f}s] {s['text']}")
        for w in s["words"]:
            alts = ", ".join(f"{a['w'] or '(delete)'} {a['p']:.2f}" for a in w["alternatives"])
            L.append(f"      {w['start_s']:7.1f}s  {w['word']:<14} {w['p']:.2f}   | {alts}")
    L.append(f"\nruntime stages {({k: round(v) for k, v in T.items()})} s; video {video_s:.0f}s")
    if sc:
        L.append(f"score: WER {sc['wer'][0]:.3f} [{sc['wer'][1]:.3f},{sc['wer'][2]:.3f}]  CER {sc['cer'][0]:.3f}  "
                 f"words {100 * sc['words_correct'][0]:.0f}%")
        if "suggestion_bar" in sc:
            b = sc["suggestion_bar"]
            L.append(f"suggestion bar: auto {100 * b['auto_words']:.1f}% | <=1 tap top-3 {100 * b['top3']['words_after_le1_tap']:.1f}% "
                     f"({b['top3']['taps_per_100_words']:.1f} taps/100 words) | top-5 {100 * b['top5']['words_after_le1_tap']:.1f}% "
                     f"({b['top5']['taps_per_100_words']:.1f}) | top-5 + names {100 * b['top5_names']['words_after_le1_tap']:.1f}%")
        for i, l in enumerate(sc["lines"]):
            L.append(f"    {i + 1}. truth: {l}\n       hyp:   {sc['hyp_by_line'][i]}")
    return "\n".join(L)


def main(src: Path, manifest: Path | None = None, truth: Path | None = None, rerun: bool = False) -> int:
    from phase0.analysis import v4_suggest as V
    cfg = json.loads(FROZEN4.read_text())
    check_frozen(cfg)
    manifest = Path(manifest) if manifest else V3MAN
    T, t_all = {}, time.time()
    tgt = prepare(src, manifest, rerun, T)
    setname = build_pool(tgt, cfg, T)
    t0 = time.time()
    segs, info = select_and_suggest(tgt, setname, cfg)
    T["select_suggest"] = time.time() - t0
    T["total"] = time.time() - t_all
    names = V.load_names()[:cfg["suggest"]["n_names"]]
    res = json.loads((tgt / "decipher.json").read_text())
    sc = None
    if truth is not None:
        z = np.load(Path(".cache/llmdec/sets") / f"{setname}_lp.npz")
        for s in segs:
            s["_lps"] = [z[f"{s['id']}__{g}"] for g in ("zs", "desk") if f"{s['id']}__{g}" in z.files]
        sc = score(segs, truth, names)
    txt = report(tgt.name, segs, names, sc, T, res["video_s"])
    print(txt)
    out = {"session": tgt.name, "source_session": src.name, "version": "v4", "frozen_config": str(FROZEN4),
           "frozen_md5": md5(FROZEN4), "manifest": str(manifest), "manifest_md5": md5(manifest), "names_quicklist": names,
           "phrases": [s["text"] for s in segs], "timing_s": T, "video_s": res["video_s"], **info,
           "segments": [{k: v for k, v in s.items() if not k.startswith("_")} for s in segs]}
    if sc is not None:
        out["score"] = sc
    (tgt / "decipher2_v4.json").write_text(json.dumps(out, indent=1, default=float))
    (tgt / "decipher2_v4.txt").write_text(txt + "\n")
    print(f"wrote {tgt / 'decipher2_v4.json'} and decipher2_v4.txt")
    return 0
