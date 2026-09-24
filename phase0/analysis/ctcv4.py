"""ctc_v4+ desk data step: fine-tune the hwtmix desk ensemble on every labelled desk session (old prompted session,
blind-1, blind-2, later calibration sessions), leave-one-session-out CV with CPU proxy decoders (Qwen-0.5B word beam,
char beam), CTC alignment checks and versioned deploy manifests in the ctc_v3 layout (<root>/deploy/manifest.json,
model paths relative to <root>). ctc_v3 files are only read (init checkpoints, zero-shot models), never written.

  PYTHONPATH=. .venv/bin/python -m phase0.analysis.ctcv4 data --root .cache/ctc_v4
  PYTHONPATH=. .venv/bin/python -m phase0.tools.gpulock -- .venv/bin/python -m phase0.analysis.ctcv4 train \
      --root .cache/ctc_v4 --name final_d31 --seeds 0,1,2
  PYTHONPATH=. .venv/bin/python -m phase0.analysis.ctcv4 loso --root .cache/ctc_v4          # CPU eval
  PYTHONPATH=. .venv/bin/python -m phase0.analysis.ctcv4 manifest --root .cache/ctc_v4 --name final_d31
"""

from __future__ import annotations

import functools
import os

import torch

if os.environ.get("CTCV3_CPU"):
    torch.backends.mps.is_available = functools.lru_cache()(lambda: False)

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from phase0.analysis import ctcv3 as C
from phase0.analysis import seqctc as S
from phase0.analysis.seqctc import BLANK, KBD, OTHER

V3 = C.ROOT
TAG = "hwtmix"
OLD = C.DESK
BLIND1 = C.BLIND1
BLIND2 = "20260913-125733-desk"
BLIND2_TWIN = "zz-ctc3-20260913-125733-desk"
RECIPE = {"steps": 300, "lr": 5e-4, "desk_w": 0.5, "ftmix": 0.3}   # = ctc_v3 final_d26 (deployed desk group)


def norm(s: str) -> str:
    from phase0.analysis.decipher import norm as n
    return n(s)


# ============================================================================ labelled desk data
def prompted_windows(sid: str):
    """Prompter sessions: phrases.jsonl shown -> done windows (the old desk session's training windows)."""
    return [(float(t0), float(t1), norm(txt)) for t0, t1, txt in S.desk_windows(C.stream(sid))]


def twin_windows(sid: str, twin: str):
    """Blind sessions: decoded segments (decipher.json of the ctc3 twin) map 1:1, in order, onto truth.txt lines."""
    st = C.stream(sid)
    dj = json.loads((C.SESS / twin / "decipher.json").read_text())
    lines = [norm(l) for l in (C.SESS / sid / "truth.txt").read_text().splitlines() if norm(l)]
    assert len(lines) == len(dj["segments"]), (sid, len(lines), len(dj["segments"]))
    return [(float(st.t[0] + r["t0"]), float(st.t[0] + r["t1"]), l) for r, l in zip(dj["segments"], lines)]


def base_sessions() -> dict:
    return {
        OLD: {"kind": "prompted", "source": "phrases.jsonl shown->done (generic prompts)", "windows": prompted_windows(OLD)},
        BLIND1: {"kind": "blind", "source": "decipher.json segments 1:1 truth.txt (decipher_score segment_to_lines)",
                 "windows": [(float(a), float(b), t) for a, b, t in C.blind1_windows()]},
        BLIND2: {"kind": "blind", "source": f"{BLIND2_TWIN}/decipher.json segments 1:1 truth.txt",
                 "windows": twin_windows(BLIND2, BLIND2_TWIN)},
    }


def save_data(f: Path, sessions: dict, note: str = ""):
    f.parent.mkdir(parents=True, exist_ok=True)
    n = sum(len(v["windows"]) for v in sessions.values())
    f.write_text(json.dumps({"what": "labelled desk phrase windows (absolute stream times, s) for desk fine-tuning",
                             "note": note, "n_phrases": n, "sessions": sessions}, indent=1))
    return n


def load_data(f: Path) -> dict:
    return json.loads(Path(f).read_text())["sessions"]


def data_hash(f: Path) -> str:
    return hashlib.md5(Path(f).read_bytes()).hexdigest()


def cmd_data(a) -> int:
    root = Path(a.root)
    n = save_data(root / "desk_data.json", base_sessions(), "ctc_v4: old desk 20 + blind-1 6 + blind-2 5")
    print(f"wrote {root / 'desk_data.json'}: {n} phrases")
    return 0


# ============================================================================ training (MPS via gpulock)
def desk_sources(r: dict, sessions: dict, desk_w: float, ftmix: float):
    """Same mixture as ctcv3.desk_sources: desk phrase windows at weight desk_w (per session proportional to its
    phrase count), keyboard sessions + 30% How-We-Type the rest."""
    groups = [(C.stream(sid), [tuple(w) for w in v["windows"]]) for sid, v in sessions.items() if v["windows"]]
    ntot = sum(len(w) for _, w in groups)
    src = [(S.PhraseSource(st, w), desk_w * len(w) / ntot) for st, w in groups]
    kb = C.kbd_sources(dict(r, mix=ftmix, syn=0.0, synsrc="kbd"), KBD)
    tot = sum(w for _, w in kb)
    return src + [(s, w / tot * (1 - desk_w)) for s, w in kb]


def run_dir(root: Path, seed: int, name: str) -> Path:
    return Path(root) / "runs" / TAG / f"seed{seed}" / name


def cmd_train(a) -> int:
    root = Path(a.root)
    dfile = Path(a.data) if a.data else root / "desk_data.json"
    excl = set(filter(None, a.exclude.split(",")))
    sessions = {k: v for k, v in load_data(dfile).items() if k not in excl}
    r = C.recipe(TAG)
    for seed in map(int, a.seeds.split(",")):
        d = run_dir(root, seed, a.name)
        d.mkdir(parents=True, exist_ok=True)
        f = d / "model.pt"
        if f.exists() and (d / "meta.json").exists():   # never silently reuse a model trained on other data/settings
            meta = json.loads((d / "meta.json").read_text())
            if (meta.get("data_md5") != data_hash(dfile) or meta.get("excluded") != sorted(excl)
                    or meta.get("steps") != a.steps):
                raise SystemExit(f"{f} exists but was trained on different data/settings ({meta.get('data')}, "
                                 f"excluded {meta.get('excluded')}, steps {meta.get('steps')}); use a new --root or --name")
        if not f.exists():
            cfg = C.mkcfg(r, seed * 1000, a.steps, a.lr)
            m = S.build(cfg)
            init = V3 / "runs" / TAG / f"seed{seed}" / "all.pt"
            m.load_state_dict(torch.load(init, map_location="cpu"))
            t0 = time.time()
            S.train(m, desk_sources(r, sessions, a.desk_w, a.ftmix), cfg, log=f"[ctcv4 {a.name} s{seed}]",
                    ckpt=d / "model.ckpt")
            torch.save(m.state_dict(), f)
            meta = {"cmd": "final", "tag": TAG, "name": a.name, "seeds": str(seed), "steps": a.steps, "lr": a.lr,
                    "desk_w": a.desk_w, "ftmix": a.ftmix, "ftsyn": 0.0, "ftsynsrc": "kbd",
                    "ntrain": sum(len(v["windows"]) for v in sessions.values()),
                    "sessions": {k: len(v["windows"]) for k, v in sessions.items()}, "excluded": sorted(excl),
                    "data": str(dfile), "data_md5": data_hash(dfile), "init": str(init), "device": str(S.device()),
                    "secs": time.time() - t0}
            (d / "meta.json").write_text(json.dumps(meta, indent=1))
        m = C.load(r, f)
        for sid in filter(None, a.infer.split(",")):
            if not (d / f"{sid}_cont.npy").exists():
                np.save(d / f"{sid}_cont.npy", C.infer_cont(m, C.stream(sid)).astype(np.float16))
        print(f"[ctcv4 {a.name} s{seed}] done", flush=True)
    return 0


# ============================================================================ posteriors (CPU)
def member_cont(root: Path, model: Path, sid: str) -> np.ndarray:
    """Continuous 30 Hz log-posteriors of one model on one session; cached (run dir first, then <root>/post)."""
    model = Path(model)
    own = model.parent / f"{sid}_cont.npy"
    if model.name == "model.pt" and own.exists() and own.stat().st_mtime >= model.stat().st_mtime:   # run dirs only
        return np.load(own).astype(np.float32)
    key = hashlib.md5(f"{model.resolve()}:{model.stat().st_mtime}".encode()).hexdigest()[:12]
    f = Path(root) / "post" / f"{key}_{sid}.npy"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        m = C.load(C.recipe(TAG), model)
        np.save(f, C.infer_cont(m, C.stream(sid)).astype(np.float16))
    return np.load(f).astype(np.float32)


def ensemble_cont(root: Path, models, sid: str) -> np.ndarray:
    got = [member_cont(root, m, sid) for m in models]
    G = min(len(g) for g in got)
    return C.logmean([g[:G] for g in got])


# ============================================================================ CTC alignment check
def merged(lp: np.ndarray) -> np.ndarray:
    """'other' joins blank (as in ctcv3.ctc_align): 28 classes, blank = 0."""
    out = np.delete(lp.astype(np.float64), OTHER, axis=1)
    out[:, BLANK] = np.logaddexp(lp[:, BLANK], lp[:, OTHER])
    return out


def ctc_logp(lp: np.ndarray, text: str) -> float:
    sy = S.text_syms(text)
    if len(sy) == 0 or len(lp) < len(sy):
        return -np.inf
    x = torch.from_numpy(merged(lp)).float()[:, None, :]
    loss = torch.nn.functional.ctc_loss(x, torch.as_tensor(sy)[None], torch.tensor([len(lp)]), torch.tensor([len(sy)]),
                                        blank=BLANK, reduction="sum", zero_infinity=False)
    return -float(loss)


def align_stats(lp: np.ndarray, text: str) -> dict:
    """logp(text) vs the best single frame path; margin per character (~0 = the evidence supports the text).
    Also greedy char CER against the text."""
    from phase0.analysis.decode import edit_distance
    m = merged(lp)
    best = float(m.max(1).sum())
    lt = ctc_logp(lp, text)
    n = max(1, len(S.text_syms(text)))
    g = S.greedy(lp)
    return {"logp": lt, "best": best, "margin_per_char": (lt - best) / n, "nll_per_char": -lt / n,
            "greedy": g, "greedy_cer": edit_distance(text, g) / max(1, len(text))}


def emission_span(lp: np.ndarray, thr: float = 0.5):
    pnb = 1 - np.exp(np.logaddexp(lp[:, BLANK], lp[:, OTHER]))
    em = np.where(pnb > thr)[0]
    return (int(em[0]), int(em[-1])) if len(em) else None


# ============================================================================ evaluation (CPU proxy decoders)
def truth_lines(sessions: dict, sid: str):
    return [w[2] for w in sorted(sessions[sid]["windows"], key=lambda w: w[0])]


def eval_rows(root: Path, rows, decs=("qwen", "char"), procs: int = 2) -> dict:
    """rows: [{label, sid, models, lines}] -> per row and decoder: WER/CER over lines (decipher.score_variant on
    the continuous stream segmented like decipher: gap 2.0 s, pad 0.5, thr 0.5)."""
    from phase0.analysis import ctcv3_eval as E   # disables MPS in this process (decoding is CPU-only)
    from phase0.analysis import seqctc2 as S2
    from phase0.analysis.decipher import score_variant
    E.HYP = Path(root) / "hyps"
    D = E.decoders(list(decs))
    jobs, plan = [], []
    for r in rows:
        L = ensemble_cont(root, r["models"], r["sid"])
        st = C.stream(r["sid"])
        times = st.t[::2][:len(L)]
        segs = S2.segments(L, times, 2.0)
        for dn in decs:
            plan.append((r, dn, len(jobs), len(segs)))
            jobs += [(D[dn], L[s0:s1]) for s0, s1 in segs]
    hy = E.decode_all(jobs, procs)
    out = {}
    for r, dn, k, n in plan:
        sv = score_variant(r["lines"], hy[k:k + n])
        res = E.summ(sv["ced"], sv["cn"], sv["wed"], sv["wn"], sv["wok"])
        res.update({"hyps": hy[k:k + n], "hyp_by_line": sv["hyp_by_line"], "sid": r["sid"], "label": r["label"],
                    "models": [str(m) for m in r["models"]]})
        out[f"{r['label']}|{r['sid']}|{dn}"] = res
    return out


def pooled(res: dict, labels, sids, dec: str, ref: str | None = None) -> dict:
    """Concatenate per-line arrays over sessions (pooled LOSO); paired bootstrap delta vs label `ref`."""
    from phase0.analysis import ctcv3_eval as E
    out = {}
    for lab in labels:
        ks = [f"{lab}|{s}|{dec}" for s in sids]
        if not all(k in res for k in ks):
            continue
        cat = {f: np.concatenate([res[k][f] for k in ks]) for f in ("ed", "n", "wed", "wn")}
        wok = 1 - cat["wed"] / np.maximum(cat["wn"], 1)
        r = E.summ(cat["ed"], cat["n"], cat["wed"], cat["wn"])
        r["words_correct"] = float(1 - cat["wed"].sum() / cat["wn"].sum())
        if ref and lab != ref and all(f"{ref}|{s}|{dec}" in res for s in sids):
            rk = [f"{ref}|{s}|{dec}" for s in sids]
            rw = np.concatenate([res[k]["wed"] for k in rk])
            rc = np.concatenate([res[k]["ed"] for k in rk])
            r["d_wer_vs_" + ref] = S.boot_delta(rw, cat["wed"], cat["wn"])
            r["d_cer_vs_" + ref] = S.boot_delta(rc, cat["ed"], cat["n"])
        del wok
        out[lab] = {k: v for k, v in r.items() if k not in ("ed", "n", "wed", "wn")}
    return out


def print_res(res: dict, pool: dict | None = None, title: str = ""):
    if title:
        print(f"== {title}")
    for k, r in res.items():
        print(f"  {k:<58} WER {r['wer']:.3f} [{r['wer_ci'][0]:.3f},{r['wer_ci'][1]:.3f}] CER {r['cer']:.3f} "
              f"words {r.get('words_correct', float('nan')):.2f}", flush=True)
    for dec, p in (pool or {}).items():
        for lab, r in p.items():
            msg = f"  POOLED {dec:<5} {lab:<22} WER {r['wer']:.3f} [{r['wer_ci'][0]:.3f},{r['wer_ci'][1]:.3f}] CER {r['cer']:.3f}"
            for kk, v in r.items():
                if kk.startswith("d_wer"):
                    msg += f" | {kk} {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]"
            print(msg, flush=True)


def cmd_loso(a) -> int:
    """Leave-one-session-out: zero-shot (3 seeds) vs fine-tuned on the other sessions (3 seeds) [+ old-only d20]."""
    root = Path(a.root)
    sessions = load_data(root / "desk_data.json")
    sids = [s for s in (a.sids.split(",") if a.sids else sessions)]
    seeds = list(map(int, a.seeds.split(",")))
    rows = []
    for sid in sids:
        lines = truth_lines(sessions, sid)
        rows.append({"label": "zs", "sid": sid, "lines": lines,
                     "models": [V3 / "runs" / TAG / f"seed{s}" / "all.pt" for s in seeds]})
        rows.append({"label": "loso_ft", "sid": sid, "lines": lines,
                     "models": [run_dir(root, s, f"loso_{sid}") / "model.pt" for s in seeds]})
        if sid != OLD and a.d20:
            rows.append({"label": "v3_d20_oldonly", "sid": sid, "lines": lines,
                         "models": [V3 / "runs" / TAG / f"seed{s}" / "final_d20" / "model.pt" for s in seeds]})
    res = eval_rows(root, rows, a.decoders.split(","), a.procs)
    pool = {dec: pooled(res, ["zs", "loso_ft"], sids, dec, ref="zs") for dec in a.decoders.split(",")}
    if a.d20:
        bl = [s for s in sids if s != OLD]
        for dec in a.decoders.split(","):
            pool[dec + "_blind"] = pooled(res, ["zs", "v3_d20_oldonly", "loso_ft"], bl, dec, ref="v3_d20_oldonly")
    out = root / "results" / f"{a.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sids": sids, "seeds": seeds, "rows": res, "pooled": pool}, indent=1, default=float))
    print_res(res, pool, a.name)
    print(f"-> {out}")
    return 0


# ============================================================================ manifests
def load_manifest_models(man_path: Path, group: str):
    man = json.loads(Path(man_path).read_text())
    base = Path(man["root"]) if man.get("root") else Path(man_path).resolve().parent.parent
    return [base / m["model"] for m in man["groups"][group]["members"]], man


def write_manifest(root: Path, name: str, seeds, version: str, out: Path | None = None, copy: bool = True,
                   note: str = "") -> Path:
    """ctc_v3-compatible manifest: groups zeroshot (ctc_v3 zero-shot models, copied) + desk (<name> models)."""
    root = Path(root)
    v3man = json.loads((V3 / "deploy" / "manifest.json").read_text())
    dd = root / "deploy"
    out = Path(out) if out else dd / "manifest.json"
    zs, desk = [], []
    for m in v3man["groups"]["zeroshot"]["members"]:
        src = V3 / m["model"]
        if copy:
            (dd / "models").mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dd / "models" / src.name)
            zs.append({**m, "model": f"deploy/models/{src.name}", "copied_from": str(src)})
        else:
            zs.append({**m, "model": str(src.resolve())})
    for s in seeds:
        src = run_dir(root, s, name) / "model.pt"
        meta = json.loads((src.parent / "meta.json").read_text())
        ent = {"recipe": C.recipe(TAG), "tag": TAG, "seed": s, "kind": name, "trained_on": meta}
        if copy:
            dst = dd / "models" / f"{TAG}_s{s}_{name}.pt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            ent["model"] = f"deploy/models/{dst.name}"
            ent["md5"] = hashlib.md5(dst.read_bytes()).hexdigest()
        else:
            ent["model"] = str(src.resolve())
        desk.append(ent)
    man = {"version": version, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": note,
           "primary": v3man["primary"], "segments": v3man["segments"],
           "groups": {"zeroshot": {"members": zs, "tta": False}, "desk": {"members": desk, "tta": False}}}
    if not copy:
        man["root"] = "/"   # absolute model paths
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, indent=1, default=str))
    return out


def cmd_manifest(a) -> int:
    f = write_manifest(Path(a.root), a.name, list(map(int, a.seeds.split(","))), a.version,
                       Path(a.out) if a.out else None, not a.no_copy, a.note)
    print(f"wrote {f}")
    return 0


# ============================================================================ CLI
def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("data")
    s.add_argument("--root", default=".cache/ctc_v4")
    s.set_defaults(fn=cmd_data)
    s = sub.add_parser("train")
    s.add_argument("--root", default=".cache/ctc_v4")
    s.add_argument("--data", default="")
    s.add_argument("--name", required=True)
    s.add_argument("--seeds", default="0,1,2")
    s.add_argument("--exclude", default="", help="comma list of session ids left out (LOSO folds)")
    s.add_argument("--infer", default="", help="comma list of session ids to write <sid>_cont.npy for")
    s.add_argument("--steps", type=int, default=RECIPE["steps"])
    s.add_argument("--lr", type=float, default=RECIPE["lr"])
    s.add_argument("--desk-w", type=float, default=RECIPE["desk_w"])
    s.add_argument("--ftmix", type=float, default=RECIPE["ftmix"])
    s.set_defaults(fn=cmd_train)
    s = sub.add_parser("loso")
    s.add_argument("--root", default=".cache/ctc_v4")
    s.add_argument("--name", default="loso")
    s.add_argument("--sids", default="")
    s.add_argument("--seeds", default="0,1,2")
    s.add_argument("--decoders", default="qwen,char")
    s.add_argument("--procs", type=int, default=2)
    s.add_argument("--d20", action="store_true", help="also the v3 old-session-only finals on the blind sessions")
    s.set_defaults(fn=cmd_loso)
    s = sub.add_parser("manifest")
    s.add_argument("--root", default=".cache/ctc_v4")
    s.add_argument("--name", required=True)
    s.add_argument("--seeds", default="0,1,2")
    s.add_argument("--version", default="v4")
    s.add_argument("--out", default="")
    s.add_argument("--no-copy", action="store_true", help="absolute model paths instead of copies (CV/alignment use)")
    s.add_argument("--note", default="")
    s.set_defaults(fn=cmd_manifest)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
