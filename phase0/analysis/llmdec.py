"""Generate-and-verify open-vocab decoder: candidates rescored by exact CTC ll + lam*LLM logp; tuned on kbd + OLD desk only.
mini: s2.sh phase0.analysis.llmdec cands {kbd|old|new} -> llmdec_mlx --set S --llm M -> llmdec tune|freeze|final"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import itertools
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(".cache/llmdec")
RES = Path("results/llmdec")
SESS = Path("data/sessions")
DC = Path(".cache/decipher")
OLD, NEW = "20260910-202149-desk", "20260912-174542-desk"
TUNE_QWEN = Path("results/seqctc2/tuning/tune_qwen_lex_hwtmix.json")
FROZEN = RES / "frozen_config.json"
FROZEN2 = RES / "frozen_config_v2.json"
CHAR = {"alpha": 0.6, "beta": 2.0, "beam": 32, "topn": 16}   # decipher.char_params() values; wider beam for n-best
GRID = {"lam": (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0), "wb": (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0),
        "cb": (0.0, 0.5, 1.0, 2.0), "K": (0, 5, 10), "prompt": ("p1", "p2", "p3", "p4")}
LLMS = ("3b", "8b")
PROMPTS_ALL = ("p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8")
GRID2 = {"prompt": ("p3", "p4", "p5", "p6", "p7"), "K": (5, 10), "lam": (1.5, 2.0, 3.0), "wb": (-1.0, 0.0, 1.0),
         "cb": (1.0, 2.0, 3.0), "mu_lex": (0.0, 1.0, 2.0), "mu_pers": (0.0, 1.0, 2.0), "nu": (0.0, 0.1, 0.25, 0.5)}


def norm(s: str) -> str:
    return " ".join("".join(c if "a" <= c <= "z" else (" " if c.isspace() or c == "-" else "") for c in s.lower()).split())


# ============================================================================ stage 1: candidates (project venv)
def beam_nbest(lp, lm, alpha, beta, beam=32, topn=16, prune=-7.0):
    """seqctc.beam_lm with the final beam returned as a ranked, whitespace-normalised n-best list."""
    from phase0.analysis.seqctc import BLANK, OTHER, SYMS
    P = lp.astype(np.float64)
    blank = np.logaddexp(P[:, BLANK], P[:, OTHER])
    beams = {"": (0.0, -np.inf)}
    for t in range(len(P)):
        row = P[t]
        cand = [c for c in range(1, 28) if row[c] > prune]
        nb = defaultdict(lambda: [-np.inf, -np.inf])
        for pre, (pb, pnb) in beams.items():
            ptot = np.logaddexp(pb, pnb)
            e = nb[pre]
            e[0] = np.logaddexp(e[0], ptot + blank[t])
            last = pre[-1] if pre else ""
            lmrow = None
            for c in cand:
                ch = SYMS[c]
                if ch == last:
                    e[1] = np.logaddexp(e[1], pnb + row[c])
                    base = pb
                else:
                    base = ptot
                if ch == " " and (not pre or last == " "):
                    continue
                if lmrow is None:
                    lmrow = lm(pre)
                e2 = nb[pre + ch]
                e2[1] = np.logaddexp(e2[1], base + row[c] + alpha * lmrow[c - 1] + beta)
        beams = dict(sorted(nb.items(), key=lambda kv: -np.logaddexp(*kv[1]))[:beam])
    out = {}
    for pre, v in sorted(beams.items(), key=lambda kv: -np.logaddexp(*kv[1])):
        k = " ".join(pre.split())
        if k and k not in out:
            out[k] = float(np.logaddexp(*v))
    return list(out.items())[:topn]


def load_set(name: str):
    """-> items [{id, ref?, lp: {group: [T, NS]}}], meta"""
    from phase0.analysis import seqctc as S
    from phase0.analysis import seqctc2 as S2
    if name == "kbd":
        items = []
        for s in S.LOSO + (S.HELD,):
            st = S.load_session(s)
            names = np.array(S2.key_names(s), dtype=object)
            fold = "held" if s == S.HELD else s[9:15]
            lps, _ = S.load_lps(OUT / "kbd" / "hwt" / "seed0" / f"{fold}.npz")
            for i, (t0, t1, sy) in enumerate(S.eval_windows(st)):
                idx = np.where((st.kt >= t0) & (st.kt <= t1))[0]
                ft = S2.final_text(names[idx])
                if "#" in ft or len(ft.split()) < 2:
                    continue
                items.append({"id": f"{fold}_{i}", "ref": ft, "lp": {"zs": lps[i]}})
        return items, {"source": "seqctc hwt LOSO/held out-of-fold posteriors, seed0 (hwtmix fold npz no longer exist)"}
    sid = OLD if name == "old" else NEW if name == "new" else name[2:]   # "s_<session id>"
    z = np.load(DC / "out" / f"{sid}_lp.npz")
    times = z["times"]
    groups = ["zs"] if name == "old" else ["zs", "desk"]   # desk models were trained on OLD -> never used there
    lps = {}
    for g in groups:
        lp = z[g].astype(np.float64)
        lps[g] = (lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)).astype(np.float32)
    if name == "old":
        segs = S2.segments(lps["zs"], times, 2.0, pad=0.5, thr=0.5)   # zero-shot-only segmentation
        rows = [json.loads(l) for l in open(SESS / sid / "phrases.jsonl")]
        meta = {"lines": [norm(r["phrase"]) for r in sorted((r for r in rows if r["event"] == "shown"), key=lambda r: r["idx"])]}
    else:
        dj = json.loads((SESS / sid / "decipher.json").read_text())
        segs = [tuple(r["frames30"]) for r in dj["segments"]]
        meta = {"session": sid, "decipher_rows": [{g: r.get(g, {}) for g in groups} for r in dj["segments"]]}
    meta["segments"] = [list(map(int, s)) for s in segs]
    items = [{"id": f"seg{k + 1}", "lp": {g: lps[g][s0:s1] for g in groups}} for k, (s0, s1) in enumerate(segs)]
    return items, meta


def cmd_cands(a) -> int:
    from phase0.analysis import autocorrect as AC
    from phase0.analysis import seqctc2 as S2
    if a.set not in ("kbd", "old") and not FROZEN2.exists():
        sys.exit("freeze the config before touching a new session")
    items, meta = load_set(a.set)
    reuse = "decipher_rows" in meta   # new sessions: Qwen-0.5B/greedy outputs from decipher.json (no MPS job here)
    if not reuse:
        S2._SC["qwen"] = AC.NLM(str(DC / "qwen2.5-0.5b"), device="cpu")
    qdec = {"kind": "qwen", "cfg": json.loads(TUNE_QWEN.read_text())["best"]}
    lm = S2.charlm()
    arrs = {}
    for k, it in enumerate(items):
        it["c"], it["secs"] = {}, {}
        for g, lp in it["lp"].items():
            t0 = time.time()
            nb = beam_nbest(lp, lm, CHAR["alpha"], CHAR["beta"], CHAR["beam"], CHAR["topn"])
            t1 = time.time()
            q = norm(meta["decipher_rows"][k][g]["qwen"]) if reuse else norm(S2.run_decoder(qdec, lp))
            t2 = time.time()
            it["c"][g] = {"char": nb, "greedy": norm(S2.run_decoder({"kind": "greedy"}, lp)), "qwen": q}
            it["secs"][g] = {"char_nbest": t1 - t0, "qwen05": t2 - t1}
            arrs[f"{it['id']}__{g}"] = lp.astype(np.float32)
        del it["lp"]
        print(a.set, it["id"], {g: (v["qwen"], v["char"][0][0] if v["char"] else "") for g, v in it["c"].items()}, flush=True)
    (OUT / "sets").mkdir(parents=True, exist_ok=True)
    (OUT / "sets" / f"{a.set}.json").write_text(json.dumps({"meta": meta, "char": CHAR, "qwen": qdec, "items": items}, indent=1))
    np.savez_compressed(OUT / "sets" / f"{a.set}_lp.npz", **arrs)
    return 0


# ============================================================================ stage 3: verify (project venv)
_B: set = set()


def blocked(t: str) -> bool:
    from phase0.analysis import seqctc2 as S2
    if not _B:
        _B.update(S2.blocklist())
    return any(S2.is_blocked(w, _B) for w in t.split())


def ctc_ll(lp: np.ndarray, texts: list[str]) -> np.ndarray:
    """exact log P(text | posteriors), blank := blank + 'other' key mass (as the decoders), optional edge spaces."""
    import torch
    import torch.nn.functional as F
    from phase0.analysis.seqctc import SYMS
    P = lp.astype(np.float64)
    M = np.concatenate([np.logaddexp(P[:, 0], P[:, 28])[:, None], P[:, 1:28]], 1)
    T = len(M)
    var, own = [], []
    for k, t in enumerate(texts):
        for v in {t, " " + t, t + " ", " " + t + " "}:
            var.append([SYMS.index(c) for c in v])
            own.append(k)
    lpt = torch.from_numpy(M).unsqueeze(1).expand(T, len(var), 28)
    tg = torch.tensor([c for v in var for c in v], dtype=torch.long)
    loss = F.ctc_loss(lpt, tg, torch.full((len(var),), T, dtype=torch.long), torch.tensor([len(v) for v in var]),
                      blank=0, reduction="none", zero_infinity=False).numpy()
    out = np.full(len(texts), -np.inf)
    for k, l in zip(own, -loss):
        out[k] = np.logaddexp(out[k], l)
    return out


def gkeys(groups):
    return list(groups) + (["ens"] if len(groups) > 1 else [])


def load_all(setname: str, llms=LLMS):
    d = json.loads((OUT / "sets" / f"{setname}.json").read_text())
    z = np.load(OUT / "sets" / f"{setname}_lp.npz")
    L = {m: json.loads(f.read_text()) for m in llms if (f := OUT / "llm" / m / f"{setname}.json").exists()}
    return d, z, L


_FZ: dict = {}


def _fuzzy(setname: str) -> dict:
    if setname not in _FZ:
        tag = os.environ.get("LLMDEC_FUZZY_TAG", "")
        f = OUT / "fuzzy" / f"{setname}{'__' + tag if tag else ''}.json"
        _FZ[setname] = json.loads(f.read_text())["items"] if f.exists() else {}
    return _FZ[setname]


def build_pools(setname: str, llms=LLMS):
    """per item, gkey: base candidates (char n-best, greedy, qwen of the group(s)) + per (llm, prompt) generated lines;
    arrays of CTC ll per model group, LLM log-prob per llm."""
    from phase0.analysis.decode import edit_distance
    d, z, L = load_all(setname, llms)
    pools = []
    t_ctc = []
    for it in d["items"]:
        groups = list(it["c"])
        P = {}
        for gk in gkeys(groups):
            gs = groups if gk == "ens" else [gk]
            base = []
            for g in gs:
                base += [t for t, _ in it["c"][g]["char"]] + [it["c"][g]["greedy"], it["c"][g]["qwen"]]
            gens = {(m, p): [norm(x) for x in L[m]["items"][it["id"]]["gens"][p].get(gk, [])] for m in L
                    for p in PROMPTS_ALL if p in L[m]["items"].get(it["id"], {}).get("gens", {})}
            texts = list(dict.fromkeys(x for x in base + [y for v in gens.values() for y in v] if x and not blocked(x)))
            nonfz = list(texts)
            fzt = [norm(x) for x in _fuzzy(setname).get(it["id"], {}).get(gk, [])]
            texts = list(dict.fromkeys(texts + [x for x in fzt if x and not blocked(x)]))
            nbase = {x for x in base if x}
            P[gk] = {"texts": texts, "base": np.array([t in nbase for t in texts]),
                     "nlex": np.array([sum(w in _lexset() for w in t.split()) for t in texts], float),
                     "npers": np.array([sum(_pers().get(w, 0) >= 2 for w in t.split()) for t in texts], float),
                     "plm": np.array([_plm_gain(t) for t in texts], float),
                     "isfz": np.array([t not in set(nonfz) for t in texts]),
                     "ntech": np.array([sum(w in _tech() for w in t.split()) for t in texts], float),
                     "fzd": np.array([0.0 if t in set(nonfz) else float(min(edit_distance(t.split(), u.split()) for u in nonfz))
                                      if nonfz else 0.0 for t in texts]),
                     "gen_rank": {k: np.array([v.index(t) if t in v else 99 for t in texts]) for k, v in gens.items()},
                     "gens": gens, "groups": gs}
        # CTC once per group over the union of texts it needs
        for g in groups:
            need = sorted({t for gk, p in P.items() if g in p["groups"] for t in p["texts"]})
            t0 = time.time()
            ll = dict(zip(need, ctc_ll(z[f"{it['id']}__{g}"], need))) if need else {}
            t_ctc.append((time.time() - t0) / max(len(need), 1))
            for gk, p in P.items():
                if g in p["groups"]:
                    p.setdefault("ctc", []).append(np.array([ll[t] for t in p["texts"]]))
        for gk, p in P.items():
            p["ctc"] = np.mean(p["ctc"], 0)
            p["lm"] = {m: np.array([L[m]["items"].get(it["id"], {}).get("lm", {}).get(t, [np.nan])[0] for t in p["texts"]])
                       for m in L}
            p["nw"] = np.array([len(t.split()) for t in p["texts"]], float)
            p["nc"] = np.array([len(t) for t in p["texts"]], float)
        pools.append({"id": it["id"], "ref": it.get("ref"), "P": P, "c": it["c"]})
    return d, pools, {"ctc_s_per_text": float(np.mean(t_ctc)) if t_ctc else None}


_LX: dict = {}


def _lexset() -> set:
    if "g" not in _LX:
        from phase0.analysis import seqctc2 as S2
        _LX["g"] = set(S2.lexicon().logp)
    return _LX["g"]


def _pers() -> dict:
    if "p" not in _LX:
        f = OUT / "personal" / "vocab.json"
        _LX["p"] = json.loads(f.read_text())["unigram"] if f.exists() else {}
    return _LX["p"]


def _tech() -> set:
    if "t" not in _LX:
        f = OUT / "techvocab" / "tech_vocab.json"
        _LX["t"] = set(json.loads(f.read_text())["terms"]) if f.exists() else set()
    return _LX["t"]


def _plm_gain(t: str, alpha: float = 0.3) -> float:
    """sum over words of log(alpha*P_personal_bigram(w|prev) + (1-alpha)*P_generic(w)) - log P_generic(w)"""
    if "v" not in _LX:
        f = OUT / "personal" / "vocab.json"
        v = json.loads(f.read_text()) if f.exists() else {"unigram": {}, "bigram": {}, "ctx": {}}
        from phase0.analysis import seqctc2 as S2
        lex = S2.lexicon().logp
        _LX["v"] = (v, float(sum(v["unigram"].values())) or 1.0, lex, min(lex.values()) - 2.0)
    v, N, lex, floor = _LX["v"]
    g = 0.0
    ws = t.split()
    for k, w in enumerate(ws):
        pg = float(np.exp(lex.get(w, floor)))
        pu = (v["unigram"].get(w, 0) + 0.1) / (N + 0.1 * 50000)
        prev = ws[k - 1] if k else ""
        c = v["ctx"].get(prev) if prev else None
        pp = (max(v["bigram"].get(f"{prev} {w}", 0) - 0.75, 0) / c[0] + 0.75 * c[1] / c[0] * pu) if c else pu
        g += min(float(np.log(alpha * pp + (1 - alpha) * pg) - np.log(pg)), 4.0)   # clip: OOV words hit the generic floor
    return g


def select(p, llm, prompt, K, lam, wb, cb, mu_lex=0.0, mu_pers=0.0, nu=0.0, lam_p=0.0, plm="p05b", use_fz=0, kappa=0.0, mu_tech=0.0):
    gr = p["gen_rank"].get((llm, prompt)) if llm else None
    ok = p["base"] | (gr < K) if gr is not None else p["base"]
    if use_fz and "isfz" in p:
        ok = ok | p["isfz"]
    lmv = p["lm"][llm] if llm and lam else 0.0
    s = p["ctc"] + (lam * lmv if llm and lam else 0.0) + wb * p["nw"] + cb * p["nc"] + mu_lex * p["nlex"] + mu_pers * p["npers"] + nu * p["plm"]
    if mu_tech and "ntech" in p:
        s = s + mu_tech * p["ntech"]
    if kappa and "fzd" in p:
        s = s - kappa * p["fzd"]
    if lam_p and plm in p["lm"]:
        s = s + lam_p * np.nan_to_num(p["lm"][plm] - np.nanmean(p["lm"][plm]), nan=-50.0)
    s = np.where(ok & np.isfinite(s), s, -np.inf)
    return p["texts"][int(np.argmax(s))] if np.isfinite(s).any() else ""


def rewrite_only(p, llm, prompt):
    g = [t for t in p["gens"][(llm, prompt)] if t and not blocked(t)]
    return g[0] if g else ""


def wer_parts(setname, d, hyps):
    from phase0.analysis.decode import edit_distance
    if setname == "old":
        from phase0.analysis.decipher import score_variant
        sv = score_variant(d["meta"]["lines"], hyps)
        return float(sv["wed"].sum()), float(sv["wn"].sum()), float(sv["ced"].sum()), float(sv["cn"].sum())
    refs = [it["ref"] for it in d["items"]]
    return (sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps)), sum(len(r.split()) for r in refs),
            sum(edit_distance(r, h) for r, h in zip(refs, hyps)), sum(len(r) for r in refs))


def cmd_tune(a) -> int:
    sets = {}
    for s in ("kbd", "old"):
        d, pools, tm = build_pools(s)
        sets[s] = (d, pools)
    res = {"sets": {s: len(v[1]) for s, v in sets.items()}, "objective": "mean of kbd WER and old-desk(zs) WER",
           "baselines": {}, "rewrite_only": {}, "trials": []}

    def evalf(fn):
        r = {}
        for s, (d, pools) in sets.items():
            we, wn, ce, cn = wer_parts(s, d, [fn(p) for p in pools])
            r[s] = {"wer": we / wn, "cer": ce / cn}
        r["obj"] = 0.5 * (r["kbd"]["wer"] + r["old"]["wer"])
        return r

    for nm, fn in {"qwen05": lambda p: p["c"]["zs"]["qwen"], "char1": lambda p: p["c"]["zs"]["char"][0][0] if p["c"]["zs"]["char"] else "",
                   "greedy": lambda p: p["c"]["zs"]["greedy"]}.items():
        res["baselines"][nm] = evalf(fn)
        print("baseline", nm, res["baselines"][nm], flush=True)
    for m in LLMS:
        for pr in GRID["prompt"]:
            res["rewrite_only"][f"{m}/{pr}"] = evalf(lambda p: rewrite_only(p["P"]["zs"], m, pr))
            print("rewrite-only", m, pr, res["rewrite_only"][f"{m}/{pr}"], flush=True)
    for m in LLMS:
        for pr, K, lam, wb, cb in itertools.product(GRID["prompt"], GRID["K"], GRID["lam"], GRID["wb"], GRID["cb"]):
            if K == 0 and pr == "p2":
                continue
            r = evalf(lambda p: select(p["P"]["zs"], m, pr, K, lam, wb, cb))
            res["trials"].append({"llm": m, "prompt": pr, "K": K, "lam": lam, "wb": wb, "cb": cb, **r})
    RES.mkdir(parents=True, exist_ok=True)
    best = {}
    for m in LLMS:
        tr = sorted((t for t in res["trials"] if t["llm"] == m), key=lambda t: (t["obj"], t["kbd"]["cer"] + t["old"]["cer"]))
        best[m] = tr[0]
        print("best", m, tr[0], flush=True)
        for t in tr[1:6]:
            print("   next", t, flush=True)
    res["best"] = best
    (RES / "tuning.json").write_text(json.dumps(res, indent=1))
    return 0


def cmd_tune2(a) -> int:
    """focused 8B generate-and-verify grid incl. context/profile prompts and lexicon/personal word bonuses"""
    m = a.llm
    llms = (m,) + ((a.plm,) if a.plm else ())
    sets = {s: build_pools(s, llms)[:2] for s in ("kbd", "old")}
    res = {"llm": m, "objective": "mean of kbd WER and old-desk(zs) WER", "grid": GRID2, "trials": []}
    grid = dict(GRID2, lam_p=(0.0, 0.5, 1.0) if a.plm else (0.0,))
    for pr, K, lam, wb, cb, ml, mp, nu, lp_ in itertools.product(*grid.values()):
        r = {}
        for s, (d, pools) in sets.items():
            if not all((m, pr) in pp["P"]["zs"]["gen_rank"] for pp in pools):
                break
            we, wn, ce, cn = wer_parts(s, d, [select(pp["P"]["zs"], m, pr, K, lam, wb, cb, ml, mp, nu, lp_, a.plm or "p05b")
                                              for pp in pools])
            r[s] = {"wer": we / wn, "cer": ce / cn}
        if len(r) < 2:
            continue
        r["obj"] = 0.5 * (r["kbd"]["wer"] + r["old"]["wer"])
        res["trials"].append({"prompt": pr, "K": K, "lam": lam, "wb": wb, "cb": cb, "mu_lex": ml, "mu_pers": mp, "nu": nu,
                              "lam_p": lp_, **r})
    tr = sorted(res["trials"], key=lambda t: (t["obj"], t["kbd"]["cer"] + t["old"]["cer"]))
    res["best"] = tr[0] if tr else None
    # best per ablation slice (same search, restricted) for the report
    def best_of(pred):
        c = [t for t in tr if pred(t)]
        return c[0] if c else None
    nop = lambda t: t["mu_pers"] == 0 and t["nu"] == 0 and t.get("lam_p", 0) == 0  # noqa: E731
    res["slices"] = {"generic_no_ctx(p3/p4)": best_of(lambda t: t["prompt"] in ("p3", "p4") and nop(t)),
                     "generic_+context(p6)": best_of(lambda t: t["prompt"] == "p6" and nop(t)),
                     "generic_best": best_of(lambda t: t["prompt"] in ("p3", "p4", "p6") and nop(t))}
    for k, v in res["slices"].items():
        print(k, v and {kk: v[kk] for kk in ("prompt", "K", "lam", "wb", "cb", "mu_lex", "mu_pers", "nu")},
              v and f"kbd {v['kbd']['wer']:.3f} old {v['old']['wer']:.3f} obj {v['obj']:.3f}", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"tuning_v2_{m}{'_' + a.plm if a.plm else ''}.json").write_text(json.dumps(res, indent=1))
    return 0


GRID3 = {"prompt": ("p7", "p8"), "use_fz": (0, 1), "kappa": (0.0, 1.0, 2.0), "lam": (2.0, 3.0, 4.0), "cb": (2.0, 3.0),
         "lam_p": (0.5, 1.0, 1.5), "mu_pers": (0.0, 1.0)}
V2_PERSONAL_FIXED = {"K": 5, "wb": -1.0, "mu_lex": 0.0, "nu": 0.0}


def cmd_tune3(a) -> int:
    """v3: fuzzy keyboard-aware candidates + jargon prompt p8 around the frozen v2 personal config (tuning sets only)"""
    llms = ("8b", "p05b")
    sets = {s: build_pools(s, llms)[:2] for s in ("kbd", "old")}
    res = {"grid": GRID3, "fixed": V2_PERSONAL_FIXED, "trials": []}
    for vals in itertools.product(*GRID3.values()):
        g = dict(zip(GRID3, vals))
        if not g["use_fz"] and g["kappa"]:
            continue
        r = {}
        for s, (d, pools) in sets.items():
            if not all(("8b", g["prompt"]) in pp["P"]["zs"]["gen_rank"] for pp in pools):
                break
            hy = [select(pp["P"]["zs"], "8b", g["prompt"], V2_PERSONAL_FIXED["K"], g["lam"], V2_PERSONAL_FIXED["wb"], g["cb"],
                         V2_PERSONAL_FIXED["mu_lex"], g["mu_pers"], V2_PERSONAL_FIXED["nu"], g["lam_p"], "p05b", g["use_fz"], g["kappa"])
                  for pp in pools]
            we, wn, ce, cn = wer_parts(s, d, hy)
            r[s] = {"wer": we / wn, "cer": ce / cn}
        if len(r) < 2:
            continue
        r["obj"] = 0.5 * (r["kbd"]["wer"] + r["old"]["wer"])
        res["trials"].append({**g, **V2_PERSONAL_FIXED, **r})
    tr = res["trials"]
    res["best_personal_kbd"] = min(tr, key=lambda t: (t["kbd"]["wer"], t["kbd"]["cer"], t["old"]["wer"]))
    res["best_personal_kbd_no_fuzzy"] = min((t for t in tr if not t["use_fz"]), key=lambda t: (t["kbd"]["wer"], t["kbd"]["cer"], t["old"]["wer"]))
    res["best_joint"] = min(tr, key=lambda t: (t["obj"], t["kbd"]["cer"] + t["old"]["cer"]))
    # pool oracle with / without fuzzy candidates
    from phase0.analysis.decode import edit_distance
    from phase0.analysis.decipher import score_variant
    orc = {}
    for s, (d, pools) in sets.items():
        if s == "kbd":
            refs = [pp["ref"] for pp in pools]
        else:
            sv = score_variant(d["meta"]["lines"], [pp["c"]["zs"]["qwen"] for pp in pools])
            refs = [" ".join(d["meta"]["lines"][i] for i in sv["seg_lines"][k]) for k in range(len(pools))]
        for nm, keep in (("pool_v2", lambda P: ~P["isfz"]), ("pool_v3_with_fuzzy", lambda P: np.ones(len(P["texts"]), bool))):
            hy = []
            for pp, ref in zip(pools, refs):
                P = pp["P"]["zs"]
                tx = [t for t, k in zip(P["texts"], keep(P)) if k]
                hy.append(min(tx, key=lambda t: (edit_distance(ref.split(), t.split()), edit_distance(ref, t))) if tx else "")
            we, wn, ce, cn = wer_parts(s, d, hy)
            orc.setdefault(s, {})[nm] = {"oracle_wer": we / wn, "oracle_cer": ce / cn}
    res["oracle"] = orc
    for k in ("best_personal_kbd", "best_personal_kbd_no_fuzzy", "best_joint"):
        t = res[k]
        print(k, {x: t[x] for x in GRID3}, f"kbd {t['kbd']['wer']:.3f} old {t['old']['wer']:.3f} obj {t['obj']:.3f}", flush=True)
    print("oracle", json.dumps(orc), flush=True)
    (RES / "tuning_v3.json").write_text(json.dumps(res, indent=1))
    return 0


def cmd_tune31(a) -> int:
    """v3.1: v3 frozen personal weights fixed; only the tech-vocab bonus mu_tech is tuned (fuzzy candidates from tag v31)"""
    assert os.environ.get("LLMDEC_FUZZY_TAG") == "v31", "set LLMDEC_FUZZY_TAG=v31"
    v3 = json.loads((RES / "frozen_config_v3.json").read_text())["personal_v3"]["run"]
    sets = {s: build_pools(s, ("8b", "p05b"))[:2] for s in ("kbd", "old")}
    res = {"fixed_v3_run": v3, "grid_mu_tech": (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0), "trials": []}
    for mt in res["grid_mu_tech"]:
        r = {}
        for s, (d, pools) in sets.items():
            hy = [select(pp["P"]["zs"], "8b", v3["prompt"], v3["K"], v3["lam"], v3["wb"], v3["cb"], v3["mu_lex"], v3["mu_pers"],
                         v3["nu"], v3["lam_p"], v3["plm"], v3["use_fz"], v3["kappa"], mt) for pp in pools]
            we, wn, ce, cn = wer_parts(s, d, hy)
            r[s] = {"wer": we / wn, "cer": ce / cn}
        r["obj"] = 0.5 * (r["kbd"]["wer"] + r["old"]["wer"])
        res["trials"].append({"mu_tech": mt, **r})
        print(f"mu_tech {mt:+.1f}  kbd {r['kbd']['wer']:.3f}/{r['kbd']['cer']:.3f}  old {r['old']['wer']:.3f}/{r['old']['cer']:.3f}  obj {r['obj']:.3f}", flush=True)
    res["best"] = min(res["trials"], key=lambda t: (t["obj"], t["kbd"]["cer"] + t["old"]["cer"], abs(t["mu_tech"])))
    from phase0.analysis.decode import edit_distance
    from phase0.analysis.decipher import score_variant
    orc = {}
    for s, (d, pools) in sets.items():
        if s == "kbd":
            refs = [pp["ref"] for pp in pools]
        else:
            sv = score_variant(d["meta"]["lines"], [pp["c"]["zs"]["qwen"] for pp in pools])
            refs = [" ".join(d["meta"]["lines"][i] for i in sv["seg_lines"][k]) for k in range(len(pools))]
        hy = [min(pp["P"]["zs"]["texts"], key=lambda t: (edit_distance(ref.split(), t.split()), edit_distance(ref, t))) for pp, ref in zip(pools, refs)]
        we, wn, ce, cn = wer_parts(s, d, hy)
        orc[s] = {"oracle_wer_pool_v31": we / wn, "oracle_cer_pool_v31": ce / cn}
    res["oracle"] = orc
    print("best", res["best"], "oracle", orc, flush=True)
    (RES / "tuning_v3_1.json").write_text(json.dumps(res, indent=1))
    return 0


def cmd_freeze(a) -> int:
    if FROZEN.exists():
        sys.exit(f"{FROZEN} exists; refusing to overwrite")
    t = json.loads((RES / "tuning.json").read_text())
    ro = {m: min(GRID["prompt"], key=lambda pr: t["rewrite_only"][f"{m}/{pr}"]["obj"]) for m in LLMS}
    gv = {m: {k: t["best"][m][k] for k in ("prompt", "K", "lam", "wb", "cb")} for m in LLMS}
    chosen = min(LLMS, key=lambda m: t["best"][m]["obj"])
    cfg = {"frozen_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "tuned_on": ["keyboard windows (hwt LOSO/held out-of-fold, 50 windows)", f"{OLD} zero-shot hwtmix continuous, zs-only segmentation"],
           "objective": t["objective"], "char_nbest": CHAR, "qwen05": json.loads(TUNE_QWEN.read_text())["best"],
           "generate_and_verify": gv, "rewrite_only_prompt": ro, "chosen_llm": chosen,
           "tuning_scores": {"gv": {m: t["best"][m] for m in LLMS}, "rewrite_only": {m: t["rewrite_only"][f"{m}/{ro[m]}"] for m in LLMS},
                             "baselines": t["baselines"]},
           "ensemble": "CTC ll = mean of zs and desk model-group ll; candidates = union; LLM prompt gets merged guesses (untuned, no data)",
           "new_session_segments": "decipher.json frames30 (same as the Qwen-0.5B baseline)",
           "disclosure": ("the designer had seen the new session's truth.txt and Qwen/char outputs (in the task brief) before "
                          "writing prompts p1-p4; prompts are generic (no domain/jargon hints). A small dry-run grid on OLD with "
                          "3b p1-p3 was run before the full tune (tuning data only). Keyboard set uses seqctc hwt LOSO posteriors "
                          "because the hwtmix fold npz files no longer exist."),
           "llm_repos": {"3b": "mlx-community/Qwen2.5-3B-Instruct-4bit", "8b": "mlx-community/Qwen3-8B-4bit"},
           "code_md5": {f: hashlib.md5(Path(f"phase0/analysis/{f}").read_bytes()).hexdigest() for f in ("llmdec.py", "llmdec_mlx.py")}}
    FROZEN.write_text(json.dumps(cfg, indent=1))
    print(json.dumps(cfg, indent=1))
    return 0


def cmd_final(a) -> int:
    from phase0.analysis.seqctc import boot_ratio
    from phase0.analysis.decipher import score_variant
    cfg = json.loads(FROZEN.read_text())
    d, pools, tm = build_pools("new")
    L = {m: json.loads((OUT / "llm" / m / "new.json").read_text()) for m in LLMS}
    variants = {}
    for gk in ("zs", "desk", "ens"):
        if gk != "ens":
            variants[f"{gk}/qwen05"] = [p["c"][gk]["qwen"] for p in pools]
            variants[f"{gk}/char1"] = [p["c"][gk]["char"][0][0] if p["c"][gk]["char"] else "" for p in pools]
        for m in LLMS:
            variants[f"{gk}/rewrite_{m}"] = [rewrite_only(p["P"][gk], m, cfg["rewrite_only_prompt"][m]) for p in pools]
            g = cfg["generate_and_verify"][m]
            variants[f"{gk}/gv_{m}"] = [select(p["P"][gk], m, g["prompt"], g["K"], g["lam"], g["wb"], g["cb"]) for p in pools]
    truth = SESS / NEW / "truth.txt"
    lines = [norm(l) for l in truth.read_text().splitlines() if norm(l)]
    out = {"frozen_config_sha": hashlib.md5(FROZEN.read_bytes()).hexdigest(), "scored_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "lines": lines, "variants": {}}
    for k, hyps in variants.items():
        sv = score_variant(lines, hyps)
        out["variants"][k] = {"wer": boot_ratio(sv["wed"], sv["wn"]), "cer": boot_ratio(sv["ced"], sv["cn"]),
                              "words_correct": boot_ratio(sv["wok"], sv["wn"]), "hyps": hyps, "hyp_by_line": sv["hyp_by_line"]}
        r = out["variants"][k]
        print(f"{k:<18} WER {r['wer'][0]:.3f} [{r['wer'][1]:.3f},{r['wer'][2]:.3f}] words {100 * r['words_correct'][0]:.0f}%  " + " | ".join(hyps), flush=True)
    # runtime per phrase (single-model zs config of each LLM)
    rt = {"ctc_s_per_text": tm["ctc_s_per_text"],
          "cands_s_per_phrase": {g: float(np.mean([it["secs"][g]["char_nbest"] + it["secs"][g]["qwen05"] for it in d["items"]])) for g in ("zs", "desk")}}
    for m in LLMS:
        its = L[m]["items"].values()
        p = cfg["generate_and_verify"][m]["prompt"]
        rt[m] = {"gen_s_per_phrase": float(np.mean([it["secs"]["gen"][p]["zs"] for it in its])),
                 "lm_s_per_text": float(np.sum([it["secs"]["lm"] for it in its]) / max(1, sum(it["secs"]["n_lm"] for it in its))),
                 "pool_size_zs": float(np.mean([len(pp["P"]["zs"]["texts"]) for pp in pools])), "load_s": L[m].get("load_s")}
        rt[m]["gv_total_s_per_phrase_zs"] = (rt["cands_s_per_phrase"]["zs"] + rt[m]["gen_s_per_phrase"]
                                             + rt[m]["pool_size_zs"] * (rt[m]["lm_s_per_text"] + rt["ctc_s_per_text"]))
    out["runtime"] = rt
    print(json.dumps(rt, indent=1))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "new_session_score.json").write_text(json.dumps(out, indent=1, default=float))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("cands")
    s.add_argument("set", help="kbd | old | new | s_<session id>")
    s.set_defaults(fn=cmd_cands)
    for nm, fn in (("tune", cmd_tune), ("freeze", cmd_freeze), ("final", cmd_final)):
        sub.add_parser(nm).set_defaults(fn=fn)
    sub.add_parser("tune3").set_defaults(fn=cmd_tune3)
    sub.add_parser("tune31").set_defaults(fn=cmd_tune31)
    s = sub.add_parser("tune2")
    s.add_argument("--llm", default="8b")
    s.add_argument("--plm", default="", help="personal LLM key with lm scores (adds lam_p to the grid)")
    s.set_defaults(fn=cmd_tune2)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
