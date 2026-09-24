"""Keyboard-aware fuzzy candidate generator over the merged (generic + personal) lexicon.
Per seed sentence, each word slot gets lexicon words within a small weighted edit distance (German QWERTZ neighbouring-key
substitutions cheap, single insert/delete cheap) + word splits/merges; prior = mixed generic/personal unigram. A beam over
slots yields sentence candidates, pre-filtered by exact CTC log-lik. Output: .cache/llmdec/fuzzy/<set>.json {id: {gk: [texts]}}.
  PYTHONPATH=. .venv/bin/python -m phase0.analysis.llmdec_fuzzy {kbd|old|s_<sid>} [--test]"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np

OUT = Path(".cache/llmdec")
FZ = OUT / "fuzzy"
ROWS = [("qwertzuiopü", 0.0), ("asdfghjklöä", 0.25), ("yxcvbnm", 0.75)]
POS = {c: (r, k + off) for r, (row, off) in enumerate(ROWS) for k, c in enumerate(row)}
CFG = {"sub_near": 0.5, "sub_far": 1.5, "indel": 0.7, "swap": 0.6, "max_cost": 2.5, "max_cost_short": 1.0, "k_per_slot": 12,
       "a_cost": 1.5, "b_prior": 0.4, "pers_bonus": 2.0, "pers_min_count": 3, "beam": 60, "per_seed": 60, "n_seeds": 8,
       "keep_ctc": 40, "w_pers": 0.3, "split_cost": 0.6}
# hand-set (not tuned); the example pairs used to sanity-check reachability came from blind-1 errors -> v3 blind-1 is NOT blind


def sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    pa, pb = POS.get(a), POS.get(b)
    if pa and pb and math.hypot(pa[0] - pb[0], pa[1] - pb[1]) <= 1.3:
        return CFG["sub_near"]
    return CFG["sub_far"]


def wdist(s: str, t: str, cap: float) -> float:
    n, m = len(s), len(t)
    if abs(n - m) * CFG["indel"] > cap:
        return cap + 1
    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1) * CFG["indel"]
    D[0, :] = np.arange(m + 1) * CFG["indel"]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            v = min(D[i - 1, j] + CFG["indel"], D[i, j - 1] + CFG["indel"], D[i - 1, j - 1] + sub_cost(s[i - 1], t[j - 1]))
            if i > 1 and j > 1 and s[i - 1] == t[j - 2] and s[i - 2] == t[j - 1]:
                v = min(v, D[i - 2, j - 2] + CFG["swap"])
            D[i, j] = v
        if D[i].min() > cap:
            return cap + 1
    return float(D[n, m])


def deletes(w: str, d: int = 2) -> set:
    out, frontier = {w}, {w}
    for _ in range(d):
        frontier = {x[:i] + x[i + 1:] for x in frontier for i in range(len(x))}
        out |= frontier
    return out


TAG = os.environ.get("LLMDEC_FUZZY_TAG", "")
TECH_F = OUT / "techvocab" / "tech_vocab.json"
TECH_PRIOR_DROP = 3.0   # tech-only terms get prior = generic floor - this (low: must not swamp common English)


def sfx() -> str:
    return f"__{TAG}" if TAG else ""


class Lex:
    def __init__(self):
        f = FZ / f"index{sfx()}.pkl"
        if f.exists():
            self.prior, self.index, self.pers = pickle.loads(f.read_bytes())
            return
        from phase0.analysis import seqctc2 as S2
        gen = S2.lexicon().logp
        voc = json.loads((OUT / "personal" / "vocab.json").read_text())["unigram"]
        N = float(sum(voc.values()))
        B = S2.blocklist()
        words = set(gen) | {w for w, c in voc.items() if c >= CFG["pers_min_count"] and w.isalpha() and w.isascii()
                            and 2 <= len(w) <= 15 and not S2.is_blocked(w, B)}
        floor = min(gen.values())
        tech = set()
        if TAG == "v31":
            tech = {w for w in json.loads(TECH_F.read_text())["terms"] if not S2.is_blocked(w, B)} - words
            words |= tech
        self.prior = {w: math.log((1 - CFG["w_pers"]) * math.exp(gen.get(w, floor - 3)) + CFG["w_pers"] * voc.get(w, 0) / N)
                      for w in words if w not in tech}
        self.prior.update({w: floor - TECH_PRIOR_DROP for w in tech})
        self.pers = {w for w in words if voc.get(w, 0) >= 3 and w not in gen}
        self.index: dict = {}
        for w in words:
            if len(w) <= 15:
                for x in deletes(w, 2 if len(w) > 3 else 1):
                    self.index.setdefault(x, []).append(w)
        FZ.mkdir(parents=True, exist_ok=True)
        f.write_bytes(pickle.dumps((self.prior, self.index, self.pers)))

    def near(self, w: str) -> list[tuple[str, float]]:
        cap = CFG["max_cost"] if len(w) > 3 else CFG["max_cost_short"]
        cands = set()
        for x in deletes(w, 2 if len(w) > 3 else 1):
            cands.update(self.index.get(x, ()))
        out = []
        for c in cands:
            if c == w:
                continue
            d = wdist(w, c, cap)
            if d <= cap:
                out.append((c, d))
        out.sort(key=lambda cd: -self.alt_score(w, cd[0], cd[1]))
        return out[:CFG["k_per_slot"]]

    def alt_score(self, w: str, c: str, d: float) -> float:
        return (-CFG["a_cost"] * d + CFG["b_prior"] * (self.prior[c] - self.prior.get(w, -25.0))
                + (CFG["pers_bonus"] if c in self.pers else 0.0))


def expand(lex: Lex, seed: str) -> list[tuple[str, float]]:
    ws = seed.split()
    if not ws:
        return []
    slots = []
    for k, w in enumerate(ws):
        alts = [((w,), 0.0)]
        pw = lex.prior.get(w, -25.0)
        for c, d in lex.near(w):
            alts.append(((c,), lex.alt_score(w, c, d)))
        for i in range(1, len(w)):   # split
            a, b = w[:i], w[i:]
            if a in lex.prior and b in lex.prior and len(a) > 1 and len(b) > 1:
                alts.append(((a, b), -CFG["a_cost"] * CFG["split_cost"] + CFG["b_prior"] * (lex.prior[a] + lex.prior[b] - pw)))
        slots.append(alts)
    beams = [((), 0.0, False)]   # (words, score, previous slot merged into this one)
    for k, alts in enumerate(slots):
        nb = []
        for words, sc, skip in beams:
            if skip:
                nb.append((words, sc, False))
                continue
            for a, s in alts:
                nb.append((words + a, sc + s, False))
            if k + 1 < len(ws):   # merge with next word
                m = ws[k] + ws[k + 1]
                if m in lex.prior:
                    nb.append((words + (m,), sc - CFG["a_cost"] * CFG["split_cost"], True))
        nb.sort(key=lambda x: -x[1])
        beams = nb[:CFG["beam"]]
    out = {}
    for words, sc, _ in beams:
        t = " ".join(words)
        out[t] = max(out.get(t, -1e9), sc)
    return sorted(out.items(), key=lambda kv: -kv[1])[:CFG["per_seed"]]


def seeds_of(it, g, gens) -> list[str]:
    c = it["c"][g]
    s = [c["qwen"]] + [t for t, _ in c["char"][:3]] + [c["greedy"]]
    for p in ("p3", "p6", "p7", "p8"):
        s += (gens.get(p, {}).get(g) or gens.get(p, {}).get("ens") or [])[:1]
    from phase0.analysis.llmdec import norm
    return list(dict.fromkeys(norm(x) for x in s if x))[:CFG["n_seeds"]]


def run_set(setname: str, llm: str = "8b") -> dict:
    from phase0.analysis.llmdec import ctc_ll
    lex = Lex()
    d = json.loads((OUT / "sets" / f"{setname}.json").read_text())
    z = np.load(OUT / "sets" / f"{setname}_lp.npz")
    lf = OUT / "llm" / llm / f"{setname}.json"
    G = json.loads(lf.read_text())["items"] if lf.exists() else {}
    res, t0 = {}, time.time()
    for it in d["items"]:
        groups = list(it["c"])
        gks = groups + (["ens"] if len(groups) > 1 else [])
        gens = G.get(it["id"], {}).get("gens", {})
        res[it["id"]] = {}
        for gk in gks:
            gs = groups if gk == "ens" else [gk]
            texts = {}
            for g in gs:
                for sd in seeds_of(it, g, gens):
                    for t, sc in expand(lex, sd):
                        texts[t] = max(texts.get(t, -1e9), sc)
            cand = list(texts)
            if not cand:
                res[it["id"]][gk] = []
                continue
            ll = np.mean([ctc_ll(z[f"{it['id']}__{g}"], cand) for g in gs], 0)
            order = np.argsort(-ll)[:CFG["keep_ctc"]]
            res[it["id"]][gk] = [cand[i] for i in order if np.isfinite(ll[i])]
    FZ.mkdir(parents=True, exist_ok=True)
    (FZ / f"{setname}{sfx()}.json").write_text(json.dumps({"cfg": CFG, "tag": TAG, "items": res,
                                                         "secs_per_item": (time.time() - t0) / max(len(d["items"]), 1)}))
    print(setname, "items", len(res), f"{(time.time() - t0) / max(len(res), 1):.1f}s/item", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("set", nargs="?")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.test:
        t0 = time.time()
        lex = Lex()
        print(f"lexicon {len(lex.prior)} words, index {len(lex.index)} keys, {time.time() - t0:.0f}s")
        for src, tgt in (("dog", "div"), ("cope", "codex"), ("enter", "center"), ("aice", "claude"), ("repot", "report"),
                         ("fineline", "timeline")):
            nb = lex.near(src)
            r = [c for c, _ in nb]
            print(f"{src:>9} -> {tgt:<9} in lexicon={tgt in lex.prior} rank={r.index(tgt) + 1 if tgt in r else None} "
                  f"cost={wdist(src, tgt, 9):.1f} top={r[:5]}")
        for s in ("can you enter the dog if you can", "aice code or cope which one is the ahead"):
            print(s, "->", [t for t, _ in expand(lex, s)[:6]])
        return 0
    run_set(a.set)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
