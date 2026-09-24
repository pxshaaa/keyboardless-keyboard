"""v4 M1: lexicon-constrained CTC word beam as an extra candidate generator (production port of swipe_p1, frozen for v4).
Prefix beam over a trie of the merged lexicon (generic + personal, blocklist applied) on the ensemble posteriors;
optional spaces (word boundary without a space emission, penalty gamma), extra spaces, unigram lookahead, personal bigram
prior (alpha), word insertion bonus (beta), name/personal boost. Returns the top-K texts per config (beam order).
The texts only EXTEND the v3 gen-verify pool; char n-best / greedy / 8B rewrites / fuzzy stay in the pool, so words
outside the lexicon remain reachable (letter-by-letter escape path)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

NEG = -1e30
MERGED = Path(".cache/personal/lexicon_merged.json")
GENERIC = Path(".cache/autocorrect/lexicon.json")
PERSONAL = Path(".cache/personal/lexicon_personal.json")
BIGRAM = Path(".cache/personal/bigram_personal.npz")
NAMES = {"claude", "bifi", "codex"}   # boost set seed as in swipe_p1 (from blind-1/2 errors: disclosed, dev only)
# frozen v4 configs (selected on dev: see results/v4/wordbeam_choice.json)
CONFIGS = {"small": dict(alpha=1.0, beta=2.0, gamma=-8.0, boost=2.0, beam=128),    # = swipe_p1 --grid small
           "big": dict(alpha=2.0, beta=4.0, gamma=-4.0, boost=2.0, beam=256, prune=-14.0)}   # = swipe_p1 --grid big --beam 256 --prune -14 (32/32 segments identical)
SEARCH = dict(prune=-8.0, topk=50, xspace=-3.0)
FILES = (MERGED, GENERIC, PERSONAL, BIGRAM, Path("models/vocab_blocklist.txt"))


def lae(a, b):
    if a < b:
        a, b = b, a
    return a if b <= NEG else a + math.log1p(math.exp(b - a))


def m28(lp: np.ndarray) -> np.ndarray:
    P = lp.astype(np.float64)
    return np.concatenate([np.logaddexp(P[:, 0], P[:, 28])[:, None], P[:, 1:28]], 1)


def ens28(lps: list[np.ndarray]) -> np.ndarray:
    M = np.mean([m28(l) for l in lps], 0)
    return M - np.logaddexp.reduce(M, axis=1, keepdims=True)


class LM:
    def __init__(self):
        from phase0.analysis import seqctc2 as S2
        lex = json.loads(MERGED.read_text())
        B = S2.blocklist()
        self.logp = {w: v for w, v in lex.items() if w.isalpha() and w.isascii() and not S2.is_blocked(w, B)}
        self.bg = S2.BigramScorer(BIGRAM)
        gen = set(json.loads(GENERIC.read_text()))
        pers = json.loads(PERSONAL.read_text())["personal"]
        self.boostset = set(NAMES) | {w for w, v in pers.items() if v.get("count", 0) >= 10 and w not in gen and w in self.logp}
        self.ch, self.word, self.best = [dict()], [False], [NEG]
        for w, lp in self.logp.items():
            n = 0
            for c in w:
                nx = self.ch[n].get(c)
                if nx is None:
                    nx = len(self.ch)
                    self.ch[n][c] = nx
                    self.ch.append(dict())
                    self.word.append(False)
                    self.best.append(NEG)
                n = nx
                self.best[n] = max(self.best[n], lp)
            self.word[n] = True
        self.best[0] = 0.0
        self._c: dict = {}

    def big(self, prev, w):
        k = (prev, w)
        v = self._c.get(k)
        if v is None:
            v = self.bg.score([((prev,) if prev else (), w)])[0] if w in self.bg.ix else self.logp.get(w, -20.0)
            self._c[k] = v
        return v


def beam(M: np.ndarray, lm: LM, cfg: dict) -> list[tuple[str, float]]:
    A, Bt, G, BO = cfg["alpha"], cfg["beta"], cfg["gamma"], cfg["boost"]
    SP = 27
    beams = {((), "", -1): [0.0, NEG, 0.0, 0]}   # key (words, partial, last) -> [pb, pnb, lm, node]
    for t in range(len(M)):
        row = M[t]
        cand = [c for c in range(1, 28) if row[c] > cfg["prune"]]
        nb: dict = {}

        def add(key, b, n, lmv, node):
            e = nb.get(key)
            if e is None:
                nb[key] = [b, n, lmv, node]
            else:
                e[0] = lae(e[0], b)
                e[1] = lae(e[1], n)
        for key, (pb, pnb, lmv, node) in beams.items():
            words, partial, last = key
            ptot = lae(pb, pnb)
            add(key, ptot + row[0], NEG, lmv, node)
            if last >= 0 and pnb > NEG:
                add(key, NEG, pnb + row[last], lmv, node)
            la = lm.best[node] if node else 0.0
            term = bool(partial) and lm.word[node]
            if term:
                wsc = A * lm.big(words[-1] if words else "", partial) + Bt + (BO if partial in lm.boostset else 0.0) - A * la
            for c in cand:
                base = pb if c == last else ptot
                if base <= NEG:
                    continue
                if c == SP:
                    if term:
                        add((words + (partial,), "", SP), NEG, base + row[SP], lmv + wsc, 0)
                    elif not partial:
                        add((words, "", SP), NEG, base + row[SP] + (cfg["xspace"] if words else 0.0), lmv, 0)
                    continue
                ch = chr(96 + c)
                nx = lm.ch[node].get(ch)
                if nx is not None:
                    add((words, partial + ch, c), NEG, base + row[c], lmv - A * la + A * lm.best[nx], nx)
                if term:
                    r = lm.ch[0].get(ch)
                    if r is not None:
                        add((words + (partial,), ch, c), NEG, base + row[c] + G, lmv + wsc + A * lm.best[r], r)
        beams = dict(sorted(nb.items(), key=lambda kv: -(lae(kv[1][0], kv[1][1]) + kv[1][2]))[:cfg["beam"]])
    fin: dict = {}
    for (words, partial, last), (pb, pnb, lmv, node) in beams.items():
        sc = lae(pb, pnb) + lmv
        if partial:
            if not lm.word[node]:
                continue
            sc += A * lm.big(words[-1] if words else "", partial) + Bt + (BO if partial in lm.boostset else 0.0) - A * lm.best[node]
            words = words + (partial,)
        txt = " ".join(words)
        fin[txt] = lae(fin.get(txt, NEG), sc)
    return sorted(fin.items(), key=lambda kv: -kv[1])


_LM: LM | None = None


def get_lm() -> LM:
    global _LM
    if _LM is None:
        _LM = LM()
    return _LM


def candidates(lps: list[np.ndarray], configs=("small", "big")) -> dict[str, list[str]]:
    """-> {config name: top-K texts (beam order)} for one segment (lps = one array per posterior group)."""
    lm, M = get_lm(), ens28(lps)
    return {c: [t for t, _ in beam(M, lm, dict(SEARCH, **CONFIGS[c]))[:SEARCH["topk"]]] for c in configs}


def _seg_job(args):
    key, lps, configs = args
    return key, candidates(lps, configs)


def run_set(setname: str, configs=("small", "big"), procs: int = 3) -> dict:
    """all segments of .cache/llmdec/sets/<setname> -> {item id: {config: [texts]}}"""
    from multiprocessing import Pool
    d = json.loads((Path(".cache/llmdec/sets") / f"{setname}.json").read_text())
    z = np.load(Path(".cache/llmdec/sets") / f"{setname}_lp.npz")
    jobs = [(it["id"], [z[f"{it['id']}__{g}"] for g in it["c"]], tuple(configs)) for it in d["items"]]
    if procs <= 1:
        return dict(map(_seg_job, jobs))
    with Pool(procs) as pool:
        return dict(pool.map(_seg_job, jobs))
