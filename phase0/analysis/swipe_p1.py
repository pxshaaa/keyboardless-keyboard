"""P1 lexicon-constrained CTC word beam (closed personal vocabulary, swipe-style): prefix beam over a trie of the merged
lexicon on the v3 posteriors; optional spaces (word boundary without a space emission, penalty gamma), extra spaces,
unigram lookahead, personal bigram prior (alpha), word insertion bonus (beta), name/personal boost.
Outputs top-K per segment, top-1 WER, pool oracle, union-with-frozen-pool oracle, and missing-word recall.
  PYTHONPATH=. .venv/bin/python -m phase0.analysis.swipe_p1 [--sets old,b1,b2] [--grid small|one] [--pad 0.5|1.0]"""
from __future__ import annotations

import argparse
import json
import math
import time
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from phase0.analysis import swipe_common as C

NEG = -1e30
MISSING = ["sorry", "claude", "bifi", "by", "deceived", "wonder", "codex", "center", "div", "does", "best", "also", "and", "shall"]


def lae(a, b):
    if a < b:
        a, b = b, a
    return a if b <= NEG else a + math.log1p(math.exp(b - a))


class LM:
    def __init__(self):
        from phase0.analysis import seqctc2 as S2
        lex = C.merged_lex()
        B = S2.blocklist()
        self.logp = {w: v for w, v in lex.items() if w.isalpha() and w.isascii() and not S2.is_blocked(w, B)}
        self.bg = S2.BigramScorer(Path(".cache/personal/bigram_personal.npz"))
        gen = set(json.loads(Path(".cache/autocorrect/lexicon.json").read_text()))
        pers = json.loads(Path(".cache/personal/lexicon_personal.json").read_text())["personal"]
        self.boostset = set(C.NAMES) | {w for w, v in pers.items() if v.get("count", 0) >= 10 and w not in gen and w in self.logp}
        # trie
        self.ch = [dict()]
        self.word = [False]
        self.best = [NEG]
        for w, lp in self.logp.items():
            n = 0
            self.best[0] = max(self.best[0], lp)
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
            if w in self.bg.ix:
                v = self.bg.score([((prev,) if prev else (), w)])[0]
            else:
                v = self.logp.get(w, -20.0)
            self._c[k] = v
        return v

    def sent(self, text, cfg):
        ws = text.split()
        s = 0.0
        for k, w in enumerate(ws):
            s += cfg["alpha"] * self.big(ws[k - 1] if k else "", w) + cfg["beta"] + (cfg["boost"] if w in self.boostset else 0.0)
        return s


def beam(M, lm: LM, cfg):
    A, Bt, G, BO = cfg["alpha"], cfg["beta"], cfg["gamma"], cfg["boost"]
    SP = 27
    beams = {((), "", -1): [NEG, 0.0, 0.0, 0]}   # key -> [pb, pnb, lm, node]; start as non-blank-0 so ptot=0
    beams[((), "", -1)] = [0.0, NEG, 0.0, 0]
    prune = cfg["prune"]
    W = cfg["beam"]
    for t in range(len(M)):
        row = M[t]
        cand = [c for c in range(1, 28) if row[c] > prune]
        nb = {}

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
        items = sorted(nb.items(), key=lambda kv: -(lae(kv[1][0], kv[1][1]) + kv[1][2]))[:W]
        beams = dict(items)
    fin = {}
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


_LM = None


def _init():
    global _LM
    _LM = LM()


def _job(args):
    sid, lps, cfg = args
    t0 = time.time()
    M = C.ens28(lps)
    out = beam(M, _LM, cfg)[:cfg["topk"]]
    texts = [t for t, _ in out]
    if texts:
        ll = C.ctc_ens(lps, texts)
        resc = [float(ll[i] + _LM.sent(t, cfg)) for i, t in enumerate(texts)]
    else:
        ll, resc = [], []
    return sid, [{"text": t, "beam": float(s), "ctc": float(ll[i]), "rescored": resc[i]} for i, (t, s) in enumerate(out)], time.time() - t0


def recut(name, pad):
    """segments re-cut from the continuous posteriors with a different start/end pad (same thresholds as mkdecipher)."""
    from phase0.analysis import seqctc2 as S2
    sdir = C.SETS[name][1]
    z = np.load(f".cache/decipher/out/{sdir}_lp.npz")
    groups = [g for g in ("zs", "desk") if g in z.files]
    comb = np.mean([z[g].astype(np.float32) for g in groups], 0)
    segs = S2.segments(comb, z["times"], 2.0, pad=pad, thr=0.5)
    return [[z[g][s0:s1].astype(np.float32) for g in groups] for s0, s1 in segs]


GRIDS = {
    "big": [dict(alpha=a, beta=b, gamma=g, boost=bo) for a, b, g, bo in
            [(2.0, 4.0, -4.0, 2.0), (2.5, 5.0, -4.0, 2.0), (3.0, 6.0, -4.0, 2.0), (2.5, 3.0, -4.0, 2.0), (2.5, 7.0, -4.0, 2.0),
             (2.5, 5.0, -8.0, 2.0), (2.5, 5.0, -4.0, 0.0), (2.5, 5.0, -4.0, 5.0), (2.0, 5.0, -4.0, 2.0)]],
    "one": [dict(alpha=1.0, beta=2.0, gamma=-4.0, boost=2.0)],
    "small": [dict(alpha=a, beta=b, gamma=g, boost=bo) for a, b, g, bo in
              [(1.0, 2.0, -4.0, 2.0), (0.5, 1.0, -4.0, 2.0), (1.5, 3.0, -4.0, 2.0), (1.0, 2.0, -8.0, 2.0),
               (1.0, 2.0, -4.0, 0.0), (1.0, 4.0, -4.0, 2.0), (1.0, 0.0, -4.0, 2.0)]],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="old,b1,b2")
    ap.add_argument("--grid", default="one")
    ap.add_argument("--pad", type=float, default=0.5)
    ap.add_argument("--beam", type=int, default=128)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--prune", type=float, default=-8.0)
    a = ap.parse_args()
    from phase0.analysis.decipher import score_variant
    from phase0.analysis.decode import edit_distance
    p0 = json.loads((C.OUT / "p0_error_anatomy.json").read_text())
    frozen_orc = {(s["set"], s["id"]): s["pool_oracle_wed"] for s in p0["segments"]}
    res = {"pad": a.pad, "beam": a.beam, "topk": a.topk, "runs": []}
    data = {}
    for name in a.sets.split(","):
        segs, lines = C.load(name)
        if a.pad != 0.5:
            rc = recut(name, a.pad)
            if len(rc) != len(segs):
                print(f"{name}: recut pad {a.pad} gives {len(rc)} segments vs {len(segs)}; skipping", flush=True)
                continue
            for s, l in zip(segs, rc):
                s["lps"] = l
        data[name] = (segs, lines)
    with Pool(a.procs, initializer=_init) as pool:
        for gi, g in enumerate(GRIDS[a.grid]):
            cfg = dict(g, prune=a.prune, beam=a.beam, topk=a.topk, xspace=-3.0)
            for name, (segs, lines) in data.items():
                t0 = time.time()
                outs = {sid: (o, sec) for sid, o, sec in pool.map(_job, [(s["id"], s["lps"], cfg) for s in segs])}
                row = {"set": name, "cfg": g, "secs_per_seg": float(np.mean([outs[s["id"]][1] for s in segs]))}
                for sel in ("beam", "rescored"):
                    hyps = [max(outs[s["id"]][0], key=lambda x: x[sel])["text"] if outs[s["id"]][0] else "" for s in segs]
                    sv = score_variant(lines, hyps)
                    e, n = C.seg_wer([(s["truth"], h) for s, h in zip(segs, hyps)])
                    row[f"top1_{sel}"] = {"wer_official": float(sv["wed"].sum() / sv["wn"].sum()), "wer_segwise": e / n,
                                          "wer_ci": C.boot_ci(sv["wed"], sv["wn"]), "hyps": hyps}
                orc, orc_u, fz, n = 0, 0, 0, 0
                miss = {}
                for s in segs:
                    tw = s["truth"].split()
                    texts = [x["text"] for x in outs[s["id"]][0]]
                    o = min([edit_distance(tw, t.split()) for t in texts] or [len(tw)])
                    orc += o
                    fzo = frozen_orc.get((name, s["id"]), len(tw))
                    fz += fzo
                    orc_u += min(o, fzo)
                    n += len(tw)
                    for w in MISSING:
                        if w in tw:
                            ranks = [k for k, t in enumerate(texts) if w in t.split()]
                            miss.setdefault(w, []).append({"id": s["id"], "best_rank": ranks[0] + 1 if ranks else None})
                    s.setdefault("cands", {})[json.dumps(g)] = texts
                row.update({"pool_oracle_wer": orc / n, "frozen_pool_oracle_wer": fz / n, "union_oracle_wer": orc_u / n,
                            "missing_word_ranks": miss, "wall_s": time.time() - t0})
                res["runs"].append(row)
                print(f"[{gi}] {name} {g} top1 beam {row['top1_beam']['wer_official']:.3f} resc {row['top1_rescored']['wer_official']:.3f} "
                      f"oracle {orc / n:.3f} frozen {fz / n:.3f} union {orc_u / n:.3f} {row['secs_per_seg']:.1f}s/seg "
                      f"miss {{{', '.join(f'{w}:{[m['best_rank'] for m in v]}' for w, v in miss.items())}}}", flush=True)
            cand_f = C.CACHE / f"p1_cands_{a.grid}_pad{a.pad}{a.tag}.json"
            cand_f.write_text(json.dumps({name: {s["id"]: s["cands"] for s in segs} for name, (segs, _) in data.items()}))
            (C.OUT / f"p1_wordbeam_{a.grid}_pad{a.pad}{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
