"""Shared harness for the swipe/word-decoding feasibility probes (P0-P4). Dev data only; prototypes, not production."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("LLMDEC_FUZZY_TAG", "")
OUT = Path("results/swipe")
CACHE = Path(".cache/swipe")
LLM = Path(".cache/llmdec")
SETS = {   # name -> (llmdec set, derived session dir, truth file, source session)
    "old": ("s_zz-v3cv-ftkf5-olddesk", "zz-v3cv-ftkf5-olddesk", "20260910-202149-desk"),
    "b1": ("s_zz-v3b1-d20zs-blind1", "zz-v3b1-d20zs-blind1", "20260912-174542-desk"),
    "b2": ("s_zz-ctc3-20260913-125733-desk", "zz-ctc3-20260913-125733-desk", "20260913-125733-desk"),
}
NAMES = {"claude", "bifi", "codex"}
SYMS = "_abcdefghijklmnopqrstuvwxyz #"


def norm(s: str) -> str:
    return " ".join("".join(c if "a" <= c <= "z" else (" " if c.isspace() or c == "-" else "") for c in s.lower()).split())


def merged_lex() -> dict:
    return json.loads(Path(".cache/personal/lexicon_merged.json").read_text())


def m28(lp: np.ndarray) -> np.ndarray:
    """29-way log posteriors -> 28-way (blank := blank + other), as ctc_ll does."""
    P = lp.astype(np.float64)
    return np.concatenate([np.logaddexp(P[:, 0], P[:, 28])[:, None], P[:, 1:28]], 1)


def ens28(lps: list[np.ndarray]) -> np.ndarray:
    """geometric mean of the groups, renormalised per frame (beam/viterbi evidence)."""
    M = np.mean([m28(l) for l in lps], 0)
    return M - np.logaddexp.reduce(M, axis=1, keepdims=True)


def ctc_ens(lps: list[np.ndarray], texts: list[str]) -> np.ndarray:
    """exact CTC log-lik, mean over model groups (the frozen v3 selection's evidence)."""
    from phase0.analysis.llmdec import ctc_ll
    texts = [t if t else " " for t in texts]
    out = []
    for i in range(0, len(texts), 256):
        out.append(np.mean([ctc_ll(l, texts[i:i + 256]) for l in lps], 0))
    return np.concatenate(out) if out else np.zeros(0)


def viterbi(M: np.ndarray, text: str):
    """CTC forced alignment. -> (path_logp, state per frame, label seq). states even=blank, odd=label k//2."""
    lab = [SYMS.index(c) for c in text]
    L = len(lab)
    S = 2 * L + 1
    T = len(M)
    ext = np.zeros(S, int)
    ext[1::2] = lab
    NEG = -1e18
    D = np.full((T, S), NEG)
    B = np.zeros((T, S), np.int8)
    D[0, 0] = M[0, 0]
    if L:
        D[0, 1] = M[0, ext[1]]
    skip = np.zeros(S, bool)
    for s in range(3, S, 2):
        skip[s] = ext[s] != ext[s - 2]
    for t in range(1, T):
        prev = D[t - 1]
        c0 = prev
        c1 = np.concatenate([[NEG], prev[:-1]])
        c2 = np.where(skip, np.concatenate([[NEG, NEG], prev[:-2]]), NEG)
        st = np.stack([c0, c1, c2])
        b = st.argmax(0)
        D[t] = st[b, np.arange(S)] + M[t, ext]
        B[t] = b
    ends = [S - 1] + ([S - 2] if S >= 2 else [])
    s = int(max(ends, key=lambda e: D[T - 1, e]))
    lp = D[T - 1, s]
    path = np.zeros(T, int)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s = s - int(B[t, s])
    return float(lp), path, lab


def letter_frames(M: np.ndarray, text: str):
    """per character of text: (first, last, peak) frame of its label state on the Viterbi path (-1 if never)."""
    _, path, lab = viterbi(M, text)
    out = []
    for k in range(len(text)):
        fr = np.where(path == 2 * k + 1)[0]
        if len(fr):
            out.append((int(fr[0]), int(fr[-1]), int(fr[np.argmax(M[fr, lab[k]])])))
        else:
            out.append((-1, -1, -1))
    return out


def word_align(ref, hyp):
    from phase0.analysis.decipher import word_align as wa
    return wa(ref, hyp)


def load(name: str):
    """-> list of segments {id, lps[zs,desk], frozen hyp, truth words assigned}, lines."""
    setname, sdir, _ = SETS[name]
    d = json.loads((LLM / "sets" / f"{setname}.json").read_text())
    z = np.load(LLM / "sets" / f"{setname}_lp.npz")
    rec = json.loads(Path(f"data/sessions/{sdir}/decipher2_v3_recommended.json").read_text())
    hyps = {r["id"]: norm(r["hyp"]) for r in rec["records"]}
    tf = Path(f"data/sessions/{sdir}/truth.txt")
    tf = tf if tf.exists() else Path(f"data/sessions/{SETS[name][2]}/truth.txt")
    lines = [norm(l) for l in tf.read_text().splitlines() if norm(l)]
    segs = []
    for it in d["items"]:
        segs.append({"id": it["id"], "lps": [z[f"{it['id']}__{g}"] for g in it["c"]], "hyp": hyps[it["id"]], "c": it["c"]})
    assign_truth(segs, lines)
    return segs, lines


def assign_truth(segs, lines):
    """truth words -> segments: word alignment of concatenated truth vs frozen output, then boundary words between
    adjacent segments moved (+-3) to maximise summed CTC log-lik."""
    rw = [w for l in lines for w in l.split()]
    hw, hs = [], []
    for k, s in enumerate(segs):
        for w in s["hyp"].split():
            hw.append(w)
            hs.append(k)
    owner = [None] * len(rw)
    for op, i, j in word_align(rw, hw):
        if op in ("ok", "sub"):
            owner[i] = hs[j]
    # fill deletions from neighbours (prefer previous), enforce monotone
    last = 0
    for i in range(len(rw)):
        if owner[i] is None:
            nxt = next((owner[k] for k in range(i + 1, len(rw)) if owner[k] is not None), last)
            owner[i] = last if i > 0 else nxt
        owner[i] = max(owner[i], last)
        last = owner[i]
    cuts = [0]
    for k in range(1, len(segs)):
        cuts.append(next((i for i in range(len(rw)) if owner[i] >= k), len(rw)))
    cuts.append(len(rw))
    for k in range(1, len(segs)):   # refine each boundary by CTC
        best = None
        for c in range(max(cuts[k - 1], cuts[k] - 3), min(cuts[k + 1], cuts[k] + 3) + 1):
            a = " ".join(rw[cuts[k - 1]:c])
            b = " ".join(rw[c:cuts[k + 1]])
            sc = ctc_ens(segs[k - 1]["lps"], [a])[0] + ctc_ens(segs[k]["lps"], [b])[0]
            if best is None or sc > best[0]:
                best = (sc, c)
        cuts[k] = best[1]
    for k, s in enumerate(segs):
        s["truth"] = " ".join(rw[cuts[k]:cuts[k + 1]])


def seg_wer(pairs):
    """[(truth, hyp)] -> word edits, n words (per-segment alignment)."""
    from phase0.analysis.decode import edit_distance
    e = sum(edit_distance(t.split(), h.split()) for t, h in pairs)
    n = sum(len(t.split()) for t, _ in pairs)
    return e, n


def boot_ci(x_err, x_n, n=5000, seed=0):
    x_err, x_n = np.asarray(x_err, float), np.asarray(x_n, float)
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(x_err), (n, len(x_err)))
    v = x_err[idx].sum(1) / np.maximum(x_n[idx].sum(1), 1)
    return [float(x_err.sum() / max(x_n.sum(), 1)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def boot_delta(a_err, b_err, x_n, n=5000, seed=0):
    a, b, w = map(lambda v: np.asarray(v, float), (a_err, b_err, x_n))
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(a), (n, len(a)))
    v = (b[idx].sum(1) - a[idx].sum(1)) / np.maximum(w[idx].sum(1), 1)
    return [float((b.sum() - a.sum()) / w.sum()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
