"""Camera-accuracy requirements under smarter autocorrect: real-prob-vector tap simulator, decoder sweep
(beam / n-best+neural LM / word-level neural / LLM), real desk re-decode. Run: python -m phase0.analysis.autocorrect <cmd>"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from phase0.analysis.decode import ALPHABET, A_INDEX, NA, SPACE, CharLM, WordLM, edit_distance

CACHE = Path(".cache/autocorrect")
OUT = Path("results/autocorrect")
MACK = CACHE / "mackenzie" / "phrases2.txt"   # MacKenzie & Soukoreff (2003) phrase set, yorku.ca/mack
PROJ = Path("phase0/phrases.txt")
LM_PATH = Path("models/charlm.npz")
RUN5 = {"KEYPRE_EXTRA": "20260911-164237-kbd", "KEYPRE_PROBS": "probs_run5"}
DESK = "20260910-202149-desk"
N_DEV = 150
EPS = 1e-12
MIN_KEY = 30            # pooled vectors (over 3 seeds) below which a key borrows a neighbour's
QWERTZ = ("qwertzuiop", "asdfghjkl", "yxcvbnm")
ACCS = (0.5, 0.6, 0.657, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
THREADS = 4


def norm(t: str) -> str:
    return re.sub(r" +", " ", re.sub(r"[^a-z ]+", " ", t.lower())).strip()


# ============================================================================ text
def texts() -> dict[str, list[str]]:
    """MacKenzie phrases: seeded shuffle -> dev 150 / test 350. Project phrases: test only."""
    ph = list(dict.fromkeys(norm(l) for l in MACK.read_text().splitlines() if l.strip()))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(ph))
    ph = [ph[i] for i in idx]
    proj = [norm(l) for l in PROJ.read_text().splitlines() if l.strip()]
    return {"dev": ph[:N_DEV], "test": ph[N_DEV:], "proj": proj}


# ============================================================================ pool
def build_pool() -> Path:
    """Every OOF fused (pose+pixel) probability vector of run 5: 4 LOSO folds + held-out
    015948, 3 seeds. Fusion weight nested exactly as keypre.cmd_keys does."""
    os.environ.update(RUN5)
    import lightgbm  # noqa: F401  before torch
    from phase0.analysis import keypre as K

    ys = {h: K.ours(h)["y"] for h in K.FOLDS}
    P = K._fuse(K._load("pix", "scratch", 1.0, K.SEEDS), K._load("pose", "scratch_abs", 1.0, K.SEEDS), ys)
    Ps, Y, S, F = [], [], [], []
    for s in P:
        for fi, h in enumerate(K.FOLDS):
            Ps.append(P[s][h]); Y.append(ys[h]); S += [s] * len(ys[h]); F += [fi] * len(ys[h])
    out = CACHE / "pool.npz"
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(out, P=np.vstack(Ps).astype(np.float64), y=np.concatenate(Y), seed=np.array(S),
             fold=np.array(F), folds=np.array(K.FOLDS), held=K.HELD)
    return out


def _keypos() -> dict[int, tuple[float, float]]:
    pos = {}
    for r, row in enumerate(QWERTZ):
        for c, ch in enumerate(row):
            pos[A_INDEX[ch]] = (c + 0.25 * r, float(r))
    pos[SPACE] = (4.5, 3.0)
    return pos


class Pool:
    def __init__(self, path: Path = CACHE / "pool.npz", folds: str = "all"):
        z = np.load(path, allow_pickle=True)
        P, y, fold = z["P"], z["y"], z["fold"]
        if folds == "held":
            m = fold == list(z["folds"]).index(str(z["held"]))
            P, y = P[m], y[m]
        P = np.maximum(P, 1e-6)
        self.logP = np.log(P / P.sum(1, keepdims=True))
        self.y = y
        self.by_key = {k: np.where(y == k)[0] for k in range(NA)}
        top = self.logP.argmax(1)
        self.ok = {k: np.where((y == k) & (top == k))[0] for k in range(NA)}
        self.bad = {k: np.where((y == k) & (top != k))[0] for k in range(NA)}
        pos = _keypos()
        self.borrow = {}
        for k in range(NA):
            if len(self.by_key[k]) >= MIN_KEY:
                continue
            cands = [j for j in range(NA) if len(self.by_key[j]) >= MIN_KEY and j != SPACE]
            self.borrow[k] = min(cands, key=lambda j: math.dist(pos[k], pos[j]))

    def draw(self, k: int, rng) -> np.ndarray:
        """A real log-prob vector whose true key is k; rare keys take a spatial neighbour's
        vector with the two keys' entries swapped, so the confusion stays local."""
        src = self.borrow.get(k, k)
        v = self.logP[self.by_key[src][rng.integers(len(self.by_key[src]))]].copy()
        if src != k:
            v[[k, src]] = v[[src, k]]
        return v

    def draw_at(self, k: int, acc: float, rng) -> np.ndarray:
        """Resampling alternative to shift(): a real correct-top-1 vector with prob acc, else a real wrong one."""
        src = self.borrow.get(k, k)
        lst = self.ok[src] if rng.random() < acc else self.bad[src]
        if not len(lst):
            lst = self.by_key[src]
        v = self.logP[lst[rng.integers(len(lst))]].copy()
        if src != k:
            v[[k, src]] = v[[src, k]]
        return v


def shift(v: np.ndarray, k: np.ndarray, b: float) -> np.ndarray:
    """True-class logit shift: q ∝ p·exp(b·[j = true]). b>0 raises accuracy while the
    competitor distribution (the spatial confusion structure) is left untouched."""
    q = v.copy()
    q[np.arange(len(k)), k] += b
    return q - np.logaddexp.reduce(q, axis=1, keepdims=True)


def calibrate(pool: Pool, chars: np.ndarray, n: int = 20000, seed: int = 123) -> dict:
    """Bisection for b per target top-1, weighting keys by the dev text's character mix."""
    rng = np.random.default_rng(seed)
    ks = chars[rng.integers(len(chars), size=n)]
    V = np.stack([pool.draw(int(k), rng) for k in ks])

    def stats(b):
        q = shift(V, ks, b)
        r = (q > q[np.arange(n), ks][:, None]).sum(1)
        return {"top1": float((r == 0).mean()), "top5": float((r < 5).mean()),
                "p_true": float(np.exp(q[np.arange(n), ks]).mean()),
                "entropy": float(-(np.exp(q) * q).sum(1).mean())}

    out = {"b0": dict(b=0.0, **stats(0.0))}
    for acc in ACCS:
        lo, hi = -6.0, 12.0
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if stats(mid)["top1"] < acc else (lo, mid)
        out[f"{acc}"] = dict(b=(lo + hi) / 2, **stats((lo + hi) / 2))
    return out


# ============================================================================ tap errors
@dataclass(frozen=True)
class TapNoise:
    name: str
    p_del: float             # P(a typed character produces no tap)
    ins: float               # expected spurious taps per character
    disp: float = 1.0        # variance/mean of spurious taps per character (1 = Poisson)
    mix: tuple = (0.5, 0.0, 0.0)   # P(spurious key = previous key, next key, space); rest unigram
    tau: float = 1.0         # spurious-tap vectors flattened by this temperature


def _n_spurious(rng, nz: TapNoise) -> int:
    if nz.ins <= 0:
        return 0
    if nz.disp <= 1.0:
        return int(rng.poisson(nz.ins))
    pp = 1.0 / nz.disp
    return int(rng.negative_binomial(nz.ins * pp / (1 - pp), pp))


def simulate(text: str, pool: Pool, noise: TapNoise, seed_seq, method: str = "shift",
             b: float = 0.0, unigram: np.ndarray | None = None, acc: float = 0.0):
    """-> (T x 27 log-probs, true key per tap, -1 spurious). Same seed -> same draws at every b."""
    rng = np.random.default_rng(seed_seq)
    rows, truth = [], []
    ks = [A_INDEX[ch] for ch in text]
    for n, k in enumerate(ks):
        keep = rng.random() >= noise.p_del
        v = pool.draw_at(k, acc, rng) if method == "resample" else pool.draw(k, rng)
        if keep:
            rows.append(v); truth.append(k)
        for _ in range(_n_spurious(rng, noise)):
            u = rng.random()
            m = noise.mix
            if u < m[0]:
                kk = k
            elif u < m[0] + m[1]:
                kk = ks[n + 1] if n + 1 < len(ks) else k
            elif u < m[0] + m[1] + m[2]:
                kk = SPACE
            else:
                kk = int(rng.choice(NA, p=unigram))
            w = pool.draw(kk, rng) / noise.tau
            rows.append(w - np.logaddexp.reduce(w)); truth.append(-1)
    if not rows:
        return np.zeros((0, NA)), np.zeros(0, int)
    V = np.stack(rows)
    t = np.array(truth)
    if method == "shift":
        V = np.where((t >= 0)[:, None], shift(V, np.where(t >= 0, t, 0), b), V)
    return V, t


# ============================================================================ decoders
@dataclass(frozen=True)
class BeamCfg:
    obs: float = 2.0
    deletion: float = -5.0
    insertion: float = -7.0
    beam: int = 30
    nbest: int = 32


_LM: dict = {}


def lms():
    if "c" not in _LM:
        _LM["c"] = CharLM.load(LM_PATH)
        _LM["w"] = WordLM.load(LM_PATH.with_suffix(".words.json"))
    return _LM["c"], _LM["w"]


def beam_nbest(obs_logp: np.ndarray, cfg: BeamCfg) -> list[tuple[str, float]]:
    """lm.beam_nbest with the incumbent unigram word model: its top-1 IS decode.beam_decode."""
    from phase0.analysis.lm import DecWeights, UnigramAdapter, beam_nbest as bn
    c, w = lms()
    if len(obs_logp) == 0:
        return [("", 0.0)]
    dw = DecWeights(obs=cfg.obs, deletion=cfg.deletion, insertion=cfg.insertion, max_deletions=3)
    return bn(obs_logp, c, UnigramAdapter(w), dw, cfg.beam, cfg.nbest)


# ============================================================================ metrics
def word_hits(ref: str, hyp: str) -> int:
    r, h = ref.split(), hyp.split()
    D = np.zeros((len(r) + 1, len(h) + 1), int)
    D[:, 0] = np.arange(len(r) + 1)
    D[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    i, j, hits = len(r), len(h), 0
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            hits += r[i - 1] == h[j - 1]
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return hits


def sent_stats(ref: str, hyp: str) -> list[int]:
    """[char edits, ref chars, word edits, ref words, word hits, exact]"""
    return [edit_distance(ref, hyp), len(ref), edit_distance(ref.split(), hyp.split()),
            len(ref.split()), word_hits(ref, hyp), int(ref == hyp)]


def summarize(S: np.ndarray, B: int = 2000, seed: int = 0) -> dict:
    S = np.asarray(S, float)
    idx = np.random.default_rng(seed).integers(0, len(S), (B, len(S)))
    T = S[idx].sum(1)
    tot = S.sum(0)
    f = lambda t: (t[..., 0] / t[..., 1], t[..., 2] / t[..., 3], t[..., 4] / t[..., 3])
    c, w, a = f(tot)
    cb, wb, ab = f(T)
    q = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return {"cer": float(c), "cer_ci": q(cb), "wer": float(w), "wer_ci": q(wb),
            "word_acc": float(a), "word_acc_ci": q(ab), "sent_acc": float(S[:, 5].mean()), "n": len(S)}


def paired(Sa: np.ndarray, Sb: np.ndarray, B: int = 2000, seed: int = 0) -> dict:
    """b - a for CER and word accuracy, same resampled sentences on both sides."""
    Sa, Sb = np.asarray(Sa, float), np.asarray(Sb, float)
    idx = np.random.default_rng(seed).integers(0, len(Sa), (B, len(Sa)))
    ta, tb = Sa[idx].sum(1), Sb[idx].sum(1)
    dc = tb[:, 0] / tb[:, 1] - ta[:, 0] / ta[:, 1]
    da = tb[:, 4] / tb[:, 3] - ta[:, 4] / ta[:, 3]
    A, Bt = Sa.sum(0), Sb.sum(0)
    q = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return {"d_cer": float(Bt[0] / Bt[1] - A[0] / A[1]), "d_cer_ci": q(dc),
            "d_word_acc": float(Bt[4] / Bt[3] - A[4] / A[3]), "d_word_acc_ci": q(da)}


# ============================================================================ desk session
def _desk_fold(args):
    import lightgbm  # noqa: F401
    from phase0.analysis import combine as CB
    from phase0.analysis import hirecall as HR
    from phase0.analysis import pipeline as pl

    CB._shim()
    HR.use_cache()
    j, tag, merged, pick = args
    HR._ST[tag] = merged
    kf = pl.kbd_fit()
    s = pl.session_path(HR.DESK)
    ev = np.array(merged[pick["sid"]]["ev"], int)
    c = CB.Cfg(alpha=pick["alpha"], obs=pick["obs"], deletion=pick["deletion"], em=True)
    _, segs, p, sess, k = CB.stack_proba(s, ev, kf, c, HR.MODE)
    tr = [x for i, x in enumerate(segs) if i != j and len(x[1])]
    st = replace(pl.FULL, deletion=pick["deletion"])
    p = pl.weakly_supervise(sess, k, p, tr, st)
    r = CB.decode_rows(p, segs, kf, c, keep={j})[0]
    text, rows = segs[j]
    return j, {"text": text, "probs": p[rows].tolist(), "alpha": pick["alpha"], "obs": pick["obs"],
               "deletion": pick["deletion"], "insertion": -7.0, "max_deletions": 3,
               "n_taps_stream": int(len(ev)), "hyp": r[1], "edits": int(r[2]), "chars": int(r[3]),
               "stream_cfg": merged[pick["sid"]]["cfg"],
               "all_segs": [[t, np.asarray(rr).tolist()] for t, rr in segs],
               "all_probs": p.tolist() if j == 0 else None}


def cmd_desk_dump(a) -> int:
    """Reproduce hirecall `cv --tags base,rec` (CER 0.448) fold by fold and keep, for every
    held-out phrase, the exact observation matrix and weights its beam decode used."""
    import lightgbm  # noqa: F401
    import multiprocessing as mp
    from phase0.analysis import combine as CB
    from phase0.analysis import hirecall as HR

    tags = ["base", "rec"]
    merged, rows, off = [], [], 0
    for t in tags:
        st = HR.load_streams(t)
        merged += [dict(x, sid=x["sid"] + off) for x in st]
        rows += [dict(r, sid=r["sid"] + off) for r in HR.load_table(t, HR.MODE)]
        off += len(st)
    tag = "+".join(tags)
    n = len(rows[0]["per"])
    picks = [CB.select(rows, set(range(n)) - {j}) for j in range(n)]
    jobs = [(j, tag, merged, picks[j]) for j in range(n)]
    out = {}
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for j, r in pool.imap_unordered(_desk_fold, jobs):
            out[j] = r
            print(f"  phrase {j:2d} taps={len(r['probs'])} CER={r['edits']/r['chars']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    res = [out[j] for j in range(n)]
    ref = json.loads((HR.HR_CACHE / "cv_all.json").read_text())
    mine = sum(r["edits"] for r in res) / sum(r["chars"] for r in res)
    same = [tuple(x) for x in ref["per"]] == [(r["edits"], r["chars"]) for r in res]
    print(f"reproduced CER {mine:.4f} vs cv_all.json {ref['cer']:.4f}; per-phrase identical: {same}")
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "desk_obs.json").write_text(json.dumps(res))
    return 0


def align_rates(obs_list: list[np.ndarray], refs: list[str], iters: int = 8) -> dict:
    """Hard-EM Viterbi alignment of taps to the known text -> del/ins rates per char and top-1/5 of
    matched taps (optimistic: the alignment prefers agreeing matches)."""
    pd, pi = 0.2, 0.5
    for _ in range(iters):
        nd = ni = nm = nc = 0
        t1 = t5 = dup = 0
        for o, ref in zip(obs_list, refs):
            y = [A_INDEX[c] for c in ref]
            T, L = len(o), len(y)
            D = np.full((T + 1, L + 1), -np.inf)
            bp = np.zeros((T + 1, L + 1), int)
            D[0, 0] = 0
            lpd, lpi = math.log(pd), math.log(pi)
            for i in range(T + 1):
                for j in range(L + 1):
                    if i == 0 and j == 0:
                        continue
                    best, arg = -np.inf, 0
                    if i and j and D[i - 1, j - 1] + o[i - 1, y[j - 1]] > best:
                        best, arg = D[i - 1, j - 1] + o[i - 1, y[j - 1]], 0
                    if j and D[i, j - 1] + lpd > best:
                        best, arg = D[i, j - 1] + lpd, 1
                    if i and D[i - 1, j] + lpi > best:
                        best, arg = D[i - 1, j] + lpi, 2
                    D[i, j], bp[i, j] = best, arg
            i, j = T, L
            path = []
            while i or j:
                m = bp[i, j]
                path.append((m, i, j))
                if m == 0:
                    i, j = i - 1, j - 1
                elif m == 1:
                    j -= 1
                else:
                    i -= 1
            path.reverse()
            last_match = None
            for m, i, j in path:
                if m == 0:
                    nm += 1
                    r = (o[i - 1] > o[i - 1, y[j - 1]]).sum()
                    t1 += r == 0
                    t5 += r < 5
                    last_match = y[j - 1]
                elif m == 1:
                    nd += 1
                else:
                    ni += 1
                    dup += last_match is not None and int(np.argmax(o[i - 1])) == last_match
            nc += L
        pd = max(nd / nc, 1e-3)
        pi = max(ni / (nm + ni), 1e-3)
    return {"p_del": nd / nc, "ins_per_char": ni / nc, "taps_per_char": (nm + ni) / nc,
            "matched_top1": t1 / max(nm, 1), "matched_top5": t5 / max(nm, 1),
            "ins_argmax_equals_prev_key": dup / max(ni, 1)}


def cmd_rates(a) -> int:
    res = json.loads((CACHE / "desk_obs.json").read_text())
    obs = [np.log(np.maximum(np.array(r["probs"]), EPS)) for r in res]
    refs = [r["text"] for r in res]
    out = {"desk_picked_streams": align_rates(obs, refs)}
    print(json.dumps(out, indent=1))
    (CACHE / "desk_rates.json").write_text(json.dumps(out, indent=1))
    return 0


# ============================================================================ lexicon
def cmd_lexicon(a) -> int:
    """Modern word list for the word decoder: word counts over the (already project-phrase-
    decontaminated) Tatoeba + Enron + Gutenberg LM corpora. Words only - no sentences."""
    cnt: Counter = Counter()
    for name in ("tatoeba", "enron", "gutenberg"):
        f = Path(".cache/lm") / f"{name}.txt"
        with f.open() as fh:
            while True:
                chunk = fh.read(8_000_000)
                if not chunk:
                    break
                cnt.update(chunk.split())
        print(name, len(cnt), flush=True)
    vocab = [(w, c) for w, c in cnt.most_common(a.top) if c >= 3 and len(w) <= 18]
    single = set("ai")
    vocab = [(w, c) for w, c in vocab if len(w) > 1 or w in single]
    tot = sum(c for _, c in vocab)
    lex = {w: math.log(c / tot) for w, c in vocab}
    (CACHE / "lexicon.json").write_text(json.dumps(lex))
    T = texts()
    for k, v in T.items():
        ws = [w for s in v for w in s.split()]
        print(f"{k}: OOV token rate {np.mean([w not in lex for w in ws]):.4f} "
              f"({sorted({w for w in ws if w not in lex})[:20]})")
    print(f"lexicon {len(lex)} words -> {CACHE/'lexicon.json'}")
    return 0


# ============================================================================ sweep: beam + n-best
# fitted so simulated desk insertions match the real ones under alignment (see cmd_insstruct)
DESK_DISP, DESK_MIX, DESK_TAU = 1.9, (0.45, 0.33, 0.08), 1.5


def noises() -> dict[str, TapNoise]:
    """clean; AMENDMENT-4 target detector; keyboard-tuned detector on 131629 (r .837, f .148);
    and the desk rates measured by aligning the picked desk streams to the known text."""
    out = {"clean": TapNoise("clean", 0.0, 0.0), "target": TapNoise("target", 0.05, 0.05),
           "kbd": TapNoise("kbd", 0.163, 0.148)}
    f = CACHE / "desk_rates.json"
    if f.exists():
        r = json.loads(f.read_text())["desk_picked_streams"]
        out["desk"] = TapNoise("desk", r["p_del"], r["ins_per_char"], DESK_DISP, DESK_MIX, DESK_TAU)
    return out


def unigram(texts_: list[str]) -> np.ndarray:
    c = np.bincount([A_INDEX[ch] for s in texts_ for ch in s], minlength=NA).astype(float) + 1
    return c / c.sum()


_POOL: dict = {}


def pool_ctx():
    if "p" not in _POOL:
        _POOL["p"] = Pool()
        _POOL["cal"] = json.loads((CACHE / "calib.json").read_text())
        _POOL["uni"] = unigram(texts()["dev"])
    return _POOL["p"], _POOL["cal"], _POOL["uni"]


SPLIT_ID = {"dev": 0, "test": 1, "proj": 2}
NOISE_ID = {"clean": 0, "target": 1, "kbd": 2, "desk": 3}


def sim_obs(split: str, i: int, text: str, noise: TapNoise, acc: float, seed: int = 0):
    pool, cal, uni = pool_ctx()
    b = cal[f"{acc}"]["b"]
    return simulate(text, pool, noise, [seed, SPLIT_ID[split], i, NOISE_ID[noise.name]], "shift", b, uni)


def _beam_job(args):
    split, i, text, noise, acc, cfg, seed = args
    V, t = sim_obs(split, i, text, noise, acc, seed)
    nb = beam_nbest(V, cfg)
    return {"split": split, "i": i, "noise": noise.name, "acc": acc, "seed": seed, "cfg": cfg.__dict__,
            "n_taps": len(V), "nbest": nb}


def run_pool(fn, jobs, procs: int, label: str = ""):
    import multiprocessing as mp
    out, t0 = [], time.time()
    with mp.get_context("spawn").Pool(procs) as pool:
        for n, r in enumerate(pool.imap_unordered(fn, jobs, chunksize=4)):
            out.append(r)
            if (n + 1) % 500 == 0:
                print(f"  {label} {n+1}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    return out


TUNE_ANCHORS = (0.6, 0.8, 0.95)
TUNE_N = 60


def cmd_tune_beam(a) -> int:
    """Coordinate search of (obs, deletion, insertion) per noise at 3 anchor accuracies, dev only."""
    dev = texts()["dev"][:TUNE_N]
    res = _load_json(CACHE / "tune_beam.json", {})
    for nz in [v for k, v in noises().items() if not a.noises or k in a.noises.split(",")]:
        for acc in [x for x in TUNE_ANCHORS if not a.accs or f"{x}" in a.accs.split(",")]:
            cfg = BeamCfg(nbest=1)
            best = None
            for field, grid in (("obs", (1.0, 2.0, 3.0, 4.0)), ("deletion", (-9.0, -5.0, -3.0, -2.0)),
                                ("insertion", (-9.0, -7.0, -4.0, -2.5, -1.5)), ("obs", (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.5, 8.0))):
                scores = {}
                for v in grid:
                    c = replace(cfg, **{field: v})
                    jobs = [("dev", i, s, nz, acc, c, 0) for i, s in enumerate(dev)]
                    rs = run_pool(_beam_job, jobs, a.procs)
                    S = np.array([sent_stats(dev[r["i"]], r["nbest"][0][0]) for r in rs])
                    scores[v] = S[:, 0].sum() / S[:, 1].sum()
                v = min(scores, key=scores.get)
                cfg = replace(cfg, **{field: v})
                best = scores[v]
                print(f"{nz.name} acc={acc} {field}: " + " ".join(f"{k:g}:{x:.3f}" for k, x in scores.items()), flush=True)
            res[f"{nz.name}|{acc}"] = {"cfg": cfg.__dict__, "dev_cer": best}
            (CACHE / "tune_beam.json").write_text(json.dumps(res, indent=1))
    return 0


def beam_cfg_for(noise: str, acc: float) -> BeamCfg:
    t = json.loads((CACHE / "tune_beam.json").read_text())
    anc = min(TUNE_ANCHORS, key=lambda x: abs(x - acc))
    return replace(BeamCfg(**t[f"{noise}|{anc}"]["cfg"]), nbest=30, beam=30)


def nbest_path(split: str) -> Path:
    return CACHE / f"nbest_{split}.jsonl"


def cmd_sweep_beam(a) -> int:
    T = texts()
    nz = noises()
    names = a.noises.split(",") if a.noises else list(nz)
    for split in a.splits.split(","):
        done = set()
        p = nbest_path(split)
        if p.exists():
            for l in p.read_text().splitlines():
                r = json.loads(l)
                done.add((r["i"], r["noise"], r["acc"], r["seed"]))
        jobs = [(split, i, s, nz[n], acc, beam_cfg_for(n, acc), sd)
                for sd in range(a.seeds) for n in names for acc in ACCS for i, s in enumerate(T[split])
                if (i, n, acc, sd) not in done]
        print(f"{split}: {len(jobs)} beam decodes", flush=True)
        import multiprocessing as mp
        t0 = time.time()
        with mp.get_context("spawn").Pool(a.procs) as pool, p.open("a") as fh:
            for k, r in enumerate(pool.imap_unordered(_beam_job, jobs, chunksize=4)):
                fh.write(json.dumps(r) + "\n")
                if (k + 1) % 500 == 0:
                    fh.flush()
                    print(f"  {k+1}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    return 0


# ============================================================================ neural LM
class NLM:
    """Local causal LM; logits only at scored positions (full-vocab logits for a batch blow up memory)."""

    def __init__(self, name: str = "Qwen/Qwen2.5-0.5B", device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if self.dev == "mps" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(self.dev).eval()
        self.name = name
        self.bos = self.tok("\n").input_ids
        self.eos = self.tok("\n").input_ids
        self._ids: dict = {}
        self.cache: dict = {}
        self.n_tokens = 0

    def ids(self, s: str) -> list[int]:
        if s not in self._ids:
            self._ids[s] = self.tok(s).input_ids if s else []
        return self._ids[s]

    def ctx_ids(self, words: tuple) -> list[int]:
        return self.bos + (self.ids(" ".join(words)) if words else [])

    @staticmethod
    def cont(words: tuple, w: str) -> str:
        return (" " + w) if words else w

    def score(self, pairs: list[tuple[tuple, str]], chunk: int = 48) -> list[float]:
        """log P(continuation | "\\n" + words) in nats; continuation '\\n' = end of sentence."""
        need = [pr for pr in dict.fromkeys(pairs) if pr not in self.cache]
        by_ctx: dict = {}
        for pr in need:
            by_ctx.setdefault(pr[0], []).append(pr[1])
        group: dict = {}
        size = 0
        for ctx, ws in by_ctx.items():
            for k in range(0, len(ws), 384):
                part = ws[k:k + 384]
                if group and (len(group) >= chunk or size + len(part) > 384 or ctx in group):
                    self._score_ctxs(group)
                    group, size = {}, 0
                group[ctx] = part
                size += len(part)
        if group:
            self._score_ctxs(group)
        if len(self.cache) > 600_000:
            keep = {pr: self.cache[pr] for pr in pairs if pr in self.cache}
            self.cache = keep
        return [self.cache[pr] for pr in pairs]

    def _score_ctxs(self, group: dict):
        torch = self.torch
        dev = self.dev
        ctxs = list(group)
        cids = [self.ctx_ids(c) for c in ctxs]
        Lc = max(map(len, cids))
        inp_np = np.zeros((len(ctxs), Lc), np.int64)
        att_np = np.zeros((len(ctxs), Lc), np.int64)
        for n, ids in enumerate(cids):
            inp_np[n, Lc - len(ids):] = ids
            att_np[n, Lc - len(ids):] = 1
        inp, att = torch.from_numpy(inp_np), torch.from_numpy(att_np)
        pos = (att.cumsum(1) - 1).clamp(min=0)
        with torch.no_grad():
            o = self.model.model(input_ids=inp.to(dev), attention_mask=att.to(dev), position_ids=pos.to(dev),
                                 use_cache=True)
            last = self.model.lm_head(o.last_hidden_state[:, -1]).float().log_softmax(-1)
            cache = o.past_key_values
            multi = []   # (ctx row, word, token ids)
            first_idx, first_tok = [], []
            for n, c in enumerate(ctxs):
                for w in group[c]:
                    x = self.eos if w == "\n" else self.ids(self.cont(c, w))
                    first_idx.append(n); first_tok.append(x[0])
                    if len(x) > 1:
                        multi.append((n, w, x))
            self.n_tokens += int(att.sum())
            fv = last[torch.tensor(first_idx, device=dev), torch.tensor(first_tok, device=dev)].cpu().numpy()
            vals = {}
            it = 0
            for n, c in enumerate(ctxs):
                for w in group[c]:
                    vals[(c, w)] = float(fv[it]); it += 1
            if multi:
                owner = torch.tensor([m[0] for m in multi], device=dev)
                Lw = max(len(m[2]) - 1 for m in multi)
                winp_np = np.zeros((len(multi), Lw), np.int64)
                watt_np = np.zeros((len(multi), Lw), np.int64)
                for r, (_, _, x) in enumerate(multi):
                    winp_np[r, :len(x) - 1] = x[:-1]
                    watt_np[r, :len(x) - 1] = 1
                winp, watt = torch.from_numpy(winp_np), torch.from_numpy(watt_np)
                cache.batch_select_indices(owner)
                full_att = torch.cat([att.to(dev)[owner], watt.to(dev)], 1)
                wpos = att.sum(1).to(dev)[owner][:, None] + torch.arange(Lw, device=dev)[None, :]
                h = self.model.model(input_ids=winp.to(dev), attention_mask=full_att, position_ids=wpos,
                                     past_key_values=cache, use_cache=True).last_hidden_state
                self.n_tokens += int(watt.sum())
                rows, cols, tgt, own = [], [], [], []
                for r, (_, _, x) in enumerate(multi):
                    for q in range(1, len(x)):
                        rows.append(r); cols.append(q - 1); tgt.append(x[q])
                sel = h[torch.tensor(rows, device=dev), torch.tensor(cols, device=dev)]
                lg = self.model.lm_head(sel).float()
                t = torch.tensor(tgt, device=dev)
                lp = (lg.gather(1, t[:, None])[:, 0] - torch.logsumexp(lg, 1)).cpu().numpy()
                add = np.zeros(len(multi))
                np.add.at(add, np.array(rows), lp)
                for r, (n, w, _) in enumerate(multi):
                    vals[(ctxs[n], w)] += float(add[r])
        self.cache.update(vals)

    def sentence_logp(self, texts_: list[str], bs: int = 160) -> list[float]:
        """log P("\\n" + text + "\\n") in nats, whole sentences batched (no context sharing needed)."""
        torch = self.torch
        dev = self.dev
        out = []
        for k in range(0, len(texts_), bs):
            part = texts_[k:k + bs]
            enc = self.tok(part).input_ids
            seqs = [self.bos + ids + self.eos for ids in enc]
            L = max(map(len, seqs))
            inp = np.zeros((len(seqs), L), np.int64)
            att = np.zeros((len(seqs), L), np.int64)
            rows, cols, tgt = [], [], []
            for n, sq in enumerate(seqs):
                inp[n, :len(sq)] = sq
                att[n, :len(sq)] = 1
                rows += [n] * (len(sq) - 1)
                cols += list(range(len(sq) - 1))
                tgt += sq[1:]
            self.n_tokens += int(att.sum())
            with torch.no_grad():
                h = self.model.model(input_ids=torch.from_numpy(inp).to(dev),
                                     attention_mask=torch.from_numpy(att).to(dev)).last_hidden_state
                sel = h[torch.tensor(rows, device=dev), torch.tensor(cols, device=dev)]
                lp = []
                for q in range(0, len(sel), 1024):
                    lg = self.model.lm_head(sel[q:q + 1024]).float()
                    t = torch.tensor(tgt[q:q + 1024], device=dev)
                    lp.append(lg.gather(1, t[:, None])[:, 0] - torch.logsumexp(lg, 1))
                lp = torch.cat(lp).cpu().numpy()
            tot = np.zeros(len(part))
            np.add.at(tot, np.array(rows), lp)
            out += tot.tolist()
        return out


# ============================================================================ word decoder
class Trie:
    """Lexicon as per-depth arrays; children of one parent are contiguous at the next depth."""

    def __init__(self, lex: dict[str, float], maxlen: int = 18):
        words = sorted(w for w in lex if len(w) <= maxlen and all(c in A_INDEX and c != " " for c in w))
        self.words = words
        self.uni = np.array([lex[w] for w in words])
        self.levels = []
        prev = {"": 0}
        for d in range(1, maxlen + 1):
            pre = sorted({w[:d] for w in words if len(w) >= d}, key=lambda p: (prev[p[:-1]], p[-1]))
            if not pre:
                break
            idx = {p: n for n, p in enumerate(pre)}
            par = np.array([prev[p[:-1]] for p in pre])
            ch = np.array([A_INDEX[p[-1]] for p in pre])
            wid = np.full(len(pre), -1)
            self.levels.append({"par": par, "ch": ch, "wid": wid, "pre": pre})
            prev = idx
        wix = {w: n for n, w in enumerate(words)}
        for lv in self.levels:
            lv["wid"] = np.array([wix.get(p, -1) for p in lv["pre"]])
            nprev = (lv["par"].max() + 1) if len(lv["par"]) else 0
            lv["start"] = np.searchsorted(lv["par"], np.arange(nprev))
            lv["end"] = np.searchsorted(lv["par"], np.arange(nprev), side="right")
            del lv["pre"]


@dataclass(frozen=True)
class WordCfg:
    obs: float = 2.0
    deletion: float = -5.0
    insertion: float = -7.0
    space_del: float = -4.0
    lm: float = 1.0
    word_bonus: float = 0.0
    uni: float = 0.3
    beam: int = 8
    k_words: int = 80        # 40 / margin 14 left 30% of true words out of every shortlist under kbd noise
    margin: float = 20.0
    level_cap: int = 6000
    lmax: int = 26


def word_lattice(O: np.ndarray, i: int, trie: Trie, c: WordCfg):
    """-> (word ids, span d, tap score) for words starting at tap i: Viterbi over the trie, taps may be
    spurious (insertion) and letters may lack a tap (deletion)."""
    T = len(O)
    W = min(T - i, c.lmax)
    dd = np.arange(W + 1)
    root = dd * c.insertion
    P = root[None, :]
    keep_par = np.array([0])
    ends_w, ends_s = [], []
    Ow = O[i:i + W]
    for lv in trie.levels:
        if not len(keep_par):
            break
        st, en = lv["start"], lv["end"]
        valid = keep_par[keep_par < len(st)]
        pos = valid
        lens = en[pos] - st[pos]
        if lens.sum() == 0:
            break
        child = np.repeat(st[pos], lens) + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
        src = np.repeat(np.searchsorted(keep_par, pos), lens)
        Pp = P[src]
        A = Pp + c.deletion
        A[:, 1:] = np.maximum(A[:, 1:], Pp[:, :-1] + Ow[:, lv["ch"][child]].T)
        N = np.maximum.accumulate(A - dd * c.insertion, axis=1) + dd * c.insertion
        best = N.max(1)
        thr = best.max() - c.margin
        m = best >= thr
        if m.sum() > c.level_cap:
            m &= best >= np.partition(best, -c.level_cap)[-c.level_cap]
        w = lv["wid"][child]
        isw = w >= 0
        if isw.any():
            ends_w.append(w[isw]); ends_s.append(N[isw])
        order = np.flatnonzero(m)
        keep_par = child[order]
        P = N[order]
        srt = np.argsort(keep_par, kind="stable")
        keep_par, P = keep_par[srt], P[srt]
    if not ends_w:
        return np.zeros(0, int), np.zeros((0, W + 1))
    return np.concatenate(ends_w), np.vstack(ends_s)


def word_decode(O_raw: np.ndarray, trie: Trie, nlm: NLM, c: WordCfg, return_nbest: int = 0):
    """Stack decoder over tap positions: hyps = word tuples, extended by lattice words + neural LM."""
    T = len(O_raw)
    if T == 0:
        return ""
    O = c.obs * O_raw
    stacks: list[dict] = [dict() for _ in range(T + 1)]
    stacks[0][()] = 0.0
    finals: dict = {}
    for i in range(T):
        if not stacks[i]:
            continue
        hyps = sorted(stacks[i].items(), key=lambda kv: -kv[1])[:c.beam]
        wid, S = word_lattice(O, i, trie, c)
        for h, sc in hyps:
            if h:
                _push(finals, h, sc + (T - i) * c.insertion)
        if not len(wid):
            continue
        W = S.shape[1] - 1
        d = np.arange(W + 1)
        # space after the word: consumed tap (if any left) or deleted
        sp_tap = np.full(W + 1, -np.inf)
        ok = i + d < T
        sp_tap[ok] = O[i + d[ok], SPACE]
        cont_tap = S + sp_tap[None, :]
        cont_del = S + c.space_del
        cand = []
        for arr, extra in ((cont_tap, 1), (cont_del, 0)):
            flat = arr.ravel()
            kk = min(6 * c.k_words, flat.size)
            fl = np.argpartition(-flat, kk - 1)[:kk]
            r, dcol = np.unravel_index(fl, arr.shape)
            for rr, dc in zip(r, dcol):
                e = i + dc + extra
                if dc == 0 or e > T or not np.isfinite(arr[rr, dc]):
                    continue
                cand.append((int(wid[rr]), e, float(arr[rr, dc])))
        fin = []
        if T - i <= W:
            col = S[:, T - i]
            kk = min(2 * c.k_words, len(col))
            for rr in np.argpartition(-col, kk - 1)[:kk]:
                if np.isfinite(col[rr]):
                    fin.append((int(wid[rr]), float(col[rr])))
        # prefilter distinct words by tap score + unigram prior
        pri: dict[int, float] = {}
        for w_, e, s_ in cand:
            pri[w_] = max(pri.get(w_, -np.inf), s_ + c.uni * trie.uni[w_])
        for w_, s_ in fin:
            pri[w_] = max(pri.get(w_, -np.inf), s_ + c.uni * trie.uni[w_])
        top = set(sorted(pri, key=pri.get, reverse=True)[:c.k_words])
        cand = [x for x in cand if x[0] in top]
        fin = [x for x in fin if x[0] in top]
        pairs = [(h, trie.words[w_]) for h, _ in hyps for w_ in top]
        lm = dict(zip(pairs, nlm.score(pairs)))
        for h, sc in hyps:
            for w_, e, s_ in cand:
                word = trie.words[w_]
                _push(stacks[e], h + (word,), sc + s_ + c.lm * lm[(h, word)] + c.word_bonus)
            for w_, s_ in fin:
                word = trie.words[w_]
                _push(finals, h + (word,), sc + s_ + c.lm * lm[(h, word)] + c.word_bonus)
    for h, sc in stacks[T].items():
        _push(finals, h, sc)
    if not finals:
        return ""
    top = sorted(finals.items(), key=lambda kv: -kv[1])[:4 * c.beam]
    eos = nlm.score([(h, "\n") for h, _ in top])
    ranked = sorted(((h, sc + c.lm * e) for (h, sc), e in zip(top, eos)), key=lambda kv: -kv[1])
    if return_nbest:
        return [(" ".join(h), s_) for h, s_ in ranked[:return_nbest]]
    return " ".join(ranked[0][0])


def _push(d: dict, k, v: float):
    if v > d.get(k, -np.inf):
        d[k] = v


_TRIE: dict = {}


def trie() -> Trie:
    if "t" not in _TRIE:
        import pickle
        f = CACHE / "trie.pkl"
        if f.exists():
            _TRIE["t"] = pickle.loads(f.read_bytes())
        else:
            _TRIE["t"] = Trie(json.loads((CACHE / "lexicon.json").read_text()))
            f.write_bytes(pickle.dumps(_TRIE["t"]))
    return _TRIE["t"]



# ============================================================================ word decoder: tune + sweep
WORD_ANCHOR = 0.75


def model_tag(name: str) -> str:
    return name.split("/")[-1].lower()


def _load_json(p: Path, default):
    return json.loads(p.read_text()) if p.exists() else default


def word_cfg_for(noise: str, tag: str) -> WordCfg:
    t = _load_json(CACHE / f"tune_word_{tag}.json", {})
    if noise == "target" and noise not in t:
        noise = "kbd"   # untuned tap-noise regime borrows the nearest tuned one (disclosed in SUMMARY)
    if noise not in t:
        return WordCfg()
    base = WordCfg()
    return replace(WordCfg(**t[noise]["cfg"]), k_words=base.k_words, margin=base.margin, level_cap=base.level_cap)


def cmd_tune_word(a) -> int:
    """Coordinate search on dev at one anchor accuracy per noise; NLM cache is shared across configs."""
    tag = model_tag(a.model)
    nlm = NLM(a.model)
    tr = trie()
    dev = texts()["dev"][:a.n]
    nz = noises()
    names = a.noises.split(",") if a.noises else ["clean", "kbd", "desk"]
    out_p = CACHE / f"tune_word_{tag}.json"
    res = _load_json(out_p, {})
    for name in names:
        Vs = [sim_obs("dev", i, s, nz[name], WORD_ANCHOR)[0] for i, s in enumerate(dev)]
        bc = beam_cfg_for(name, WORD_ANCHOR)
        cfg = WordCfg(obs=min(bc.obs, 3.0), deletion=bc.deletion, insertion=bc.insertion)
        best = None
        for field, grid in (("obs", (1.0, 2.0, 3.5)), ("lm", (0.6, 1.0, 1.5)),
                            ("word_bonus", (-3.0, 0.0, 3.0)),
                            ("insertion", (cfg.insertion - 2, cfg.insertion, cfg.insertion + 2))):
            scores = {}
            for v in dict.fromkeys(grid):
                c = replace(cfg, **{field: v})
                t0 = time.time()
                S = np.array([sent_stats(s, word_decode(V, tr, nlm, c)) for s, V in zip(dev, Vs)])
                scores[v] = S[:, 0].sum() / S[:, 1].sum()
                print(f"  {name} {field}={v:g} CER={scores[v]:.3f} ({time.time()-t0:.0f}s)", flush=True)
            cfg = replace(cfg, **{field: min(scores, key=scores.get)})
            best = min(scores.values())
        res[name] = {"cfg": cfg.__dict__, "dev_cer": best, "anchor": WORD_ANCHOR, "n": len(dev)}
        out_p.write_text(json.dumps(res, indent=1))
        print(f"{name}: {cfg} dev CER {best:.3f}", flush=True)
    return 0


def cmd_sweep_word(a) -> int:
    tag = model_tag(a.model)
    nlm = NLM(a.model)
    tr = trie()
    T = texts()
    nz = noises()
    names = a.noises.split(",") if a.noises else ["clean", "kbd", "desk"]
    accs = [float(x) for x in a.accs.split(",")] if a.accs else list(ACCS)
    plan_f = CACHE / "sweep_word_plan.json"
    plan = json.loads(plan_f.read_text()) if plan_f.exists() else [[n, accs, a.n] for n in names]
    for split in a.splits.split(","):
        p = CACHE / f"word_{tag}_{split}.jsonl"
        done = {(r["i"], r["noise"], r["acc"]) for r in map(json.loads, p.read_text().splitlines())} \
            if p.exists() else set()
        t0, k = time.time(), 0
        with p.open("a") as fh:
            for name, p_accs, p_n in plan:
                cfg = word_cfg_for(name, tag)
                sents = T[split][:p_n] if p_n else T[split]
                for acc in p_accs:
                    for i, s in enumerate(sents):
                        if (i, name, acc) in done:
                            continue
                        V, _ = sim_obs(split, i, s, nz[name], acc)
                        t1 = time.time()
                        h = word_decode(V, tr, nlm, cfg)
                        fh.write(json.dumps({"i": i, "noise": name, "acc": acc, "hyp": h,
                                             "sec": time.time() - t1}) + "\n")
                        k += 1
                        if k % 100 == 0:
                            fh.flush()
                            print(f"  {split} {name} acc={acc} {k} done ({time.time()-t0:.0f}s)", flush=True)
    return 0


# ============================================================================ n-best neural rescoring
def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def nlm_scores(tag: str, model: str, hyps: set[str], nlm=None) -> dict[str, float]:
    p = CACHE / f"nlm_sent_{tag}.json"
    have = _load_json(p, {})
    need = sorted(h for h in hyps if h not in have)
    if need:
        nlm = nlm or NLM(model)
        t0 = time.time()
        for k in range(0, len(need), 4096):
            part = need[k:k + 4096]
            have.update(zip(part, nlm.sentence_logp(part)))
            if (k // 4096) % 5 == 0:
                print(f"  scored {k+len(part)}/{len(need)} ({time.time()-t0:.0f}s)", flush=True)
                p.write_text(json.dumps(have))
        p.write_text(json.dumps(have))
    return have


LAMS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
GAMS = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)


def rescore_pick(nb: list, lm: dict, lam: float, gam: float) -> str:
    return max(nb, key=lambda x: x[1] + lam * lm[x[0]] + gam * len(x[0].split()))[0]


def tune_rescore(recs: list[dict], texts_: list[str], lm: dict) -> tuple[float, float, float]:
    best = (np.inf, 0.0, 0.0)
    for lam in LAMS:
        for gam in GAMS:
            e = n = 0
            for r in recs:
                ref = texts_[r["i"]]
                e += edit_distance(ref, rescore_pick(r["nbest"], lm, lam, gam))
                n += len(ref)
            best = min(best, (e / n, lam, gam))
    return best


def cmd_rescore(a) -> int:
    tag = model_tag(a.model)
    T = texts()
    recs = {sp: load_jsonl(nbest_path(sp)) for sp in ("dev", "test", "proj")}
    hyps = {h for rs in recs.values() for r in rs for h, _ in r["nbest"]}
    print(f"{len(hyps)} distinct n-best strings", flush=True)
    lm = nlm_scores(tag, a.model, hyps)
    out = {}
    for name in sorted({r["noise"] for r in recs["dev"]}):
        dev = [r for r in recs["dev"] if r["noise"] == name]
        cer, lam, gam = tune_rescore(dev, T["dev"], lm)
        out[name] = {"lam": lam, "gam": gam, "dev_cer": cer}
        print(f"{name}: lambda={lam} gamma={gam} dev CER {cer:.3f}", flush=True)
    (CACHE / f"rescore_{tag}.json").write_text(json.dumps(out, indent=1))
    return 0


# ============================================================================ real desk session
def desk_inputs():
    res = json.loads((CACHE / "desk_obs.json").read_text())
    return res, [np.log(np.maximum(np.array(r["probs"]), EPS)) for r in res], [r["text"] for r in res]


def cmd_desk_eval(a) -> int:
    """(a) the picked beam decode (must reproduce 0.448), (b) its n-best rescored, (c) word decoder;
    decoder settings come from simulated-desk dev tuning, never from the desk phrases."""
    from phase0.analysis.decode import Weights, beam_decode
    tag = model_tag(a.model)
    res, obs, refs = desk_inputs()
    c, w = lms()
    rows = {"beam": [], "nbest": []}
    lists = []
    for r, o in zip(res, obs):
        wt = Weights(obs=r["obs"], deletion=r["deletion"], insertion=-7.0, max_deletions=3)
        hyp = beam_decode(o, c, w, wt, 30)
        rows["beam"].append(hyp)
        nb = beam_nbest(o, BeamCfg(obs=r["obs"], deletion=r["deletion"], insertion=-7.0, beam=30, nbest=30))
        lists.append(nb)
        rows["nbest"].append(nb[0][0])
    nlm = NLM(a.model)
    lm = nlm_scores(tag, a.model, {h for nb in lists for h, _ in nb}, nlm)
    rs = _load_json(CACHE / f"rescore_{tag}.json", {})
    lam, gam = rs["desk"]["lam"], rs["desk"]["gam"]
    rows[f"nbest+{tag}"] = [rescore_pick(nb, lm, lam, gam) for nb in lists]
    rows["nbest_oracle"] = [min(nb, key=lambda x: edit_distance(ref, x[0]))[0] for nb, ref in zip(lists, refs)]
    lopo = []
    for j in range(len(refs)):
        tr_ = [{"i": k, "nbest": lists[k]} for k in range(len(refs)) if k != j]
        _, l_, g_ = tune_rescore(tr_, refs, lm)
        lopo.append(rescore_pick(lists[j], lm, l_, g_))
    rows[f"nbest+{tag}_lopo"] = lopo
    tr = trie()
    for name in ("desk", "kbd"):
        cfg = word_cfg_for(name, tag)
        rows[f"word_{tag}[{name}-tuned]"] = [word_decode(o, tr, nlm, cfg) for o in obs]
    stats = {k: np.array([sent_stats(ref, h) for ref, h in zip(refs, v)]) for k, v in rows.items()}
    out = {"n_phrases": len(refs), "rescore_weights": [lam, gam], "decoders": {}}
    for k, S in stats.items():
        out["decoders"][k] = {**summarize(S), "vs_beam": paired(stats["beam"], S),
                              "hyps": rows[k]}
        m = out["decoders"][k]
        print(f"{k:<34} CER {m['cer']:.3f} [{m['cer_ci'][0]:.3f},{m['cer_ci'][1]:.3f}] WER {m['wer']:.3f} "
              f"word-acc {m['word_acc']:.3f}  dCER vs beam {m['vs_beam']['d_cer']:+.3f} "
              f"[{m['vs_beam']['d_cer_ci'][0]:+.3f},{m['vs_beam']['d_cer_ci'][1]:+.3f}]", flush=True)
    out["refs"] = refs
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"desk_{tag}.json").write_text(json.dumps(out, indent=1))
    for ref, *hs in list(zip(refs, *[rows[k] for k in rows]))[:8]:
        print(f"  typed {ref!r}\n    " + "\n    ".join(f"{k:<30} {h!r}" for k, h in zip(rows, hs)))
    return 0


# ============================================================================ LLM correction (few cells)
LLM_PROMPT = ("These are candidate readings of one short English sentence typed on an invisible keyboard; "
              "they contain typing and recognition errors. Reply with only the most likely intended sentence, "
              "lowercase, no punctuation.\nCandidates:\n{cands}\nSentence:")


def cmd_llm(a) -> int:
    """Small local instruct model rewrites the top-10 n-best; greedy, no desk or test tuning."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = a.model if "Instruct" in a.model else "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to("mps").eval()
    tag = model_tag(name)
    cells = [tuple(x.split(":")) for x in a.cells.split(",")]
    T = texts()
    jobs = []
    for split in a.splits.split(","):
        for r in load_jsonl(nbest_path(split)):
            if (r["noise"], f"{r['acc']}") in cells and (not a.n or r["i"] < a.n):
                jobs.append((split, T[split][r["i"]], r))
    desk = []
    if "desk_real" in a.cells:
        res, obs, refs = desk_inputs()
        for r, o, ref in zip(res, obs, refs):
            nb = beam_nbest(o, BeamCfg(obs=r["obs"], deletion=r["deletion"], insertion=-7.0, nbest=30))
            desk.append(("desk_real", ref, {"noise": "desk_real", "acc": "real", "i": len(desk), "nbest": nb}))
    out_p = CACHE / f"llm_{tag}.jsonl"
    done = {(r["split"], r["noise"], str(r["acc"]), r["i"]) for r in load_jsonl(out_p)}
    t0 = time.time()
    with out_p.open("a") as fh:
        for k, (split, ref, r) in enumerate(jobs + desk):
            if (split, r["noise"], str(r["acc"]), r["i"]) in done:
                continue
            cands = "\n".join(f"- {h}" for h, _ in r["nbest"][:10] if h)
            msgs = [{"role": "user", "content": LLM_PROMPT.format(cands=cands or "- ")}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True)["input_ids"].to("mps")
            with torch.no_grad():
                g = m.generate(ids, max_new_tokens=40, do_sample=False)
            hyp = norm(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).split("\n")[0])
            fh.write(json.dumps({"split": split, "noise": r["noise"], "acc": r["acc"], "i": r["i"],
                                 "hyp": hyp, "ref": ref}) + "\n")
            if (k + 1) % 50 == 0:
                fh.flush()
                print(f"  {k+1}/{len(jobs)+len(desk)} ({time.time()-t0:.0f}s)", flush=True)
    return 0


# ============================================================================ report
def _stats_by_cell(recs, texts_, pick, seed: int = 0) -> dict:
    cells: dict = {}
    for r in recs:
        if r.get("seed", 0) != seed:
            continue
        cells.setdefault((r["noise"], float(r["acc"])), []).append((r["i"], sent_stats(texts_[r["i"]], pick(r))))
    return {k: np.array([s for _, s in sorted(v)]) for k, v in cells.items()}


def cmd_report(a) -> int:
    tag = model_tag(a.model)
    T = texts()
    rs = _load_json(CACHE / f"rescore_{tag}.json", {})
    lm = _load_json(CACHE / f"nlm_sent_{tag}.json", {})
    out = {"meta": {"text": "MacKenzie phrase set: dev 150 / test 350 (seeded); + 20 project phrases",
                    "noises": {k: v.__dict__ for k, v in noises().items()},
                    "calibration": _load_json(CACHE / "calib.json", {}),
                    "desk_rates": _load_json(CACHE / "desk_rates.json", {}),
                    "tune_beam": _load_json(CACHE / "tune_beam.json", {}),
                    "tune_word": _load_json(CACHE / f"tune_word_{tag}.json", {}),
                    "rescore": rs, "nlm": a.model}, "splits": {}}
    for split in ("test", "proj"):
        nb = load_jsonl(nbest_path(split))
        dec = {"beam": _stats_by_cell(nb, T[split], lambda r: r["nbest"][0][0])}
        if rs and lm:
            dec[f"nbest+{tag}"] = _stats_by_cell(
                nb, T[split], lambda r: rescore_pick(r["nbest"], lm, rs[r["noise"]]["lam"], rs[r["noise"]]["gam"]))
            dec["nbest_oracle"] = _stats_by_cell(
                nb, T[split], lambda r: min(r["nbest"], key=lambda x: edit_distance(T[split][r["i"]], x[0]))[0])
        wr = load_jsonl(CACHE / f"word_{tag}_{split}.jsonl")
        if wr:
            dec[f"word+{tag}"] = _stats_by_cell(wr, T[split], lambda r: r["hyp"])
        for f in CACHE.glob("llm_*.jsonl"):
            seen, lr = set(), []
            for r in load_jsonl(f):   # llm rows carry no seed; the first per sentence came from the seed-0 n-best
                k = (r["split"], r["noise"], r["acc"], r["i"])
                if r["split"] == split and k not in seen:
                    seen.add(k)
                    lr.append(r)
            if lr:
                dec[f.stem.replace("llm_", "llm+")] = _stats_by_cell(lr, T[split], lambda r: r["hyp"])
        rows = []
        for d, cells in dec.items():
            for (nz, acc), S in sorted(cells.items()):
                base = dec["beam"].get((nz, acc))
                row = {"decoder": d, "noise": nz, "acc": acc, **summarize(S)}
                if d != "beam" and base is not None and len(base) == len(S):
                    row["vs_beam"] = paired(base, S)
                elif d != "beam" and base is not None:
                    row["note"] = f"subset n={len(S)}; paired vs beam on the same sentences"
                    row["vs_beam"] = paired(base[:len(S)], S)
                rows.append(row)
        out["splits"][split] = rows
    s1 = _stats_by_cell(load_jsonl(nbest_path("test")), T["test"], lambda r: r["nbest"][0][0], seed=1)
    s0 = _stats_by_cell(load_jsonl(nbest_path("test")), T["test"], lambda r: r["nbest"][0][0], seed=0)
    out["seed_check_beam_test"] = [{"noise": k[0], "acc": k[1], "seed0": summarize(s0[k]), "seed1": summarize(v),
                                    "seed1_minus_seed0": paired(s0[k], v)}
                                   for k, v in sorted(s1.items()) if k in s0 and len(s0[k]) == len(v)]
    dk = _load_json(OUT / f"desk_{tag}.json", {})
    if dk:
        out["real_desk_beam"] = {k: dk["decoders"]["beam"][k] for k in ("cer", "word_acc")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sweep.json").write_text(json.dumps(out, indent=1))
    plot(out, OUT / "curve.png")
    for r in out["splits"]["test"]:
        v = r.get("vs_beam", {})
        print(f"{r['decoder']:<22}{r['noise']:<8}{r['acc']:<6} CER {r['cer']:.3f} word-acc {r['word_acc']:.3f} "
              f"[{r['word_acc_ci'][0]:.3f},{r['word_acc_ci'][1]:.3f}] sent {r['sent_acc']:.2f} n={r['n']}"
              + (f"  dWA {v['d_word_acc']:+.3f} [{v['d_word_acc_ci'][0]:+.3f},{v['d_word_acc_ci'][1]:+.3f}]" if v else ""))
    return 0


SERIES = {"beam": "#2a78d6", "nbest+": "#eb6834", "word+": "#1baf7a", "llm+": "#4a3aa7"}
NOISE_TITLE = {"clean": "perfect tap detection", "target": "5% missed / 5% extra taps",
               "kbd": "keyboard detector (16% missed, 15% extra)", "desk": "desk detector (20% missed, 130% extra)"}


def plot(out: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in out["splits"]["test"] if not r["decoder"].endswith("oracle")]
    noises_ = [n for n in ("clean", "target", "kbd", "desk") if any(r["noise"] == n for r in rows)]
    fig, axes = plt.subplots(2, len(noises_), figsize=(4.2 * len(noises_), 7.4), sharex=True, sharey="row")
    axes = np.atleast_2d(axes)
    decs = list(dict.fromkeys(r["decoder"] for r in rows))
    for col, nz in enumerate(noises_):
        for row_i, (key, lab) in enumerate((("word_acc", "words correct"), ("cer", "character error rate"))):
            ax = axes[row_i, col]
            for d in decs:
                pts = sorted((r["acc"], r[key], r[key + "_ci"]) for r in rows if r["decoder"] == d and r["noise"] == nz)
                if not pts:
                    continue
                color = next(v for k, v in SERIES.items() if d.startswith(k))
                x = [p[0] * 100 for p in pts]
                ax.fill_between(x, [p[2][0] for p in pts], [p[2][1] for p in pts], color=color, alpha=0.12, lw=0)
                ax.plot(x, [p[1] for p in pts], color=color, lw=2, marker="o", ms=4, label=d)
            ax.axvline(65.7, color="#8a8984", lw=1, ls=":")
            real = out.get("real_desk_beam")
            if nz == "desk" and real:
                ax.axhline(real[key], color="#52514e", lw=1, ls="--")
                ax.text(51, real[key], "real desk session (beam)", fontsize=7, color="#52514e", va="bottom")
            if row_i == 0:
                ax.axhspan(0.8, 0.9, color="#8a8984", alpha=0.08, lw=0)
                ax.set_title(NOISE_TITLE[nz], fontsize=10, color="#0b0b0b")
            ax.grid(color="#e6e5e0", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if col == 0:
                ax.set_ylabel(lab, color="#52514e")
            if row_i == 1:
                ax.set_xlabel("raw key accuracy (top-1, %)", color="#52514e")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Held-out MacKenzie phrases simulated from real per-tap key probabilities (beam, n-best: n=350; "
                 "word-level: n=30-100); dotted = current held-out model, band = 80-90% words", fontsize=10,
                 color="#52514e")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


# ============================================================================ simulator robustness
def _robust_job(args):
    i, text, name, acc = args
    pool, cal, uni = pool_ctx()
    nz = noises()[name]
    V, t = simulate(text, pool, nz, [0, SPLIT_ID["test"], i, NOISE_ID[name]], "resample", 0.0, uni, acc)
    m = t >= 0
    r = (V[m] > V[m][np.arange(m.sum()), t[m]][:, None]).sum(1) if m.any() else np.zeros(0)
    return {"i": i, "noise": name, "acc": acc, "hyp": beam_nbest(V, replace(beam_cfg_for(name, acc), nbest=1))[0][0],
            "top1": float((r == 0).mean()) if len(r) else 0.0, "top5": float((r < 5).mean()) if len(r) else 0.0}


def cmd_robust(a) -> int:
    """Same sentences/noise, key accuracy set by resampling real vectors instead of the logit shift."""
    T = texts()["test"][:150]
    cells = [("clean", 0.6), ("clean", 0.8), ("clean", 0.9), ("kbd", 0.6), ("kbd", 0.8), ("kbd", 0.9)]
    jobs = [(i, s, n, acc) for n, acc in cells for i, s in enumerate(T)]
    rs = run_pool(_robust_job, jobs, a.procs)
    shift_recs = {(r["i"], r["noise"], r["acc"]): r for r in load_jsonl(nbest_path("test")) if r["seed"] == 0}
    out = []
    for n, acc in cells:
        cur = sorted((r for r in rs if r["noise"] == n and r["acc"] == acc), key=lambda r: r["i"])
        Sr = np.array([sent_stats(T[r["i"]], r["hyp"]) for r in cur])
        Ss = np.array([sent_stats(T[r["i"]], shift_recs[(r["i"], n, acc)]["nbest"][0][0]) for r in cur])
        row = {"noise": n, "acc": acc, "resample": summarize(Sr), "shift": summarize(Ss),
               "resample_minus_shift": paired(Ss, Sr),
               "resample_vec_top1": float(np.mean([r["top1"] for r in cur])),
               "resample_vec_top5": float(np.mean([r["top5"] for r in cur]))}
        out.append(row)
        print(f"{n} {acc}: word-acc shift {row['shift']['word_acc']:.3f} resample {row['resample']['word_acc']:.3f} "
              f"d={row['resample_minus_shift']['d_word_acc']:+.3f} {row['resample_minus_shift']['d_word_acc_ci']} "
              f"| resample top1 {row['resample_vec_top1']:.3f} top5 {row['resample_vec_top5']:.3f}", flush=True)
    (CACHE / "robust.json").write_text(json.dumps(out, indent=1))
    return 0



# ============================================================================ memorisation probe
def ngram_logp(text: str) -> float:
    c, _ = lms()
    t, tot = text + " ", 0.0
    for j, ch in enumerate(t):
        tot += float(c.logprobs((" " + t[:j])[-5:])[A_INDEX[ch]])
    return tot


def cmd_probe(a) -> int:
    """Neural-LM vs char 6-gram bits/char on MacKenzie, project phrases and word-shuffled MacKenzie."""
    T = texts()
    nlm = NLM(a.model)
    rng = np.random.default_rng(0)
    sets = {"mackenzie_test": T["test"], "project_phrases": T["proj"],
            "mackenzie_test_word_shuffled": [" ".join(rng.permutation(x.split())) for x in T["test"]]}
    out = {}
    for k, v in sets.items():
        chars = sum(len(x) + 1 for x in v)
        nb = -sum(nlm.sentence_logp(v)) / chars / math.log(2)
        gb = -sum(ngram_logp(x) for x in v) / chars / math.log(2)
        out[k] = {"n": len(v), "nlm_bits_per_char": nb, "ngram_bits_per_char": gb, "nlm_over_ngram": nb / gb}
        print(f"{k:<30} n={len(v):<4} NLM {nb:.3f} b/c  6-gram {gb:.3f} b/c  ratio {nb/gb:.3f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"probe_{model_tag(a.model)}.json").write_text(json.dumps(out, indent=1))
    return 0



# ============================================================================ CLI
def cmd_pool(a) -> int:
    p = build_pool()
    pool = Pool(p)
    T = texts()
    chars = np.array([A_INDEX[c] for s in T["dev"] for c in s])
    cal = calibrate(pool, chars)
    held = Pool(p, "held")
    cal["held_only_b0"] = calibrate(held, chars)["b0"]
    cal["borrow"] = {ALPHABET[k]: ALPHABET[v] for k, v in pool.borrow.items()}
    (CACHE / "calib.json").write_text(json.dumps(cal, indent=1))
    print(json.dumps(cal, indent=1))
    return 0


COMMANDS = {"pool": cmd_pool, "lexicon": cmd_lexicon, "desk-dump": cmd_desk_dump, "rates": cmd_rates,
            "tune-beam": cmd_tune_beam, "sweep-beam": cmd_sweep_beam, "tune-word": cmd_tune_word,
            "sweep-word": cmd_sweep_word, "rescore": cmd_rescore, "desk-eval": cmd_desk_eval,
            "llm": cmd_llm, "report": cmd_report, "robust": cmd_robust,
            "probe": cmd_probe}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.autocorrect")
    ap.add_argument("cmd", choices=sorted(COMMANDS))
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--top", type=int, default=120_000)
    ap.add_argument("--splits", default="test")
    ap.add_argument("--noises", default="")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--accs", default="")
    ap.add_argument("--cells", default="kbd:0.8,kbd:0.9,desk:0.8,desk:0.9,clean:0.657")
    a, rest = ap.parse_known_args(argv)
    a.rest = rest
    return COMMANDS[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
