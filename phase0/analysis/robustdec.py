"""Noise-robust word decoder: pair-HMM tap channel (deletions, bursts of spurious taps repeating prev/next key)
in a trie word lattice + Qwen word stack search. Run: python -m phase0.analysis.robustdec {tune|sweep|desk|report}"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass, replace, asdict
from pathlib import Path

import numpy as np

from phase0.analysis import autocorrect as AC
from phase0.analysis.decode import A_INDEX, NA, SPACE, edit_distance

CACHE = Path(".cache/robustdec")
OUT = Path("results/robustdec")
BLOCK = Path("models/vocab_blocklist.txt")
SUFFIXES = ("", "s", "es", "ed", "er", "ers", "ing", "in", "y")
ACCS = (0.7, 0.8, 0.9)
NOISES = ("clean", "target", "kbd", "desk")
ANCHOR = 0.8


# ============================================================================ vocabulary
def blocklist() -> set[str]:
    return {l.strip() for l in BLOCK.read_text().splitlines() if l.strip() and not l.startswith("#")}


def is_blocked(w: str, B: set[str]) -> bool:
    return any(w.endswith(s) and w[:len(w) - len(s)] in B for s in SUFFIXES if len(w) > len(s))


_TR: dict = {}


def trie() -> AC.Trie:
    """autocorrect's lexicon minus the blocklist (and inflections)."""
    if "t" not in _TR:
        f = CACHE / "trie_filtered.pkl"
        if f.exists():
            _TR["t"] = pickle.loads(f.read_bytes())
        else:
            lex = json.loads((AC.CACHE / "lexicon.json").read_text())
            B = blocklist()
            lex = {w: v for w, v in lex.items() if not is_blocked(w, B)}
            _TR["t"] = AC.Trie(lex)
            CACHE.mkdir(parents=True, exist_ok=True)
            f.write_bytes(pickle.dumps(_TR["t"]))
    return _TR["t"]


def cmd_vocab(a) -> int:
    lex = json.loads((AC.CACHE / "lexicon.json").read_text())
    B = blocklist()
    nb = sum(is_blocked(w, B) for w in lex)
    T = AC.texts()
    hits = {k: sum(is_blocked(w, B) for s in v for w in s.split()) for k, v in T.items()}
    tr = trie()
    print(f"lexicon {len(lex)} words, blocked {nb} (words not printed); trie words {len(tr.words)}; "
          f"blocked tokens in reference text: {hits}")
    return 0


# ============================================================================ channel + decoder
@dataclass(frozen=True)
class RCfg:
    obs: float = 2.0          # exponent on real-tap key probabilities
    ins_obs: float = 1.0      # exponent on spurious-tap key probabilities
    p_del: float = 0.2        # P(char has no real tap)
    p_sdel: float = 0.2       # P(space has no real tap)
    ins: float = 1.3          # expected spurious taps per gap (geometric)
    m_prev: float = 0.45      # spurious key = previous char
    m_next: float = 0.33      # spurious key = next char
    m_space: float = 0.08     # spurious key = space; rest unigram
    agg: str = "sum"          # marginalise ("sum") or Viterbi ("max") over alignments inside a word
    lm: float = 1.0
    word_bonus: float = 0.0
    uni: float = 0.3
    beam: int = 8
    k_words: int = 50
    ends: int = 4             # end positions pushed per shortlisted word
    margin: float = 12.0
    level_cap: int = 5000
    lmax: int = 36            # max taps per word
    soft: float = 0.0         # weight on centred detector logit, real-tap side (real desk only)
    dup: float = 0.0          # bonus for spurious reading of a same-finger tap <120 ms after the previous

    def logs(self):
        q = self.ins / (1.0 + self.ins)
        m_uni = max(1.0 - self.m_prev - self.m_next - self.m_space, 1e-6)
        return dict(ld=math.log(max(self.p_del, 1e-6)), lm_=math.log(max(1 - self.p_del, 1e-6)),
                    lsd=math.log(max(self.p_sdel, 1e-6)), lsm=math.log(max(1 - self.p_sdel, 1e-6)),
                    lq=math.log(max(q, 1e-9)), lstop=math.log(1 - q),
                    lmix=np.log(np.maximum([self.m_prev, self.m_next, self.m_space, m_uni], 1e-9)))


def channel(O_raw: np.ndarray, c: RCfg, uni: np.ndarray, feats: dict | None = None):
    """-> match scores M (T,27) and cumulative spurious-tap scores CS (T+1,27,27) for (prev, next) keys."""
    L = c.logs()
    T = len(O_raw)
    O_raw = O_raw.astype(np.float32)
    M = c.obs * O_raw + np.float32(L["lm_"])
    if feats is not None and c.soft:
        M += np.float32(c.soft) * feats["zlogit"][:, None]
    Sp = c.ins_obs * O_raw
    U = np.logaddexp.reduce(Sp + np.log(uni)[None, :], axis=1)
    mx = L["lmix"]
    base = np.logaddexp(mx[2] + Sp[:, SPACE], mx[3] + U)                 # (T,)
    prev = mx[0] + Sp[:, :, None]                                          # (T,27,1)
    nxt = mx[1] + Sp[:, None, :]                                           # (T,1,27)
    spur = np.logaddexp(np.logaddexp(prev, nxt), base[:, None, None]) + np.float32(L["lq"])
    if feats is not None and c.dup:
        spur += np.float32(c.dup) * feats["dup"][:, None, None]
    CS = np.zeros((T + 1, NA, NA), np.float32)
    np.cumsum(spur, axis=0, out=CS[1:])
    return M, CS, L


def _gap(P, CSi, pc, ch, agg, lstop):
    """P (n,W+1) state after the previous char; gap of spurious taps keyed (pc, ch) -> state before ch."""
    C = CSi[:, pc, ch].T                                                   # (n,W+1)
    if agg == "max":
        return np.maximum.accumulate(P - C, axis=1) + C + lstop
    return _lcse(P - C) + C + lstop


def _lcse(X):
    """Row-wise log-cumsum-exp in the linear domain (terms >700 nats below the row max underflow to 0)."""
    m = X.max(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        Y = np.exp(X - m)
        np.cumsum(Y, axis=1, out=Y)
        return np.log(Y) + m


def word_lattice(M, CS, L, i: int, tr: AC.Trie, c: RCfg):
    """Words starting after a boundary at tap i. -> wid, E (n,W+1) score of the word incl. its trailing gap
    to the space and ending exactly before tap i+d, last char per word."""
    T = len(M)
    W = min(T - i, c.lmax)
    agg = np.maximum if c.agg == "max" else np.logaddexp
    CSi = CS[i:i + W + 1] - CS[i][None]                                     # (W+1,27,27)
    Mi = M[i:i + W]
    P = np.full((1, W + 1), -np.inf, np.float32)
    P[0, 0] = 0.0
    pch = np.array([SPACE])
    keep = np.array([0])
    out_w, out_s, out_c = [], [], []
    for lv in tr.levels:
        if not len(keep):
            break
        st, en = lv["start"], lv["end"]
        ok = keep < len(st)
        keep, P, pch = keep[ok], P[ok], pch[ok]
        lens = en[keep] - st[keep]
        if lens.sum() == 0:
            break
        child = np.repeat(st[keep], lens) + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
        src = np.repeat(np.arange(len(keep)), lens)
        ch = lv["ch"][child]
        G = _gap(P[src], CSi, pch[src], ch, c.agg, L["lstop"])
        N = G + L["ld"]
        N[:, 1:] = agg(N[:, 1:], G[:, :-1] + Mi[:, ch].T)
        w = lv["wid"][child]
        isw = w >= 0
        if isw.any():
            out_w.append(w[isw]); out_s.append(N[isw]); out_c.append(ch[isw])
        colbest = N.max(0)
        rel = (N - colbest[None, :]).max(1)
        m = rel >= -c.margin
        if m.sum() > c.level_cap:
            m &= rel >= np.partition(rel, -c.level_cap)[-c.level_cap]
        order = np.flatnonzero(m)
        keep, P, pch = child[order], N[order], ch[order]
        srt = np.argsort(keep, kind="stable")
        keep, P, pch = keep[srt], P[srt], pch[srt]
    if not out_w:
        return None
    return np.concatenate(out_w), np.vstack(out_s), np.concatenate(out_c), CSi, W


def decode(O_raw: np.ndarray, tr: AC.Trie, nlm, c: RCfg, uni: np.ndarray, nbest: int = 0, feats=None):
    T = len(O_raw)
    if T == 0:
        return "" if not nbest else [("", 0.0)]
    M, CS, L = channel(O_raw, c, uni, feats)
    agg = np.maximum if c.agg == "max" else np.logaddexp
    stacks: list[dict] = [dict() for _ in range(T + 1)]
    stacks[0][()] = 0.0
    finals: dict = {}
    for i in range(T):
        if not stacks[i]:
            continue
        hyps = sorted(stacks[i].items(), key=lambda kv: -kv[1])[:c.beam]
        lat = word_lattice(M, CS, L, i, tr, c)
        if lat is None:
            continue
        wid, N, lastc, CSi, W = lat
        # trailing gap to the space, then the space tap (consumed) or a deleted space
        G = _gap(N, CSi, lastc, np.full(len(lastc), SPACE), c.agg, L["lstop"])
        cont = G + np.float32(L["lsd"])                   # column = taps consumed incl. the space
        cont[:, 1:] = agg(cont[:, 1:], G[:, :-1] + (M[i:i + W, SPACE] - L["lm_"] + L["lsm"])[None, :])
        # sentence end: every remaining tap spurious (prev = next = last char); only if the word can reach T
        fin = np.full(len(wid), -np.inf)
        if T - i <= W:
            tail = CS[T][None] - CS[i:T + 1]                                   # (T-i+1,27,27)
            tl = tail[:, lastc, lastc].T                                       # (n, T-i+1)
            fin = agg.reduce(N[:, :T - i + 1] + tl, axis=1) + L["lstop"] if c.agg == "sum" \
                else (N[:, :T - i + 1] + tl).max(1) + L["lstop"]
        colbest = np.maximum(cont.max(0), -1e30)
        rel = np.maximum((cont - colbest[None, :]).max(1), fin - (fin.max() if np.isfinite(fin).any() else 0))
        pri = rel + c.uni * tr.uni[wid]
        # distinct words: keep best row per word (a word id appears once per trie node, so rows are unique)
        kk = min(c.k_words, len(pri))
        top = np.argpartition(-pri, kk - 1)[:kk]
        top = top[np.isfinite(pri[top])]
        if not len(top):
            continue
        words = [tr.words[wid[r]] for r in top]
        pairs = [(h, w) for h, _ in hyps for w in words]
        lmv = dict(zip(pairs, nlm.score(pairs)))
        ne = min(c.ends, W)
        ends = np.argpartition(-(cont[top, 1:] - colbest[None, 1:]), ne - 1, axis=1)[:, :ne] + 1
        for h, sc in hyps:
            for n_, (r, w) in enumerate(zip(top, words)):
                base = sc + c.lm * lmv[(h, w)] + c.word_bonus
                hw = h + (w,)
                for d in ends[n_]:
                    v = cont[r, d]
                    if np.isfinite(v) and i + d <= T:
                        AC._push(stacks[i + d], hw, base + float(v))
                if np.isfinite(fin[r]):
                    AC._push(finals, hw, base + float(fin[r]))
    for h, sc in stacks[T].items():   # ended on a space tap: trailing space is fine
        if h:
            AC._push(finals, h, sc)
    if not finals:
        return "" if not nbest else [("", 0.0)]
    top = sorted(finals.items(), key=lambda kv: -kv[1])[:4 * c.beam]
    eos = nlm.score([(h, "\n") for h, _ in top])
    ranked = sorted(((h, sc + c.lm * e) for (h, sc), e in zip(top, eos)), key=lambda kv: -kv[1])
    if nbest:
        return [(" ".join(h), s) for h, s in ranked[:nbest]]
    return " ".join(ranked[0][0])


def text_score(O_raw: np.ndarray, text: str, c: RCfg, uni: np.ndarray) -> float:
    """Channel log-likelihood of a whole text given the taps (same pair-HMM as the decoder, no LM)."""
    T = len(O_raw)
    M, CS, L = channel(O_raw, c, uni)
    agg = np.maximum if c.agg == "max" else np.logaddexp
    ys = [A_INDEX[ch] for ch in text]
    P = np.full((1, T + 1), -np.inf, np.float32)
    P[0, 0] = 0.0
    pc = SPACE
    for y in ys:
        G = _gap(P, CS, np.array([pc]), np.array([y]), c.agg, L["lstop"])
        ld, lm_ = (L["lsd"], L["lsm"]) if y == SPACE else (L["ld"], L["lm_"])
        N = G + ld
        N[:, 1:] = agg(N[:, 1:], G[:, :-1] + M[:, y] - L["lm_"] + lm_)
        P, pc = N, y
    tail = (CS[T][None] - CS)[:, pc, pc]
    v = P[0] + tail
    return float((np.logaddexp.reduce(v) if c.agg == "sum" else v.max()) + L["lstop"])


def cmd_diag(a) -> int:
    """Real desk: channel score of the reference vs the beam hypothesis vs this decoder's hypothesis."""
    res, obs, refs = AC.desk_inputs()
    old = json.loads((AC.OUT / "desk_qwen2.5-0.5b.json").read_text())["decoders"]["beam"]["hyps"]
    mine = {r["j"]: r["hyp"] for r in AC.load_jsonl(CACHE / f"desk_{a.lm}.jsonl") if r["cfg_key"] == "tuned"}
    cfg = cfg_for("desk", a.lm)
    rows = []
    for j, (o, ref) in enumerate(zip(obs, refs)):
        r = {"j": j, "taps": len(o), "chars": len(ref), "ref": text_score(o, ref, cfg, uni()),
             "beam": text_score(o, old[j], cfg, uni()) if old[j] else None}
        if j in mine and mine[j]:
            r["mine"] = text_score(o, mine[j], cfg, uni())
        rows.append(r)
        print(j, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    b = [r for r in rows if r.get("beam") is not None]
    print(f"ref channel >= beam hyp channel in {sum(r['ref'] >= r['beam'] for r in b)}/{len(b)}; "
          f">= own hyp in {sum(r['ref'] >= r['mine'] for r in rows if 'mine' in r)}/{sum('mine' in r for r in rows)}")
    return 0


def cmd_tapfeat(a) -> int:
    """Per real-desk tap: time, detector probability and attributed finger (for soft-tap / duplicate features)."""
    import lightgbm  # noqa: F401  before torch
    from phase0.analysis import desk_tune as dt
    from phase0.analysis import hirecall as HR
    from phase0.analysis import pipeline as pl

    res = json.loads((AC.CACHE / "desk_obs.json").read_text())
    d = pl.frame_probs(pl.session_path(HR.DESK))
    streams = {json.dumps(x["cfg"] if isinstance(x["cfg"], dict) else eval(x["cfg"]), sort_keys=True): x
               for x in HR.load_streams("base")}
    out = []
    for r in res:
        st = streams[json.dumps(r["stream_cfg"], sort_keys=True)]
        ev = np.array(st["ev"], int)
        assert len(ev) == r["n_taps_stream"]
        seg = dict((t, rows) for t, rows in r["all_segs"])
        rows = np.array([x for t, x in r["all_segs"] if t == r["text"]][0], int)
        assert len(rows) == len(r["probs"])
        k = ev[rows]
        fing = [dt.attribute(d["P"], int(q), d["score"]) for q in k]
        out.append({"text": r["text"], "t": d["t"][k].tolist(), "p": d["p"][k].tolist(),
                    "hand": [int(h) for h, _ in fing], "finger": [int(f) for _, f in fing]})
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "desk_tapfeat.json").write_text(json.dumps(out))
    dt_ = np.concatenate([np.diff(o["t"]) for o in out])
    same = np.concatenate([(np.diff(o["finger"]) == 0) & (np.diff(o["hand"]) == 0) for o in out])
    print(f"{len(out)} phrases; inter-tap s median {np.median(dt_):.3f}; <120ms {np.mean(dt_ < 0.12):.2f}; "
          f"same finger&hand {same.mean():.2f}; same & <120ms {np.mean(same & (dt_ < 0.12)):.2f}")
    return 0


# ============================================================================ cheap LM for smoke tests
class UniLM:
    """Stand-in for NLM: log unigram word probability (no context)."""

    def __init__(self, tr: AC.Trie):
        self.u = dict(zip(tr.words, tr.uni))

    def score(self, pairs):
        return [-3.0 if w == "\n" else float(self.u.get(w, -20.0)) for _, w in pairs]


def load_nlm(kind: str):
    if kind == "uni":
        return UniLM(trie())
    path = CACHE / "qwen2.5-0.5b"
    return AC.NLM(str(path) if path.exists() else "Qwen/Qwen2.5-0.5B")


# ============================================================================ configs
def channel_for(noise: str) -> dict:
    """Channel parameters = the simulator's measured tap-error rates (desk ones were measured by aligning the
    real desk taps to text; mix fitted to the aligned real insertions). No dev/test decoding used here."""
    nz = AC.noises()[noise]
    ins = max(nz.ins, 0.01)
    if nz.mix[1] > 0:   # desk: explicit prev/next/space mix
        mp, mn, ms = nz.mix
    else:
        mp, mn, ms = nz.mix[0], 0.0, 0.0
    return dict(p_del=max(nz.p_del, 0.01), p_sdel=max(nz.p_del, 0.01), ins=ins, m_prev=mp, m_next=mn,
                m_space=ms, ins_obs=1.0 / nz.tau)


def base_cfg(noise: str) -> RCfg:
    return replace(RCfg(), **channel_for(noise))


def cfg_for(noise: str, tag: str = "qwen") -> RCfg:
    t = AC._load_json(CACHE / f"tune_{tag}.json", {})
    if noise in t:
        return RCfg(**t[noise]["cfg"])
    return base_cfg(noise)


_UNI: dict = {}


def uni() -> np.ndarray:
    if "u" not in _UNI:
        _UNI["u"] = AC.unigram(AC.texts()["dev"])
    return _UNI["u"]


# ============================================================================ tune (sim dev only)
TUNE_GRID = (("obs", (1.0, 1.5, 2.0, 3.0)), ("lm", (0.6, 1.0, 1.5)), ("word_bonus", (-2.0, 0.0, 2.0)),
             ("ins_obs", (0.5, 1.0)), ("agg", ("sum", "max")), ("obs", None), ("lm", None))


def cmd_tune(a) -> int:
    nlm = load_nlm(a.lm)
    tag = a.lm
    tr = trie()
    dev = AC.texts()["dev"][:a.n]
    out_p = CACHE / f"tune_{tag}.json"
    res = AC._load_json(out_p, {})
    steps = TUNE_GRID
    if a.steps:
        grids = dict(TUNE_GRID[:-2])
        steps = tuple((f.rstrip("*"), None if f.endswith("*") else grids[f]) for f in a.steps.split(","))
    for name in a.noises.split(","):
        nz = AC.noises()[name]
        Vs = [AC.sim_obs("dev", i, s, nz, ANCHOR)[0] for i, s in enumerate(dev)]
        cfg = replace(base_cfg(name), **json.loads(a.start)) if a.start else base_cfg(name)
        prev = res.get(name, {})
        log = prev.get("log", []) if a.start else []
        memo = {json.dumps(r["cfg"], sort_keys=True): r["cer"] for r in log if "cfg" in r}
        for field, grid in steps:
            v0 = getattr(cfg, field)
            if grid is None:   # refine around the current value
                grid = (round(v0 * 0.7, 3), v0, round(v0 * 1.4, 3))
            scores = {}
            for v in dict.fromkeys((v0,) + tuple(grid)):   # the current value always competes
                cc = replace(cfg, **{field: v})
                key = json.dumps(asdict(cc), sort_keys=True)
                if key in memo:
                    scores[v] = memo[key]
                    continue
                t0 = time.time()
                S = np.array([AC.sent_stats(s, decode(V, tr, nlm, cc, uni())) for s, V in zip(dev, Vs)])
                scores[v] = memo[key] = S[:, 0].sum() / S[:, 1].sum()
                wa = S[:, 4].sum() / S[:, 3].sum()
                log.append({"field": field, "value": v, "cfg": asdict(cc), "cer": scores[v], "word_acc": wa,
                            "sec": time.time() - t0})
                print(f"  {name} {field}={v} CER={scores[v]:.3f} words={wa:.3f} ({time.time()-t0:.0f}s)", flush=True)
            cfg = replace(cfg, **{field: min(scores, key=scores.get)})
            res[name] = {"cfg": asdict(cfg), "dev_cer": scores[getattr(cfg, field)], "anchor": ANCHOR,
                         "n": len(dev), "log": log}
            out_p.write_text(json.dumps(res, indent=1))
        print(f"{name}: {cfg} dev CER {res[name]['dev_cer']:.3f}", flush=True)
    return 0


# ============================================================================ sweep (sim test)
def cmd_sweep(a) -> int:
    nlm = load_nlm(a.lm)
    tr = trie()
    T = AC.texts()
    nz = AC.noises()
    p = CACHE / f"sweep_{a.lm}_{a.split}.jsonl"
    done = {(r["i"], r["noise"], r["acc"]) for r in AC.load_jsonl(p)}
    accs = [float(x) for x in a.accs.split(",")]
    t0, k = time.time(), 0
    with p.open("a") as fh:
        for rnd in range(0, a.n, a.chunk):          # interleave cells so partial runs cover every cell
            for name in a.noises.split(","):
                cfg = cfg_for(name, a.lm)
                for acc in accs:
                    for i in range(rnd, min(rnd + a.chunk, a.n)):
                        if (i, name, acc) in done:
                            continue
                        s = T[a.split][i]
                        V, _ = AC.sim_obs(a.split, i, s, nz[name], acc)
                        t1 = time.time()
                        h = decode(V, tr, nlm, cfg, uni())
                        fh.write(json.dumps({"i": i, "noise": name, "acc": acc, "hyp": h, "n_taps": len(V),
                                             "sec": time.time() - t1}) + "\n")
                        fh.flush()
                        k += 1
                        if k % 25 == 0:
                            print(f"  {a.split} round {rnd} {name} {acc} {k} done ({time.time()-t0:.0f}s)", flush=True)
    return 0


# ============================================================================ real desk
DESK_GRID = (("obs", (0.75, 1.0, 1.5, 2.0)), ("lm", (0.4, 0.6, 1.0)), ("ins_obs", (0.5, 1.0)),
             ("soft", (0.5, 1.0)), ("dup", (1.0,)))


def desk_feats() -> list[dict]:
    f = CACHE / "desk_tapfeat.json"
    if not f.exists():
        return [None] * 20
    out = []
    for r in json.loads(f.read_text()):
        p = np.clip(np.array(r["p"]), 1e-3, 1 - 1e-3)
        z = np.log(p / (1 - p))
        t = np.array(r["t"])
        fid = np.array(r["finger"]) * 2 + np.array(r["hand"])
        dup = np.r_[False, (np.diff(t) < 0.12) & (np.diff(fid) == 0)]
        out.append({"zlogit": (z - z.mean()).astype(np.float32), "dup": dup.astype(np.float32)})
    return out


def cmd_desk(a) -> int:
    """(a) sim-desk-dev-tuned config as is; (b) leave-one-phrase-out over a small grid around it."""
    nlm = load_nlm(a.lm)
    tr = trie()
    res, obs, refs = AC.desk_inputs()
    p = CACHE / f"desk_{a.lm}.jsonl"
    have = {(r["cfg_key"], r["j"]): r for r in AC.load_jsonl(p)}
    base = cfg_for("desk", a.lm)
    cfgs = {"tuned": base}
    for field, grid in DESK_GRID:
        for v in grid:
            cc = replace(base, **{field: v})
            if cc != base:
                cfgs[f"{field}={v}"] = cc
    if a.only_tuned:
        cfgs = {"tuned": base}
    feats = desk_feats()
    t0 = time.time()
    with p.open("a") as fh:
        for key, cc in cfgs.items():
            for j, o in enumerate(obs):
                if (key, j) in have:
                    continue
                t1 = time.time()
                nb = decode(o, tr, nlm, cc, uni(), nbest=10, feats=feats[j])
                r = {"cfg_key": key, "cfg": asdict(cc), "j": j, "hyp": nb[0][0], "nbest": nb, "sec": time.time() - t1}
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                have[(key, j)] = r
                print(f"  desk {key} phrase {j} CER {edit_distance(refs[j], nb[0][0])/len(refs[j]):.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    return 0


# ============================================================================ report
def cmd_report(a) -> int:
    T = AC.texts()
    split = a.split
    rows = AC.load_jsonl(CACHE / f"sweep_{a.lm}_{split}.jsonl")
    beam = {(r["i"], r["noise"], float(r["acc"])): r["nbest"][0][0]
            for r in AC.load_jsonl(AC.nbest_path(split)) if r.get("seed", 0) == 0}
    word = {(r["i"], r["noise"], float(r["acc"])): r["hyp"]
            for r in AC.load_jsonl(AC.CACHE / f"word_qwen2.5-0.5b_{split}.jsonl")}
    out = {"meta": {"tune": {k: {kk: vv for kk, vv in v.items() if kk != "log"}
                             for k, v in AC._load_json(CACHE / f"tune_{a.lm}.json", {}).items()},
                    "split": split, "lm": a.lm}, "cells": []}
    for name in NOISES:
        for acc in ACCS:
            cur = sorted((r for r in rows if r["noise"] == name and float(r["acc"]) == acc), key=lambda r: r["i"])
            if not cur:
                continue
            ref = [T[split][r["i"]] for r in cur]
            S = np.array([AC.sent_stats(x, r["hyp"]) for x, r in zip(ref, cur)])
            cell = {"noise": name, "acc": acc, "robust": AC.summarize(S),
                    "sec_per_sentence": float(np.mean([r["sec"] for r in cur]))}
            if all((r["i"], name, acc) in beam for r in cur):
                Sb = np.array([AC.sent_stats(x, beam[(r["i"], name, acc)]) for x, r in zip(ref, cur)])
                cell["beam_same_sentences"] = AC.summarize(Sb)
                cell["robust_minus_beam"] = AC.paired(Sb, S)
            wsub = [(x, r) for x, r in zip(ref, cur) if (r["i"], name, acc) in word]
            if wsub:
                Sw = np.array([AC.sent_stats(x, word[(r["i"], name, acc)]) for x, r in wsub])
                Sr = np.array([AC.sent_stats(x, r["hyp"]) for x, r in wsub])
                cell["old_word_same_sentences"] = AC.summarize(Sw)
                cell["robust_minus_old_word"] = AC.paired(Sw, Sr)
            out["cells"].append(cell)
            b = cell.get("robust_minus_beam", {})
            print(f"{name:<7}{acc:<5} n={len(cur):<4} robust words {cell['robust']['word_acc']:.3f} "
                  f"[{cell['robust']['word_acc_ci'][0]:.3f},{cell['robust']['word_acc_ci'][1]:.3f}] CER "
                  f"{cell['robust']['cer']:.3f} | beam words {cell.get('beam_same_sentences', {}).get('word_acc', float('nan')):.3f} "
                  f"dWA {b.get('d_word_acc', float('nan')):+.3f} {b.get('d_word_acc_ci', '')} "
                  + (f"| old word {cell['old_word_same_sentences']['word_acc']:.3f} (n={len(wsub)}) dWA "
                     f"{cell['robust_minus_old_word']['d_word_acc']:+.3f} {cell['robust_minus_old_word']['d_word_acc_ci']}"
                     if wsub else "") + f" | {cell['sec_per_sentence']:.1f}s", flush=True)
    # real desk
    dk = AC.load_jsonl(CACHE / f"desk_{a.lm}.jsonl")
    if dk:
        res, obs, refs = AC.desk_inputs()
        old = json.loads((AC.OUT / "desk_qwen2.5-0.5b.json").read_text())["decoders"]
        Sbeam = np.array([AC.sent_stats(r, h) for r, h in zip(refs, old["beam"]["hyps"])])
        by = {}
        for r in dk:
            by.setdefault(r["cfg_key"], {})[r["j"]] = r["hyp"]
        desk = {"beam_0.448": AC.summarize(Sbeam)}
        full = {k: v for k, v in by.items() if len(v) == len(refs)}
        if "tuned" in full:
            S = np.array([AC.sent_stats(refs[j], full["tuned"][j]) for j in range(len(refs))])
            desk["robust_simdev_tuned"] = {**AC.summarize(S), "vs_beam": AC.paired(Sbeam, S),
                                           "hyps": [full["tuned"][j] for j in range(len(refs))]}
        if len(full) > 1:
            keys = sorted(full)
            E = {k: np.array([edit_distance(refs[j], full[k][j]) for j in range(len(refs))]) for k in keys}
            pick, hyps = [], []
            for j in range(len(refs)):
                tot = {k: (E[k].sum() - E[k][j]) for k in keys}
                kb = min(keys, key=lambda k: (tot[k], k != "tuned"))
                pick.append(kb)
                hyps.append(full[kb][j])
            S = np.array([AC.sent_stats(refs[j], hyps[j]) for j in range(len(refs))])
            desk["robust_lopo"] = {**AC.summarize(S), "vs_beam": AC.paired(Sbeam, S), "picked": pick, "hyps": hyps,
                                   "grid": keys}
            desk["grid_cer_in_sample"] = {k: float(E[k].sum() / sum(map(len, refs))) for k in keys}
        for k, v in desk.items():
            if isinstance(v, dict) and "cer" in v:
                vb = v.get("vs_beam", {})
                print(f"desk {k:<22} CER {v['cer']:.3f} [{v['cer_ci'][0]:.3f},{v['cer_ci'][1]:.3f}] words "
                      f"{v['word_acc']:.3f} " + (f"dCER {vb['d_cer']:+.3f} [{vb['d_cer_ci'][0]:+.3f},{vb['d_cer_ci'][1]:+.3f}]" if vb else ""))
        out["real_desk"] = desk
        out["real_desk"]["refs"] = refs
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"report_{a.lm}_{split}.json").write_text(json.dumps(out, indent=1))
    return 0


def cmd_plot(a) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = json.loads((OUT / f"report_{a.lm}_{a.split}.json").read_text())
    fig, axes = plt.subplots(1, len(NOISES), figsize=(16, 4.2), sharey=True)
    series = (("beam_same_sentences", "beam (current)", "#2a78d6"),
              ("old_word_same_sentences", "word-level + Qwen (old)", "#8a8984"),
              ("robust", "noisy-channel word decoder", "#1baf7a"))
    for ax, nz in zip(axes, NOISES):
        cells = sorted((c for c in rep["cells"] if c["noise"] == nz), key=lambda c: c["acc"])
        for key, lab, col in series:
            pts = [(c["acc"] * 100, c[key]["word_acc"], c[key]["word_acc_ci"]) for c in cells if key in c]
            if not pts:
                continue
            x = [q[0] for q in pts]
            ax.fill_between(x, [q[2][0] for q in pts], [q[2][1] for q in pts], color=col, alpha=0.12, lw=0)
            ax.plot(x, [q[1] for q in pts], color=col, lw=2, marker="o", ms=4, label=lab)
        ax.set_title(AC.NOISE_TITLE[nz], fontsize=9)
        ax.set_xlabel("raw key accuracy (top-1, %)")
        ax.grid(color="#e6e5e0", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("words correct (test)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "words_vs_keys.png", dpi=150, facecolor="#fcfcfb")
    return 0


# ============================================================================ CLI
def cmd_smoke(a) -> int:
    nlm = load_nlm(a.lm)
    tr = trie()
    T = AC.texts()
    nz = AC.noises()
    for name in a.noises.split(","):
        cfg = cfg_for(name, a.lm)
        for i in range(a.n):
            s = T["dev"][i]
            V, t = AC.sim_obs("dev", i, s, nz[name], ANCHOR)
            t0 = time.time()
            h = decode(V, tr, nlm, cfg, uni())
            print(f"{name} taps={len(V)} chars={len(s)} {time.time()-t0:.1f}s CER {edit_distance(s, h)/len(s):.3f}\n"
                  f"  ref {s!r}\n  hyp {h!r}", flush=True)
    return 0


COMMANDS = {"tune": cmd_tune, "sweep": cmd_sweep, "desk": cmd_desk, "report": cmd_report, "vocab": cmd_vocab,
            "smoke": cmd_smoke, "diag": cmd_diag, "tapfeat": cmd_tapfeat, "plot": cmd_plot}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.robustdec")
    ap.add_argument("cmd", choices=sorted(COMMANDS))
    ap.add_argument("--lm", default="qwen", choices=("qwen", "uni"))
    ap.add_argument("--noises", default="desk")
    ap.add_argument("--accs", default="0.7,0.8,0.9")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--only-tuned", action="store_true")
    ap.add_argument("--start", default="", help="JSON cfg overrides to resume tuning from")
    ap.add_argument("--steps", default="", help="fields to tune, 'field*' = refine around current")
    a = ap.parse_args(argv)
    return COMMANDS[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
