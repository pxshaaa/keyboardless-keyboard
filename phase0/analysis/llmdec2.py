"""LLM-guided open-vocabulary CTC decoding (MLX). Token-synchronous beam search over LLM tokens; every extension is scored
by exact CTC prefix log-prob (vectorised over time) + lam * LLM log-prob + word/lexicon/personal bonuses. Candidate tokens
= LLM top-N  U  tokens reachable in a char trie under the CTC prefix score (so out-of-lexicon words like 'codex' are
reachable). Runs in the llmdec MLX venv (numpy + mlx only).
  mini: PYTHONPATH=. .cache/llmdec/venv/bin/python -m phase0.analysis.llmdec2 run --llm 8b --sets kbd,old --grid A"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

OUT = Path(".cache/llmdec")
RES = Path("results/llmdec")
PERS = OUT / "personal"
REPOS = {"05b": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
         "8b": "mlx-community/Qwen3-8B-4bit", "14b": "mlx-community/Qwen3-14B-4bit"}
ALPHA = "abcdefghijklmnopqrstuvwxyz "
NINF = -np.inf


def norm(s: str) -> str:
    return " ".join("".join(c if "a" <= c <= "z" else (" " if c.isspace() or c == "-" else "") for c in s.lower()).split())


# ============================================================================ CTC prefix scorer
class CTCPrefix:
    """Exact CTC prefix/full log-probs for char strings over G posterior streams (mean over streams).
    'other' key mass joins blank; an optional leading and trailing space are absorbed."""

    def __init__(self, lps: list[np.ndarray]):
        T = min(len(l) for l in lps)
        L = np.stack([np.maximum(l[:T].astype(np.float64), -60.0) for l in lps])     # G,T,29
        self.X = L[:, :, 1:28]                                                          # G,T,27 (a-z, space)
        blank = np.logaddexp(L[:, :, 0], L[:, :, 28])
        self.Bc = np.concatenate([np.zeros((len(lps), 1)), np.cumsum(blank, 1)], 1)      # G,T+1
        self.G, self.T = len(lps), T
        self.n_ext = 0
        n0 = np.full_like(self.Bc, NINF)
        sp = self._ext(n0, self.Bc, "", [26])
        root_n, root_b = sp[0][:, :, 0], np.logaddexp(self.Bc, sp[1][:, :, 0])
        self.st = {"": (root_n, root_b, np.zeros(self.G), np.full(self.G, NINF))}

    def _ext(self, n_p, b_p, last, cs):
        """extend one parent by chars cs -> n,b (G,T+1,k), prefix (G,k), full (G,k)"""
        idx = [ALPHA.index(c) if isinstance(c, str) else c for c in cs]
        tot = np.logaddexp(n_p, b_p)
        phi = np.stack([b_p if (last and ALPHA[i] == last) else tot for i in idx], -1)      # G,T+1,k
        e = self.X[:, :, idx]                                                               # G,T,k
        E = np.concatenate([np.zeros((self.G, 1, len(idx))), np.cumsum(e, 1)], 1)         # G,T+1,k
        with np.errstate(invalid="ignore"):
            acc = np.logaddexp.accumulate(phi[:, :-1] - E[:, :-1], axis=1)
        n = np.full(phi.shape, NINF)
        n[:, 1:] = E[:, 1:] + acc
        B = self.Bc[:, :, None]
        with np.errstate(invalid="ignore"):
            acc2 = np.logaddexp.accumulate(n[:, :-1] - B[:, :-1], axis=1)
        b = np.full(phi.shape, NINF)
        b[:, 1:] = B[:, 1:] + acc2
        pre = np.logaddexp.reduce(phi[:, :-1] + e, axis=1)
        full = np.logaddexp(n[:, -1], b[:, -1])
        self.n_ext += len(idx)
        return n, b, pre, full

    def get(self, s: str):
        r = self.st.get(s)
        if r is not None:
            return r
        k = len(s)
        while k > 0 and s[:k] not in self.st:   # iterative: long pool texts would overflow the recursion limit
            k -= 1
        for j in range(k + 1, len(s) + 1):
            p = self.st[s[:j - 1]]
            n, b, pre, full = self._ext(p[0], p[1], s[j - 2] if j > 1 else " ", [s[j - 1]])
            self.st[s[:j]] = (n[:, :, 0], b[:, :, 0], pre[:, 0], full[:, 0])
        return self.st[s]

    def children(self, s: str, cs: list[str]):
        todo = [c for c in cs if s + c not in self.st]
        if todo:
            p = self.get(s)
            n, b, pre, full = self._ext(p[0], p[1], s[-1] if s else " ", todo)
            for k, c in enumerate(todo):
                self.st[s + c] = (n[:, :, k], b[:, :, k], pre[:, k], full[:, k])
        return [float(self.st[s + c][2].mean()) for c in cs]

    def prefix(self, s: str) -> float:
        return float(self.get(s)[2].mean()) if s else 0.0

    def final(self, s: str) -> float:
        a, b = self.get(s)[3], self.get(s + " ")[3]
        return float(np.logaddexp(a, b).mean())


# ============================================================================ vocab / LLM
class TokTable:
    PUNCT = {",", ".", "'", "!", "?", ":", ";", "-"}

    def __init__(self, tok, V: int, cache_f: Path | None = None):
        if cache_f and cache_f.exists():
            strs = json.loads(cache_f.read_text())
        else:
            strs = [tok.decode([i]) for i in range(min(V, len(getattr(tok, "_tokenizer", tok))))]
            if cache_f:
                cache_f.parent.mkdir(parents=True, exist_ok=True)
                cache_f.write_text(json.dumps(strs))
        self.chars: dict[int, str] = {}
        pat = re.compile(r" ?[A-Za-z]+(?:'[A-Za-z]+)?|'[A-Za-z]+")
        for i, s in enumerate(strs):
            if pat.fullmatch(s):
                self.chars[i] = s.lower().replace("'", "")
            elif s in self.PUNCT:
                self.chars[i] = ""
        self.valid = np.array(sorted(self.chars), dtype=np.int64)
        eos = set()
        for t in ("<|im_end|>", "<|endoftext|>", "\n"):
            ids = tok.encode(t, add_special_tokens=False) if hasattr(tok, "encode") else []
            if len(ids) == 1:
                eos.add(ids[0])
        for i in (getattr(tok, "eos_token_ids", None) or []):
            eos.add(int(i))
        self.eos = np.array(sorted(eos), dtype=np.int64)
        self.trie: dict = {}
        for i, c in self.chars.items():
            if not c:
                continue
            node = self.trie
            for ch in c:
                node = node.setdefault(ch, {})
            node.setdefault("$", []).append(i)


class CatCache:
    """minimal KV cache for one-token decoding steps (no mask needed for a single query token)"""

    def __init__(self, k, v, off):
        self.keys, self.values, self.offset = k, v, off

    def update_and_fetch(self, k, v):
        import mlx.core as mx
        self.keys = mx.concatenate([self.keys, k], axis=2)
        self.values = mx.concatenate([self.values, v], axis=2)
        self.offset += k.shape[2]
        return self.keys, self.values


class LLM:
    def __init__(self, key: str, bs: int = 4, base: str | None = None, adapter: str | None = None):
        from huggingface_hub import snapshot_download
        from mlx_lm import load
        from phase0.analysis import mlxsafe
        mlxsafe.cap()
        import mlx.core as mx
        self.mx = mx
        t0 = time.time()
        src = base or REPOS[key]
        path = src if Path(src).exists() else snapshot_download(src, local_files_only=True)
        self.model, self.tok = load(path, adapter_path=adapter) if adapter else load(path)
        self.key, self.bs = key, bs
        self.load_s = time.time() - t0
        lg = self.model(mx.array([[0]]))
        self.V = int(lg.shape[-1])
        self.tt = TokTable(self.tok, self.V, OUT / "toktab" / f"{key}{'_' + Path(src).name if base else ''}.json")
        self.chat = key != "raw"
        self.n_fwd, self.fwd_s = 0, 0.0

    def head(self, h):
        m = self.model
        if hasattr(m, "lm_head"):
            return m.lm_head(h)
        return m.model.embed_tokens.as_linear(h)

    def prompt_ids(self, user: str | None, raw_prefix: str | None) -> list[int]:
        if raw_prefix is not None:
            return self.tok.encode(raw_prefix)
        txt = self.tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True,
                                           tokenize=False, enable_thinking=False)
        return self.tok.encode(txt)

    def prefill(self, ids: list[int]):
        mx = self.mx
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(self.model)
        h = self.model.model(mx.array([ids]), cache=cache)
        lp = self.head(h[:, -1:, :]).astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        kv = [(c.keys[..., :len(ids), :], c.values[..., :len(ids), :]) for c in cache]
        mx.eval(lp, *[x for p in kv for x in p])
        return kv, len(ids), np.array(lp[0, 0])

    def extend(self, parents: list, toks: list[int]):
        """feed one token after each parent KV state -> (child KV states, next-token log-probs); chunks of <= bs"""
        mx = self.mx
        kvs, lps = [], []
        t0 = time.time()
        for k in range(0, len(parents), self.bs):
            par, tk = parents[k:k + self.bs], toks[k:k + self.bs]
            M = len(par)
            caches = [CatCache(mx.concatenate([p[l][0] for p in par], 0) if M > 1 else par[0][l][0],
                               mx.concatenate([p[l][1] for p in par], 0) if M > 1 else par[0][l][1],
                               int(par[0][l][0].shape[2])) for l in range(len(par[0]))]
            h = self.model.model(mx.array([[t] for t in tk]), cache=caches)
            lg = self.head(h[:, -1:, :]).astype(mx.float32)
            lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            mx.eval(lg, *[c.keys for c in caches], *[c.values for c in caches])
            a = np.array(lg[:, 0, :])
            for i in range(M):
                kvs.append([(c.keys[i:i + 1], c.values[i:i + 1]) if M > 1 else (c.keys, c.values) for c in caches])
                lps.append(a[i])
        self.n_fwd += len(parents)
        self.fwd_s += time.time() - t0
        return kvs, lps

    def score_texts(self, kv0, lp0: np.ndarray, seqs: list[list[int]]) -> list[float]:
        """sum log P(tokens | prompt) + max EOS log-prob at the end; right-padded batches of <= bs (causal => padding inert)"""
        mx = self.mx
        out = []
        t0 = time.time()
        eos = mx.array(self.tt.eos)
        for k in range(0, len(seqs), self.bs):
            ch = seqs[k:k + self.bs]
            M, L = len(ch), max(len(x) for x in ch)
            caches = [CatCache(mx.repeat(kk, M, axis=0), mx.repeat(vv, M, axis=0), int(kk.shape[2])) for kk, vv in kv0]
            arr = mx.array([x + [0] * (L - len(x)) for x in ch])
            h = self.model.model(arr, cache=caches)
            lg = self.head(h).astype(mx.float32)
            lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)                         # M,L,V
            nxt = mx.take_along_axis(lg[:, :-1, :], arr[:, 1:, None], axis=-1)[..., 0] if L > 1 else mx.zeros((M, 0))
            eos_lp = mx.max(lg[:, :, :][..., eos], axis=-1)                             # M,L
            mx.eval(nxt, eos_lp)
            nxt, eos_lp = np.array(nxt), np.array(eos_lp)
            for i, x in enumerate(ch):
                n = len(x)
                out.append(float(lp0[x[0]] + nxt[i, :n - 1].sum() + eos_lp[i, n - 1]))
            del caches, h, lg
        self.n_fwd += sum(len(x) for x in seqs)
        self.fwd_s += time.time() - t0
        return out

    def step(self, pc, P: int, seqs: list[tuple]) -> list[np.ndarray]:
        """next-token log-probs after prompt + each (equal-length) generated token sequence"""
        mx = self.mx
        from mlx_lm.models.cache import KVCache
        out = []
        t0 = time.time()
        for k in range(0, len(seqs), self.bs):
            ch = seqs[k:k + self.bs]
            M = len(ch)
            caches = []
            for c in pc:
                c2 = KVCache()
                c2.keys = mx.repeat(c.keys[..., :P, :], M, axis=0)
                c2.values = mx.repeat(c.values[..., :P, :], M, axis=0)
                c2.offset = P
                caches.append(c2)
            h = self.model.model(mx.array([list(s) for s in ch]), cache=caches)
            lg = self.head(h[:, -1:, :]).astype(mx.float32)
            lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            mx.eval(lg)
            a = np.array(lg[:, 0, :])
            out += [a[i] for i in range(M)]
            del caches, h, lg
        self.n_fwd += len(seqs)
        self.fwd_s += time.time() - t0
        return out


# ============================================================================ word-level knowledge
class Words:
    def __init__(self):
        lex = json.loads(Path(".cache/autocorrect/lexicon.json").read_text())
        B = set()
        bf = Path("models/vocab_blocklist.txt")
        if bf.exists():
            B = {l.strip() for l in bf.read_text().splitlines() if l.strip() and not l.startswith("#")}
        self.generic = {w: v for w, v in lex.items() if w.isalpha() and w.isascii() and w not in B}
        self.pers_vocab: dict = {}
        self.pers_bi: dict = {}
        self.pers_uni_n = 1.0
        if (PERS / "vocab.json").exists():
            d = json.loads((PERS / "vocab.json").read_text())
            self.pers_vocab = d["unigram"]
            self.pers_uni_n = float(sum(self.pers_vocab.values()))
            self.pers_bi = d.get("bigram", {})
            self.pers_ctx = d.get("ctx", {})
        self.gen_min = min(self.generic.values())

    def lm_mix(self, prev: str, w: str, alpha: float) -> float:
        """log( alpha * P_personal(w | prev) + (1-alpha) * P_generic(w) ) with abs-discount personal bigram"""
        pg = math.exp(self.generic.get(w, self.gen_min - 2.0))
        cu = self.pers_vocab.get(w, 0) + 0.1
        pu = cu / (self.pers_uni_n + 0.1 * 50000)
        c = self.pers_ctx.get(prev) if prev else None
        if c:
            cb = self.pers_bi.get(prev + " " + w, 0)
            pp = max(cb - 0.75, 0) / c[0] + 0.75 * c[1] / c[0] * pu
        else:
            pp = pu
        return math.log(alpha * pp + (1 - alpha) * pg)


# ============================================================================ decoder
@dataclass(frozen=True)
class DCfg:
    prompt: str = "chat"          # raw | chat | guess
    ctx: int = 0                  # previous decoded lines in prompt
    profile: bool = False
    lam: float = 1.0
    wb: float = 0.0               # per completed word
    cb: float = 0.0               # per typed character
    stop_margin: float = 40.0
    pool: int = 0                 # 1: also score the generate-and-verify pool (char n-best, greedy, qwen05, stored LLM rewrites)
    pool_llm: str = ""            # whose stored rewrites to add ('' = same key as the decoding LLM)
    mu_lex: float = 0.0           # completed word in generic lexicon
    mu_pers: float = 0.0          # completed word in personal vocab (count>=2)
    nu_ng: float = 0.0            # weight of personal+generic word LM log-prob per completed word
    ng_alpha: float = 0.5
    beam: int = 6
    topn: int = 24
    margin: float = 6.0
    maxnodes: int = 160
    groups: str = "zs"            # '+'-joined posterior groups (mean CTC score)

    def name(self) -> str:
        d = asdict(self)
        return ",".join(f"{k}={v}" for k, v in d.items() if DCfg.__dataclass_fields__[k].default != v) or "default"


@dataclass
class Hyp:
    toks: tuple
    chars: str
    llm: float
    bonus: float
    score: float


INTRO = "Someone typed a short line of text"


def user_prompt(cfg: DCfg, prev: list[str], guesses: list[str], profile: str) -> str:
    parts = []
    if cfg.profile and profile:
        parts.append(f"About the writer: {profile}")
    if cfg.ctx and prev:
        parts.append("Lines they typed just before, in order (may contain recognition errors):\n" + "\n".join(prev[-cfg.ctx:]))
    if cfg.prompt == "guess" and guesses:
        parts.append("A noisy recognizer read the new line as (letters may be wrong, missing or extra):\n"
                     + "\n".join(guesses[:4]))
    parts.append(f"{INTRO}. Write the line they typed. Reply with the line only, lowercase, no punctuation.")
    return "\n\n".join(parts)


def word_terms(cfg: DCfg, W: Words, prev_word: str, word: str) -> float:
    s = cfg.wb
    if cfg.mu_lex and word in W.generic:
        s += cfg.mu_lex
    if cfg.mu_pers and W.pers_vocab.get(word, 0) >= 2:
        s += cfg.mu_pers
    if cfg.nu_ng:
        s += cfg.nu_ng * W.lm_mix(prev_word, word, cfg.ng_alpha)
    return s


def completed_words(old: str, new: str):
    """words completed when extending old -> new (a space typed after a letter)"""
    out = []
    for k in range(len(old), len(new)):
        if new[k] == " " and k > 0 and new[k - 1] != " ":
            ws = new[:k].split()
            out.append((ws[-2] if len(ws) > 1 else "", ws[-1]))
    return out


def decode(llm: LLM, S: CTCPrefix, W: Words, cfg: DCfg, prompt_ids: list[int], memo: dict, max_steps: int,
           pool: list[str] | None = None):
    tt = llm.tt
    kv0, P, lp0 = llm.prefill(prompt_ids)
    beams = [Hyp((), "", 0.0, 0.0, 0.0)]
    state = {(): (kv0, lp0)}
    finished: list[tuple[float, str]] = []
    stale = 0
    for step in range(max_steps):
        cands: dict[str, Hyp] = {}
        for h in beams:
            lpv = state[h.toks][1]
            if h.chars and h.chars[-1] != " ":
                ws = h.chars.split()
                wt = word_terms(cfg, W, ws[-2] if len(ws) > 1 else "", ws[-1])
                fs = S.final(h.chars) + cfg.lam * (h.llm + float(lpv[tt.eos].max())) + h.bonus + wt
                finished.append((fs, h.chars))
            base = S.prefix(h.chars)
            ids = set(tt.valid[np.argpartition(-lpv[tt.valid], cfg.topn)[:cfg.topn]].tolist())
            # CTC-guided trie walk (best-first)
            start = h.chars
            heap = [(-base, "", tt.trie)]
            nodes = 0
            while heap and nodes < cfg.maxnodes:
                negs, u, node = heapq.heappop(heap)
                nodes += 1
                if "$" in node and u:
                    ids.update(node["$"])
                kids = [c for c in node if c != "$"]
                if not kids:
                    continue
                cur = start + (u[1:] if (not start and u.startswith(" ")) else u)
                ok = [c for c in kids if not (c == " " and (not cur or cur.endswith(" ")))]
                if not ok:
                    continue
                scs = S.children(cur, ok)
                for c, sc in zip(ok, scs):
                    if sc >= base - cfg.margin:
                        heapq.heappush(heap, (-sc, u + c, node[c]))
            for i in ids:
                c = tt.chars[i]
                if not h.chars and c.startswith(" "):
                    c = c[1:]
                if c.startswith(" ") and h.chars.endswith(" "):
                    continue
                new = h.chars + c
                if not new and c == "" and not h.chars:
                    continue
                pre = S.prefix(new)
                if not np.isfinite(pre):
                    continue
                bonus = h.bonus + cfg.cb * len(c) + sum(word_terms(cfg, W, a, b) for a, b in completed_words(h.chars, new))
                llm_v = h.llm + float(lpv[i])
                sc = pre + cfg.lam * llm_v + bonus
                o = cands.get(new)
                if o is None or sc > o.score:
                    cands[new] = Hyp(h.toks + (i,), new, llm_v, bonus, sc)
        prev_beams = beams
        beams = sorted(cands.values(), key=lambda x: -x.score)[:cfg.beam]
        if not beams:
            beams = prev_beams
            break
        kvs, lps = llm.extend([state[h.toks[:-1]][0] for h in beams], [h.toks[-1] for h in beams])
        state = {h.toks: (kv, lp) for h, kv, lp in zip(beams, kvs, lps)}
        if finished and max(finished)[0] > beams[0].score + cfg.stop_margin:
            stale += 1
            if stale >= 3:
                break
        else:
            stale = 0
    for h in beams:   # finalise whatever is still active (step limit)
        if h.chars and h.chars[-1] != " ":
            ws = h.chars.split()
            wt = word_terms(cfg, W, ws[-2] if len(ws) > 1 else "", ws[-1])
            e = float(state[h.toks][1][tt.eos].max()) if h.toks in state else -5.0
            finished.append((S.final(h.chars) + cfg.lam * (h.llm + e) + h.bonus + wt, h.chars))
    src = {c: "beam" for _, c in finished}
    if pool:
        texts = [t for t in dict.fromkeys(norm(x) for x in pool) if t and len(t) <= 200 and np.isfinite(S.final(t))]
        if texts:
            ids = [llm.tok.encode(t) for t in texts]
            keep = [(t, x) for t, x in zip(texts, ids) if x]
            lls = llm.score_texts(kv0, lp0, [x for _, x in keep])
            for (t, _), ll in zip(keep, lls):
                ws = t.split()
                wt = sum(word_terms(cfg, W, ws[k - 1] if k else "", w) for k, w in enumerate(ws))
                finished.append((S.final(t) + cfg.lam * ll + cfg.cb * len(t) + wt, t))
                src.setdefault(t, "pool")
    if not finished:
        return (beams[0].chars if beams else ""), {"steps": step + 1}
    fs, best = max(finished)
    ranked = sorted({c: s for s, c in sorted(finished)}.items(), key=lambda kv: -kv[1])[:8]
    return " ".join(best.split()), {"steps": step + 1, "score": fs, "nbest": ranked, "src": src.get(best, "beam")}


# ============================================================================ sets
def load_items(setname: str):
    d = json.loads((OUT / "sets" / f"{setname}.json").read_text())
    z = np.load(OUT / "sets" / f"{setname}_lp.npz")
    extra = OUT / "sets" / f"{setname}_extra_lp.npz"
    ze = np.load(extra) if extra.exists() else None
    items = []
    for it in d["items"]:
        lps = {g: z[f"{it['id']}__{g}"] for g in it["c"]}
        if ze is not None:
            for k in ze.files:
                if k.startswith(it["id"] + "__"):
                    lps[k.split("__", 1)[1]] = ze[k]
        fold = it["id"].rsplit("_", 1)[0] if setname == "kbd" else "sess"
        items.append({"id": it["id"], "ref": it.get("ref"), "c": it["c"], "lps": lps, "fold": fold})
    return d, items


def guesses_of(it, groups):
    g0 = groups[0] if groups[0] in it["c"] else next(iter(it["c"]))
    c = it["c"][g0]
    return list(dict.fromkeys([x for x in [c["qwen"]] + [t for t, _ in c["char"][:4]] if x]))


_GV: dict = {}


def gv_pool(setname: str, it, groups, key: str) -> list[str]:
    out = []
    for g in groups:
        if g in it["c"]:
            c = it["c"][g]
            out += [t for t, _ in c["char"]] + [c["greedy"], c["qwen"]]
    f = OUT / "llm" / key / f"{setname}.json"
    if (key, setname) not in _GV:
        _GV[(key, setname)] = json.loads(f.read_text())["items"] if f.exists() else {}
    r = _GV[(key, setname)].get(it["id"])
    if r:
        gk = groups[0] if len(groups) == 1 else "ens"
        for p, v in r["gens"].items():
            out += v.get(gk, v.get(groups[0], []))
    return out


def run_set(llm: LLM, W: Words, setname: str, cfgs: list[DCfg], out_f: Path, profile: str):
    d, items = load_items(setname)
    done = set()
    if out_f.exists():
        for l in out_f.read_text().splitlines():
            r = json.loads(l)
            done.add((r["cfg"], r["set"], r["id"]))
    hist: dict = {}
    if out_f.exists():
        for l in out_f.read_text().splitlines():
            r = json.loads(l)
            hist.setdefault((r["cfg"], r["set"], r.get("fold")), []).append(r["hyp"])
    with out_f.open("a") as fh:
        for it in items:
            memo: dict = {}
            for cfg in cfgs:
                nm = cfg.name()
                if (nm, setname, it["id"]) in done:
                    continue
                groups = cfg.groups.split("+")
                if any(g not in it["lps"] for g in groups):
                    continue
                t0 = time.time()
                S = CTCPrefix([it["lps"][g] for g in groups])
                prev = hist.get((nm, setname, it["fold"]), [])
                if cfg.prompt == "raw":
                    ids = llm.prompt_ids(None, "\n".join(prev[-cfg.ctx:] if cfg.ctx else []) + "\n")
                else:
                    ids = llm.prompt_ids(user_prompt(cfg, prev, guesses_of(it, groups), profile), None)
                greedy_len = len(it["c"][groups[0]]["greedy"]) if groups[0] in it["c"] else 30
                max_steps = max(10, greedy_len + 12)
                f0, s0 = llm.n_fwd, llm.fwd_s
                pool = gv_pool(setname, it, groups, cfg.pool_llm or llm.key) if cfg.pool else None
                hyp, info = decode(llm, S, W, cfg, ids, memo, max_steps, pool)
                rec = {"cfg": nm, "set": setname, "id": it["id"], "fold": it["fold"], "hyp": hyp, "ref": it.get("ref"),
                       "secs": time.time() - t0, "llm_s": llm.fwd_s - s0, "n_fwd": llm.n_fwd - f0, "n_ext": S.n_ext,
                       "prompt_tokens": len(ids), **{k: v for k, v in info.items()}}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                hist.setdefault((nm, setname, it["fold"]), []).append(hyp)
                print(f"{setname} {it['id']} [{nm}] {rec['secs']:.1f}s llm {rec['llm_s']:.1f}s | {hyp} || {it.get('ref')}",
                      flush=True)
            del memo


GRIDS = {
    "smoke": [DCfg()],
    "A": [DCfg(prompt=p, lam=l) for p in ("raw", "chat", "guess") for l in (0.5, 1.0)],
}


def parse_cfgs(spec: str) -> list[DCfg]:
    if spec in GRIDS:
        return GRIDS[spec]
    out = []
    for part in spec.split(";"):
        kw = {}
        if part.strip() == "default":
            out.append(DCfg())
            continue
        for kv in filter(None, part.split(",")):
            k, v = kv.split("=")
            ft = DCfg.__dataclass_fields__[k].type
            kw[k] = (v == "True") if ft in ("bool", bool) else int(v) if ft in ("int", int) else float(v) if ft in ("float", float) else v
        out.append(DCfg(**kw))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--llm", required=True, choices=tuple(REPOS))
    r.add_argument("--sets", default="kbd,old")
    r.add_argument("--grid", default="smoke", help="grid name or 'k=v,k=v;k=v' configs")
    r.add_argument("--out", default=None)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--base", default=None, help="override model repo/path (e.g. a personal LoRA base)")
    r.add_argument("--adapter", default=None, help="mlx_lm LoRA adapter directory")
    a = ap.parse_args(argv)
    if any(x not in ("kbd", "old") for x in a.sets.split(",")) and not (RES / "frozen_config_v2.json").exists():
        raise SystemExit("freeze the config before touching a new session")
    llm = LLM(a.llm, base=a.base, adapter=a.adapter)
    W = Words()
    prof_f = PERS / "profile_summary.txt"
    profile = prof_f.read_text().strip() if prof_f.exists() else ""
    cfgs = parse_cfgs(a.grid)
    out_f = Path(a.out or (OUT / "dec2" / f"{a.llm}.jsonl"))
    out_f.parent.mkdir(parents=True, exist_ok=True)
    print(f"[llmdec2] {a.llm} load {llm.load_s:.1f}s V={llm.V} valid={len(llm.tt.valid)} eos={llm.tt.eos.tolist()} "
          f"cfgs={[c.name() for c in cfgs]}", flush=True)
    for s in a.sets.split(","):
        run_set(llm, W, s, cfgs, out_f, profile)
    from phase0.analysis import mlxsafe
    print(f"[llmdec2] done; peak MLX memory {mlxsafe.peak_gb():.2f} GB; fwd {llm.n_fwd} in {llm.fwd_s:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
