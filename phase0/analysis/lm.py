"""Stronger LMs and decoding for the tap decoder; the keyprobs observation model is reused as-is.
Nothing here is fitted, tuned or selected on phase0/phrases.txt; `contamination` proves it."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import tarfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from phase0.analysis.decode import (
    ALPHABET,
    A_INDEX,
    LM_ORDER,
    NA,
    CharLM,
    KeyProbsSpatial,
    UniformSpatial,
    WordLM,
    Weights,
    desk_segments,
    edit_distance,
    fetch_corpus,
    kbd_segments,
    read_jsonl,
)

CORPUS_DIR = Path("data/lm")
MODERN_DIR = CORPUS_DIR / "modern"
CACHE_DIR = Path(".cache/lm")
PHRASES = Path("phase0/phrases.txt")
SESSIONS = Path("data/sessions")
BOS = "<s>"


# --------------------------------------------------------------------------- corpora
def normalize(text: str) -> str:
    return re.sub(r" +", " ", re.sub(r"[^a-z ]+", " ", text.lower()))


def _cached(name: str, build) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / f"{name}.txt"
    if not f.exists():
        f.write_text(build(), encoding="utf-8")
    return f.read_text(encoding="utf-8")


def corpus_gutenberg() -> str:
    """The incumbent: 10 Project Gutenberg novels, ~5.5M chars of Victorian prose."""
    return _cached("gutenberg", fetch_corpus)


def _join(sents: list[str]) -> str:
    return re.sub(r" +", " ", " ".join(s.strip() for s in sents if s.strip()))


def decontaminate(sents: list[str], name: str) -> list[str]:
    """Drop any source sentence containing a test phrase; the count is printed, never hidden."""
    bad = test_phrases()
    keep, dropped = [], 0
    for s in sents:
        if any(b in s for b in bad):
            dropped += 1
            continue
        keep.append(s)
    print(f"[decontaminate:{name}] dropped {dropped} of {len(sents)} sentences containing a test phrase")
    return keep


def corpus_tatoeba() -> str:
    """Modern conversational English sentences (Tatoeba eng export, CC-BY 2.0 FR)."""

    def build() -> str:
        src = _find("eng_sentences.tsv")
        out = []
        for line in src.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = line.split("\t")
            if len(p) >= 3:
                out.append(normalize(p[2]))
        return _join(decontaminate(out, "tatoeba"))

    return _cached("tatoeba", build)


def corpus_enron() -> str:
    """Business email register: bodies of the Enron mail corpus, headers and quotes stripped."""

    def build() -> str:
        src = _find("enron_mail_20150507.tar.gz", "enron.tar.gz")
        import contextlib
        out, n = [], 0
        with contextlib.suppress(tarfile.TarError, EOFError, OSError), \
                tarfile.open(src, "r|gz") as tf:  # truncated archive still yields a usable prefix
            for m in tf:
                if not m.isfile():
                    continue
                try:
                    fh = tf.extractfile(m)
                except (tarfile.TarError, EOFError, OSError):
                    break
                if fh is None:
                    continue
                body = _email_body(fh.read().decode("utf-8", errors="ignore"))
                if body:
                    out += [normalize(x) for x in re.split(r"[.!?\n]+", body)]
                    n += 1
                if n >= 250_000:
                    break
        return _join(decontaminate(out, "enron"))



    return _cached("enron", build)


_QUOTE = re.compile(r"^\s*(>|\||-{3,}|_{3,}|from:|to:|cc:|sent:|subject:|\*{3,})", re.I)
_BOILER = re.compile(r"(forwarded by|original message|this e-?mail .{0,40}confiden)", re.I)


def _email_body(raw: str) -> str:
    _, _, body = raw.partition("\n\n")
    lines = []
    for ln in body.splitlines():
        if _QUOTE.match(ln) or _BOILER.search(ln):
            break
        lines.append(ln)
    return " ".join(lines)


def _find(*names: str) -> Path:
    for d in (MODERN_DIR, CORPUS_DIR, Path("/tmp/lmwork/corpus")):
        for n in names:
            if (d / n).exists():
                return d / n
    raise SystemExit(f"missing corpus file {names[0]}; put it in {MODERN_DIR}")


CORPORA = {"gutenberg": corpus_gutenberg, "tatoeba": corpus_tatoeba, "enron": corpus_enron}


def split_corpus(text: str, dev_frac: float = 0.02, seed: int = 0) -> tuple[str, str]:
    """Chunk-wise train/dev split so interpolation weights are tuned on held-out text."""
    chunks = [text[i:i + 20_000] for i in range(0, len(text), 20_000)]
    rng = random.Random(seed)
    idx = list(range(len(chunks)))
    rng.shuffle(idx)
    k = max(1, int(len(chunks) * dev_frac))
    dev = {i for i in idx[:k]}
    return ("".join(c for i, c in enumerate(chunks) if i not in dev),
            "".join(c for i, c in enumerate(chunks) if i in dev))


# -------------------------------------------------------------------- contamination
def test_phrases() -> list[str]:
    return [normalize(l).strip() for l in PHRASES.read_text().splitlines() if l.strip()]


def contamination(text: str, phrases: list[str], ngram: int = 5) -> dict:
    """Exact-phrase hits plus word-n-gram hits: how much of the test set is in this corpus."""
    hits = {p: text.count(" " + p + " ") for p in phrases}
    grams = set()
    for i in range(0, len(text), 1_000_000):  # n-gram set over the whole corpus, chunked
        ws = text[max(0, i - 100):i + 1_000_000].split()
        grams |= {" ".join(ws[j:j + ngram]) for j in range(len(ws) - ngram + 1)}
    covered = {}
    for p in phrases:
        ws = p.split()
        g = [" ".join(ws[j:j + ngram]) for j in range(len(ws) - ngram + 1)]
        covered[p] = (sum(x in grams for x in g), len(g))
    return {"exact": hits, "ngram": covered, "n": ngram}


def print_contamination(name: str, rep: dict):
    ex = sum(rep["exact"].values())
    hit_p = sum(1 for v in rep["exact"].values() if v)
    num = sum(a for a, _ in rep["ngram"].values())
    den = sum(b for _, b in rep["ngram"].values())
    print(f"  {name:11s} exact-phrase hits={ex} (in {hit_p}/{len(rep['exact'])} phrases)  "
          f"{rep['n']}-gram coverage={num}/{den} = {num/max(1,den):.3f}")
    for p, v in rep["exact"].items():
        if v:
            print(f"      !! {v}x {p!r}")


# ----------------------------------------------------------------------- char models
class MixCharLM:
    """Interpolates char n-grams at the probability level; mixing counts instead would let the
    larger corpus swallow the smaller one context by context."""

    def __init__(self, models: list[CharLM], weights: list[float]):
        w = np.array(weights, float)
        self.models, self.w = models, w / w.sum()
        self.order = max(m.order for m in models)
        self._cache: dict[str, np.ndarray] = {}

    def logprobs(self, ctx: str) -> np.ndarray:
        hit = self._cache.get(ctx)
        if hit is not None:
            return hit
        p = np.zeros(NA)
        for m, w in zip(self.models, self.w):
            p += w * np.exp(m.logprobs(ctx))
        out = np.log(np.maximum(p, 1e-300))
        if len(self._cache) < 400_000:
            self._cache[ctx] = out
        return out


def char_logppl(lm, text: str, limit: int = 200_000) -> float:
    """Mean -log2 P(char) under the model; the number the interpolation weights minimise."""
    t = text[:limit]
    tot = 0.0
    ctx = " "
    for ch in t:
        c = A_INDEX.get(ch)
        if c is None:
            continue
        tot += lm.logprobs(ctx[-(LM_ORDER - 1):])[c]
        ctx += ch
    return -tot / max(1, len(t)) / math.log(2)


def tune_mix(models: list[CharLM], dev: str, grid: int = 11) -> tuple[list[float], float]:
    """Grid-search the mixture weight on held-out *text* (never on the test phrases)."""
    if len(models) == 1:
        return [1.0], char_logppl(models[0], dev)
    best, bw = math.inf, None
    for i in range(grid):
        a = i / (grid - 1)
        v = char_logppl(MixCharLM(models, [1 - a, a]), dev)
        if v < best:
            best, bw = v, [1 - a, a]
    return bw, best


# ----------------------------------------------------------------------- word models
class WordBigramLM:
    """Word bigram with stupid backoff + the incumbent prefix trie; the unigram trie only says
    "is this a word", which is where the fluent-filler outputs come from."""

    def __init__(self, uni: dict[str, float], bi: dict[str, dict[str, float]],
                 prefixes: set[str], oov: float, alpha: float = 0.4):
        self.uni, self.bi, self.prefixes, self.oov, self.alpha = uni, bi, prefixes, oov, alpha

    @classmethod
    def train(cls, text: str, top: int = 60_000, min_bi: int = 2):
        words = text.split()
        cnt = Counter(words)
        vocab = {w for w, _ in cnt.most_common(top)}
        tot = sum(cnt[w] for w in vocab)
        uni = {w: math.log(cnt[w] / tot) for w in vocab}
        pair = Counter()
        prev = BOS
        for w in words:
            if w in vocab:
                pair[(prev, w)] += 1
                prev = w
            else:
                prev = BOS
        ctx_tot = Counter()
        for (a, _), n in pair.items():
            ctx_tot[a] += n
        bi: dict[str, dict[str, float]] = defaultdict(dict)
        for (a, b), n in pair.items():
            if n >= min_bi:
                bi[a][b] = math.log(n / ctx_tot[a])
        prefixes = set()
        for w in vocab:
            for i in range(1, len(w) + 1):
                prefixes.add(w[:i])
        return cls(uni, dict(bi), prefixes, math.log(0.5 / tot))

    def word_score(self, w: str, prev: str = BOS) -> float:
        row = self.bi.get(prev)
        if row is not None and w in row:
            return row[w]
        return math.log(self.alpha) + self.uni.get(w, self.oov)

    def is_prefix(self, w: str) -> bool:
        return w == "" or w in self.prefixes


class UnigramAdapter:
    """Wraps the incumbent WordLM so the beam can treat both word models identically."""

    def __init__(self, wl: WordLM):
        self.wl = wl

    def word_score(self, w: str, prev: str = BOS) -> float:
        return self.wl.word_score(w)

    def is_prefix(self, w: str) -> bool:
        return self.wl.is_prefix(w)


# -------------------------------------------------------------------------- decoding
@dataclass(frozen=True)
class DecWeights:
    obs: float = 1.0
    char: float = 1.0
    word: float = 0.6
    prefix_penalty: float = 2.5
    insertion: float = -7.0
    deletion: float = -9.0
    max_deletions: int = 2
    length_bonus: float = 0.0  # per-character reward, counteracts the beam's bias to short output

    @classmethod
    def from_weights(cls, w: Weights) -> "DecWeights":
        return cls(w.obs, w.char, w.word, w.prefix_penalty, w.insertion, w.deletion,
                   w.max_deletions, 0.0)


def _ctx(text: str) -> str:
    return (" " + text)[-(LM_ORDER - 1):]


def _last(text: str) -> str:
    return text.rsplit(" ", 1)[-1]


def _prev_word(text: str) -> str:
    ws = text.split(" ")[:-1]
    ws = [w for w in ws if w]
    return ws[-1] if ws else BOS


def _extra(text: str, ch: str, word_lm, w: DecWeights) -> float:
    if word_lm is None:
        return 0.0
    if ch == " ":
        done = _last(text)
        return w.word * word_lm.word_score(done, _prev_word(text)) if done else -abs(w.deletion)
    return 0.0 if word_lm.is_prefix(_last(text) + ch) else -w.prefix_penalty


def _push(d: dict, key: str, score: float):
    if score > d.get(key, -math.inf):
        d[key] = score


def _topk(d: dict[str, float], beam: int) -> dict[str, float]:
    if len(d) <= beam:
        return d
    return dict(sorted(d.items(), key=lambda kv: -kv[1])[:beam])


def _close_deletions(hyps, char_lm, word_lm, w: DecWeights, beam: int):
    out = dict(hyps)
    frontier = hyps
    for _ in range(w.max_deletions):
        nxt: dict[str, float] = {}
        for text, sc in frontier.items():
            lm = char_lm.logprobs(_ctx(text)) if char_lm else np.zeros(NA)
            for c in range(NA):
                _push(nxt, text + ALPHABET[c],
                      sc + w.deletion + w.char * lm[c] + _extra(text, ALPHABET[c], word_lm, w)
                      + w.length_bonus)
        frontier = _topk(nxt, beam)
        for t, s in frontier.items():
            _push(out, t, s)
    return _topk(out, beam)


def beam_nbest(obs_logp: np.ndarray, char_lm, word_lm, w: DecWeights = DecWeights(),
               beam: int = 30, nbest: int = 1) -> list[tuple[str, float]]:
    """Same tap-level lattice as decode.beam_decode, plus word-bigram context, a length
    bonus and an n-best list for downstream rescoring."""
    hyps = {"": 0.0}
    for i in range(len(obs_logp) + 1):
        hyps = _close_deletions(hyps, char_lm, word_lm, w, beam)
        if i == len(obs_logp):
            break
        nxt: dict[str, float] = {}
        row = obs_logp[i]
        for text, sc in hyps.items():
            _push(nxt, text, sc + w.insertion)
            lm = char_lm.logprobs(_ctx(text)) if char_lm else np.zeros(NA)
            for c in range(NA):
                _push(nxt, text + ALPHABET[c],
                      sc + w.obs * row[c] + w.char * lm[c] + _extra(text, ALPHABET[c], word_lm, w)
                      + w.length_bonus)
        hyps = _topk(nxt, beam)
    final = []
    for t, s in hyps.items():
        tail = _last(t)
        if word_lm and tail:
            s += w.word * word_lm.word_score(tail, _prev_word(t))
        final.append((t.strip(), s))
    final.sort(key=lambda kv: -kv[1])
    out, seen = [], set()
    for t, s in final:  # strip() can collide, keep the best score per surface form
        if t not in seen:
            seen.add(t)
            out.append((t, s))
    return out[:nbest]


# ---------------------------------------------------------------------- neural rescore
class NeuralLM:
    """GPT-2 / DistilGPT-2 as an n-best rescorer. Too slow in-beam on CPU, fine on 30 strings."""

    def __init__(self, name: str = "distilgpt2"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name).eval()
        self.name = name

    def logp(self, texts: list[str]) -> list[float]:
        """Total log P(text) in nats, conditioned on a neutral newline prefix."""
        torch = self.torch
        out = []
        with torch.no_grad():
            for t in texts:
                ids = self.tok("\n" + t, return_tensors="pt").input_ids
                if ids.shape[1] < 2:
                    out.append(0.0)
                    continue
                lg = torch.log_softmax(self.model(ids).logits[0, :-1], -1)
                out.append(float(lg[torch.arange(ids.shape[1] - 1), ids[0, 1:]].sum()))
        return out


# ------------------------------------------------------------------------ evaluation
def load_segments(session: str, taps_file: str | None = None, control: bool = False,
                  taps_root: Path | None = None) -> list[tuple[str, list[dict]]]:
    sd = SESSIONS / session
    root = (taps_root / session) if taps_root else sd
    taps = read_jsonl(root / (taps_file or "taps.jsonl"))
    return kbd_segments(sd, taps) if control else desk_segments(sd, taps)


def obs_matrices(session: str, segs, spatial: str) -> list[np.ndarray]:
    """Precompute the observation log-probs once; every LM/decoder config reuses them."""
    sp = UniformSpatial() if spatial == "uniform" else KeyProbsSpatial()
    return [sp.logp(SESSIONS / session, taps) if taps else np.zeros((0, NA)) for _, taps in segs]


def score_rows(rows: list[tuple[str, str]]) -> tuple[float, float]:
    ce = sum(edit_distance(r, h) for r, h in rows)
    cn = sum(len(r) for r in rows)
    we = sum(edit_distance(r.split(), h.split()) for r, h in rows)
    wn = sum(len(r.split()) for r in rows)
    return ce / max(1, cn), we / max(1, wn)


def bootstrap_ci(rows, B: int = 2000, seed: int = 0, metric: str = "cer") -> tuple[float, float, float]:
    """Segment-level bootstrap of the corpus-level rate (edits and chars resampled together)."""
    if metric == "cer":
        pairs = [(edit_distance(r, h), len(r)) for r, h in rows]
    else:
        pairs = [(edit_distance(r.split(), h.split()), len(r.split())) for r, h in rows]
    e = np.array([p[0] for p in pairs], float)
    n = np.array([max(1, p[1]) for p in pairs], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(e), size=(B, len(e)))
    boot = e[idx].sum(1) / n[idx].sum(1)
    return float(e.sum() / n.sum()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_delta_ci(rows_a, rows_b, B: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """CI on CER(b) - CER(a) with the same resampled segments on both sides."""
    ea = np.array([edit_distance(r, h) for r, h in rows_a], float)
    eb = np.array([edit_distance(r, h) for r, h in rows_b], float)
    n = np.array([max(1, len(r)) for r, _ in rows_a], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ea), size=(B, len(ea)))
    d = eb[idx].sum(1) / n[idx].sum(1) - ea[idx].sum(1) / n[idx].sum(1)
    return float(eb.sum() / n.sum() - ea.sum() / n.sum()), \
        float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def decode_segments(segs, obs, char_lm, word_lm, w: DecWeights, beam: int,
                    nbest: int = 1) -> tuple[list[tuple[str, str]], list[list[tuple[str, float]]]]:
    rows, lists = [], []
    for (ref, taps), o in zip(segs, obs):
        if not taps:
            rows.append((ref, ""))
            lists.append([("", 0.0)])
            continue
        nb = beam_nbest(o, char_lm, word_lm, w, beam, nbest)
        rows.append((ref, nb[0][0]))
        lists.append(nb)
    return rows, lists


# ------------------------------------------------------------------------- model zoo
def build_models(spec: str, dev_ppl: bool = False, cache: dict | None = None):
    """spec: 'gutenberg', 'tatoeba', 'enron', or 'a+b' for a tuned interpolation."""
    cache = cache if cache is not None else {}

    def one(name: str):
        if name not in cache:
            base, _, cap = name.partition("@")
            text = CORPORA[base]()
            if cap:  # size-cap so "modern beats Victorian" is not just "bigger corpus wins"
                text = text[:int(float(cap) * 1e6)]
            tr, dv = split_corpus(text)
            cache[name] = (CharLM.train(tr, LM_ORDER), tr, dv)
        return cache[name]

    parts = spec.split("+")
    built = [one(p) for p in parts]
    dev = "".join(b[2] for b in built)
    models = [b[0] for b in built]
    if len(models) == 1:
        clm, weights = models[0], [1.0]
    else:
        weights, _ = tune_mix(models, dev)
        clm = MixCharLM(models, weights)
    text_all = " ".join(b[1] for b in built)
    return clm, text_all, weights, dev


# ------------------------------------------------------------------------------- CLI
def cmd_contamination(a) -> int:
    ph = test_phrases()
    print(f"contamination check against {len(ph)} test phrases in {PHRASES}")
    for name in a.corpora:
        t = CORPORA[name]()
        print(f"  {name}: {len(t):,} chars")
        print_contamination(name, contamination(t, ph))
    return 0


def cmd_ppl(a) -> int:
    """Held-out char perplexity of each variant on modern dev text (no test phrases)."""
    _, dev_modern = split_corpus(CORPORA[a.dev]())
    cache: dict = {}
    print(f"dev = held-out {a.dev} text ({len(dev_modern):,} chars)")
    for spec in a.specs:
        clm, _, wts, _ = build_models(spec, cache=cache)
        print(f"  {spec:22s} weights={[round(x,2) for x in wts]} "
              f"dev bits/char={char_logppl(clm, dev_modern):.3f}")
    return 0


def _eval_one(session, taps_file, control, spatial, clm, wlm, w, beam, nbest, taps_root):
    segs = load_segments(session, taps_file, control, taps_root)
    obs = obs_matrices(session, segs, spatial)
    return segs, decode_segments(segs, obs, clm, wlm, w, beam, nbest)


def cmd_eval(a) -> int:
    taps_root = Path(a.taps_root) if a.taps_root else None
    cache: dict = {}
    base_rows: dict[str, list] = {}
    for spec in a.specs:
        clm, text, wts, _ = build_models(spec, cache=cache)
        wlm = None
        if a.word == "bigram":
            wlm = WordBigramLM.train(text)
        elif a.word == "unigram":
            wlm = UnigramAdapter(WordLM.train(text))
        w = DecWeights(obs=a.obs_weight, char=a.char_weight, word=a.word_weight,
                       insertion=a.insertion, deletion=a.deletion,
                       max_deletions=a.max_deletions, length_bonus=a.length_bonus)
        for session, control in zip(a.sessions, a.control):
            for spatial in (["uniform", a.spatial] if a.lm_only else [a.spatial]):
                t0 = time.time()
                segs, (rows, lists) = _eval_one(session, a.taps, control, spatial, clm, wlm,
                                                w, a.beam, a.nbest, taps_root)
                c, ci_lo, ci_hi = bootstrap_ci(rows)
                wv, _, _ = bootstrap_ci(rows, metric="wer")
                tag = f"{spec}/{a.word}/{spatial}"
                key = f"{session}|{spatial}"
                extra = ""
                if key in base_rows:
                    d, dlo, dhi = paired_delta_ci(base_rows[key], rows)
                    extra = f"  d={d:+.3f} [{dlo:+.3f},{dhi:+.3f}]"
                else:
                    base_rows[key] = rows
                print(f"{session[:18]:18s} {tag:34s} CER={c:.3f} [{ci_lo:.3f},{ci_hi:.3f}] "
                      f"WER={wv:.3f}{extra}  ({time.time()-t0:.1f}s)")
                if a.show:
                    for (ref, hyp) in rows[:a.show]:
                        print(f"      {hyp[:72]!r}\n      {'':>2}truth {ref[:72]!r}")
    return 0


def cmd_sweep(a) -> int:
    """Decoder hyperparameter sweep. Selection is reported per session, never mixed."""
    taps_root = Path(a.taps_root) if a.taps_root else None
    clm, text, _, _ = build_models(a.spec)
    wlm = WordBigramLM.train(text) if a.word == "bigram" else UnigramAdapter(WordLM.train(text))
    base = DecWeights(obs=a.obs_weight, char=a.char_weight, word=a.word_weight,
                      insertion=a.insertion, deletion=a.deletion,
                      max_deletions=a.max_deletions, length_bonus=a.length_bonus)
    for session, control in zip(a.sessions, a.control):
        segs = load_segments(session, a.taps, control, taps_root)
        obs = obs_matrices(session, segs, a.spatial)
        print(f"### sweep {a.param} on {session} (spec={a.spec}, word={a.word})")
        for v in a.values:
            v = int(v) if a.param in ("beam", "max_deletions") else v
            w = base if a.param == "beam" else replace(base, **{a.param: v})
            beam = int(v) if a.param == "beam" else a.beam
            t0 = time.time()
            rows, _ = decode_segments(segs, obs, clm, wlm, w, beam)
            c, lo, hi = bootstrap_ci(rows)
            print(f"  {a.param}={v:<8} CER={c:.3f} [{lo:.3f},{hi:.3f}]  ({time.time()-t0:.1f}s)")
    return 0


def cmd_rescore(a) -> int:
    """Generate n-best with the char/word decoder, rescore with a neural LM, report both."""
    taps_root = Path(a.taps_root) if a.taps_root else None
    clm, text, _, _ = build_models(a.spec)
    wlm = WordBigramLM.train(text) if a.word == "bigram" else UnigramAdapter(WordLM.train(text))
    w = DecWeights(obs=a.obs_weight, char=a.char_weight, word=a.word_weight,
                   insertion=a.insertion, deletion=a.deletion,
                   max_deletions=a.max_deletions, length_bonus=a.length_bonus)
    nlm = NeuralLM(a.neural)
    for session, control in zip(a.sessions, a.control):
        segs = load_segments(session, a.taps, control, taps_root)
        obs = obs_matrices(session, segs, a.spatial)
        rows, lists = decode_segments(segs, obs, clm, wlm, w, a.beam, a.nbest)
        neural = [nlm.logp([t for t, _ in nb]) for nb in lists]
        c0, lo0, hi0 = bootstrap_ci(rows)
        print(f"{session}: 1-best CER={c0:.3f} [{lo0:.3f},{hi0:.3f}]  nbest={a.nbest} {a.neural}")
        oracle = [(ref, min((t for t, _ in nb), key=lambda t: edit_distance(ref, t)))
                  for (ref, _), nb in zip(segs, lists)]
        co, loo, hio = bootstrap_ci(oracle)
        print(f"   n-best oracle CER={co:.3f} [{loo:.3f},{hio:.3f}]  (ceiling for any rescorer)")
        for lam in a.lambdas:
            resc = []
            for (ref, _), nb, ns in zip(segs, lists, neural):
                best = max(range(len(nb)), key=lambda i: nb[i][1] + lam * ns[i]
                           + a.neural_len * len(nb[i][0]))
                resc.append((ref, nb[best][0]))
            c, lo, hi = bootstrap_ci(resc)
            d, dlo, dhi = paired_delta_ci(rows, resc)
            print(f"   lambda={lam:<5} CER={c:.3f} [{lo:.3f},{hi:.3f}]  "
                  f"d={d:+.3f} [{dlo:+.3f},{dhi:+.3f}]")
    return 0


def align_score(obs: np.ndarray, text: str, w: DecWeights) -> float:
    """Best score for forcing `text` onto this tap sequence under the same insert/delete lattice."""
    L = len(text)
    cols = np.array([A_INDEX.get(c, A_INDEX[" "]) for c in text], int)
    prev = np.full(L + 1, -math.inf)
    prev[0] = 0.0
    for j in range(1, L + 1):
        prev[j] = prev[j - 1] + w.deletion
    for i in range(1, len(obs) + 1):
        cur = np.full(L + 1, -math.inf)
        cur[0] = prev[0] + w.insertion
        row = obs[i - 1]
        for j in range(1, L + 1):
            cur[j] = max(prev[j] + w.insertion, prev[j - 1] + w.obs * row[cols[j - 1]],
                         cur[j - 1] + w.deletion)
        prev = cur
    return float(prev[L])


def cmd_ceiling(a) -> int:
    """Diagnostic upper bound: a perfect closed-vocabulary LM that knows the 20 candidate phrases."""
    taps_root = Path(a.taps_root) if a.taps_root else None
    w = DecWeights(obs=a.obs_weight, insertion=a.insertion, deletion=a.deletion)
    for session, control in zip(a.sessions, a.control):
        segs = load_segments(session, a.taps, control, taps_root)
        cands = sorted({r for r, _ in segs}) if a.candidates == "self" else test_phrases()
        obs = obs_matrices(session, segs, a.spatial)
        rows, ranks = [], []
        for (ref, taps), o in zip(segs, obs):
            pool = cands if ref in cands else cands + [ref]
            sc = sorted(((align_score(o, c, w), c) for c in pool), reverse=True)
            rows.append((ref, sc[0][1]))
            ranks.append(1 + [c for _, c in sc].index(ref))
        c, lo, hi = bootstrap_ci(rows)
        print(f"{session} closed-vocab({len(cands)}) oracle-LM CER={c:.3f} [{lo:.3f},{hi:.3f}]  "
              f"exact={sum(r==h for r,h in rows)}/{len(rows)}  "
              f"median rank of truth={int(np.median(ranks))}/{len(cands)}")
    return 0


def _common(p):
    p.add_argument("--sessions", nargs="+", default=["20260910-202149-desk"])
    p.add_argument("--control", nargs="+", type=int, default=None)
    p.add_argument("--taps", default="taps_pos.jsonl")
    p.add_argument("--taps-root", default=None, help="read taps from here instead of the session dir")
    p.add_argument("--spatial", choices=["keyprobs", "uniform"], default="keyprobs")
    p.add_argument("--word", choices=["none", "unigram", "bigram"], default="bigram")
    p.add_argument("--beam", type=int, default=30)
    p.add_argument("--nbest", type=int, default=1)
    p.add_argument("--obs-weight", type=float, default=1.0)
    p.add_argument("--char-weight", type=float, default=1.0)
    p.add_argument("--word-weight", type=float, default=0.6)
    p.add_argument("--insertion", type=float, default=-7.0)
    p.add_argument("--deletion", type=float, default=-9.0)
    p.add_argument("--max-deletions", type=int, default=2)
    p.add_argument("--length-bonus", type=float, default=0.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("contamination", help="grep every corpus for the test phrases")
    c.add_argument("--corpora", nargs="+", default=list(CORPORA))
    c.set_defaults(func=cmd_contamination)

    p = sub.add_parser("ppl", help="held-out char perplexity per LM variant")
    p.add_argument("--specs", nargs="+", default=["gutenberg", "tatoeba", "gutenberg+tatoeba"])
    p.add_argument("--dev", default="tatoeba")
    p.set_defaults(func=cmd_ppl)

    e = sub.add_parser("eval", help="CER/WER with bootstrap CIs per LM variant")
    _common(e)
    e.add_argument("--specs", nargs="+", default=["gutenberg"])
    e.add_argument("--lm-only", action="store_true", help="also report the uniform-observation floor")
    e.add_argument("--show", type=int, default=0)
    e.set_defaults(func=cmd_eval)

    s = sub.add_parser("sweep", help="sweep one decoder hyperparameter")
    _common(s)
    s.add_argument("--spec", default="gutenberg")
    s.add_argument("--param", required=True)
    s.add_argument("--values", nargs="+", type=float, required=True)
    s.set_defaults(func=cmd_sweep)

    r = sub.add_parser("rescore", help="n-best + neural rescoring")
    _common(r)
    r.add_argument("--spec", default="gutenberg")
    r.add_argument("--neural", default="distilgpt2")
    r.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0])
    r.add_argument("--neural-len", type=float, default=0.0)
    r.set_defaults(func=cmd_rescore)

    cl = sub.add_parser("ceiling", help="diagnostic: what a perfect closed-vocabulary LM would buy")
    _common(cl)
    cl.add_argument("--candidates", choices=["phrases", "self"], default="phrases")
    cl.set_defaults(func=cmd_ceiling)

    a = ap.parse_args(argv)
    if getattr(a, "sessions", None) and a.control is None:
        a.control = [1 if s.endswith("kbd") else 0 for s in a.sessions]
    if getattr(a, "param", None) == "beam":
        a.values = [int(v) for v in a.values]
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
