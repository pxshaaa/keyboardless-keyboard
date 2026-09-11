"""Decode typed text from camera taps: per-key spatial model + char/word LM + beam search.
Nothing here ever reads phase0/phrases.txt or a phrase's own text for fitting."""

from __future__ import annotations

import argparse
import json
import math
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phase0.analysis.analyze_drift import MODIFIERS, pair_events, read_jsonl

ALPHABET = "abcdefghijklmnopqrstuvwxyz "
A_INDEX = {c: i for i, c in enumerate(ALPHABET)}
SPACE = A_INDEX[" "]
NA = len(ALPHABET)
BASE = NA + 1  # +1 keeps the LM's start-of-text pad symbol out of the alphabet
PAD = NA
KEY_ALIAS = {"space": " "}

FINGERTIPS = (4, 8, 12, 16, 20)
POSE_JOINTS = (0,) + FINGERTIPS
LM_ORDER = 6
LM_BOOKS = (1342, 2701, 11, 98, 84, 1661, 76, 174, 345, 1080)
LM_DIR = Path("data/lm")
CORPUS_NOTE = "Project Gutenberg public-domain prose: " + ", ".join(f"#{b}" for b in LM_BOOKS)


# --------------------------------------------------------------------------- data
def load_pose(session: Path) -> tuple[np.ndarray, np.ndarray]:
    """(frame_ids, P[frame, hand, joint, xyz]) from landmarks.parquet."""
    import pyarrow.parquet as pq

    t = pq.read_table(session / "landmarks.parquet", columns=["i", "hand", "joint", "x", "y", "z"])
    i = t.column("i").to_numpy()
    hand = t.column("hand").to_numpy()
    joint = t.column("joint").to_numpy()
    frames = np.unique(i)
    inv = np.searchsorted(frames, i)
    P = np.full((len(frames), 2, 21, 3), np.nan, np.float32)
    for k, ch in enumerate("xyz"):
        P[inv, hand, joint, k] = t.column(ch).to_numpy()
    return frames, P


def pose_features(session: Path, taps: list[dict]) -> np.ndarray:
    frames, P = load_pose(session)
    out = np.full((len(taps), 2 * len(POSE_JOINTS) * 3), np.nan, np.float32)
    for n, tap in enumerate(taps):
        k = int(np.searchsorted(frames, tap["i"]))
        if k >= len(frames) or frames[k] != tap["i"]:
            continue
        out[n] = np.concatenate([P[k, s, list(POSE_JOINTS), :].ravel() for s in (0, 1)])
    return out


def labelled_taps(session: Path, taps_name: str = "taps.jsonl") -> tuple[list[dict], list[str]]:
    """Pair keydowns to taps (+/-80 ms, 1:1) -> the taps that carry a known key label."""
    keys = [r for r in read_jsonl(session / "keys.jsonl") if r.get("event") == "down"]
    taps = read_jsonl(session / taps_name)
    usable = [i for i, r in enumerate(keys) if r.get("key") not in MODIFIERS and r.get("key") != "unknown"]
    pairs = pair_events([keys[i]["t"] for i in usable], [r["t"] for r in taps])
    out_taps, out_keys = [], []
    for a, b in pairs:
        key = KEY_ALIAS.get(keys[usable[a]]["key"], keys[usable[a]]["key"])
        if key in A_INDEX:
            out_taps.append(taps[b])
            out_keys.append(key)
    return out_taps, out_keys


# ------------------------------------------------------------------- spatial models
class UniformSpatial:
    """Honest floor: the vision system contributes nothing."""

    name = "uniform"

    def logp(self, session: Path, taps: list[dict]) -> np.ndarray:
        return np.full((len(taps), NA), -math.log(NA), np.float64)


class GaussianSpatial:
    """One bivariate Gaussian per key over the tap's reported fingertip (x, y)."""

    name = "gauss"

    def __init__(self, mu, prec, logdet, keys, fallback):
        self.mu, self.prec, self.logdet, self.keys, self.fallback = mu, prec, logdet, keys, fallback

    @classmethod
    def fit(cls, xy: np.ndarray, labels: list[str], min_n: int = 12, shrink: float = 0.35):
        pooled = np.cov(xy.T) + np.eye(2) * 4.0
        gmu = xy.mean(0)
        mu = np.tile(gmu, (NA, 1))
        cov = np.tile(pooled, (NA, 1, 1))
        seen = np.zeros(NA, bool)
        for c in set(labels):
            v = xy[np.array([l == c for l in labels])]
            if len(v) < min_n:
                continue
            mu[A_INDEX[c]] = v.mean(0)
            # shrink toward the pooled covariance: 20-100 samples cannot support a free 2x2
            cov[A_INDEX[c]] = (1 - shrink) * (np.cov(v.T) + np.eye(2) * 4.0) + shrink * pooled
            seen[A_INDEX[c]] = True
        prec = np.linalg.inv(cov)
        logdet = np.linalg.slogdet(cov)[1]
        return cls(mu, prec, logdet, seen, ~seen)

    def logp(self, session: Path, taps: list[dict]) -> np.ndarray:
        x = np.array([[t["x"], t["y"]] for t in taps], float)
        d = x[:, None, :] - self.mu[None, :, :]
        q = np.einsum("nkj,kjl,nkl->nk", d, self.prec, d)
        ll = -0.5 * (q + self.logdet[None, :])
        # keys never observed keep the pooled Gaussian, which is deliberately uninformative
        return ll - _logsumexp(ll, 1, keepdims=True)

    def save(self, path: Path):
        np.savez(path, kind="gauss", mu=self.mu, prec=self.prec, logdet=self.logdet,
                 keys=self.keys, fallback=self.fallback)

    @classmethod
    def load(cls, z):
        return cls(z["mu"], z["prec"], z["logdet"], z["keys"], z["fallback"])


class PoseSpatial:
    """All 12 fingertip/wrist positions at the tap frame; the tap file's own (finger, x, y)
    names a near-random finger, so the per-key Gaussian above mostly sees that noise."""

    name = "pose"

    def __init__(self, model, classes):
        self.model, self.classes = model, classes

    @classmethod
    def fit(cls, feats: np.ndarray, labels: list[str], min_n: int = 12):
        import lightgbm as lgb

        cnt = Counter(labels)
        keep = np.array([cnt[l] >= min_n for l in labels])
        y = np.array([A_INDEX[l] for l in labels])[keep]
        classes = np.unique(y)
        remap = {c: i for i, c in enumerate(classes)}
        m = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.08, num_leaves=15,
                               min_child_samples=8, verbose=-1)
        m.fit(feats[keep], np.array([remap[v] for v in y]))
        return cls(m, classes)

    def logp(self, session: Path, taps: list[dict]) -> np.ndarray:
        p = self.model.predict_proba(pose_features(session, taps))
        out = np.full((len(taps), NA), 1e-6)
        out[:, self.classes] = np.maximum(p, 1e-6)
        return np.log(out / out.sum(1, keepdims=True))

    def save(self, path: Path):
        import joblib

        joblib.dump({"kind": "pose", "model": self.model, "classes": self.classes}, path)


class KeyProbsSpatial:
    """The tap file's own key distribution from `tap_pos apply`; the pose->(x,y)->Gaussian
    route loses most of the pose signal (0.214 vs 0.383 LOSO top-1), so read it directly."""

    name = "keyprobs"

    def __init__(self, floor: float = 1e-5):
        self.floor = floor

    def logp(self, session: Path, taps: list[dict]) -> np.ndarray:
        out = np.full((len(taps), NA), self.floor)
        missing = 0
        for i, t in enumerate(taps):
            kp = t.get("key_probs")
            if not kp:
                missing += 1  # no distribution -> that tap stays uniform, it is not evidence
                continue
            seen, tot = [], 0.0
            for k, v in kp.items():
                c = A_INDEX.get(KEY_ALIAS.get(k, k))
                if c is None:
                    continue
                out[i, c] = max(float(v), self.floor)
                seen.append(c)
                tot += float(v)
            # top-k truncation: spread the unlisted tail evenly rather than pretending it is zero
            rest = [c for c in range(NA) if c not in seen]
            if rest and tot < 1.0:
                out[i, rest] = max((1.0 - tot) / len(rest), self.floor)
        if missing == len(taps) and taps:
            raise SystemExit("--spatial keyprobs: no tap carries key_probs - run `tap_pos apply` first")
        return np.log(out / out.sum(1, keepdims=True))


def _logsumexp(a, axis=None, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    s = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return s if keepdims else np.squeeze(s, axis)


def load_spatial(path: Path):
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=True)
        return GaussianSpatial.load(z)
    import joblib

    b = joblib.load(path)
    return PoseSpatial(b["model"], b["classes"])


# ------------------------------------------------------------------- language model
def fetch_corpus() -> str:
    LM_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for b in LM_BOOKS:
        f = LM_DIR / f"g{b}.txt"
        if not f.exists():
            for url in (f"https://www.gutenberg.org/files/{b}/{b}-0.txt",
                        f"https://www.gutenberg.org/cache/epub/{b}/pg{b}.txt"):
                try:
                    f.write_bytes(urllib.request.urlopen(url, timeout=60).read())
                    break
                except Exception:
                    continue
        if f.exists():
            parts.append(f.read_text(encoding="utf-8", errors="ignore"))
    if not parts:
        raise SystemExit(f"no LM corpus in {LM_DIR} and it could not be downloaded")
    text = re.sub(r"[^a-z ]+", " ", " ".join(parts).lower())
    return re.sub(r" +", " ", text)


class CharLM:
    """Stupid-backoff character n-gram; counts held as sorted int64 code arrays."""

    def __init__(self, tables, order, alpha=0.4):
        self.tables, self.order, self.alpha = tables, order, alpha
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    @classmethod
    def train(cls, text: str, order: int = LM_ORDER, prune_from: int = 4):
        seq = np.array([A_INDEX.get(c, SPACE) for c in text], np.int64)
        tables = {}
        for k in range(1, order + 1):
            code = np.zeros(len(seq) - k + 1, np.int64)
            for j in range(k):
                code = code * BASE + seq[j:len(seq) - k + 1 + j]
            u, c = np.unique(code, return_counts=True)
            if k >= prune_from:  # 6-gram tail is mostly singletons; dropping it halves the file
                m = c >= 2
                u, c = u[m], c[m]
            tables[k] = (u, c.astype(np.int64))
        # order-0 unigram totals for the final backoff
        return cls(tables, order)

    def _count(self, k: int, code: int) -> int:
        u, c = self.tables[k]
        i = np.searchsorted(u, code)
        return int(c[i]) if i < len(u) and u[i] == code else 0

    def _row(self, k: int, ctx_code: int) -> np.ndarray:
        """Counts of every alphabet symbol following the length-(k-1) context."""
        u, c = self.tables[k]
        lo = np.searchsorted(u, ctx_code * BASE)
        hi = np.searchsorted(u, ctx_code * BASE + NA)
        out = np.zeros(NA, np.int64)
        if hi > lo:
            out[u[lo:hi] - ctx_code * BASE] = c[lo:hi]
        return out

    def logprobs(self, ctx: str) -> np.ndarray:
        """log P(next | ctx) for the whole alphabet, stupid backoff over orders."""
        ctx = ctx[-(self.order - 1):]
        key = (len(ctx), _encode(ctx))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        p = np.zeros(NA)
        weight = 1.0
        remaining = np.ones(NA, bool)
        for k in range(len(ctx) + 1, 0, -1):
            ctx_code = _encode(ctx[len(ctx) - (k - 1):]) if k > 1 else 0
            row = self._row(k, ctx_code)
            tot = row.sum()
            if tot == 0:
                continue
            hits = (row > 0) & remaining
            p[hits] = weight * row[hits] / tot
            remaining &= ~hits
            if not remaining.any():
                break
            weight *= self.alpha
        p[remaining] = weight * 1e-7
        out = np.log(p / p.sum())
        if len(self._cache) < 400_000:
            self._cache[key] = out
        return out

    def save(self, path: Path):
        d = {"order": np.int64(self.order)}
        for k, (u, c) in self.tables.items():
            d[f"u{k}"], d[f"c{k}"] = u, c
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path: Path):
        z = np.load(path)
        order = int(z["order"])
        return cls({k: (z[f"u{k}"], z[f"c{k}"]) for k in range(1, order + 1)}, order)


def _encode(s: str) -> int:
    code = 0
    for ch in s:
        code = code * BASE + A_INDEX.get(ch, PAD)
    return code


class WordLM:
    """Unigram word scores plus a prefix trie, so partial words steer the beam early."""

    def __init__(self, logp: dict[str, float], prefixes: set[str], oov: float):
        self.logp, self.prefixes, self.oov = logp, prefixes, oov

    @classmethod
    def train(cls, text: str, top: int = 60_000):
        cnt = Counter(text.split())
        common = cnt.most_common(top)
        tot = sum(n for _, n in common)
        logp = {w: math.log(n / tot) for w, n in common}
        prefixes = set()
        for w, _ in common:
            for i in range(1, len(w) + 1):
                prefixes.add(w[:i])
        return cls(logp, prefixes, math.log(0.5 / tot))

    def word_score(self, w: str) -> float:
        return self.logp.get(w, self.oov)

    def is_prefix(self, w: str) -> bool:
        return w == "" or w in self.prefixes

    def save(self, path: Path):
        path.write_text(json.dumps({"oov": self.oov, "logp": self.logp}))

    @classmethod
    def load(cls, path: Path):
        d = json.loads(path.read_text())
        prefixes = set()
        for w in d["logp"]:
            for i in range(1, len(w) + 1):
                prefixes.add(w[:i])
        return cls(d["logp"], prefixes, d["oov"])


# ------------------------------------------------------------------------- decoding
@dataclass(frozen=True)
class Weights:
    obs: float = 1.0
    char: float = 1.0
    word: float = 0.6
    prefix_penalty: float = 2.5
    insertion: float = -7.0   # log-cost of declaring a detected tap spurious
    deletion: float = -9.0    # log-cost of inventing a character with no tap
    max_deletions: int = 2


def beam_decode(obs_logp: np.ndarray, char_lm: CharLM | None, word_lm: WordLM | None,
                w: Weights = Weights(), beam: int = 40, argmax_obs: bool = False) -> str:
    """Beam search over taps; each tap may emit a character or be dropped, and characters
    may be emitted with no tap, so the decoder survives over- and under-detection."""
    if argmax_obs:
        hard = np.full_like(obs_logp, math.log(1e-4))
        hard[np.arange(len(obs_logp)), obs_logp.argmax(1)] = math.log(1 - 1e-4 * (NA - 1))
        obs_logp = hard

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
                      sc + w.obs * row[c] + w.char * lm[c] + _extra(text, ALPHABET[c], word_lm, w))
        hyps = _topk(nxt, beam)
    best = max(hyps, key=lambda t: hyps[t] + (w.word * word_lm.word_score(_last(t)) if word_lm and _last(t) else 0.0))
    return best.strip()


def _ctx(text: str) -> str:
    return (" " + text)[-(LM_ORDER - 1):]


def _last(text: str) -> str:
    return text.rsplit(" ", 1)[-1]


def _extra(text: str, ch: str, word_lm: WordLM | None, w: Weights) -> float:
    if word_lm is None:
        return 0.0
    if ch == " ":
        done = _last(text)
        return w.word * word_lm.word_score(done) if done else -abs(w.deletion)
    return 0.0 if word_lm.is_prefix(_last(text) + ch) else -w.prefix_penalty


def _push(d: dict, key: str, score: float):
    if score > d.get(key, -math.inf):
        d[key] = score


def _topk(d: dict[str, float], beam: int) -> dict[str, float]:
    if len(d) <= beam:
        return d
    return dict(sorted(d.items(), key=lambda kv: -kv[1])[:beam])


def _close_deletions(hyps, char_lm, word_lm, w, beam):
    """Let hypotheses emit up to max_deletions characters that no tap was detected for."""
    out = dict(hyps)
    frontier = hyps
    for _ in range(w.max_deletions):
        nxt: dict[str, float] = {}
        for text, sc in frontier.items():
            lm = char_lm.logprobs(_ctx(text)) if char_lm else np.zeros(NA)
            for c in range(NA):
                _push(nxt, text + ALPHABET[c],
                      sc + w.deletion + w.char * lm[c] + _extra(text, ALPHABET[c], word_lm, w))
        frontier = _topk(nxt, beam)
        for t, s in frontier.items():
            _push(out, t, s)
    return _topk(out, beam)


# -------------------------------------------------------------------------- metrics
def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    return edit_distance(ref, hyp) / max(1, len(ref))


def wer(ref: str, hyp: str) -> float:
    r = ref.split()
    return edit_distance(r, hyp.split()) / max(1, len(r))


# -------------------------------------------------------------------------- segments
def desk_segments(session: Path, taps: list[dict]) -> list[tuple[str, list[dict]]]:
    """Each phrase's [shown, done) window is decoded independently against its known text."""
    rows = read_jsonl(session / "phrases.jsonl")
    shown = {r["idx"]: r for r in rows if r["event"] == "shown"}
    done = {r["idx"]: r for r in rows if r["event"] == "done"}
    out = []
    for idx in sorted(shown):
        if idx not in done:
            continue
        t0, t1 = shown[idx]["t"], done[idx]["t"]
        out.append((shown[idx]["phrase"], [t for t in taps if t0 <= t["t"] < t1]))
    return out


def kbd_segments(session: Path, taps: list[dict], gap: float = 1.5, min_chars: int = 15):
    """Control: cut the keyboard session at typing pauses and use the real keystrokes as truth."""
    keys = [dict(r, key=KEY_ALIAS.get(r["key"], r["key"])) for r in read_jsonl(session / "keys.jsonl")
            if r.get("event") == "down" and KEY_ALIAS.get(r.get("key"), r.get("key")) in A_INDEX]
    segs, cur = [], []
    for r in keys:
        if cur and r["t"] - cur[-1]["t"] > gap:
            segs.append(cur)
            cur = []
        cur.append(r)
    if cur:
        segs.append(cur)
    out = []
    for s in segs:
        text = "".join(r["key"] for r in s).strip()
        if len(text) < min_chars:
            continue
        t0, t1 = s[0]["t"] - 0.2, s[-1]["t"] + 0.4
        out.append((text, [t for t in taps if t0 <= t["t"] <= t1]))
    return out


# ------------------------------------------------------------------------------ CLI
def cmd_fit(a) -> int:
    xy, feats, labels = [], [], []
    for s in a.sessions:
        sd = Path(s)
        taps, keys = labelled_taps(sd, a.taps_name)
        print(f"{sd.name}: {len(taps)} labelled taps")
        xy.append(np.array([[t["x"], t["y"]] for t in taps], float))
        if a.spatial == "pose":
            feats.append(pose_features(sd, taps))
        labels += keys
    xy = np.vstack(xy)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.spatial == "gauss":
        GaussianSpatial.fit(xy, labels).save(Path(a.out))
    else:
        PoseSpatial.fit(np.vstack(feats), labels).save(Path(a.out))
    print(f"spatial[{a.spatial}] from {len(labels)} labelled taps -> {a.out}")

    lm_path, wl_path = Path(a.lm_out), Path(a.lm_out).with_suffix(".words.json")
    if a.rebuild_lm or not lm_path.exists():
        text = fetch_corpus()
        print(f"LM corpus: {len(text):,} chars ({CORPUS_NOTE})")
        CharLM.train(text, a.lm_order).save(lm_path)
        WordLM.train(text).save(wl_path)
        print(f"char {a.lm_order}-gram -> {lm_path}; word unigram -> {wl_path}")
    return 0


def cmd_spatial_report(a) -> int:
    """Cross-validated top-k of the spatial model alone: is there any per-key signal?"""
    from sklearn.model_selection import StratifiedKFold

    sd = Path(a.session)
    taps, labels = labelled_taps(sd, a.taps_name)
    y = np.array([A_INDEX[l] for l in labels])
    cnt = Counter(y)
    keep = np.array([cnt[v] >= 12 for v in y])
    taps = [t for t, k in zip(taps, keep) if k]
    labels = [l for l, k in zip(labels, keep) if k]
    y = y[keep]
    xy = np.array([[t["x"], t["y"]] for t in taps], float)
    feats = pose_features(sd, taps)
    print(f"{sd.name}: {len(y)} labelled taps, {len(set(y))} keys, "
          f"majority={max(Counter(y).values())/len(y):.3f}, chance={1/len(set(y)):.3f}")
    for name in ("gauss", "pose"):
        t1 = t5 = 0
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(xy, y):
            if name == "gauss":
                m = GaussianSpatial.fit(xy[tr], [labels[i] for i in tr])
                lp = m.logp(sd, [taps[i] for i in te])
            else:
                m = PoseSpatial.fit(feats[tr], [labels[i] for i in tr])
                p = m.model.predict_proba(feats[te])
                lp = np.full((len(te), NA), -20.0)
                lp[:, m.classes] = np.log(np.maximum(p, 1e-9))
            order = np.argsort(-lp, 1)
            t1 += int((order[:, 0] == y[te]).sum())
            t5 += int(sum(y[te][i] in order[i, :5] for i in range(len(te))))
        print(f"  {name:6s} top1={t1/len(y):.3f} top5={t5/len(y):.3f}")
    return 0


def _decode_all(segments, spatial, char_lm, word_lm, session, w, beam, argmax_obs, label, quiet=False):
    rows = []
    for text, taps in segments:
        if not taps:
            rows.append((text, "", 1.0, 1.0, 0))
            continue
        lp = spatial.logp(session, taps)
        hyp = beam_decode(lp, char_lm, word_lm, w, beam, argmax_obs)
        rows.append((text, hyp, cer(text, hyp), wer(text, hyp), len(taps)))
    tot_c = sum(edit_distance(r[0], r[1]) for r in rows)
    ref_c = sum(len(r[0]) for r in rows)
    tot_w = sum(edit_distance(r[0].split(), r[1].split()) for r in rows)
    ref_w = sum(len(r[0].split()) for r in rows)
    if not quiet:
        print(f"\n### {label}")
        print(f"{'#':>2} {'taps':>4} {'chars':>5} {'CER':>7} {'WER':>7}  decoded")
        for i, (ref, hyp, c, wv, n) in enumerate(rows):
            print(f"{i:>2} {n:>4} {len(ref):>5} {c:>7.3f} {wv:>7.3f}  {hyp[:70]!r}")
            print(f"{'':>21}{'truth':>7}  {ref[:70]!r}")
    return tot_c / max(1, ref_c), tot_w / max(1, ref_w), rows


class OracleSpatial:
    """Diagnostic: perfect (or `acc`-accurate) key identity on taps that a keystroke pairs to."""

    name = "oracle"

    def __init__(self, session: Path, taps_name: str, acc: float):
        taps, labels = labelled_taps(session, taps_name)
        self.lab = {round(t["t"], 6): l for t, l in zip(taps, labels)}
        self.acc = acc

    def logp(self, session: Path, taps: list[dict]) -> np.ndarray:
        o = np.full((len(taps), NA), 1.0 / NA)
        for i, t in enumerate(taps):
            k = self.lab.get(round(t["t"], 6))
            if k is not None:
                o[i] = (1 - self.acc) / (NA - 1)
                o[i, A_INDEX[k]] = self.acc
        return np.log(o / o.sum(1, keepdims=True))


def cmd_decode(a) -> int:
    session = Path(a.session)
    taps = read_jsonl(Path(a.taps) if a.taps else session / "taps.jsonl")
    if a.spatial == "uniform":
        spatial = UniformSpatial()
    elif a.spatial == "keyprobs":
        spatial = KeyProbsSpatial()
    elif a.spatial == "oracle":
        spatial = OracleSpatial(session, Path(a.taps).name if a.taps else "taps.jsonl", a.oracle_acc)
    else:
        spatial = load_spatial(Path(a.model))
    char_lm = None if a.no_lm else CharLM.load(Path(a.lm))
    word_lm = None if (a.no_lm or a.no_word_lm) else WordLM.load(Path(a.lm).with_suffix(".words.json"))
    segments = kbd_segments(session, taps) if a.control else desk_segments(session, taps)
    w = Weights(obs=a.obs_weight, char=a.char_weight, word=a.word_weight,
                insertion=a.insertion, deletion=a.deletion)
    label = (f"{session.name} [{Path(a.taps).name if a.taps else 'taps.jsonl'}] "
             f"spatial={a.spatial} lm={'off' if a.no_lm else 'on'} beam={a.beam}"
             f"{' argmax' if a.argmax else ''}")
    c, wv, _ = _decode_all(segments, spatial, char_lm, word_lm, session, w, a.beam, a.argmax, label)
    print(f"\nOVERALL {label}: CER={c:.3f} WER={wv:.3f} over {len(segments)} segments")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit the spatial model on keyboard sessions + build the LM")
    f.add_argument("--sessions", nargs="+", required=True)
    f.add_argument("--spatial", choices=["gauss", "pose"], default="gauss")
    f.add_argument("--taps-name", default="taps.jsonl")
    f.add_argument("--out", default=None)
    f.add_argument("--lm-out", default="models/charlm.npz")
    f.add_argument("--lm-order", type=int, default=LM_ORDER)
    f.add_argument("--rebuild-lm", action="store_true")
    f.set_defaults(func=cmd_fit)

    d = sub.add_parser("decode", help="decode a session and score it against its known text")
    d.add_argument("session")
    d.add_argument("--taps", default=None, help="alternative taps.jsonl (e.g. taps_desk.jsonl)")
    d.add_argument("--model", default="models/spatial_gauss.npz")
    d.add_argument("--spatial", choices=["gauss", "pose", "uniform", "oracle", "keyprobs"],
                   default="gauss")
    d.add_argument("--oracle-acc", type=float, default=1.0, help="--spatial oracle: per-key accuracy")
    d.add_argument("--lm", default="models/charlm.npz")
    d.add_argument("--beam", type=int, default=30)
    d.add_argument("--argmax", action="store_true", help="ablation: collapse the key distribution")
    d.add_argument("--no-lm", action="store_true")
    d.add_argument("--no-word-lm", action="store_true")
    d.add_argument("--control", action="store_true", help="score a kbd session against its keystrokes")
    d.add_argument("--obs-weight", type=float, default=1.0)
    d.add_argument("--char-weight", type=float, default=1.0)
    d.add_argument("--word-weight", type=float, default=0.6)
    d.add_argument("--insertion", type=float, default=-7.0)
    d.add_argument("--deletion", type=float, default=-9.0)
    d.set_defaults(func=cmd_decode)

    s = sub.add_parser("spatial-report", help="cross-validated per-key accuracy of the spatial model")
    s.add_argument("session")
    s.add_argument("--taps-name", default="taps.jsonl")
    s.set_defaults(func=cmd_spatial_report)

    a = ap.parse_args(argv)
    if a.cmd == "fit" and a.out is None:
        a.out = f"models/spatial_{a.spatial}." + ("npz" if a.spatial == "gauss" else "joblib")
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
