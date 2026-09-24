"""v4 M7 suggestion bar: per-word alternatives from the scored gen-verify pool + names quick list + tap metrics.
Posterior over the pool = softmax(score / T) of the frozen v3 selection score (eligible candidates only). Every candidate is
word-aligned to the chosen sentence (char-similarity substitution cost); the candidate words aligned to a chosen word form that
slot's alternative, inserted words attach to a neighbouring slot (the non-matching one, else the previous), "" = delete the word.
Slot marginals = summed candidate posteriors per distinct alternative.
  names list: PYTHONPATH=. .venv/bin/python -m phase0.analysis.v4_suggest names"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

NAMES_FILE = Path("results/v4/names_quicklist.json")
N_NAMES = 12
MAX_CANDS = 400   # pool members (by posterior) aligned per segment


# ---------------------------------------------------------------- scoring (identical to llmdec.select)
def pool_scores(p: dict, run: dict) -> np.ndarray:
    llm, prompt, K, lam = run["llm"], run["prompt"], run["K"], run["lam"]
    gr = p["gen_rank"].get((llm, prompt)) if llm else None
    ok = p["base"] | (gr < K) if gr is not None else p["base"]
    if run.get("use_fz", 0) and "isfz" in p:
        ok = ok | p["isfz"]
    lmv = p["lm"][llm] if llm and lam else 0.0
    s = (p["ctc"] + (lam * lmv if llm and lam else 0.0) + run["wb"] * p["nw"] + run["cb"] * p["nc"]
         + run.get("mu_lex", 0.0) * p["nlex"] + run.get("mu_pers", 0.0) * p["npers"] + run.get("nu", 0.0) * p["plm"])
    if run.get("mu_tech", 0.0) and "ntech" in p:
        s = s + run["mu_tech"] * p["ntech"]
    if run.get("kappa", 0.0) and "fzd" in p:
        s = s - run["kappa"] * p["fzd"]
    plm = run.get("plm", "p05b")
    if run.get("lam_p", 0.0) and plm in p["lm"]:
        s = s + run["lam_p"] * np.nan_to_num(p["lm"][plm] - np.nanmean(p["lm"][plm]), nan=-50.0)
    return np.where(ok & np.isfinite(s), s, -np.inf)


# ---------------------------------------------------------------- alignment
def lev(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def align(h: list[str], c: list[str]):
    """-> ops [(kind, i_h|None, j_c|None)], kind in m (match/sub), d (h word without c word), i (extra c word)"""
    n, m = len(h), len(c)
    D = np.zeros((n + 1, m + 1))
    D[:, 0], D[0, :] = np.arange(n + 1), np.arange(m + 1)
    sub = [[0.0 if x == y else 0.5 + lev(x, y) / max(len(x), len(y)) for y in c] for x in h]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j - 1] + sub[i - 1][j - 1], D[i - 1, j] + 1, D[i, j - 1] + 1)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(D[i, j] - (D[i - 1, j - 1] + sub[i - 1][j - 1])) < 1e-9:
            ops.append(("m", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and abs(D[i, j] - (D[i - 1, j] + 1)) < 1e-9:
            ops.append(("d", i - 1, None))
            i -= 1
        else:
            ops.append(("i", None, j - 1))
            j -= 1
    return ops[::-1]


def spans(h: list[str], c: list[str]):
    """-> (per chosen slot: candidate span string, per slot: aligned candidate word equals the chosen word)"""
    sp = [[] for _ in h]
    eq = [False] * len(h)
    runs, pending, prev = [], [], None
    for kind, i, j in align(h, c):
        if kind == "i":
            pending.append(c[j])
            continue
        if pending:
            runs.append((prev, i, pending))
            pending = []
        if kind == "m":
            sp[i].append(c[j])
            eq[i] = c[j] == h[i]
        prev = i
    if pending:
        runs.append((prev, None, pending))
    for a, b, ws in runs:
        if a is None and b is None:
            continue
        tgt = b if a is None else a if b is None else (b if (eq[a] and not eq[b]) else a)
        sp[tgt] = sp[tgt] + ws if tgt == a else ws + sp[tgt]
    return [" ".join(x) for x in sp], eq


# ---------------------------------------------------------------- suggestions
def cand_spans(chosen: str, texts: list[str], scores: np.ndarray) -> list[tuple[int, list[str]]]:
    """top MAX_CANDS eligible candidates by score (temperature-invariant order) -> [(pool index, per-slot span)]"""
    h = chosen.split()
    idx = np.where(np.isfinite(scores))[0]
    idx = idx[np.argsort(-scores[idx], kind="stable")][:MAX_CANDS]
    return [(int(i), spans(h, texts[i].split())[0]) for i in idx] if h else []


def slot_posteriors(chosen: str, texts: list[str], scores: np.ndarray, T: float, cached=None) -> list[dict]:
    h = chosen.split()
    cs = cached if cached is not None else cand_spans(chosen, texts, scores)
    if not h or not cs:
        return [{w: 1.0} for w in h]
    z = np.array([scores[i] for i, _ in cs]) / T
    w = np.exp(z - z.max())
    w /= w.sum()
    slot = [defaultdict(float) for _ in h]
    for (_, sp), wk in zip(cs, w):
        for j, s_ in enumerate(sp):
            slot[j][s_] += float(wk)
    return [dict(sorted(d.items(), key=lambda kv: -kv[1])) for d in slot]


def word_entries(chosen: str, post: list[dict], times: list[tuple[float, float]], k: int = 5) -> list[dict]:
    out = []
    for j, w in enumerate(chosen.split()):
        alts = [{"w": a, "p": round(p, 4)} for a, p in post[j].items() if a != w][:k]
        out.append({"word": w, "p": round(post[j].get(w, 0.0), 4), "alternatives": alts,
                    "start_s": round(times[j][0], 2) if times else None, "end_s": round(times[j][1], 2) if times else None})
    return out


# ---------------------------------------------------------------- word times (CTC forced alignment of the chosen text)
SYMS = "_abcdefghijklmnopqrstuvwxyz #"


def word_frames(M: np.ndarray, text: str) -> list[tuple[int, int]]:
    lab = [SYMS.index(c) for c in text]
    L, T = len(lab), len(M)
    S = 2 * L + 1
    if L == 0 or T == 0:
        return []
    ext = np.zeros(S, int)
    ext[1::2] = lab
    NEG = -1e18
    D = np.full((T, S), NEG)
    B = np.zeros((T, S), np.int8)
    D[0, 0], D[0, 1] = M[0, 0], M[0, ext[1]]
    skip = np.zeros(S, bool)
    for s in range(3, S, 2):
        skip[s] = ext[s] != ext[s - 2]
    for t in range(1, T):
        prev = D[t - 1]
        st = np.stack([prev, np.concatenate([[NEG], prev[:-1]]), np.where(skip, np.concatenate([[NEG, NEG], prev[:-2]]), NEG)])
        b = st.argmax(0)
        D[t] = st[b, np.arange(S)] + M[t, ext]
        B[t] = b
    s = S - 1 if S < 2 or D[T - 1, S - 1] >= D[T - 1, S - 2] else S - 2
    path = np.zeros(T, int)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(B[t, s])
    out, k = [], 0
    for w in text.split(" "):
        fr = np.where((path >= 2 * k + 1) & (path <= 2 * (k + len(w)) - 1) & (path % 2 == 1))[0]
        out.append((int(fr[0]), int(fr[-1])) if len(fr) else (-1, -1))
        k += len(w) + 1
    # words without own frames (never emitted): interpolate between neighbours
    for j, (a, b) in enumerate(out):
        if a < 0:
            pa = next((out[i][1] for i in range(j - 1, -1, -1) if out[i][0] >= 0), 0)
            nb = next((out[i][0] for i in range(j + 1, len(out)) if out[i][0] >= 0), T - 1)
            out[j] = (pa, max(pa, nb))
    return out


# ---------------------------------------------------------------- metrics
def seg_metrics(truth: str, chosen: str, post: list[dict], names: list[str], ks=(3, 5)) -> dict:
    t, h = truth.split(), chosen.split()
    r = {"n": len(t), "edits": lev(h, t)}
    if not h:
        r.update({"auto": 0, **{f"top{k}": {"words": 0, "taps": 0, "edits": len(t)} for k in ks},
                  "top5_names": {"words": 0, "taps": 0, "edits": len(t)}, "slots": []})
        return r
    tsp, eq = spans(h, t)
    r["auto"] = int(sum(eq))
    slots = []
    for j, w in enumerate(h):
        if tsp[j] == w:
            continue
        alts = [a for a in post[j] if a != w]
        rank = alts.index(tsp[j]) + 1 if tsp[j] in alts else None
        slots.append({"j": j, "hyp": w, "truth": tsp[j], "rank": rank, "gain": len(tsp[j].split()) - int(eq[j]),
                      "name": tsp[j] in names[:N_NAMES]})
    r["slots"] = slots
    for key, fix in [(f"top{k}", (lambda s, k=k: s["rank"] is not None and s["rank"] <= k)) for k in ks] + \
                    [("top5_names", lambda s: (s["rank"] is not None and s["rank"] <= 5) or s["name"])]:
        fixed = [s for s in slots if fix(s)]
        new = list(h)
        for s in fixed:
            new[s["j"]] = s["truth"]
        r[key] = {"words": r["auto"] + sum(s["gain"] for s in fixed), "taps": len(fixed),
                  "edits": lev(" ".join(new).split(), t)}
    return r


def summarise(rows: list[dict], ks=(3, 5)) -> dict:
    n = sum(r["n"] for r in rows)
    out = {"n_words": n, "auto_words": sum(r["auto"] for r in rows) / n, "wer_segwise": sum(r["edits"] for r in rows) / n}
    for key in [f"top{k}" for k in ks] + ["top5_names"]:
        out[key] = {"words_after_le1_tap": sum(r[key]["words"] for r in rows) / n,
                    "taps_per_100_words": 100 * sum(r[key]["taps"] for r in rows) / n,
                    "wer_after": sum(r[key]["edits"] for r in rows) / n}
    return out


# ---------------------------------------------------------------- names quick list (built once, frozen)
def fold(s: str) -> str:
    s = s.replace("ß", "ss")
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def build_names() -> dict:
    """proper-noun-like words among profile.json's top words + topics: Title-case mid-sentence in >=25% of English-corpus
    uses (>=10 uses; >=45% if the word is in the generic lexicon), rare in the German corpus (de/en uses <= 0.1, drops German
    nouns), profile language en, >=3 letters. Sorted by profile count; top N_NAMES form the quick list."""
    prof = json.loads(Path(".cache/personal/profile.json").read_text())
    cand: dict = {}
    for k, v in prof.items():
        if not isinstance(v, list) or not v:
            continue
        if isinstance(v[0], dict) and "word" in v[0]:
            for e in v:
                cand.setdefault(e["word"], [0, e.get("lang", "en")])[0] += e["count"]
        elif "topics" in k and isinstance(v[0], list):
            for w, c in v:
                cand.setdefault(w, [0, "en"])[0] += c
    gen = set(json.loads(Path(".cache/autocorrect/lexicon.json").read_text()))
    st = {}
    for lang in ("en", "de"):
        title, tot = Counter(), Counter()
        for line in open(f".cache/personal/corpus_{lang}.txt", errors="ignore"):
            for tk in re.findall(r"[^\W\d_]+", fold(line))[1:]:
                lw = tk.lower()
                if lw in cand:
                    tot[lw] += 1
                    title[lw] += tk[0].isupper() and tk[1:].islower()
        st[lang] = (title, tot)
    rows = []
    for w, (c, lang) in cand.items():
        ne, nd = st["en"][1][w], st["de"][1][w]
        rate = st["en"][0][w] / ne if ne >= 10 else 0.0
        if lang == "en" and len(w) >= 3 and rate >= (0.45 if w in gen else 0.25) and nd / max(ne, 1) <= 0.1:
            rows.append({"word": w, "count": c, "title_rate": round(rate, 2), "de_ratio": round(nd / max(ne, 1), 2)})
    rows.sort(key=lambda r: -r["count"])
    res = {"rule": build_names.__doc__, "n": N_NAMES, "names": [r["word"] for r in rows[:N_NAMES]], "all": rows}
    NAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    NAMES_FILE.write_text(json.dumps(res, indent=1))
    return res


def load_names() -> list[str]:
    return json.loads(NAMES_FILE.read_text())["names"]


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["names"]:
        r = build_names()
        print(r["names"])
        print([(x["word"], x["count"], x["title_rate"]) for x in r["all"]])
