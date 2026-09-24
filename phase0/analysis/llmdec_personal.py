"""Personal vocab + profile for llmdec2 from .cache/personal (local only, never committed) -> .cache/llmdec/personal/.
Drops the keyboard sessions' own text and any sentence sharing a word 4-gram with a tuning/test reference (no leakage)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

P = Path(".cache/personal")
OUT = Path(".cache/llmdec/personal")


def grams(ws, n=4):
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def main() -> int:
    from phase0.analysis.llmdec import norm
    leak = set()
    kbd = json.loads(Path(".cache/llmdec/sets/kbd.json").read_text())
    texts = [it["ref"] for it in kbd["items"]]
    texts += [l for l in Path("phase0/phrases.txt").read_text().splitlines()]
    texts += [l for l in Path("data/sessions/20260912-174542-desk/truth.txt").read_text().splitlines()]
    for t in texts:
        leak |= grams(norm(t).split())
    uni, bi = Counter(), Counter()
    st = Counter()
    for l in open(P / "sentences.jsonl"):
        r = json.loads(l)
        if r["src"] == "keys":
            st["drop_keys"] += 1
            continue
        ws = norm(r["proj"]).split()
        if grams(ws) & leak:
            st["drop_leak4gram"] += 1
            continue
        st["kept"] += 1
        uni.update(ws)
        bi.update(f"{a} {b}" for a, b in zip(ws, ws[1:]))
    ctx = Counter()
    types = Counter()
    bi = {k: v for k, v in bi.items() if v >= 2}
    for k, v in bi.items():
        a = k.split()[0]
        ctx[a] += v
        types[a] += 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vocab.json").write_text(json.dumps({"unigram": dict(uni), "bigram": bi,
                                                "ctx": {a: [ctx[a], types[a]] for a in ctx}, "stats": dict(st)}))
    prof = json.loads((P / "profile.json").read_text())
    langs = prof["languages"]["words_pct_all"]
    about = prof.get("about", "a software developer")  # one-line self-description, kept in the private profile.json
    summary = (f"{about}; writes mostly English ({langs['en']:.0f}%) with some German ({langs['de']:.0f}%), "
               f"informal and lowercase, often instructions to AI coding agents.")
    (OUT / "profile_summary.txt").write_text(summary + "\n")
    print(dict(st), "vocab", len(uni), "bigrams>=2", len(bi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
