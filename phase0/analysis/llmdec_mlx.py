"""MLX half of llmdec: LLM rewrites of CTC n-best lists (no user context) + LLM log-prob of every candidate.
mini: cd ~/cvt && PYTHONPATH=. nice -n 10 .cache/llmdec/venv/bin/python -m phase0.analysis.llmdec_mlx --set old --llm 3b"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

OUT = Path(".cache/llmdec")
FROZEN = Path("results/llmdec/frozen_config_v2.json")
REPOS = {"3b": "mlx-community/Qwen2.5-3B-Instruct-4bit", "8b": "mlx-community/Qwen3-8B-4bit"}
KMAX = 10

INTRO = ("A recognizer tried to read an English sentence that someone typed. It is unreliable: the guesses below contain "
         "wrong, missing or extra letters, and words can be merged or split.")
EXTRA = {"p1": "", "p2": " Letters that are neighbours on a QWERTY keyboard are often confused with each other. "
                        "Stay close to the letters in the guesses.",
         "p3": " Do not copy the guesses: every line you write must be a fluent English sentence with every word spelled "
               "correctly, reconstructed from the letters and the context.",
         "p4": " Do not copy the guesses: fix garbled words so each line is correct English. Example: guesses "
               "'plese sned me teh fiel', 'please sne me the fiel', 'plase send me th file' -> please send me the file"}


def norm(s: str) -> str:
    return " ".join("".join(c if "a" <= c <= "z" else (" " if c.isspace() or c == "-" else "") for c in s.lower()).split())


def guesses(c: dict, groups: list[str], n_char: int = 8) -> list[str]:
    out = [c[g]["qwen"] for g in groups]
    lists = [[t for t, _ in c[g]["char"]] for g in groups]
    for k in range(max(map(len, lists)) if lists else 0):
        for l in lists:
            if k < len(l):
                out.append(l[k])
    out = list(dict.fromkeys(x for x in out if x))
    return out[:n_char * len(groups) + len(groups)]


CTX_PROMPTS = {"p5": ("p3", True, False), "p6": ("p3", False, True), "p7": ("p4", True, True),
               "p8": ("p4", True, True, True)}   # base, profile, context[, personal jargon list]
JARGON_F = OUT / "personal" / "jargon.txt"
PROFILE_F = OUT / "personal" / "profile_summary.txt"


def prompt_text(tok, gs: list[str], p: str, prev: list[str] | None = None) -> str:
    base, use_prof, use_ctx, use_jar = (CTX_PROMPTS.get(p, (p, False, False)) + (False,))[:4]
    head = ""
    if use_jar and JARGON_F.exists():
        head += "Words this user commonly types: " + ", ".join(JARGON_F.read_text().split()) + "\n\n"
    if use_prof and PROFILE_F.exists():
        head += "About the writer: " + PROFILE_F.read_text().strip() + "\n\n"
    if use_ctx and prev:
        head += ("Lines the same person typed just before, in order (machine-read, may contain errors):\n"
                 + "\n".join(prev[-3:]) + "\n\n")
    body = (head + INTRO + EXTRA[base] + "\n\nGuesses (best first):\n" + "\n".join(f"{i + 1}. {g}" for i, g in enumerate(gs))
            + f"\n\nWrite the {KMAX} most likely sentences the person actually typed, most likely first. One per line, "
              "lowercase letters and spaces only, no numbering, no punctuation, no explanations.")
    return tok.apply_chat_template([{"role": "user", "content": body}], add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)


def parse(txt: str) -> list[str]:
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    out = []
    for l in txt.splitlines():
        l = norm(re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", l))
        if l and l not in out:
            out.append(l)
    return out[:KMAX]


def lm_scores(model, tok, texts: list[str], bs: int = 4) -> dict:
    import mlx.core as mx
    nl = tok.encode("\n")
    seqs = {t: nl + tok.encode(t) + nl for t in texts}
    order = sorted(texts, key=lambda t: len(seqs[t]))
    out = {}
    for i in range(0, len(order), bs):
        b = order[i:i + bs]
        L = max(len(seqs[t]) for t in b)
        arr = [seqs[t] + [0] * (L - len(seqs[t])) for t in b]
        x = mx.array(arr)
        logits = model(x).astype(mx.float32)
        lp = logits[:, :-1, :] - mx.logsumexp(logits[:, :-1, :], axis=-1, keepdims=True)
        g = mx.take_along_axis(lp, x[:, 1:, None], axis=-1).squeeze(-1)
        mask = mx.array([[1.0 if j < len(seqs[t]) - 1 else 0.0 for j in range(L - 1)] for t in b])
        s = (g * mask).sum(axis=1)
        mx.eval(s)
        for t, v in zip(b, s.tolist()):
            out[t] = [float(v), len(seqs[t]) - 1]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="kbd | old | new | s_<session id>")
    ap.add_argument("--llm", required=True, help="3b | 8b | any key with --base (e.g. p05b for the personal LoRA)")
    ap.add_argument("--base", default=None, help="model repo/path overriding REPOS[llm]")
    ap.add_argument("--adapter", default=None, help="mlx_lm LoRA adapter dir")
    ap.add_argument("--lm-only", action="store_true", help="no generation: LM log-prob of the base pool + --pool-from rewrites")
    ap.add_argument("--pool-from", default="8b", help="with --lm-only: also score texts generated by this llm key")
    ap.add_argument("--ctx-prompts", action="store_true", help="also generate p5-p7 (profile / cross-phrase context)")
    ap.add_argument("--prompts", default="", help="restrict generation to these prompts (frozen pipeline)")
    ap.add_argument("--gks", default="", help="restrict generation to these posterior groups, e.g. ens (frozen pipeline)")
    a = ap.parse_args(argv)
    if a.set not in ("kbd", "old") and not FROZEN.exists():
        sys.exit("freeze the config before touching a new session")
    from huggingface_hub import snapshot_download
    from mlx_lm import generate, load
    from phase0.analysis import mlxsafe
    mlxsafe.cap()
    d = json.loads((OUT / "sets" / f"{a.set}.json").read_text())
    f = OUT / "llm" / a.llm / f"{a.set}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    src = a.base or REPOS[a.llm]
    res = json.loads(f.read_text()) if f.exists() else {"repo": src, "adapter": a.adapter, "items": {}}
    t0 = time.time()
    path = src if Path(src).exists() else snapshot_download(src, local_files_only=True)
    model, tok = load(path, adapter_path=a.adapter) if a.adapter else load(path)
    res["load_s"] = time.time() - t0
    if a.lm_only:
        other = OUT / "llm" / a.pool_from / f"{a.set}.json"
        G = json.loads(other.read_text())["items"] if other.exists() else {}
        tag = os.environ.get("LLMDEC_FUZZY_TAG", "")
        fzf = OUT / "fuzzy" / f"{a.set}{'__' + tag if tag else ''}.json"
        FZI = json.loads(fzf.read_text())["items"] if fzf.exists() else {}
        for it in d["items"]:
            r = res["items"].setdefault(it["id"], {"gens": {}, "secs": {"gen": {}}, "lm": {}})
            texts = set()
            for g in it["c"]:
                texts |= {t for t, _ in it["c"][g]["char"]} | {it["c"][g]["greedy"], it["c"][g]["qwen"]}
            for v in G.get(it["id"], {}).get("gens", {}).values():
                for x in v.values():
                    texts |= set(x)
            for x in FZI.get(it["id"], {}).values():
                texts |= set(x)
            texts = sorted(t for t in texts if t and t not in r["lm"])
            t1 = time.time()
            r["lm"].update(lm_scores(model, tok, texts) if texts else {})
            r["secs"]["lm"] = r["secs"].get("lm", 0.0) + time.time() - t1
            r["secs"]["n_lm"] = r["secs"].get("n_lm", 0) + len(texts)
            print(a.set, a.llm, it["id"], f"lm {len(texts)} texts", flush=True)
        f.write_text(json.dumps(res, indent=1))
        return 0
    PROMPTS = list(EXTRA) + (list(CTX_PROMPTS) if a.ctx_prompts else [])
    if a.prompts:
        want = a.prompts.split(",")
        if any(CTX_PROMPTS.get(x, (x, False, False))[2] for x in want) and "p3" not in want:   # p6/p7/p8 read p3
            want = ["p3"] + want   # context prompts read the previous items' p3 rewrite
        PROMPTS = [x for x in PROMPTS if x in want]
    prev_by_fold: dict = {}
    for it in d["items"]:
        fold = it["id"].rsplit("_", 1)[0] if a.set == "kbd" else "sess"
        r = res["items"].get(it["id"])
        prev = prev_by_fold.setdefault(fold, [])
        if r and all(p in r["gens"] for p in PROMPTS):
            g0 = list(it["c"])[0]
            prev.append((r["gens"].get("p3", {}).get(g0) or [it["c"][g0]["qwen"]])[0])
            continue
        r = r or {"gens": {}, "raw": {}, "prompts": {}, "secs": {"gen": {}}, "lm": {}}
        groups = list(it["c"])
        gks = groups + (["ens"] if len(groups) > 1 else [])
        if a.gks:
            gks = [g for g in gks if g in a.gks.split(",")]
        for p in PROMPTS:
            if p in r["gens"]:
                continue
            r["gens"][p], r["raw"][p], r["secs"]["gen"][p] = {}, {}, {}
            for gk in gks:
                gs = guesses(it["c"], groups if gk == "ens" else [gk])
                pr = prompt_text(tok, gs, p, prev)
                t0 = time.time()
                txt = generate(model, tok, prompt=pr, max_tokens=24 * KMAX, verbose=False)
                r["secs"]["gen"][p][gk] = time.time() - t0
                r["raw"][p][gk], r["gens"][p][gk] = txt, parse(txt)
                r["prompts"].setdefault(gk, gs)
        texts = set()
        for g in groups:
            texts |= {t for t, _ in it["c"][g]["char"]} | {it["c"][g]["greedy"], it["c"][g]["qwen"]}
        for p in r["gens"]:
            for v in r["gens"][p].values():
                texts |= set(v)
        texts = sorted(t for t in texts if t and t not in r["lm"])
        t0 = time.time()
        r["lm"].update(lm_scores(model, tok, texts) if texts else {})
        r["secs"]["lm"] = r["secs"].get("lm", 0.0) + time.time() - t0
        r["secs"]["n_lm"] = r["secs"].get("n_lm", 0) + len(texts)
        res["items"][it["id"]] = r
        g0 = groups[0]
        prev.append((r["gens"].get("p3", {}).get(g0) or [it["c"][g0]["qwen"]])[0])
        print(a.set, a.llm, it["id"], f"gen {sum(v for x in r['secs']['gen'].values() for v in x.values()):.1f}s "
              f"lm {r['secs']['lm']:.1f}s/{len(texts)}", next(iter(r["gens"].values()), {}).get(groups[0], [])[:2], flush=True)
        f.write_text(json.dumps(res, indent=1))
    f.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
