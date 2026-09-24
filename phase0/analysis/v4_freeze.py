"""Freeze decoder v4 from results/v4/dev_eval.json (dev = old desk CV-safe + blind-1 + blind-2; nothing held out).
  PYTHONPATH=. .venv/bin/python -m phase0.analysis.v4_freeze freeze
  PYTHONPATH=. .venv/bin/python -m phase0.analysis.v4_freeze summary [--dry <twin session dir>]"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

OUT = Path("results/v4")
FROZEN4 = Path("results/llmdec/frozen_config_v4.json")
ENFORCED = [  # decoder side: decipher2_v4 refuses to run if any of these changed
    "phase0/analysis/decipher2_ctc3.py", "phase0/analysis/decipher2_v4.py", "phase0/analysis/wordbeam_v4.py",
    "phase0/analysis/v4_suggest.py", "phase0/analysis/predict_manifest.py", "phase0/analysis/decipher2.py",
    "phase0/analysis/llmdec.py", "phase0/analysis/llmdec_mlx.py", "phase0/analysis/llmdec_fuzzy.py", "phase0/analysis/llmdec2.py",
    "phase0/analysis/mlxsafe.py", "phase0/analysis/decipher.py", "phase0/analysis/seqctc2.py",
    "results/llmdec/frozen_config_v2.json", "results/llmdec/frozen_config_v3.json", "results/v4/names_quicklist.json",
    ".cache/personal/lexicon_merged.json", ".cache/personal/lexicon_personal.json", ".cache/personal/bigram_personal.npz",
    ".cache/autocorrect/lexicon.json", "models/vocab_blocklist.txt", ".cache/llmdec/personal/vocab.json"]
RECORDED = [  # CTC stage + scoring helpers: recorded, not enforced (a retrained manifest may legitimately change them)
    ".cache/ctc_v3/predict.py", ".cache/ctc_v3/deploy/manifest.json", "phase0/analysis/ctcv3_eval.py",
    "phase0/analysis/swipe_common.py", "phase0/analysis/v4_dev.py", "phase0/analysis/v4_freeze.py"]


def md5(p) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def choose(ev: dict) -> tuple[str, float]:
    P = ev["pooled"]
    temps = [str(t) for t in ev["temps"]]
    # T: best pooled <=1-tap top-3 words on the frozen v3 pool AND the v4 pools (mean), tiebreak lower truth-slot NLL
    def t_key(T):
        return (sum(P[k]["by_T"][T]["top3_words"] for k in P) / len(P), -sum(P[k]["by_T"][T]["nll"] for k in P) / len(P))
    T = max(temps, key=t_key)
    # word-beam config: fewest pooled word errors (automatic); tiebreak <=1-tap top-3 words at T; then smaller pool
    tag = min((k for k in P if k != "v3"), key=lambda k: (P[k]["word_errors"], -P[k]["by_T"][T]["top3_words"], P[k]["mean_pool"]))
    return tag, float(T)


def freeze() -> int:
    from phase0.analysis import v4_suggest as V
    from phase0.analysis import wordbeam_v4 as W
    ev = json.loads((OUT / "dev_eval.json").read_text())
    tag, T = choose(ev)
    cfgs = {"v4s": ["small"], "v4b": ["big"], "v4u": ["small", "big"]}[tag]
    v3 = json.loads(Path("results/llmdec/frozen_config_v3.json").read_text())
    cfg = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "version": "v4",
        "dev_sets_declared": ["old desk 20260910-202149-desk, CV-safe out-of-fold v3 posteriors (s_zz-v3cv-ftkf5-olddesk)",
                              "blind-1 20260912-174542-desk (s_zz-v3b1-d20zs-blind1)", "blind-2 20260913-125733-desk (s_zz-ctc3-20260913-125733-desk)"],
        "tuned_on_dev": {"wordbeam_configs": "chosen among small / big / small+big (fewest pooled word errors, tiebreak <=1-tap top-3)",
                         "suggest_T": "chosen among " + str(ev["temps"]) + " (best pooled <=1-tap top-3 words, tiebreak truth-slot NLL)"},
        "not_tuned": "selection weights = frozen v3 personal (results/llmdec/frozen_config_v3.json 'run'); word-beam parameters = swipe_p1 "
                     "configs as probed; fuzzy generator = v3; names rule hand-set",
        "run": v3["run"],
        "base_frozen_config": {"path": "results/llmdec/frozen_config_v3.json", "md5": md5("results/llmdec/frozen_config_v3.json")},
        "wordbeam": {"configs": cfgs, "params": {c: dict(W.SEARCH, **W.CONFIGS[c]) for c in cfgs}, "topk_per_config": W.SEARCH["topk"],
                     "lexicon": "lexicon_merged.json (generic + personal, blocklist applied), boost = personal count>=10 non-generic + names seed"},
        "suggest": {"T": T, "k": 5, "n_names": V.N_NAMES, "max_cands": V.MAX_CANDS,
                    "posterior": "softmax(frozen v3 selection score / T) over eligible pool candidates, marginalised per aligned word slot"},
        "escape_path": "char n-best, greedy, Qwen-0.5B and 8B rewrites, fuzzy candidates stay in the pool: out-of-lexicon words remain reachable",
        "ctc_default_manifest": {"path": ".cache/ctc_v3/deploy/manifest.json", "md5": md5(".cache/ctc_v3/deploy/manifest.json"),
                                 "override": "--manifest <path> (predict_manifest.py wrapper; twin session zz-ctcm<md5[:8]>-<id>)"},
        "dev_results": {k: {"word_errors": v["word_errors"], "wer": v["wer"], "top3_words_at_T": v["by_T"][str(T)]["top3_words"],
                            "top5_words_at_T": v["by_T"][str(T)]["top5_words"]} for k, v in ev["pooled"].items()},
        "disclosure": "All three desk sets are dev: the word-beam configs were inspected on all of them in the swipe study, the names seed "
                      "(claude/bifi/codex) came from blind-1/2 errors, T and the config were chosen on them here. v4 numbers on these sets are "
                      "NOT blind estimates; a fresh sealed blind-3 is required for an unbiased number.",
        "md5": {f: md5(f) for f in ENFORCED},
        "md5_recorded_not_enforced": {f: md5(f) for f in RECORDED},
    }
    FROZEN4.write_text(json.dumps(cfg, indent=1))
    print("frozen", FROZEN4, "wordbeam", cfgs, "T", T, "md5", md5(FROZEN4))
    return 0


def summary(dry: str | None) -> int:
    ev = json.loads((OUT / "dev_eval.json").read_text())
    cfg = json.loads(FROZEN4.read_text())
    tag = {("small",): "v4s", ("big",): "v4b", ("small", "big"): "v4u"}[tuple(cfg["wordbeam"]["configs"])]
    T = str(cfg["suggest"]["T"])
    names = {"old": "old desk (CV-safe)", "b1": "blind-1", "b2": "blind-2"}
    rows = {}
    for s, label in names.items():
        rows[label] = {}
        for ver, k in (("v3", "v3"), ("v4", tag)):
            r = ev["sets"][s][k]
            b = r["by_T"][T]
            rows[label][ver] = {"n_words": r["n_words"], "official_wer": r["official_wer"], "official_words_correct": r["official_words"],
                                "auto_words": b["auto_words"],
                                "le1_tap_top3_words": b["top3"]["words_after_le1_tap"], "le1_tap_top5_words": b["top5"]["words_after_le1_tap"],
                                "le1_tap_top5_plus_names_words": b["top5_names"]["words_after_le1_tap"],
                                "taps_per_100_words_top3": b["top3"]["taps_per_100_words"], "taps_per_100_words_top5": b["top5"]["taps_per_100_words"],
                                "wer_after_taps_top3": b["top3"]["wer_after"], "wer_after_taps_top5": b["top5"]["wer_after"],
                                "mean_pool": r["mean_pool"]}
    out = {"frozen_config": str(FROZEN4), "frozen_config_md5": md5(FROZEN4), "wordbeam": cfg["wordbeam"]["configs"], "T": cfg["suggest"]["T"],
           "command": "PYTHONPATH=. .venv/bin/python -m phase0.analysis.decipher2_ctc3 data/sessions/<id> --version v4 [--manifest <path>] [--score <truth.txt>]",
           "metric_definitions": {
               "auto_words": "truth words matched by the chosen text (per-segment truth assignment, as swipe_p0)",
               "official_words_correct": "decipher.score_variant over concatenated lines",
               "le1_tap_topk_words": "truth words correct after at most one tap per wrong word slot: the slot's truth span (incl. attached "
                                     "missing words, or delete) is among the slot's top-k alternatives (displayed word excluded)",
               "taps_per_100_words": "successful one-tap corrections per 100 truth words"},
           "dev_declared": cfg["dev_sets_declared"], "disclosure": cfg["disclosure"], "sets": rows,
           "pooled": {k: ev["pooled"][k] for k in ("v3", tag)}}
    if dry:
        d = json.loads((Path(dry) / "decipher2_v4.json").read_text())
        out["dry_run"] = {"session": d["session"], "video_s": d["video_s"], "timing_s": d["timing_s"],
                          "score": {k: d.get("score", {}).get(k) for k in ("wer", "words_correct", "suggestion_bar")}}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(rows, indent=1, default=float)[:4000])
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "freeze":
        raise SystemExit(freeze())
    dry = sys.argv[sys.argv.index("--dry") + 1] if "--dry" in sys.argv else None
    raise SystemExit(summary(dry))
