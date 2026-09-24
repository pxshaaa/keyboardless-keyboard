"""Aggregate llmdec v2 results into results/llmdec/report_v2.json + a printed table (tuning sets, blind-1, runtime)."""

from __future__ import annotations

import json
from pathlib import Path

RES = Path("results/llmdec")


def load(name):
    f = RES / name
    return json.loads(f.read_text()) if f.exists() else None


def wc(x):
    return None if x is None else round(1 - x, 3)


def main() -> int:
    rows = []

    def add(label, kbd=None, old=None, note="", s=None):
        rows.append({"variant": label, "kbd_wer": kbd and kbd.get("wer"), "kbd_cer": kbd and kbd.get("cer"),
                     "old_wer": old and old.get("wer"), "old_cer": old and old.get("cer"), "s_per_phrase": s, "note": note})

    t = load("tuning.json")
    b = t["baselines"]
    add("CTC char 6-gram top-1", b["char1"]["kbd"], b["char1"]["old"])
    add("current Qwen-0.5B closed-lexicon word decoder", b["qwen05"]["kbd"], b["qwen05"]["old"], s=1.5)
    add("gen-verify 3B (v1 best)", t["best"]["3b"]["kbd"], t["best"]["3b"]["old"], "p4,K10,lam1.5,wb-2", s=15)
    add("gen-verify 8B (v1 best)", t["best"]["8b"]["kbd"], t["best"]["8b"]["old"], "p3,K5,lam2,cb2", s=60)
    w = load("wordlex_personal.json")
    add("Qwen-0.5B + personal lexicon", {k: w["kbd"]["generic+personal"][k] for k in ("wer", "cer")},
        {k: w["old"]["generic+personal"][k] for k in ("wer", "cer")})
    wl = load("wordlm_personal.json")
    add("Qwen-0.5B + personal word LM (a=0.2, tuned on kbd)", wl["kbd"]["personal_bigram_alpha0.2"], wl["old"]["personal_bigram_alpha0.2"])
    for f, lab in (("dec2_C_3b.json", "constrained 3B"), ("dec2_C2_3b.json", "constrained 3B high-lam")):
        d = load(f)
        k, v = min(((k, v) for k, v in d.items() if isinstance(v, dict) and "obj" in v), key=lambda kv: kv[1]["obj"])
        add(f"{lab} best", v["kbd"], v["old"], k, s=v["old"]["s_per_item"])
    d = load("dec2_D_3b.json")
    for k, v in d.items():
        if isinstance(v, dict) and "kbd" in v and "lam=3.0" in k and "mu_lex=3.0" in k and "cb=1.0" in k:
            add(f"constrained 3B ablation: {k}", v.get("kbd"), v.get("old") if "wer" in v.get("old", {}) else None)
    t2 = load("tuning_v2_8b.json")
    for sl, v in t2["slices"].items():
        add(f"gen-verify 8B v2 {sl}", v["kbd"], v["old"], ",".join(f"{x}={v[x]}" for x in ("prompt", "K", "lam", "wb", "cb")))
    fz = load("frozen_config_v2.json")
    for k in ("generic", "personal"):
        add(f"FROZEN {k}", fz[k]["tuning"]["kbd"], fz[k]["tuning"]["old"], json.dumps(fz[k]["run"]))
    t3 = load("tuning_v3.json")
    if t3:
        for k in ("best_personal_kbd_no_fuzzy", "best_personal_kbd", "best_joint"):
            v = t3[k]
            add(f"v3 {k} (fuzzy={v['use_fz']}, prompt={v['prompt']})", v["kbd"], v["old"],
                ",".join(f"{x}={v[x]}" for x in ("use_fz", "kappa", "lam", "cb", "lam_p", "mu_pers")))
    f3 = load("frozen_config_v3.json")
    if f3:
        add("FROZEN personal_v3", f3["personal_v3"]["tuning"]["kbd"], f3["personal_v3"]["tuning"]["old"], json.dumps(f3["personal_v3"]["run"]))
    out = {"tuning_rows": rows, "oracle": {"gv_pool": load("oracle_gv_pool.json"), "constrained_3b": load("oracle_constrained_3b_old.json"),
                                           "blind1_pool_v2": load("oracle_blind1_pool.json"), "tuning_v3": t3 and t3["oracle"]},
           "frozen_v3": f3,
           "frozen": fz, "status": load("status_v2.json")}
    bl = load("blind1_score_v2.json")
    if bl:
        out["blind1"] = {k: {"wer": v["wer"], "cer": v["cer"], "words_correct": v["words_correct"], "hyps": v["hyps"]}
                         for k, v in bl["variants"].items()}
        out["blind1_meta"] = {"scored_at": bl["scored_at"], "disclosure": bl["disclosure"], "frozen_md5": bl["frozen_config_md5"]}
    b3 = load("blind1_score_v3.json")
    if b3:
        out["blind1_v3_NOT_BLIND"] = {k: {"wer": v["wer"], "cer": v["cer"], "words_correct": v["words_correct"], "hyps": v["hyps"]}
                                      for k, v in b3["variants"].items()}
        out["blind1_v3_oracle"] = b3.get("oracle")
    (RES / "report_v2.json").write_text(json.dumps(out, indent=1, default=float))
    f = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    print(f"{'variant':<72} kbd WER/CER    old WER/CER")
    for r in rows:
        print(f"{r['variant'][:72]:<72} {f(r['kbd_wer'])}/{f(r['kbd_cer'])}  {f(r['old_wer'])}/{f(r['old_cer'])}")
    if b3:
        print("\nblind-1 v3 (NOT blind: generator designed after blind-1 errors):")
        for k, v in out["blind1_v3_NOT_BLIND"].items():
            print(f"{k[:72]:<72} WER {v['wer'][0]:.3f} CER {v['cer'][0]:.3f} words {100 * v['words_correct'][0]:.0f}%")
        print("oracle", out["blind1_v3_oracle"])
    if bl:
        print("\nblind-1 (scored once, frozen):")
        for k, v in out["blind1"].items():
            print(f"{k[:72]:<72} WER {v['wer'][0]:.3f} CER {v['cer'][0]:.3f} words {100 * v['words_correct'][0]:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
