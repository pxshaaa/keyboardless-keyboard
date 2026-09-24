"""Freeze v3 (results/llmdec/frozen_config_v3.json; v2 file untouched). personal_v3 = best tune3 config on the user-written
keyboard windows (tiebreak kbd CER, then old WER), around the frozen v2 personal camera weights; generic = v2 generic.
Design of the fuzzy generator was informed by blind-1 errors -> any blind-1 number for v3 is NOT blind."""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

RES = Path("results/llmdec")
OUTF = RES / "frozen_config_v3.json"


def main() -> int:
    if OUTF.exists():
        raise SystemExit(f"{OUTF} exists; refusing to overwrite")
    v2 = json.loads((RES / "frozen_config_v2.json").read_text())
    t3 = json.loads((RES / "tuning_v3.json").read_text())
    b = t3["best_personal_kbd"]
    run = {"method": "gv", "llm": "8b", "groups": "ens", "prompt": b["prompt"], "K": b["K"], "lam": b["lam"], "wb": b["wb"],
           "cb": b["cb"], "mu_lex": b["mu_lex"], "mu_pers": b["mu_pers"], "nu": b["nu"], "lam_p": b["lam_p"], "plm": "p05b",
           "use_fz": b["use_fz"], "kappa": b["kappa"]}
    v2p = v2["personal"]["tuning"]
    better = b["kbd"]["wer"] < v2p["kbd"]["wer"] - 1e-9
    cfg = {"frozen_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "tuned_on": v2["tuned_on"], "selection": "personal_v3 = argmin kbd WER over tune3 grid (tiebreak kbd CER, old WER)",
           "personal_v3": {"run": run, "tuning": {k: b[k] for k in ("kbd", "old", "obj")}},
           "personal_v3_no_fuzzy": t3["best_personal_kbd_no_fuzzy"], "best_joint": t3["best_joint"],
           "v2_personal": v2["personal"], "generic": v2["generic"],
           "oracle_tuning": t3["oracle"],
           "recommended_for_blind2": "personal_v3" if better else "v2 personal",
           "run": run if better else v2["personal"]["run"],
           "disclosure": ("fuzzy generator hand-set; its reachability examples (dog/div, cope/codex, enter/center, aice/claude) "
                          "came from blind-1 errors, so blind-1 is NOT blind for v3. Weights tuned on kbd + old desk only. "
                          "Jargon list = top personal words (decontaminated personal vocab, count>=20, not in generic lexicon)."),
           "fuzzy_cfg": json.loads((Path(".cache/llmdec/fuzzy") / "old.json").read_text())["cfg"],
           "code_md5": {n: hashlib.md5(Path(f"phase0/analysis/{n}").read_bytes()).hexdigest()
                        for n in ("llmdec.py", "llmdec_mlx.py", "llmdec_fuzzy.py", "decipher2.py")}}
    OUTF.write_text(json.dumps(cfg, indent=1, default=float))
    print("personal_v3", json.dumps(run), f"kbd {b['kbd']['wer']:.3f} old {b['old']['wer']:.3f} | v2 personal kbd {v2p['kbd']['wer']:.3f}")
    print("recommended:", cfg["recommended_for_blind2"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
