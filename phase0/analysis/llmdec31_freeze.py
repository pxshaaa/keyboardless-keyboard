"""Freeze v3.1 = frozen v3 personal + general developer vocabulary in the fuzzy lexicon + tuned tech-vocab bonus mu_tech.
Writes results/llmdec/frozen_config_v3_1.json (v2/v3 untouched). mu_tech chosen on mean(kbd WER, old WER) over tuning sets."""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

RES = Path("results/llmdec")
OUTF = RES / "frozen_config_v3_1.json"


def main() -> int:
    if OUTF.exists():
        raise SystemExit(f"{OUTF} exists; refusing to overwrite")
    v3 = json.loads((RES / "frozen_config_v3.json").read_text())
    t = json.loads((RES / "tuning_v3_1.json").read_text())
    b = t["best"]
    run = {**v3["personal_v3"]["run"], "fuzzy_tag": "v31", "mu_tech": b["mu_tech"]}
    base = v3["personal_v3"]["tuning"]
    tv = json.loads(Path(".cache/llmdec/techvocab/tech_vocab.json").read_text())
    cfg = {"frozen_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "personal_v3_1": {"run": run, "tuning": {k: b[k] for k in ("kbd", "old", "obj")}},
           "personal_v3": v3["personal_v3"], "generic": v3["generic"], "run": run,
           "tech_vocab": {k: tv[k] for k in ("sources", "n_terms", "rules")},
           "tuning_file": "results/llmdec/tuning_v3_1.json",
           "delta_vs_v3": {"kbd_wer": b["kbd"]["wer"] - base["kbd"]["wer"], "old_wer": b["old"]["wer"] - base["old"]["wer"]},
           "disclosure": "tech vocab from public sources, not from blind-1 errors; only mu_tech tuned (kbd + old desk)",
           "code_md5": {n: hashlib.md5(Path(f"phase0/analysis/{n}").read_bytes()).hexdigest()
                        for n in ("llmdec.py", "llmdec_mlx.py", "llmdec_fuzzy.py", "decipher2.py")}}
    OUTF.write_text(json.dumps(cfg, indent=1, default=float))
    print("v3.1", json.dumps(run), f"kbd {b['kbd']['wer']:.3f} old {b['old']['wer']:.3f} | v3 kbd {base['kbd']['wer']:.3f} old {base['old']['wer']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
