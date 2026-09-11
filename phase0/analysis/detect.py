"""One entry point for tap detection: run the best available detector for a session.

Exists so the Phase 0 pipeline and PROTOCOL.md have a single command that always uses the
current best model, and so `analyze_drift` (which reads taps.jsonl) gets the good detector
rather than the weak kinematic one. Measured on a held-out session: gb F1 76.3% vs
detect_taps F1 43.4%.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MODEL = Path("models/taps_gb.joblib")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("--detector", choices=["gb", "kinematic"], default="gb")
    ap.add_argument("--keep-existing", action="store_true",
                    help="do not overwrite an existing taps.jsonl")
    a = ap.parse_args(argv)

    dest = a.session / "taps.jsonl"
    if dest.exists() and a.keep_existing:
        print(f"{dest} exists; leaving it alone")
        return 0

    if a.detector == "gb":
        if not MODEL.exists():
            print(f"ERROR: {MODEL} missing. Train it first:\n"
                  "  python -m phase0.analysis.taps_gb train --sessions <s1> <s2>",
                  file=sys.stderr)
            return 2
        rc = subprocess.call([sys.executable, "-m", "phase0.analysis.taps_gb",
                              "predict", str(a.session)])
        if rc != 0:
            return rc
        src = a.session / "taps_gb.jsonl"
    else:
        rc = subprocess.call([sys.executable, "-m", "phase0.analysis.detect_taps",
                              str(a.session)])
        return rc  # detect_taps already writes taps.jsonl

    if not src.exists():
        print(f"ERROR: {src} was not produced", file=sys.stderr)
        return 2
    shutil.copyfile(src, dest)
    n = sum(1 for _ in open(dest))
    print(f"{a.detector}: {n} taps -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
