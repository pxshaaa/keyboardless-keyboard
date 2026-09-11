"""Score a taps.jsonl against keys.jsonl. The one metric set every detector is judged on.

Recall/precision use a proper 1:1 assignment within ±80 ms (analyze_drift.pair_events).
"Still FP" = taps fired >300 ms from any keydown -- the ones a human sees while resting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phase0.analysis.analyze_drift import pair_events

WINDOW_S = 0.080
STILL_S = 0.300


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def keydown_times(session: Path) -> np.ndarray:
    ks = load_jsonl(session / "keys.jsonl")
    return np.sort(np.array([k["t"] for k in ks if k["event"] == "down"], dtype=float))


def score(kt: np.ndarray, tt: np.ndarray, t_lo: float | None = None, t_hi: float | None = None) -> dict:
    """Restrict to [t_lo, t_hi) when given, so a hold-out slice can be scored honestly."""
    if t_lo is not None or t_hi is not None:
        lo = -np.inf if t_lo is None else t_lo
        hi = np.inf if t_hi is None else t_hi
        kt = kt[(kt >= lo) & (kt < hi)]
        tt = tt[(tt >= lo) & (tt < hi)]
    tt = np.sort(tt)
    pairs = pair_events(kt.tolist(), tt.tolist(), WINDOW_S) if len(kt) and len(tt) else []
    hit = len(pairs)
    errs = np.array([tt[j] - kt[i] for i, j in pairs]) if pairs else np.array([])
    still_fp = int(sum(np.min(np.abs(kt - t)) > STILL_S for t in tt)) if len(kt) and len(tt) else len(tt)
    recall = hit / len(kt) if len(kt) else 0.0
    prec = hit / len(tt) if len(tt) else 0.0
    f1 = 2 * recall * prec / (recall + prec) if (recall + prec) else 0.0
    return {
        "keydowns": int(len(kt)), "taps": int(len(tt)), "hits": hit,
        "recall": recall, "precision": prec, "f1": f1,
        "still_fp": still_fp, "still_fp_rate": still_fp / len(tt) if len(tt) else 0.0,
        "median_abs_err_ms": float(1000 * np.median(np.abs(errs))) if len(errs) else None,
        "mean_signed_err_ms": float(1000 * np.mean(errs)) if len(errs) else None,
    }


def fmt(name: str, s: dict) -> str:
    err = "-" if s["median_abs_err_ms"] is None else f'{s["median_abs_err_ms"]:.0f}ms'
    return (f"{name:<14} keys={s['keydowns']:<4} taps={s['taps']:<4} hits={s['hits']:<4} "
            f"R={100*s['recall']:5.1f}%  P={100*s['precision']:5.1f}%  F1={100*s['f1']:5.1f}%  "
            f"stillFP={s['still_fp']} ({100*s['still_fp_rate']:.0f}%)  err={err}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session", type=Path)
    ap.add_argument("--taps", type=Path, default=None, help="default: <session>/taps.jsonl")
    ap.add_argument("--split", type=float, default=0.6,
                    help="fraction of the session (by time) treated as train; the rest is hold-out")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--clip", action="store_true",
                    help="ignore taps outside [first key-0.1s, last key+0.1s] (hands arriving/leaving)")
    a = ap.parse_args(argv)
    kt = keydown_times(a.session)
    taps = load_jsonl(a.taps or a.session / "taps.jsonl")
    tt = np.array([t["t"] for t in taps], dtype=float)
    t0, t1 = kt.min(), kt.max()
    if a.clip:
        tt = tt[(tt >= t0 - 0.1) & (tt <= t1 + 0.1)]
    cut = t0 + a.split * (t1 - t0)
    out = {"all": score(kt, tt), "train": score(kt, tt, None, cut), "holdout": score(kt, tt, cut, None)}
    if a.json:
        print(json.dumps(out, indent=1))
    else:
        for k, v in out.items():
            print(fmt(k, v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
