"""Missed and extra taps per arm on the held-out kbd session at matched densities (desk has no tap truth).
Run: python -m phase0.analysis.touch_missextra ARM [ARM ...]  -> results/contact/missextra.json"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from phase0.analysis import touch_common as tc
from phase0.analysis.analyze_drift import pair_events

DENS = (1.0, 1.3, 2.0)


def at_density(p, gate, t, kt, d):
    lo_t, hi_t = kt.min() - 0.1, kt.max() + 0.1
    best = None
    for th in np.geomspace(0.005, 0.95, 160):
        ev = tc.extract(p, gate, th)
        tt = t[ev]
        tt = tt[(tt >= lo_t) & (tt <= hi_t)]
        if best is None or abs(len(tt) - d * len(kt)) < abs(len(best) - d * len(kt)):
            best = tt
    hits = len(pair_events(kt.tolist(), np.sort(best).tolist(), 0.08))
    return {"miss_rate": 1 - hits / len(kt), "extra_per_char": (len(best) - hits) / len(kt),
            "taps_per_char": len(best) / len(kt)}


def main(argv=None) -> int:
    arms = (argv or sys.argv[1:])
    src = Path(sys.argv[0]).parent if False else tc.CACHE
    t, kt = tc.frames(tc.KBD_TEST)[1], tc.kt(tc.KBD_TEST)
    out_p = Path("results/contact/missextra.json")
    out = json.loads(out_p.read_text()) if out_p.exists() else {}
    for arm in arms:
        rows = {str(d): [] for d in DENS}
        for s in (0, 1, 2):
            f = src / f"kbdtest_probs_{arm}_s{s}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            for d in DENS:
                rows[str(d)].append(at_density(z["p"], z["gate"], t, kt, d))
        out[arm] = {d: {k: float(np.mean([r[k] for r in v])) for k in v[0]} for d, v in rows.items() if v}
        print(arm, " ".join(f"@{d}: miss={v['miss_rate']:.3f} extra/char={v['extra_per_char']:.3f}"
                             for d, v in out[arm].items()), flush=True)
    out_p.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
