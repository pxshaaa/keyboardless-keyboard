"""Aggregate results/contact/arm_*.json into one table with paired phrase-bootstrap deltas vs base.
Run: python -m phase0.analysis.touch_report  -> results/contact/table.md, table.json"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from phase0.analysis import touch_common as tc

RES = Path("results/contact")
HIRECALL_BEST = 0.448


def _mean_ci(v):
    v = np.asarray(v, float)
    if len(v) == 1:
        return f"{v[0]:.3f}"
    return f"{v.mean():.3f} (seeds {', '.join(f'{x:.3f}' for x in v)})"


def seed_avg_paired(base_rows, arm_rows, key, ratio=False):
    """Average the per-phrase metric over seeds for both arms, then paired phrase bootstrap."""
    if ratio:
        eb = np.mean([[x[0] for x in r[key]["per"]] for r in base_rows], 0)
        ea = np.mean([[x[0] for x in r[key]["per"]] for r in arm_rows], 0)
        n = np.array([x[1] for x in base_rows[0][key]["per"]], float)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(n), (10000, len(n)))
        d = (ea[idx].sum(1) - eb[idx].sum(1)) / n[idx].sum(1)
        return float((ea.sum() - eb.sum()) / n.sum()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    b = np.mean([r[key]["per"] for r in base_rows], 0)
    a = np.mean([r[key]["per"] for r in arm_rows], 0)
    return tc.boot_paired(b, a)


def main(argv=None) -> int:
    arms = {}
    for f in sorted(RES.glob("arm_*.json")):
        arms[f.stem[4:]] = json.loads(f.read_text())
    if "base" not in arms:
        print("no base arm yet")
        return 1
    base = arms["base"]
    lines = ["| arm | kbd held-out F1 (mean over seeds) | desk count err @1 tap/char | d vs base [95% CI] | "
             "oracle CER (LOPO-selected, noisy) | d vs base [95% CI] | oracle CER, density-matched @1.0+1.25 taps/char | d vs base [95% CI] | "
             "end-to-end CER | d vs base [95% CI] |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    table = {}
    for name, A in arms.items():
        seeds = sorted(set(A) & set(base))
        rows_a = [A[s] for s in seeds]
        rows_b = [base[s] for s in seeds]
        f1 = [r["kbd"]["f1"] for r in rows_a]
        ce = [r["count"]["count_err"] for r in rows_a]
        dce = seed_avg_paired(rows_b, rows_a, "count") if name != "base" else None
        o_seeds = [s for s in seeds if "oracle" in A[s] and "oracle" in base[s]]
        oc = [A[s]["oracle"]["oracle_cer"] for s in o_seeds]
        doc = seed_avg_paired([base[s] for s in o_seeds], [A[s] for s in o_seeds], "oracle", True) \
            if o_seeds and name != "base" else None
        d_seeds = [s for s in seeds if "oracle_dens" in A[s] and "oracle_dens" in base[s]]
        REACH = ("1.0", "1.25")  # densities every arm reaches; the self-trained model saturates by 1.5

        def pooled(r):
            return [(sum(r["oracle_dens"][d]["per"][j][0] for d in REACH),
                     sum(r["oracle_dens"][d]["per"][j][1] for d in REACH)) for j in range(20)]
        od = [sum(e for e, _ in pooled(A[s])) / sum(n for _, n in pooled(A[s])) for s in d_seeds]
        dod = None
        if d_seeds and name != "base":
            wrap = lambda rows: [{"x": {"per": pooled(r)}} for r in rows]
            dod = seed_avg_paired(wrap([base[s] for s in d_seeds]), wrap([A[s] for s in d_seeds]), "x", True)
        e_seeds = [s for s in seeds if "e2e" in A[s] and "e2e" in base[s]]
        ec = [A[s]["e2e"]["cer"] for s in e_seeds]
        dec = seed_avg_paired([base[s] for s in e_seeds], [A[s] for s in e_seeds], "e2e", True) \
            if e_seeds and name != "base" else None
        fmt = lambda d: "-" if d is None else f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"
        lines.append(f"| {name} | {100*np.mean(f1):.1f} ({', '.join(f'{100*x:.1f}' for x in f1)}) | "
                     f"{np.mean(ce):.3f} | {fmt(dce)} | {np.mean(oc) if oc else float('nan'):.3f} "
                     f"(n={len(oc)}) | {fmt(doc)} | {np.mean(od) if od else float('nan'):.3f} (n={len(od)}) | {fmt(dod)} | {np.mean(ec) if ec else float('nan'):.3f} (n={len(ec)}) | {fmt(dec)} |")
        table[name] = {"seeds": seeds, "kbd_f1": f1, "count_err": ce, "d_count": dce, "oracle": oc,
                       "d_oracle": doc, "oracle_dens": od, "d_oracle_dens": dod, "e2e": ec, "d_e2e": dec}
    txt = "\n".join(lines) + f"\n\nhirecall's published best end-to-end desk CER: {HIRECALL_BEST}\n"
    print(txt)
    (RES / "table.md").write_text(txt)
    (RES / "table.json").write_text(json.dumps(table, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
