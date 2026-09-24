"""Collect results/normalize/*.json into one markdown numbers table (results/normalize/numbers.md).
Run: python -m phase0.analysis.normalize_report"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RES = Path("results/normalize")


def _load(name: str):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def _ci(v) -> str:
    return f"[{v[0]:.3f}, {v[1]:.3f}]" if v else ""


def _delta(v) -> str:
    return f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" if v else ""


def section_diag(out: list[str]) -> None:
    d = _load("diag_shift.json")
    if not d:
        return
    out += ["## Label-free frame quality (diagnostic)", "",
            "Per-key oracle-finger contact centroids, day-1 sessions vs 20260911-164237, in key pitches.", "",
            "| frame | keys | median shift | after best translation |", "|---|---|---|---|"]
    out += [f"| {r['frame']} | {r['keys']} | {r['median_shift_pitch']:.2f} | {r['median_resid_pitch']:.2f} |" for r in d]
    out.append("")


def section_loso(out: list[str]) -> None:
    d = _load("loso_frames.json")
    if not d:
        return
    out += ["## Frames: cross-day and LOSO per-key top-1", "",
            "x-day = train on the three 2026-09-10 sessions, test on 20260911-164237 (camera re-mounted).", "",
            "| frame | model | x-day top-1 [95% CI] | seeds | LOSO4 | held 015948 | d x-day vs abs | d LOSO4 vs abs |",
            "|---|---|---|---|---|---|---|---|"]
    for r in d:
        out.append(f"| {r['frame']} | {r['model']} | {r['new']:.3f} {_ci(r['new_ci'])} | {len(r['new_seeds'])} | "
                   f"{r['loso4']:.3f} | {r['held']:.3f} | {_delta(r.get('d_new_vs_abs'))} | "
                   f"{_delta(r.get('d_loso4_vs_abs'))} |")
    out.append("")


def section_vs_coral(out: list[str]) -> None:
    d = _load("tap_g_vs_coral.json")
    if not d:
        return
    out += ["## tap_g vs label-free CORAL (paired, 3 seeds)", "",
            "| model | x-day 164237 | LOSO4 |", "|---|---|---|"]
    out += [f"| {m} | {_delta(d[m]['new_164237'])} | {_delta(d[m]['loso4'])} |" for m in ("pose", "fused")]
    out.append("")


def section_enrol(out: list[str]) -> None:
    rows = []
    for p in sorted(RES.glob("enrol_*.json")):
        rows += [dict(r, file=p.stem) for r in json.loads(p.read_text())]
    if not rows:
        return
    out += ["## Enrolment (first N labelled taps of the new setup; scored on taps after #200)", "",
            "| source | frame | method | N | pose | fused [95% CI] | d fused vs N=0 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['file']} | {r['frame']} | {r['method']} | {r['n']} | {r['pose']:.3f} | "
                   f"{r['fused']:.3f} {_ci(r['fused_ci'])} | {_delta(r['d_vs_n0'])} |")
    out.append("")


def section_warmup(out: list[str]) -> None:
    for p in sorted(RES.glob("warmup_*.json")):
        d = json.loads(p.read_text())
        rows = [r for r in d if "fused_windows" in r]
        if not rows:
            continue
        out += [f"## Warm-up: unlabelled detected taps before the frame locks in ({p.stem})", "",
                "| frame | detected taps | windows | fused mean | min | max |", "|---|---|---|---|---|---|"]
        for r in rows:
            w = r["fused_windows"]
            out.append(f"| {r['frame']} | {r['m']} | {len(w)} | {np.mean(w):.3f} | {np.min(w):.3f} | {np.max(w):.3f} |")
        out.append("")


def section_perturb(out: list[str]) -> None:
    for p in sorted(RES.glob("perturb_*.json")):
        d = json.loads(p.read_text())
        cols = [k for k in d[0] if k != "perturb"]
        out += [f"## Synthetic geometric shift, pose-only top-1 ({p.stem})", "",
                "| perturbation | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        out += ["| " + r["perturb"] + " | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |" for r in d]
        out.append("")


def section_augment(out: list[str]) -> None:
    for p in sorted(RES.glob("augment_*.json")):
        d = json.loads(p.read_text())
        out += [f"## Hand-size augmentation, tap_g frame, on 20260911-164237 ({p.stem})", "",
                "| hand scale | model | seeds | pose | fused |", "|---|---|---|---|---|"]
        for h in sorted({r["hand"] for r in d}):
            for m in sorted({r["model"] for r in d}):
                rr = [r for r in d if r["hand"] == h and r["model"] == m]
                out.append(f"| x{h} | {m} | {len(rr)} | {np.mean([r['pose'] for r in rr]):.3f} | "
                           f"{np.mean([r['fused'] for r in rr]):.3f} |")
        out.append("")


def section_selftrain(out: list[str]) -> None:
    for p in sorted(RES.glob("selftrain_*.json")):
        d = json.loads(p.read_text())
        keys = sorted({(r["frame"], r["method"], r["round"]) for r in d},
                      key=lambda k: (k[0], ["base", "oracle", "nolm", "lm"].index(k[1]), k[2]))
        out += [f"## Self-training on the unlabelled first half, scored on the second half ({p.stem})", "",
                "| frame | method | round | seeds | pose | fused | kbd CER | pseudo-labels | pseudo precision |",
                "|---|---|---|---|---|---|---|---|---|"]
        for fr, m, rnd in keys:
            rr = [r for r in d if (r["frame"], r["method"], r["round"]) == (fr, m, rnd)]
            npl = np.mean([r["n_pseudo"] for r in rr]) if "n_pseudo" in rr[0] else float("nan")
            prec = np.nanmean([r["pseudo_prec"] for r in rr]) if "pseudo_prec" in rr[0] else float("nan")
            out.append(f"| {fr} | {m} | {rnd} | {len(rr)} | {np.mean([r['pose'] for r in rr]):.3f} | "
                       f"{np.mean([r['fused'] for r in rr]):.3f} | {np.mean([r['cer'] for r in rr]):.3f} | "
                       f"{npl:.0f} | {prec:.3f} |")
        out.append("")


def section_desk(out: list[str]) -> None:
    for p in sorted(x for x in RES.glob("desk_*.json") if x.stem != "desk_pooled"):
        d = json.loads(p.read_text())
        out += [f"## Desk CER, nested leave-one-phrase-out ({p.stem}); official hirecall 0.448", "",
                "| spec | EM on 19 phrase texts | CER mean over seeds | per seed | paired delta vs first spec, per seed |",
                "|---|---|---|---|---|"]
        for r in d:
            out.append(f"| {r['spec']} | {r['em']} | {np.mean(r['cer_seeds']):.3f} | "
                       f"{', '.join(f'{x:.3f}' for x in r['cer_seeds'])} | "
                       f"{'; '.join(_delta(x) for x in r['delta_vs_base'])} |")
        out.append("")


def section_desk_pooled(out: list[str]) -> None:
    d = _load("desk_pooled.json")
    if not d:
        return
    out += ["## Desk CER, seeds 0-2 averaged per phrase, paired over 20 phrases", "", d["note"], "",
            "| spec | EM on 19 phrase texts | CER [95% CI] | d vs abs+coral (matched) | d vs official hirecall 0.448 |",
            "|---|---|---|---|---|"]
    out += [f"| {r['spec']} | {r['em_text']} | {r['cer_seed_avg'][0]:.3f} {_ci(r['cer_seed_avg'][1:])} | "
            f"{_delta(r['d_vs_abs+coral'])} | {_delta(r['d_vs_official_0.448'])} |" for r in d["rows"]]
    out.append("")


def main() -> int:
    out = ["# normalize.py numbers", ""]
    for sec in (section_diag, section_loso, section_vs_coral, section_warmup, section_enrol, section_perturb, section_augment,
                section_selftrain, section_desk, section_desk_pooled):
        sec(out)
    text = "\n".join(out)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "numbers.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
