"""Re-tune the taps_gb event extractor for the bare-desk condition, scored against the only
desk ground truth there is: phrase text + phrase window. Run: diagnose|tune|apply SESSION"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths

from phase0.analysis.analyze_drift import (
    TAP_COUNT_RATIO_RANGE,
    _phrase_windows,
    normalize_char,
    read_jsonl,
)
from phase0.analysis.detect_taps import FINGERTIP_JOINTS
from phase0.analysis.taps_gb import (
    DEFAULT_MODEL,
    _d1,
    assemble,
    build_groups,
    flex_velocity,
    load_frames,
    smooth,
)

KBD_REF = ("20260910-131629-kbd", "20260910-015948-kbd")
HOLDOUT_KBD = Path("data/sessions/20260910-015948-kbd")
CFG_NAME = "desk_cfg.json"
DESK_CFG = {"smooth": 1, "gate_thr": 0.55, "refractory": 6, "n_consec": 1,
            "thr": 0.475, "refractory_ms": 160.0}
MISSING_HAND_FRAC = 0.25  # above this a phrase window is called aborted, not mistyped


# ---------------------------------------------------------------- probabilities
def probs(session: Path, model: Path = DEFAULT_MODEL) -> dict:
    """-> {frames, t, p, gate, P} for a session, from the trained taps_gb model."""
    b = joblib.load(model)
    frames, t, P = load_frames(session)
    X = assemble(build_groups(P), b["groups"])
    p = b["model"].predict_proba(X)[:, 1]
    gate = b["gate"].predict_proba(X)[:, 1] if b.get("gate") is not None else None
    return {"frames": frames, "t": t, "p": p, "gate": gate, "P": P, "cfg": b["cfg"]}


def gated(d: dict, smooth_w: int, gate_thr: float) -> np.ndarray:
    ps = smooth(d["p"], smooth_w)
    if gate_thr > 0 and d["gate"] is not None:
        ps = np.where(d["gate"] >= gate_thr, ps, 0.0)
    return ps


def run_length(mask: np.ndarray) -> np.ndarray:
    """Length of the contiguous True run containing each index (0 where False), vectorised."""
    m = mask.astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    out = np.zeros(len(mask), int)
    for s, e in zip(starts, ends):
        out[s:e] = e - s
    return out


def pick(p: np.ndarray, t: np.ndarray, cfg: dict) -> np.ndarray:
    """taps_gb.pick_events plus a wall-clock refractory; frame counts are not time here."""
    idx, _ = find_peaks(p, height=cfg["thr"], distance=cfg.get("refractory", 1) or 1)
    nc = cfg.get("n_consec", 1)
    if nc > 1 and len(idx):
        idx = idx[run_length(p >= cfg["thr"] * 0.6)[idx] >= nc]
    return _time_refractory(idx, t, cfg.get("refractory_ms", 0.0))


def _time_refractory(idx: np.ndarray, t: np.ndarray, ms: float) -> np.ndarray:
    if ms <= 0 or not len(idx):
        return idx
    keep, last = [], -np.inf
    for i in idx:
        if t[i] - last >= ms / 1000.0:
            keep.append(i)
            last = t[i]
    return np.array(keep, dtype=int)


# ---------------------------------------------------------------- phrase truth
def phrase_windows(session: Path) -> list[dict]:
    out = []
    for w in _phrase_windows(read_jsonl(session / "phrases.jsonl")):
        chars = [c for c in (normalize_char(ch) for ch in w["phrase"]) if c]
        out.append({**w, "chars": chars, "n_chars": len(chars)})
    return out


def mark_aborted(windows: list[dict], d: dict) -> list[dict]:
    """A phrase where a hand is missing from the frame for a large share of the window was
    not typed at all; scoring the extractor on it would be scoring the wrong thing."""
    t, P = d["t"], d["P"]
    for w in windows:
        m = (t >= w["t0"]) & (t < w["t1"])
        pres = np.isfinite(P[m][:, :, 0, 0])
        w["frac_two_hands"] = float(pres.all(1).mean()) if m.sum() else 0.0
        w["aborted"] = w["frac_two_hands"] < 1.0 - MISSING_HAND_FRAC
    return windows


def ratios(windows: list[dict], tt: np.ndarray) -> list[float]:
    return [((tt >= w["t0"]) & (tt < w["t1"])).sum() / max(w["n_chars"], 1) for w in windows]


def score_ratios(windows: list[dict], rs: list[float]) -> tuple[int, float, int]:
    """-> (phrases passing the analyze_drift gate, sum |log ratio|, scorable phrases)."""
    lo, hi = TAP_COUNT_RATIO_RANGE
    live = [(w, r) for w, r in zip(windows, rs) if not w["aborted"]]
    npass = sum(1 for _, r in live if lo <= r <= hi)
    err = sum(abs(np.log(max(r, 1e-3))) for _, r in live)
    return npass, float(err), len(live)


# ---------------------------------------------------------------- finger attribution
def finger_score(P: np.ndarray, kind: str) -> np.ndarray:
    """[F,2,5] score whose argmax names the tapping finger. 'flexvel' is what taps_gb uses."""
    if kind == "flexvel":
        return np.abs(np.nan_to_num(flex_velocity(P), nan=-1.0))
    tips = P[:, :, FINGERTIP_JOINTS, 1]
    return np.nan_to_num(_d1(tips.reshape(len(P), -1)).reshape(len(P), 2, 5), nan=-1e9)


def attribute(P: np.ndarray, k: int, score: np.ndarray) -> tuple[int, int]:
    s, f = np.unravel_index(int(np.argmax(score[k])), score[k].shape)
    return int(s), FINGERTIP_JOINTS[f]


def thumb_at_space(session: Path, d: dict, kind: str) -> tuple[float, float] | None:
    """P(thumb chosen) at true space keydowns vs elsewhere - the anchor analyze_drift relies on."""
    keys = read_jsonl(session / "keys.jsonl")
    kd = [k for k in keys if k["event"] == "down"]
    if len(kd) < 50:
        return None
    sc = finger_score(d["P"], kind)
    ki = np.clip(np.searchsorted(d["t"], [k["t"] for k in kd]), 1, len(d["t"]) - 2)
    is_sp = np.array([k["key"] == "space" for k in kd])
    thumb = np.array([attribute(d["P"], i, sc)[1] == 4 for i in ki])
    return float(thumb[is_sp].mean()), float(thumb[~is_sp].mean())


# ---------------------------------------------------------------- diagnose
def iti_stats(tt: np.ndarray) -> dict:
    iti = np.diff(np.sort(tt)) * 1000
    bins = [(0, 50), (50, 70), (70, 100), (100, 150), (150, 200), (200, 300), (300, np.inf)]
    return {
        "n": int(len(tt)),
        "pct": {q: float(np.percentile(iti, q)) for q in (5, 10, 25, 50, 75, 90)},
        "hist": {f"{lo}-{hi}": float(((iti >= lo) & (iti < hi)).mean()) for lo, hi in bins},
        "iti": iti,
    }


def shape_stats(p: np.ndarray, t: np.ndarray, thr: float) -> dict:
    """Peak morphology at the operating threshold, with no distance filter at all."""
    idx, _ = find_peaks(p, height=thr)
    if len(idx) < 3:
        return {}
    prom = peak_prominences(p, idx)[0]
    _, _, lo, hi = peak_widths(p, idx, rel_height=0.5)
    ax = np.arange(len(t))
    wms = (np.interp(hi, ax, t) - np.interp(lo, ax, t)) * 1000
    valley = np.array([p[idx[i]:idx[i + 1] + 1].min() for i in range(len(idx) - 1)])
    gap = (t[idx[1:]] - t[idx[:-1]]) * 1000
    near = gap < 150
    return {
        "peaks": int(len(idx)),
        "prom_med": float(np.median(prom)),
        "fwhm_ms": [float(np.percentile(wms, q)) for q in (25, 50, 75)],
        "near_pairs": int(near.sum()),
        "valley_med": float(np.median(valley[near])) if near.any() else float("nan"),
        "deep_split_frac": float((valley[near] < 0.5 * thr).mean()) if near.any() else float("nan"),
    }


def frame_jitter(t: np.ndarray) -> dict:
    dt = np.diff(t) * 1000
    span3 = (t[3:] - t[:-3]) * 1000
    return {
        "dt_lt_8ms": float((dt < 8).mean()),
        "dt_median_ms": float(np.median(dt)),
        "span3_p05_ms": float(np.percentile(span3, 5)),
        "span3_med_ms": float(np.median(span3)),
    }


def cmd_diagnose(a) -> int:
    d = probs(a.session, a.model)
    cfg = dict(d["cfg"])
    pg = gated(d, cfg["smooth"], cfg["gate_thr"])
    ev = pick(pg, d["t"], cfg)
    print(f"champion cfg: {cfg}\n")

    print("== frame-clock jitter (a frame-count refractory is only as long as the clock allows)")
    refs = {}
    for name in (a.session.name, *KBD_REF):
        dd = d if name == a.session.name else probs(a.session.parent / name, a.model)
        refs[name] = dd
        j = frame_jitter(dd["t"])
        print(f"  {name:<22} dt<8ms {100*j['dt_lt_8ms']:4.1f}%  median dt {j['dt_median_ms']:.2f}ms  "
              f"3-frame span p05 {j['span3_p05_ms']:5.1f}ms median {j['span3_med_ms']:.1f}ms")

    print("\n== inter-tap interval, champion settings")
    itis = {}
    for name, dd in refs.items():
        tt = dd["t"][pick(gated(dd, cfg["smooth"], cfg["gate_thr"]), dd["t"], cfg)]
        s = iti_stats(tt)
        itis[name] = s
        print(f"  {name:<22} n={s['n']:<5} median {s['pct'][50]:6.1f}ms  " +
              "  ".join(f"{k}:{100*v:4.1f}%" for k, v in s["hist"].items()))

    print("\n== probability waveform shape at thr (no distance filter)")
    for name, dd in refs.items():
        s = shape_stats(gated(dd, cfg["smooth"], cfg["gate_thr"]), dd["t"], cfg["thr"])
        print(f"  {name:<22} peaks={s['peaks']:<5} prom={s['prom_med']:.3f}  "
              f"FWHM p25/50/75={s['fwhm_ms'][0]:.0f}/{s['fwhm_ms'][1]:.0f}/{s['fwhm_ms'][2]:.0f}ms  "
              f"pairs<150ms={s['near_pairs']:<5} valley={s['valley_med']:.3f}  "
              f"deep-split={100*s['deep_split_frac']:.0f}%")

    print("\n== finger attribution: P(thumb picked) at true space keydowns vs elsewhere")
    for name, dd in refs.items():
        for kind in ("flexvel", "tipvy"):
            r = thumb_at_space(a.session.parent / name, dd, kind)
            if r:
                print(f"  {name:<22} {kind:<8} space={r[0]:.3f}  non-space={r[1]:.3f}")

    print("\n== per-phrase, champion settings")
    W = mark_aborted(phrase_windows(a.session), d)
    rs = ratios(W, d["t"][ev])
    for w, r in zip(W, rs):
        print(f"  {w['idx']:>2}  chars={w['n_chars']:<4} taps={int(r*w['n_chars']):<4} ratio={r:.2f}  "
              f"dur={w['t1']-w['t0']:5.2f}s  two-hands={100*w['frac_two_hands']:5.1f}%"
              f"{'   <- ABORTED, hands left the frame' if w['aborted'] else ''}")

    for w in W:
        if w["aborted"]:
            _explain_aborted(w, d)
    _plot_iti(itis, Path("data/sessions/desk_iti.png"), a.session.name, W, refs)
    return 0


def _explain_aborted(w: dict, d: dict) -> None:
    t, P = d["t"], d["P"]
    m = (t >= w["t0"]) & (t < w["t1"])
    ts, pres = t[m], np.isfinite(P[m][:, :, 0, 0])
    print(f"\n  phrase {w['idx']} timeline (1 s buckets, s from window start):")
    for lo in np.arange(0, ts[-1] - ts[0] + 1e-9, 1.0):
        k = (ts - ts[0] >= lo) & (ts - ts[0] < lo + 1)
        if k.sum() < 2:
            continue
        g = d["gate"][m][k]
        print(f"    +{lo:4.1f}s n={k.sum():3d}  left={100*pres[k,0].mean():3.0f}%  "
              f"right={100*pres[k,1].mean():3.0f}%  gate median={np.median(g):.2f}")


def _plot_iti(itis: dict, out: Path, desk_name: str, W: list[dict], refs: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from phase0.analysis.eval_taps import keydown_times

    fig, ax = plt.subplots(figsize=(9, 5))
    edges = np.arange(0, 405, 10)
    for name, s in itis.items():
        style = dict(lw=2.4, color="crimson") if name == desk_name else dict(lw=1.3, alpha=0.85)
        ax.hist(np.clip(s["iti"], 0, 400), bins=edges, density=True, histtype="step",
                label=f"{name}: detected (n={s['n']}, med {s['pct'][50]:.0f}ms)", **style)

    live = [w for w in W if not w["aborted"]]
    desk_true = 1000 * sum(w["t1"] - w["t0"] for w in live) / sum(w["n_chars"] for w in live)
    ax.axvline(desk_true, color="crimson", ls="--", lw=1.6,
               label=f"desk: mean interval per typed char = {desk_true:.0f}ms")
    meds = []
    for name in KBD_REF:
        iti = np.diff(keydown_times(Path("data/sessions") / name)) * 1000
        meds.append(float(np.median(iti[iti < 1000])))
    ax.axvline(float(np.mean(meds)), color="k", ls=":", lw=1.5,
               label=f"kbd: true median inter-key = {meds[0]:.0f}/{meds[1]:.0f}ms")
    ax.set_xlabel("interval (ms)")
    ax.set_ylabel("density")
    ax.set_title("Detected inter-tap intervals vs what was actually typed (keyboard-tuned extractor)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


# ---------------------------------------------------------------- tune
GRID = {
    "smooth": (1, 3, 5),
    "gate_thr": (0.55, 0.75),
    "refractory": (3, 4, 6, 8, 10, 12),
    "refractory_ms": (0.0, 50.0, 80.0, 100.0, 130.0, 160.0),
    "n_consec": (1, 2, 3),
    "thr": tuple(np.round(np.arange(0.45, 0.901, 0.025), 4)),
}


def sweep(session: Path, d: dict, grid: dict = GRID) -> list[dict]:
    W = mark_aborted(phrase_windows(session), d)
    t = d["t"]
    rows = []
    for w_s in grid["smooth"]:
        for gt in grid["gate_thr"]:
            pg = gated(d, w_s, gt)
            for rf in grid["refractory"]:
                idx, props = find_peaks(pg, height=min(grid["thr"]), distance=rf)
                h = props["peak_heights"]
                for nc in grid["n_consec"]:
                    for thr in grid["thr"]:
                        cand = idx[h >= thr]
                        if nc > 1 and len(cand):
                            cand = cand[run_length(pg >= thr * 0.6)[cand] >= nc]
                        for ms in grid["refractory_ms"]:
                            ev = _time_refractory(cand, t, ms)
                            rs = ratios(W, t[ev])
                            npass, err, live = score_ratios(W, rs)
                            rows.append({"cfg": {"smooth": w_s, "gate_thr": gt, "refractory": rf,
                                                 "n_consec": nc, "thr": float(thr),
                                                 "refractory_ms": ms},
                                         "n_taps": int(len(ev)), "npass": npass, "live": live,
                                         "logerr": err, "ratios": rs})
    rows.sort(key=lambda r: (-r["npass"], r["logerr"]))
    return rows


def kbd_score(cfg: dict, d: dict, session: Path = HOLDOUT_KBD) -> dict:
    """eval_taps --clip on a held-out keyboard session, so a desk gain that wrecks kbd shows."""
    from phase0.analysis.eval_taps import keydown_times, score

    tt = d["t"][pick(gated(d, cfg["smooth"], cfg["gate_thr"]), d["t"], cfg)]
    kt = keydown_times(session)
    tt = tt[(tt >= kt.min() - 0.1) & (tt <= kt.max() + 0.1)]
    return score(kt, tt)


def cmd_tune(a) -> int:
    d = probs(a.session, a.model)
    W = mark_aborted(phrase_windows(a.session), d)
    live = [w["idx"] for w in W if not w["aborted"]]
    print(f"scorable phrases: {live}  (aborted, excluded: {[w['idx'] for w in W if w['aborted']]})\n")
    rows = sweep(a.session, d)
    champion = dict(d["cfg"], refractory_ms=0.0)
    kd = probs(HOLDOUT_KBD, a.model)

    top_pass = rows[0]["npass"]
    seen, cands = set(), []
    for r in rows:
        if r["npass"] < top_pass:
            break
        key = (r["n_taps"], tuple(round(x, 3) for x in r["ratios"]))
        if key in seen:
            continue
        seen.add(key)
        r["kbd"] = kbd_score(r["cfg"], kd)
        cands.append(r)

    hdr = (f"{'smooth':>6} {'gate':>5} {'rf':>3} {'rf_ms':>6} {'nc':>3} {'thr':>6} {'taps':>5} "
           f"{'pass':>5} {'|log|':>6} {'kbdF1':>6} {'kbdR':>6} {'kbdP':>6}  per-phrase ratio")
    print(f"{len(cands)} distinct settings reach {top_pass}/{len(live)} phrases in range\n")
    print(f"-- best desk ratio fit --\n{hdr}")
    for r in cands[:a.top]:
        _row(r)
    print(f"\n-- same, ranked by held-out keyboard F1 --\n{hdr}")
    for r in sorted(cands, key=lambda r: -r["kbd"]["f1"])[:a.top]:
        _row(r)

    print("\n-- champion (keyboard-tuned) for reference --")
    print(hdr)
    ev = pick(gated(d, champion["smooth"], champion["gate_thr"]), d["t"], champion)
    rs = ratios(W, d["t"][ev])
    npass, err, _ = score_ratios(W, rs)
    _row({"cfg": champion, "n_taps": int(len(ev)), "npass": npass, "logerr": err, "ratios": rs,
          "kbd": kbd_score(champion, kd)})

    key = (lambda r: r["logerr"]) if a.select == "ratio" else (lambda r: -r["kbd"]["f1"])
    best = min(cands, key=key)["cfg"]
    out = a.session / CFG_NAME
    out.write_text(json.dumps(best, indent=1) + "\n")
    print(f"\nbest desk cfg -> {out}\n  {best}")
    return 0


def _row(r: dict) -> None:
    c, k = r["cfg"], r.get("kbd")
    ks = f"{100*k['f1']:6.1f} {100*k['recall']:6.1f} {100*k['precision']:6.1f}" if k else " " * 20
    print(f"{c['smooth']:>6} {c['gate_thr']:>5} {c['refractory']:>3} {c['refractory_ms']:>6.0f} "
          f"{c['n_consec']:>3} {c['thr']:>6.3f} {r['n_taps']:>5} {r['npass']:>5} {r['logerr']:>6.2f} "
          f"{ks}  " + " ".join(f"{x:.2f}" for x in r["ratios"]))


# ---------------------------------------------------------------- apply
def cmd_apply(a) -> int:
    cfg_path = a.session / CFG_NAME
    if a.cfg:
        cfg = json.loads(a.cfg)
    else:
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else dict(DESK_CFG)
    d = probs(a.session, a.model)
    ev = pick(gated(d, cfg["smooth"], cfg["gate_thr"]), d["t"], cfg)
    sc = finger_score(d["P"], a.finger)
    out = a.out or a.session / "taps_desk.jsonl"
    with open(out, "w") as fh:
        for k in ev:
            s, tip = attribute(d["P"], k, sc)
            fh.write(json.dumps({
                "t": float(d["t"][k]), "hand": int(s), "finger": int(tip),
                "x": float(np.nan_to_num(d["P"][k, s, tip, 0])),
                "y": float(np.nan_to_num(d["P"][k, s, tip, 1])),
                "conf": float(np.nan_to_num(d["P"][k, s, tip, 2])), "i": int(d["frames"][k]),
            }) + "\n")
    W = mark_aborted(phrase_windows(a.session), d)
    rs = ratios(W, d["t"][ev])
    npass, err, live = score_ratios(W, rs)
    print(f"wrote {len(ev)} events -> {out}")
    print(f"cfg {cfg}")
    print(f"per-phrase ratio: " + " ".join(f"{x:.2f}" for x in rs))
    print(f"{npass}/{live} scorable phrases inside {TAP_COUNT_RATIO_RANGE}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("diagnose", "tune", "apply"):
        s = sub.add_parser(name)
        s.add_argument("session", type=Path)
        s.add_argument("--model", type=Path, default=DEFAULT_MODEL)
        if name == "tune":
            s.add_argument("--top", type=int, default=12)
            s.add_argument("--select", choices=("ratio", "kbd"), default="ratio",
                           help="tie-break among settings that pass the most phrases")
        if name == "apply":
            s.add_argument("--out", type=Path, default=None)
            s.add_argument("--cfg", type=str, default=None, help="JSON cfg, overrides desk_cfg.json")
            s.add_argument("--finger", choices=("flexvel", "tipvy"), default="flexvel")
    a = ap.parse_args(argv)
    return {"diagnose": cmd_diagnose, "tune": cmd_tune, "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
