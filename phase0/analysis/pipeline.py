"""One end-to-end bare-desk decoding pipeline, tuned and scored on desk CER (never keyboard F1).
Run: python -m phase0.analysis.pipeline {sweep | cv | ablate | detectors | examples | probe}"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from phase0.analysis import desk_tune as dt
from phase0.analysis import tap_pos as tp

# .cache/*.pkl and models/*.pkl were pickled by tap_pos running as __main__
for _n in ("Sess", "KeyClf", "XYReg", "GaussXY", "Selector"):
    setattr(sys.modules["__main__"], _n, getattr(tp, _n))

from phase0.analysis import adapt as ad  # noqa: E402
from phase0.analysis import contact as ct  # noqa: E402
from phase0.analysis import seqdecode as sq  # noqa: E402
from phase0.analysis.decode import (  # noqa: E402
    NA,
    CharLM,
    Weights,
    WordLM,
    beam_decode,
    desk_segments,
    edit_distance,
    wer,
)

DESK = "20260910-202149-desk"
KBD_TRAIN = ("20260910-015217-kbd", "20260910-021315-kbd", "20260910-131629-kbd")
BASE_MODEL = Path("models/tap_pos.pkl")
LM_PATH = Path("models/charlm.npz")
OLD_TAPS = "taps_fix5.jsonl"  # the previous detector's preserved 591-event stream
EPS = 1e-12


# ------------------------------------------------------------------ stack definition
@dataclass(frozen=True)
class Stack:
    """Which components are switched on, plus their operating points."""

    coral: bool = True
    contact_w: float = 0.25
    em: bool = True
    motor: bool = False
    spatial: bool = True
    lam: float = 0.5
    iters: int = 6
    l2: float = 4.0
    beam: int = 30
    deletion: float = -9.0
    insertion: float = -7.0
    max_del: int = 3
    frame: str = "local"

    def weights(self) -> Weights:
        return Weights(insertion=self.insertion, deletion=self.deletion,
                       max_deletions=self.max_del)


FULL = Stack(contact_w=0.0)  # the contact expert is measured below and does not earn a place
SWEEP = Stack(contact_w=0.0, em=False)
CHAMP_MS = 0.0  # the keyboard-tuned extractor carries no wall-clock refractory


# ------------------------------------------------------------------ keyboard-only fits
class KbdFit:
    """Everything fitted on keyboard sessions. Nothing here ever sees a desk phrase."""

    def __init__(self, sessions=KBD_TRAIN, model_path: Path = BASE_MODEL, frame: str = "local"):
        self.sessions = tuple(sessions)
        self.frame = frame
        self.model = ad.load_base_model(model_path)
        self._contact = None
        self._motor = None
        self._src = None
        self._lm = None

    @property
    def src_moments(self):
        if self._src is None:
            m = self.model
            S = np.vstack([tp.features(d["sess"], d["k"], m.mode, m.joints, m.offsets)
                           for d in (tp.dataset(s) for s in self.sessions)])
            self._src = (S.mean(0), S.std(0) + 1e-6)
        return self._src

    @property
    def contact(self):
        if self._contact is None:
            ds = [ct.load(s) for s in self.sessions]
            clf = ct.FingerClf.fit(ds)
            self._contact = (clf, ct._fit_gauss(ds, "oracle", self.frame, 0, clf))
        return self._contact

    @property
    def motor(self):
        if self._motor is None:
            self._motor = sq.fit_motor(list(self.sessions))
        return self._motor

    @property
    def lm(self):
        if self._lm is None:
            self._lm = (CharLM.load(LM_PATH), WordLM.load(LM_PATH.with_suffix(".words.json")))
        return self._lm


_KF: dict = {}


def kbd_fit(frame: str = "local") -> KbdFit:
    if frame not in _KF:
        _KF[frame] = KbdFit(frame=frame)
    return _KF[frame]


# ------------------------------------------------------------------ event extraction
_FRAMES: dict = {}


def frame_probs(session: Path) -> dict:
    key = str(session)
    if key not in _FRAMES:
        d = dt.probs(session)
        d["score"] = dt.finger_score(d["P"], "flexvel")
        _FRAMES[key] = d
    return _FRAMES[key]


def champion_cfg(session: Path) -> dict:
    """The extractor as taps_gb tuned it on keyboard F1."""
    return dict(frame_probs(session)["cfg"], refractory_ms=CHAMP_MS)


def event_idx(d: dict, cfg: dict) -> np.ndarray:
    return dt.pick(dt.gated(d, cfg["smooth"], cfg["gate_thr"]), d["t"], cfg)


def taps_from(d: dict, ev: np.ndarray) -> list[dict]:
    out = []
    for k in ev:
        s, tip = dt.attribute(d["P"], k, d["score"])
        out.append({"t": float(d["t"][k]), "hand": int(s), "finger": int(tip),
                    "x": float(np.nan_to_num(d["P"][k, s, tip, 0])),
                    "y": float(np.nan_to_num(d["P"][k, s, tip, 1])),
                    "i": int(d["frames"][k])})
    return out


# ------------------------------------------------------------------ spatial evidence
def spatial_proba(sess, k: np.ndarray, kf: KbdFit, st: Stack) -> np.ndarray:
    """pose -> P(key), optionally CORAL-standardised, optionally mixed with the contact expert."""
    m = kf.model
    X = tp.features(sess, k, m.mode, m.joints, m.offsets)
    if st.coral:
        ms, ss = kf.src_moments
        X = (X - X.mean(0)) / (X.std(0) + 1e-6) * ss + ms
    p = m.model.predict_proba(X)
    out = np.full((len(k), NA), 1e-6)
    out[:, m.classes] = np.maximum(p, 1e-6)
    out /= out.sum(1, keepdims=True)
    if st.contact_w > 0:
        clf, g = kf.contact
        q = np.exp(ct.implied_logp(g, {"sess": sess, "k": k}, kf.frame, 0, clf))
        out = (1 - st.contact_w) * out + st.contact_w * q
        out /= out.sum(1, keepdims=True)
    return out


class _AdaptTarget:
    """The only thing adapt.adapt() reads off a Target is the reduced pose feature block."""

    def __init__(self, sess, k):
        self.X = tp.features(sess, k, "rel", ad.ADAPT_JOINTS, ad.ADAPT_OFFSETS)


def weakly_supervise(sess, k, base: np.ndarray, train_segs, st: Stack) -> np.ndarray:
    if not train_segs:
        return base
    return ad.adapt(_AdaptTarget(sess, k), train_segs, base, st.iters, st.lam, st.l2,
                    "em", verbose=False)[1]


# ------------------------------------------------------------------ decoding
def segments(session: Path, taps: list[dict]):
    """-> [(text, tap-row indices)] for every prompted phrase, in prompt order."""
    idx = {round(t["t"], 6): i for i, t in enumerate(taps)}
    return [(text, np.array([idx[round(x["t"], 6)] for x in s], int))
            for text, s in desk_segments(session, taps)]


def decode_one(proba: np.ndarray, rows: np.ndarray, times: np.ndarray | None,
               kf: KbdFit, st: Stack) -> str:
    if len(rows) == 0:
        return ""
    lp = np.log(np.maximum(proba[rows], EPS))
    if not st.spatial:
        lp = np.full_like(lp, -np.log(NA))
    clm, wlm = kf.lm
    if st.motor:
        return sq.joint_beam_decode(lp, times[rows], clm, wlm, kf.motor, st.weights(),
                                    sq.MotorWeights(), st.beam)[0]
    return beam_decode(lp, clm, wlm, st.weights(), st.beam)


def score_phrases(segs, proba, times, kf, st, keep=None) -> list[tuple]:
    """-> [(ref, hyp, edit distance, len(ref))] for the requested phrase indices."""
    out = []
    for j, (text, rows) in enumerate(segs):
        if keep is not None and j not in keep:
            continue
        hyp = decode_one(proba, rows, times, kf, st)
        out.append((text, hyp, edit_distance(text, hyp), len(text)))
    return out


def pooled(rows) -> float:
    return sum(r[2] for r in rows) / max(1, sum(r[3] for r in rows))


# ------------------------------------------------------------------ the pipeline
def run(session: Path, cfg: dict, st: Stack, kf: KbdFit, taps: list[dict] | None = None):
    """-> (taps, segs, base proba, times). One pass of everything that is fold-independent."""
    if taps is None:
        d = frame_probs(session)
        taps = taps_from(d, event_idx(d, cfg))
    sess = tp.load_sess(str(session))
    k = sess.rows(taps)
    base = spatial_proba(sess, k, kf, st)
    times = np.array([t["t"] for t in taps], float)
    return taps, segments(session, taps), base, times, sess, k


def cv_score(session: Path, cfg: dict, st: Stack, kf: KbdFit,
             taps: list[dict] | None = None) -> list[tuple]:
    """Leave-one-phrase-out. The EM stage only ever sees the other 19 phrases' text."""
    taps, segs, base, times, sess, k = run(session, cfg, st, kf, taps)
    if not st.em:
        return score_phrases(segs, base, times, kf, st)
    out = []
    for j in range(len(segs)):
        tr = [s for i, s in enumerate(segs) if i != j and len(s[1])]
        p = weakly_supervise(sess, k, base, tr, st)
        out += score_phrases(segs, p, times, kf, st, keep={j})
    return out


# ------------------------------------------------------------------ bootstrap
def boot_ci(rows, n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    e = np.array([r[2] for r in rows], float)
    r = np.array([r[3] for r in rows], float)
    idx = np.random.default_rng(seed).integers(0, len(e), (n, len(e)))
    v = e[idx].sum(1) / np.maximum(r[idx].sum(1), 1)
    return (float(e.sum() / max(1, r.sum())),
            float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


def boot_delta(a, b, n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Paired over phrases: CER(b) - CER(a), so a negative delta means b is better."""
    ea, ra = np.array([r[2] for r in a], float), np.array([r[3] for r in a], float)
    eb = np.array([r[2] for r in b], float)
    idx = np.random.default_rng(seed).integers(0, len(ea), (n, len(ea)))
    den = np.maximum(ra[idx].sum(1), 1)
    v = eb[idx].sum(1) / den - ea[idx].sum(1) / den
    return float(eb.sum() / ra.sum() - ea.sum() / ra.sum()), \
        float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# ------------------------------------------------------------------ extractor sweep
GRID = {
    "smooth": (1, 3),
    "gate_thr": (0.25, 0.45, 0.60),
    "refractory": (3, 4, 6, 8),
    "n_consec": (1, 2, 3),
    "thr": tuple(np.round(np.arange(0.30, 0.751, 0.05), 3)),
    "refractory_ms": (0.0, 80.0, 120.0, 160.0),
}
RATIO_BAND = (0.5, 2.2)


def candidate_cfgs(session: Path, grid: dict = GRID) -> list[tuple[dict, np.ndarray]]:
    """Every grid point, deduped by the event set it actually produces and filtered to a
    plausible taps/char band, so the CER sweep decodes each distinct tap stream once."""
    d = frame_probs(session)
    n_chars = sum(len(t) for t, _ in desk_segments(session, taps_from(d, event_idx(d, dict(
        d["cfg"], refractory_ms=0.0)))))
    seen, out = set(), []
    for sm in grid["smooth"]:
        for gt in grid["gate_thr"]:
            pg = dt.gated(d, sm, gt)
            for rf in grid["refractory"]:
                from scipy.signal import find_peaks
                idx, props = find_peaks(pg, height=min(grid["thr"]), distance=rf)
                h = props["peak_heights"]
                for nc in grid["n_consec"]:
                    for thr in grid["thr"]:
                        cand = idx[h >= thr]
                        if nc > 1 and len(cand):
                            cand = cand[dt.run_length(pg >= thr * 0.6)[cand] >= nc]
                        for ms in grid["refractory_ms"]:
                            ev = dt._time_refractory(cand, d["t"], ms)
                            if not RATIO_BAND[0] <= len(ev) / n_chars <= RATIO_BAND[1]:
                                continue
                            key = ev.tobytes()
                            if key in seen:
                                continue
                            seen.add(key)
                            out.append(({"smooth": sm, "gate_thr": gt, "refractory": rf,
                                         "n_consec": nc, "thr": float(thr),
                                         "refractory_ms": ms}, ev))
    return out


DELETIONS = (-9.0, -5.0, -3.0, -2.0)


def sweep_table(session: Path, st: Stack, kf: KbdFit, grid: dict = GRID,
                deletions=DELETIONS, verbose: bool = True, shard: int = 0,
                nshards: int = 1) -> list[dict]:
    """Per-phrase edit distances for every distinct tap stream x deletion cost, with EM off:
    fold-independent, so any fold's train/test split is a subset sum of this table."""
    d = frame_probs(session)
    cands = candidate_cfgs(session, grid)[shard::nshards]
    if verbose:
        print(f"shard {shard}/{nshards}: {len(cands)} tap streams x {len(deletions)} "
              f"deletion costs", flush=True)
    rows = []
    for n, (cfg, ev) in enumerate(cands):
        taps = taps_from(d, ev)
        _, segs, base, times, _, _ = run(session, cfg, replace(st, em=False), kf, taps)
        for dl in deletions:
            sd = replace(st, em=False, deletion=dl)
            r = score_phrases(segs, base, times, kf, sd)
            rows.append({"cfg": cfg, "deletion": dl, "n_taps": len(taps),
                         "per": [(x[2], x[3]) for x in r], "cer": pooled(r)})
        if verbose and (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(cands)} best {min(x['cer'] for x in rows):.3f}", flush=True)
    return rows


def select_cfg(rows: list[dict], train: set[int]) -> dict:
    """argmin pooled CER over the training phrases only."""
    best, bc = None, np.inf
    for r in rows:
        e = sum(r["per"][j][0] for j in train)
        n = sum(r["per"][j][1] for j in train)
        c = e / max(1, n)
        if c < bc:
            best, bc = r, c
    return best


def taps_table(session: Path, taps: list[dict], st: Stack, kf: KbdFit,
               deletions=DELETIONS) -> list[dict]:
    """The same table for a fixed event stream, so a detector we cannot re-run still gets its
    deletion cost chosen by cross-validation rather than inherited."""
    _, segs, base, times, _, _ = run(session, {}, replace(st, em=False), kf, taps)
    out = []
    for dl in deletions:
        r = score_phrases(segs, base, times, kf, replace(st, em=False, deletion=dl))
        out.append({"cfg": None, "deletion": dl, "n_taps": len(taps),
                    "per": [(x[2], x[3]) for x in r], "cer": pooled(r)})
    return out


def nested_cv(session: Path, rows: list[dict], st: Stack, kf: KbdFit,
              verbose: bool = True, taps: list[dict] | None = None) -> list[tuple]:
    """Leave-one-phrase-out with the extractor and deletion cost chosen inside each fold. The
    held-out phrase contributes to neither the tuning nor the weak supervision."""
    d = frame_probs(session)
    n = len(rows[0]["per"])
    out = []
    for j in range(n):
        pick = select_cfg(rows, set(range(n)) - {j})
        sd = replace(st, deletion=pick["deletion"])
        fixed = taps if taps is not None else taps_from(d, event_idx(d, pick["cfg"]))
        _, segs, base, times, sess, k = run(session, pick["cfg"], sd, kf, fixed)
        p = base
        if sd.em:
            p = weakly_supervise(sess, k, base, [s for i, s in enumerate(segs)
                                                 if i != j and len(s[1])], sd)
        r = score_phrases(segs, p, times, kf, sd, keep={j})[0]
        out.append(r)
        if verbose:
            c = pick["cfg"] or {}
            print(f"  phrase{j:>2} thr={c.get('thr', float('nan')):.2f} "
                  f"rf={c.get('refractory', 0)} nc={c.get('n_consec', 0)} "
                  f"ms={c.get('refractory_ms', 0):.0f} del={pick['deletion']:.0f} "
                  f"taps={len(fixed)} CER={r[2]/max(1,r[3]):.3f}", flush=True)
    return out


# ------------------------------------------------------------------ reporting helpers
def report(name: str, rows) -> dict:
    c, lo, hi = boot_ci(rows)
    w = sum(edit_distance(r[0].split(), r[1].split()) for r in rows) / \
        max(1, sum(len(r[0].split()) for r in rows))
    print(f"{name:<34} CER={c:.3f} [{lo:.3f}, {hi:.3f}]  WER={w:.3f}")
    return {"name": name, "cer": c, "lo": lo, "hi": hi, "wer": w,
            "per": [(r[2], r[3]) for r in rows]}


def session_path(sid: str) -> Path:
    return tp.sess_path(sid)


# ------------------------------------------------------------------ CLI
def cmd_sweep(a) -> int:
    s = session_path(a.session)
    kf = kbd_fit()
    rows = sweep_table(s, replace(SWEEP, contact_w=a.contact_w), kf,
                       shard=a.shard, nshards=a.nshards)
    if a.nshards > 1:
        Path(a.json).write_text(json.dumps(rows))
        print(f"shard {a.shard} -> {a.json}")
        return 0
    rows.sort(key=lambda r: r["cer"])
    nchar = sum(x[1] for x in rows[0]["per"])
    print(f"\n{'thr':>6}{'rf':>4}{'nc':>4}{'gate':>6}{'sm':>4}{'rf_ms':>7}{'del':>6}{'taps':>6}"
          f"{'taps/char':>10}{'CER(all)':>10}")
    for r in rows[:a.top]:
        c = r["cfg"]
        print(f"{c['thr']:>6.2f}{c['refractory']:>4}{c['n_consec']:>4}{c['gate_thr']:>6.2f}"
              f"{c['smooth']:>4}{c['refractory_ms']:>7.0f}{r['deletion']:>6.0f}{r['n_taps']:>6}"
              f"{r['n_taps']/nchar:>10.2f}{r['cer']:>10.3f}")
    print("\n-- CER against detection density (all candidates) --")
    print(f"{'taps/char band':>16}{'n cfg':>7}{'best CER':>10}{'median CER':>12}")
    for lo in np.arange(0.5, 2.2, 0.25):
        v = [r for r in rows if lo <= r["n_taps"] / nchar < lo + 0.25]
        if v:
            print(f"{f'{lo:.2f}-{lo+0.25:.2f}':>16}{len(v):>7}"
                  f"{min(x['cer'] for x in v):>10.3f}{np.median([x['cer'] for x in v]):>12.3f}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows))
        print(f"\nsweep table -> {a.json}")
    return 0


def _table(a) -> list[dict]:
    if a.table and Path(a.table).exists():
        return json.loads(Path(a.table).read_text())
    rows = sweep_table(session_path(a.session), SWEEP, kbd_fit())
    if a.table:
        Path(a.table).write_text(json.dumps(rows))
    return rows


def cmd_cv(a) -> int:
    s = session_path(a.session)
    kf = kbd_fit()
    rows = _table(a)
    print("\n-- nested leave-one-phrase-out, full stack --")
    r = nested_cv(s, rows, FULL, kf)
    print()
    report("FULL (cfg tuned per fold)", r)
    _examples(r, a.examples)
    return 0


ABLATIONS = (
    ("no camera (LM only)", replace(FULL, spatial=False, em=False, coral=False), "champ"),
    ("+ pose key_probs, kbd cfg", replace(FULL, em=False, coral=False), "champ"),
    ("+ desk-tuned extractor", replace(FULL, em=False, coral=False), "cv"),
    ("+ CORAL", replace(FULL, em=False), "cv"),
    ("+ weak supervision (EM)", FULL, "cv"),
    ("+ contact expert @0.25", replace(FULL, contact_w=0.25), "cv"),
    ("+ motor layer", replace(FULL, contact_w=0.25, motor=True), "cv"),
)

DROPS = (
    ("drop desk-tuned extractor", FULL, "champ_tuned"),
    ("drop CORAL", replace(FULL, coral=False), "cv"),
    ("drop weak supervision", replace(FULL, em=False), "cv"),
)


def cmd_ablate(a) -> int:
    s = session_path(a.session)
    kf = kbd_fit()
    rows = _table(a)
    champ = champion_cfg(s)

    d = frame_probs(s)
    champ_taps = taps_from(d, event_idx(d, champ))
    ctab = None

    def go(st, how):
        nonlocal ctab
        if how == "champ":
            return cv_score(s, champ, st, kf)
        if how == "champ_tuned":  # same keyboard-tuned events, deletion cost still CV-chosen
            ctab = ctab or taps_table(s, champ_taps, st, kf)
            return nested_cv(s, ctab, st, kf, False, taps=champ_taps)
        return nested_cv(s, rows, st, kf, False)

    out, prev = [], None
    print("\n== additive ladder (leave-one-phrase-out throughout) ==")
    for name, st, how in ABLATIONS:
        r = go(st, how)
        d = report(name, r)
        if prev is not None:
            dd, lo, hi = boot_delta(prev, r)
            print(f"{'':<34}   delta = {dd:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        out.append((name, r, d))
        prev = r
    full = out[4][1]
    print("\n== drop-one from the full stack ==")
    for name, st, how in DROPS:
        r = go(st, how)
        report(name, r)
        dd, lo, hi = boot_delta(full, r)
        print(f"{'':<34}   delta vs full = {dd:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    if a.json:
        Path(a.json).write_text(json.dumps([{"name": n, **d} for n, _, d in out]))
    return 0


def cmd_detectors(a) -> int:
    """Newest detector at its CV-tuned operating point vs the previous detector's event stream."""
    from phase0.analysis.analyze_drift import read_jsonl

    s = session_path(a.session)
    kf = kbd_fit()
    rows = _table(a)
    new = nested_cv(s, rows, FULL, kf, verbose=False)
    report("newest detector, tuned", new)

    old = [{k: t[k] for k in ("t", "hand", "finger", "x", "y", "i")}
           for t in read_jsonl(s / a.old_taps)]
    orows = taps_table(s, old, FULL, kf)
    r = nested_cv(s, orows, FULL, kf, verbose=False, taps=old)
    report(f"previous detector ({len(old)} events)", r)
    dd, lo, hi = boot_delta(new, r)
    print(f"{'':<34}   delta vs newest = {dd:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    nchar = sum(x[3] for x in new)
    n = len(new)
    dens = [select_cfg(rows, set(range(n)) - {j})["n_taps"] / nchar for j in range(n)]
    print(f"\ntaps/char: newest tuned {np.median(dens):.2f} (median over folds), "
          f"previous {len(old)/nchar:.2f}")
    return 0


CEIL_CFGS = ((0.50, 4, 3), (0.45, 4, 2), (0.40, 4, 1), (0.35, 3, 1), (0.30, 3, 1))


def ceiling_row(session: Path, cfg: dict, st: Stack, kf: KbdFit) -> tuple[int, float, float, float]:
    """What this tap stream could give if the spatial model were replaced by the true text's
    own monotone alignment. Uses the scored phrase's text, so it is a ceiling, never a result."""
    d = frame_probs(session)
    taps = taps_from(d, event_idx(d, cfg))
    _, segs, base, times, _, _ = run(session, cfg, st, kf, taps)
    nch = sum(len(t) for t, _ in segs)
    live = [x for x in segs if len(x[1])]
    par = ad.AlignParams.from_counts(len(taps), nch)
    q, _, _, _ = ad.align_all(np.log(np.maximum(base, EPS)), live, par)
    return (len(taps), len(taps) / max(1, nch),
            pooled(score_phrases(segs, q, times, kf, st)),
            pooled(score_phrases(segs, base, times, kf, st)))


def cmd_probe(a) -> int:
    """Where the loss actually sits: per-tap key accuracy and control CER on a keyboard session
    no model was fitted on, against the same stack's desk numbers."""
    from phase0.analysis.decode import kbd_segments
    from phase0.analysis.analyze_drift import read_jsonl

    kf = kbd_fit()
    sid = a.holdout
    sp = session_path(sid)
    taps = read_jsonl(sp / "taps.jsonl")
    sess = tp.load_sess(sid)
    k = sess.rows(taps)
    d = tp.dataset(sid)
    kk = sess.rows(d["taps"])
    print(f"{sid}: {len(taps)} taps, {len(d['keys'])} carry a keystroke label")
    for name, st in (("raw", replace(FULL, coral=False, em=False)),
                     ("CORAL", replace(FULL, em=False))):
        p = spatial_proba(sess, kk, kf, st)
        print(f"  per-tap top1 [{name:5s}] = {tp.topk(p, d['y'], 1):.3f}  "
              f"top5 = {tp.topk(p, d['y'], 5):.3f}")
    idx = {round(t["t"], 6): i for i, t in enumerate(taps)}
    segs = [(t, np.array([idx[round(x["t"], 6)] for x in s], int))
            for t, s in kbd_segments(sp, taps)]
    times = np.array([t["t"] for t in taps], float)
    for dl in (-9.0, -3.0):
        st = replace(FULL, em=False, deletion=dl)
        r = score_phrases(segs, spatial_proba(sess, k, kf, st), times, kf, st)
        print(f"  control CER (keyboard present, del={dl:.0f}) = {pooled(r):.3f} "
              f"over {len(segs)} bursts, {len(taps)/max(1,sum(x[3] for x in r)):.2f} taps/char")

    print(f"\n{a.session}: text-aligned ceiling against detection density\n"
          f"{'thr':>6}{'rf':>4}{'nc':>4}{'taps':>7}{'taps/char':>11}{'ceiling':>9}{'actual':>9}")
    ds = session_path(a.session)
    st = replace(FULL, em=False, deletion=-3.0)
    for thr, rf, nc in CEIL_CFGS:
        cfg = {"thr": thr, "smooth": 1, "refractory": rf, "n_consec": nc,
               "gate_thr": 0.25, "refractory_ms": 0.0}
        n, dens, ceil, act = ceiling_row(ds, cfg, st, kf)
        print(f"{thr:>6.2f}{rf:>4}{nc:>4}{n:>7}{dens:>11.2f}{ceil:>9.3f}{act:>9.3f}", flush=True)
    return 0


def _examples(rows, n: int) -> None:
    if n <= 0:
        return
    print(f"\n-- {min(n, len(rows))} decoded phrases --")
    for ref, hyp, e, r in rows[:n]:
        print(f"  CER={e/max(1,r):.3f} WER={wer(ref, hyp):.3f}")
        print(f"    typed:    {ref!r}")
        print(f"    produced: {hyp!r}")


def cmd_examples(a) -> int:
    s = session_path(a.session)
    kf = kbd_fit()
    rows = _table(a)
    r = nested_cv(s, rows, FULL, kf, verbose=False)
    report("FULL", r)
    _examples(r, a.examples or 20)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.pipeline", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("sweep", cmd_sweep), ("cv", cmd_cv), ("ablate", cmd_ablate),
                     ("detectors", cmd_detectors), ("examples", cmd_examples),
                     ("probe", cmd_probe)):
        p = sub.add_parser(name)
        p.add_argument("--session", default=DESK)
        p.add_argument("--table", default="data/sessions/desk_sweep.json")
        p.add_argument("--json", default=None)
        p.add_argument("--top", type=int, default=20)
        p.add_argument("--examples", type=int, default=6)
        p.add_argument("--contact-w", type=float, default=0.0)
        p.add_argument("--old-taps", default=OLD_TAPS)
        p.add_argument("--holdout", default="20260910-015948-kbd")
        p.add_argument("--shard", type=int, default=0)
        p.add_argument("--nshards", type=int, default=1)
        p.set_defaults(func=fn)
    a = ap.parse_args(argv)
    t0 = time.time()
    r = a.func(a)
    print(f"\n[{time.time()-t0:.1f}s]", file=sys.stderr)
    return r


if __name__ == "__main__":
    sys.exit(main())
