"""Joint decoding over the tap stream: motor transitions + inter-key timing over decode.py.
Run: python -m phase0.analysis.seqdecode {stats | taps | decode | ablate}"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phase0.analysis.analyze_drift import MODIFIERS, read_jsonl
from phase0.analysis.decode import (
    A_INDEX,
    ALPHABET,
    NA,
    CharLM,
    KeyProbsSpatial,
    Weights,
    WordLM,
    beam_decode,
    cer,
    desk_segments,
    edit_distance,
    kbd_segments,
    labelled_taps,
)
from phase0.analysis.finger_id import HAND_NAMES, KEY_LABEL

TRAIN_SESSIONS = ("20260910-015217-kbd", "20260910-021315-kbd", "20260910-131629-kbd")
HELDOUT_SESSION = "20260910-015948-kbd"
DESK_SESSION = "20260910-202149-desk"

# 9 motor classes: thumb (either hand) + 4 fingers x 2 hands, matching finger_id.CLASSES9
CLASSES9 = [("Either", 0)] + [(h, f) for h in HAND_NAMES for f in (1, 2, 3, 4)]
CLS_NAME = ["thumb"] + [f"{h[0]}-{n}" for h in HAND_NAMES
                        for n in ("index", "middle", "ring", "pinky")]
NC = len(CLASSES9)
THUMB = 0
TRANS_KINDS = ("same_finger", "same_hand", "alt_hand", "thumb")

# char -> motor class, from the touch-typing key map; the only place the layout is used
CHAR_CLS = np.full(NA, -1, int)
for _c in ALPHABET:
    _lab = KEY_LABEL.get("space" if _c == " " else _c)
    if _lab is not None:
        CHAR_CLS[A_INDEX[_c]] = CLASSES9.index(_lab)
CHAR_HAND = np.array([-1 if c <= 0 else (c - 1) // 4 for c in CHAR_CLS])


def sess_path(sid: str) -> Path:
    p = Path(sid)
    return p if p.exists() else Path("data/sessions") / sid


def trans_kind(a: int, b: int) -> str:
    if a == THUMB or b == THUMB:
        return "thumb"
    if a == b:
        return "same_finger"
    return "same_hand" if (a - 1) // 4 == (b - 1) // 4 else "alt_hand"


KIND_ID = {k: i for i, k in enumerate(TRANS_KINDS)}
# precomputed [NC,NC] kind ids so the beam never calls trans_kind in its inner loop
KIND_MAT = np.array([[KIND_ID[trans_kind(a, b)] for b in range(NC)] for a in range(NC)])


def _lse(a: np.ndarray) -> float:
    m = float(a.max())
    return m + float(np.log(np.exp(a - m).sum()))


# ------------------------------------------------------------------ ground truth
def key_sequence(session: Path) -> tuple[list[str], np.ndarray]:
    """(chars, keydown times) for the decodable alphabet, in order."""
    rows = [r for r in read_jsonl(session / "keys.jsonl") if r.get("event") == "down"]
    ch, t = [], []
    for r in rows:
        k = r.get("key")
        if k in MODIFIERS or k == "unknown":
            continue
        c = " " if k == "space" else k
        if c in A_INDEX:
            ch.append(c)
            t.append(float(r["t"]))
    return ch, np.asarray(t, float)


# ------------------------------------------------------------------ motor statistics
@dataclass
class MotorStats:
    """log P(class_b | class_a) and a per-kind log-normal over the inter-key interval."""

    logtrans: np.ndarray          # [NC, NC]
    mu: np.ndarray                # [4] mean of log dt per kind
    sd: np.ndarray                # [4]
    prior: np.ndarray             # [NC] log unigram class prior
    counts: dict
    iki: dict

    def logtime(self, dt: float) -> np.ndarray:
        """log density of this interval under each transition kind."""
        x = math.log(max(dt, 1e-3))
        return -0.5 * ((x - self.mu) / self.sd) ** 2 - np.log(self.sd) - 0.5 * math.log(2 * math.pi)

    def ratio(self, prev: int, dt: float | None, trans_w: float, time_w: float) -> np.ndarray:
        """Motor evidence as a log-ratio against the class marginal, so it re-ranks the LM's
        distribution instead of adding a constant emission cost that the beam reads as noise."""
        out = trans_w * (self.logtrans[prev] - self.prior)
        if dt is not None and time_w:
            tl = self.logtime(dt)[KIND_MAT[prev]]
            base = _lse(self.prior + tl)
            out = out + time_w * (tl - base)
        return out

    def to_json(self) -> dict:
        return {"logtrans": self.logtrans.tolist(), "mu": self.mu.tolist(),
                "sd": self.sd.tolist(), "prior": self.prior.tolist(),
                "counts": self.counts, "iki": self.iki}


def fit_motor(sessions: list[str], max_dt: float = 2.0, alpha: float = 1.0) -> MotorStats:
    pair = np.zeros((NC, NC))
    uni = np.zeros(NC)
    logs: dict[str, list[float]] = defaultdict(list)
    raw: dict[str, list[float]] = defaultdict(list)
    for sid in sessions:
        ch, t = key_sequence(sess_path(sid))
        cls = [int(CHAR_CLS[A_INDEX[c]]) for c in ch]
        for c in cls:
            uni[c] += 1
        for i in range(1, len(cls)):
            a, b = cls[i - 1], cls[i]
            pair[a, b] += 1
            dt = float(t[i] - t[i - 1])
            # a pause between phrases is not a motor transition, it is a decision to stop
            if 0.0 < dt <= max_dt:
                k = trans_kind(a, b)
                logs[k].append(math.log(dt))
                raw[k].append(dt)
    sm = pair + alpha
    logtrans = np.log(sm / sm.sum(1, keepdims=True))
    prior = np.log((uni + alpha) / (uni + alpha).sum())
    mu = np.zeros(len(TRANS_KINDS))
    sd = np.ones(len(TRANS_KINDS))
    allv = np.concatenate([np.asarray(v) for v in logs.values()]) if logs else np.zeros(1)
    for k, i in KIND_ID.items():
        v = np.asarray(logs.get(k, []))
        # fewer than 20 samples cannot support a free mean; fall back to the pooled interval
        mu[i], sd[i] = (v.mean(), max(v.std(), 0.15)) if len(v) >= 20 else (allv.mean(), max(allv.std(), 0.15))
    iki = {k: {"n": len(raw.get(k, [])),
               "median_ms": float(np.median(raw[k]) * 1000) if raw.get(k) else float("nan"),
               "mean_ms": float(np.mean(raw[k]) * 1000) if raw.get(k) else float("nan"),
               "p25_ms": float(np.percentile(raw[k], 25) * 1000) if raw.get(k) else float("nan"),
               "p75_ms": float(np.percentile(raw[k], 75) * 1000) if raw.get(k) else float("nan")}
           for k in TRANS_KINDS}
    counts = {"pairs": int(pair.sum()), "keys": int(uni.sum())}
    return MotorStats(logtrans, mu, sd, prior, counts, iki)


# ------------------------------------------------------------------ joint beam search
@dataclass(frozen=True)
class MotorWeights:
    trans: float = 1.0       # weight on log P(finger_b | finger_a)
    time: float = 1.0        # weight on log p(dt | transition kind)
    lm_discount: float = 0.0  # subtracted from the char LM weight when the motor layer is on


def joint_beam_decode(obs_logp: np.ndarray, times: np.ndarray, char_lm: CharLM | None,
                      word_lm: WordLM | None, motor: MotorStats | None,
                      w: Weights = Weights(), mw: MotorWeights = MotorWeights(),
                      beam: int = 30) -> tuple[str, list[tuple[int, int]]]:
    """decode.py's beam plus a motor state: each hypothesis remembers the finger class and
    the timestamp of its last tap-emitted character and pays for the transition to the next."""
    from phase0.analysis.decode import _ctx, _extra, _last

    if motor is None:
        text = beam_decode(obs_logp, char_lm, word_lm, w, beam)
        return text, []

    # hypothesis key = (text, last motor class, index of the tap that emitted it)
    hyps: dict[tuple, tuple[float, tuple]] = {("", -1, -1): (0.0, ())}
    lmcache: dict[str, np.ndarray] = {}

    def lmrow(text: str) -> np.ndarray:
        if char_lm is None:
            return np.zeros(NA)
        c = _ctx(text)
        r = lmcache.get(c)
        if r is None:
            r = lmcache[c] = char_lm.logprobs(c)
        return r

    for i in range(len(obs_logp) + 1):
        hyps = _close_motor_deletions(hyps, char_lm, word_lm, motor, w, mw, beam, lmrow)
        if i == len(obs_logp):
            break
        row = obs_logp[i]
        nxt: dict[tuple, tuple[float, tuple]] = {}
        for (text, pcls, pidx), (sc, tr) in hyps.items():
            _push2(nxt, (text, pcls, pidx), sc + w.insertion, tr)
            lm = lmrow(text)
            mrow = (motor.ratio(pcls, float(times[i] - times[pidx]), mw.trans, mw.time)
                    if pidx >= 0 else np.zeros(NC))
            for c in range(NA):
                cc = int(CHAR_CLS[c])
                if cc < 0:
                    continue
                _push2(nxt, (text + ALPHABET[c], cc, i),
                       sc + w.obs * row[c] + (w.char - mw.lm_discount) * lm[c]
                       + _extra(text, ALPHABET[c], word_lm, w) + float(mrow[cc]),
                       tr + ((i, c),))
        hyps = _topk2(nxt, beam)
    best = max(hyps, key=lambda k: hyps[k][0] + (w.word * word_lm.word_score(_last(k[0]))
                                                 if word_lm and _last(k[0]) else 0.0))
    return best[0].strip(), list(hyps[best][1])


def _push2(d: dict, key, score: float, trace: tuple):
    cur = d.get(key)
    if cur is None or score > cur[0]:
        d[key] = (score, trace)


def _topk2(d: dict, beam: int) -> dict:
    if len(d) <= beam:
        return d
    return dict(sorted(d.items(), key=lambda kv: -kv[1][0])[:beam])


def _close_motor_deletions(hyps, char_lm, word_lm, motor, w, mw, beam, lmrow):
    """Characters emitted with no tap: they still move the finger, but carry no interval."""
    from phase0.analysis.decode import _extra

    out = dict(hyps)
    frontier = hyps
    for _ in range(w.max_deletions):
        nxt: dict = {}
        for (text, pcls, pidx), (sc, tr) in frontier.items():
            lm = lmrow(text)
            mrow = motor.ratio(pcls, None, mw.trans, 0.0) if pcls >= 0 else np.zeros(NC)
            for c in range(NA):
                cc = int(CHAR_CLS[c])
                if cc < 0:
                    continue
                _push2(nxt, (text + ALPHABET[c], cc, pidx),
                       sc + w.deletion + w.char * lm[c] + _extra(text, ALPHABET[c], word_lm, w)
                       + float(mrow[cc]), tr)
        frontier = _topk2(nxt, beam)
        for k, v in frontier.items():
            _push2(out, k, v[0], v[1])
    return _topk2(out, beam)


# ------------------------------------------------------------------ per-tap evaluation
def tap_truth(session: Path, taps_name: str) -> tuple[list[dict], list[str]]:
    return labelled_taps(session, taps_name)


def per_tap_scores(taps: list[dict], truth: list[str], pred: list[int]) -> dict:
    """key / finger / hand accuracy of an integer key prediction per tap."""
    y = np.array([A_INDEX[c] for c in truth])
    p = np.asarray(pred)
    ok = p >= 0
    n = max(1, len(p))
    hit = {"key": p[ok] == y[ok], "finger": CHAR_CLS[p[ok]] == CHAR_CLS[y[ok]],
           "hand": CHAR_HAND[p[ok]] == CHAR_HAND[y[ok]]}
    out = {"n": len(p), "covered": float(ok.mean()) if len(p) else float("nan")}
    for k, v in hit.items():
        # an uncovered tap (the beam called it spurious) is scored as an error, not dropped
        out[k] = float(v.sum()) / n
        out[k + "_cov"] = float(v.mean()) if ok.any() else float("nan")
    return out


def keyprobs_matrix(session: Path, taps: list[dict]) -> np.ndarray:
    return KeyProbsSpatial().logp(session, taps)


# ------------------------------------------------------------------ segments
def segments_for(session: Path, taps: list[dict], control: bool):
    return kbd_segments(session, taps) if control else desk_segments(session, taps)


def decode_segments(segments, session: Path, char_lm, word_lm, motor, w, mw, beam, oracle=None):
    rows, tot_c, ref_c = [], 0, 0
    for text, taps in segments:
        if not taps:
            rows.append((text, "", []))
            tot_c += len(text)
            ref_c += len(text)
            continue
        lp = keyprobs_matrix(session, taps)
        if oracle is not None:
            lp = lp + oracle.bonus(taps)
        ts = np.array([t["t"] for t in taps], float)
        hyp, trace = joint_beam_decode(lp, ts, char_lm, word_lm, motor, w, mw, beam)
        rows.append((text, hyp, trace))
        tot_c += edit_distance(text, hyp)
        ref_c += len(text)
    return tot_c / max(1, ref_c), rows


# ------------------------------------------------------------------ CLI
def _load_lm(no_lm: bool, no_word: bool, lm: str):
    if no_lm:
        return None, None
    c = CharLM.load(Path(lm))
    wl = None if no_word else WordLM.load(Path(lm).with_suffix(".words.json"))
    return c, wl


def cmd_stats(a) -> int:
    ms = fit_motor(list(a.sessions))
    print(f"fitted on {', '.join(a.sessions)}: {ms.counts['keys']} keydowns, "
          f"{ms.counts['pairs']} transitions\n")
    print(f"{'transition':<14}{'n':>7}{'median ms':>11}{'mean ms':>10}{'p25':>8}{'p75':>8}")
    for k in TRANS_KINDS:
        d = ms.iki[k]
        print(f"{k:<14}{d['n']:>7}{d['median_ms']:>11.1f}{d['mean_ms']:>10.1f}"
              f"{d['p25_ms']:>8.1f}{d['p75_ms']:>8.1f}")
    tot = sum(ms.iki[k]["n"] for k in TRANS_KINDS)
    nz = {k: ms.iki[k]["n"] / max(1, tot) for k in TRANS_KINDS}
    print("\nshare of transitions: " + "  ".join(f"{k}={v:.3f}" for k, v in nz.items()))
    nonthumb = tot - ms.iki["thumb"]["n"]
    if nonthumb:
        print(f"hand-alternation rate (excluding thumb/space): "
              f"{ms.iki['alt_hand']['n'] / nonthumb:.3f}")
    print("\nfinger transition matrix P(next | prev), rows=prev:")
    print(f"{'':<10}" + "".join(f"{n:>9}" for n in CLS_NAME))
    P = np.exp(ms.logtrans)
    for i, n in enumerate(CLS_NAME):
        print(f"{n:<10}" + "".join(f"{v:>9.3f}" for v in P[i]))
    print("\ntransition kind predicted from the interval alone:")
    for tag, sids in (("fit", list(a.sessions)), *[("held", [s]) for s in a.check]):
        tp = timing_power(ms, sids)
        print(f"  {tag:<5}{','.join(s[9:] for s in sids):<26} n={tp['n']:>5} "
              f"prior_acc={tp['prior_acc']:.3f} bayes_acc={tp['bayes_acc']:.3f} "
              f"info={tp['info_nats']:.3f} of {tp['H_prior_nats']:.3f} nats")
    if a.json:
        Path(a.json).write_text(json.dumps(ms.to_json(), indent=1))
    if a.check:
        for sid in a.check:
            ch, t = key_sequence(sess_path(sid))
            cls = [int(CHAR_CLS[A_INDEX[c]]) for c in ch]
            per = defaultdict(list)
            for i in range(1, len(cls)):
                dt = t[i] - t[i - 1]
                if 0 < dt <= 2.0:
                    per[trans_kind(cls[i - 1], cls[i])].append(dt)
            print(f"\n[held-out check] {sid}: " + "  ".join(
                f"{k}={np.median(per[k]) * 1000:.0f}ms(n={len(per[k])})" for k in TRANS_KINDS if per[k]))
    return 0


def timing_power(ms: MotorStats, sessions: list[str], max_dt: float = 2.0) -> dict:
    """How much can the interval alone say about the transition kind? Bayes accuracy and
    mutual information against the kind prior, on sessions the model may or may not have seen."""
    prior = np.array([ms.iki[k]["n"] for k in TRANS_KINDS], float)
    prior = np.log(prior / prior.sum())
    y, post = [], []
    for sid in sessions:
        ch, t = key_sequence(sess_path(sid))
        cls = [int(CHAR_CLS[A_INDEX[c]]) for c in ch]
        for i in range(1, len(cls)):
            dt = float(t[i] - t[i - 1])
            if not 0 < dt <= max_dt:
                continue
            y.append(KIND_ID[trans_kind(cls[i - 1], cls[i])])
            lp = prior + ms.logtime(dt)
            post.append(lp - _lse(lp))
    y = np.asarray(y)
    P = np.asarray(post)
    base = float((y == np.argmax(prior)).mean())
    acc = float((P.argmax(1) == y).mean())
    h0 = -float((np.exp(prior) * prior).sum())
    h1 = -float(P[np.arange(len(y)), y].mean())
    return {"n": len(y), "prior_acc": base, "bayes_acc": acc,
            "H_prior_nats": h0, "H_given_dt_nats": h1, "info_nats": h0 - h1}


class FingerOracle:
    """Diagnostic ceiling: the true motor class of each tap, from the keylogger."""

    def __init__(self, session: Path, taps_name: str, penalty: float = 12.0):
        taps, keys = labelled_taps(session, taps_name)
        self.cls = {round(t["t"], 6): int(CHAR_CLS[A_INDEX[k]]) for t, k in zip(taps, keys)}
        self.penalty = penalty

    def bonus(self, taps: list[dict]) -> np.ndarray:
        out = np.zeros((len(taps), NA))
        for i, t in enumerate(taps):
            c = self.cls.get(round(t["t"], 6))
            if c is not None:
                out[i] = np.where(CHAR_CLS == c, 0.0, -self.penalty)
        return out


def cmd_taps(a) -> int:
    """Per-tap key/finger/hand accuracy: independent argmax vs joint decoding."""
    session = sess_path(a.session)
    taps = read_jsonl(session / a.taps_name)
    gt_taps, gt_keys = tap_truth(session, a.taps_name)
    truth = {round(t["t"], 6): k for t, k in zip(gt_taps, gt_keys)}
    motor = fit_motor(list(a.fit_sessions))
    char_lm, word_lm = _load_lm(a.no_lm, a.no_word_lm, a.lm)
    w = Weights()
    segs = segments_for(session, taps, control=True)

    base_pred, joint_pred, ref = [], [], []
    mw = MotorWeights(trans=a.trans_weight, time=a.time_weight, lm_discount=a.lm_discount)
    for _, stap in segs:
        if not stap:
            continue
        lp = keyprobs_matrix(session, stap)
        ts = np.array([t["t"] for t in stap], float)
        _, trace = joint_beam_decode(lp, ts, char_lm, word_lm, motor, w, mw, a.beam)
        assign = dict(trace)
        arg = lp.argmax(1)
        for i, tp in enumerate(stap):
            k = truth.get(round(tp["t"], 6))
            if k is None:
                continue
            ref.append(k)
            base_pred.append(int(arg[i]))
            joint_pred.append(int(assign.get(i, -1)))
    b = per_tap_scores([], ref, base_pred)
    j = per_tap_scores([], ref, joint_pred)
    print(f"{session.name} [{a.taps_name}] {len(ref)} keystroke-paired taps  "
          f"(lm={'off' if a.no_lm else 'on'} trans={a.trans_weight} time={a.time_weight})")
    hdr = f"{'':<10}{'n':>6}{'key':>8}{'finger':>8}{'hand':>8}{'cov':>7}{'key|c':>8}{'fing|c':>8}"
    print(hdr)
    for nm, r in (("argmax", b), ("joint", j)):
        print(f"{nm:<10}{r['n']:>6}{r['key']:>8.3f}{r['finger']:>8.3f}{r['hand']:>8.3f}"
              f"{r['covered']:>7.3f}{r['key_cov']:>8.3f}{r['finger_cov']:>8.3f}")
    return 0


def cmd_decode(a) -> int:
    session = sess_path(a.session)
    taps = read_jsonl(session / a.taps_name)
    motor = None if a.no_motor else fit_motor(list(a.fit_sessions))
    char_lm, word_lm = _load_lm(a.no_lm, a.no_word_lm, a.lm)
    w = Weights()
    mw = MotorWeights(trans=a.trans_weight, time=a.time_weight, lm_discount=a.lm_discount)
    segs = segments_for(session, taps, a.control)
    oracle = FingerOracle(session, a.taps_name) if a.oracle_finger else None
    c, rows = decode_segments(segs, session, char_lm, word_lm, motor, w, mw, a.beam, oracle)
    if a.verbose:
        for ref, hyp, _ in rows:
            print(f"  {cer(ref, hyp):.3f}  {hyp[:70]!r}\n         {ref[:70]!r}")
    print(f"OVERALL {session.name} motor={'off' if a.no_motor else f'{a.trans_weight}/{a.time_weight}'} "
          f"lm={'off' if a.no_lm else 'on'}: CER={c:.3f} over {len(segs)} segments")
    return 0


ABLATIONS = (("none", 0.0, 0.0), ("trans", 1.0, 0.0), ("time", 0.0, 1.0), ("both", 1.0, 1.0))


def cmd_ablate(a) -> int:
    motor = fit_motor(list(a.fit_sessions))
    w = Weights()
    targets = []
    for spec in a.targets:
        sid, ctrl = (spec[:-1], True) if spec.endswith("!") else (spec, False)
        targets.append((sess_path(sid), ctrl))
    for lm_on in (True, False):
        char_lm, word_lm = _load_lm(not lm_on, a.no_word_lm, a.lm)
        for sp, ctrl in targets:
            taps = read_jsonl(sp / a.taps_name)
            segs = segments_for(sp, taps, ctrl)
            out, per = [], {}
            for name, tw, mt in ABLATIONS:
                m = None if name == "none" else motor
                mw = MotorWeights(trans=tw, time=mt, lm_discount=a.lm_discount)
                c, rows = decode_segments(segs, sp, char_lm, word_lm, m, w, mw, a.beam)
                per[name] = np.array([[edit_distance(r, h), len(r)] for r, h, _ in rows], float)
                out.append(f"{name}={c:.3f}")
            print(f"{sp.name:<26} lm={'on ' if lm_on else 'off'}  " + "  ".join(out), flush=True)
            if a.bootstrap:
                rng = np.random.default_rng(0)
                idx = rng.integers(0, len(segs), (a.bootstrap, len(segs)))
                base = per["none"][idx]
                for name in ("trans", "time", "both"):
                    d = (per[name][idx][:, :, 0].sum(1) / base[:, :, 1].sum(1)
                         - base[:, :, 0].sum(1) / base[:, :, 1].sum(1))
                    print(f"{'':<26}   d({name}) = {d.mean():+.3f} "
                          f"[{np.percentile(d, 5):+.3f}, {np.percentile(d, 95):+.3f}]")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stats", help="measure the motor structure in keys.jsonl")
    s.add_argument("--sessions", nargs="+", default=list(TRAIN_SESSIONS))
    s.add_argument("--check", nargs="*", default=[], help="also print IKI on these (not fitted)")
    s.add_argument("--json", default=None)
    s.set_defaults(func=cmd_stats)

    def common(p):
        p.add_argument("--fit-sessions", nargs="+", default=list(TRAIN_SESSIONS))
        p.add_argument("--taps-name", default="taps_pos.jsonl")
        p.add_argument("--lm", default="models/charlm.npz")
        p.add_argument("--beam", type=int, default=30)
        p.add_argument("--trans-weight", type=float, default=1.0)
        p.add_argument("--time-weight", type=float, default=1.0)
        p.add_argument("--lm-discount", type=float, default=0.0)
        p.add_argument("--no-lm", action="store_true")
        p.add_argument("--no-word-lm", action="store_true")

    t = sub.add_parser("taps", help="per-tap key/finger accuracy, argmax vs joint")
    t.add_argument("session")
    common(t)
    t.set_defaults(func=cmd_taps)

    d = sub.add_parser("decode", help="CER with the motor layer on")
    d.add_argument("session")
    d.add_argument("--control", action="store_true")
    d.add_argument("--no-motor", action="store_true")
    d.add_argument("--oracle-finger", action="store_true", help="ceiling: true motor class per tap")
    d.add_argument("--verbose", action="store_true")
    common(d)
    d.set_defaults(func=cmd_decode)

    b = sub.add_parser("ablate", help="none/trans/time/both x lm on/off on several sessions")
    b.add_argument("targets", nargs="+", help="session id; suffix ! to score it as a kbd control")
    b.add_argument("--bootstrap", type=int, default=0, help="segment-resampling CI on the delta")
    common(b)
    b.set_defaults(func=cmd_ablate)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
