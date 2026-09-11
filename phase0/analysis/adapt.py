"""Weakly-supervised adaptation: the typed TEXT is known, the tap->character alignment is not.
Run: python -m phase0.analysis.adapt {em | cv | proxy | probe | perm} ..."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from phase0.analysis.analyze_drift import read_jsonl
from phase0.analysis.decode import (
    A_INDEX,
    ALPHABET,
    NA,
    CharLM,
    KeyProbsSpatial,
    Weights,
    WordLM,
    beam_decode,
    desk_segments,
    edit_distance,
    kbd_segments,
    labelled_taps,
)
from phase0.analysis import tap_pos as tp

# .cache and models/*.pkl were pickled by tap_pos running as __main__
for _n in ("Sess", "KeyClf", "XYReg", "GaussXY", "Selector"):
    setattr(sys.modules["__main__"], _n, getattr(tp, _n))

DESK = "20260910-202149-desk"
KBD_TRAIN = ("20260910-015217-kbd", "20260910-021315-kbd", "20260910-131629-kbd")
LM_PATH = Path("models/charlm.npz")
EPS = 1e-12
NEG = -1e30

# Reduced pose view for the adapted head: 591 desk taps cannot support tap_pos's 1266 columns.
ADAPT_JOINTS = tp.JOINTS_TIPS
ADAPT_OFFSETS = (-4, 0, 4)


# --------------------------------------------------------------------- alignment lattice
@dataclass(frozen=True)
class AlignParams:
    """Tap-level transition model. p_ins: tap emitted no character (spurious detection).
    p_skip: geometric chance that the next character had no tap at all (missed detection)."""

    p_ins: float = 0.08
    p_skip: float = 0.20
    max_skip: int = 3

    @staticmethod
    def from_counts(n_taps: int, n_chars: int, p_ins: float = 0.08) -> "AlignParams":
        """Skip rate is not free: (1-p_ins)*taps/(1-p_skip) characters must get consumed."""
        s = 1.0 - (1.0 - p_ins) * max(n_taps, 1) / max(n_chars, 1)
        return AlignParams(p_ins=p_ins, p_skip=float(np.clip(s, 0.01, 0.5)))


def _lse(v: np.ndarray) -> float:
    m = float(np.max(v))
    if m <= NEG / 2:
        return NEG
    return m + float(np.log(np.exp(v - m).sum()))


def forward_backward(obs_logp: np.ndarray, text: str, par: AlignParams):
    """Soft monotonic alignment of N taps to M known characters.
    -> (loglik, q[N,NA] emission posteriors, q_ins[N]); lattice state j = characters consumed."""
    N = len(obs_logp)
    codes = [A_INDEX[c] for c in text if c in A_INDEX]
    M = len(codes)
    l_ins = math.log(max(par.p_ins, EPS))
    l_emit = math.log(max(1.0 - par.p_ins, EPS))
    l_skip = math.log(max(par.p_skip, EPS))
    l_stay = math.log(max(1.0 - par.p_skip, EPS))
    # cost of consuming d characters on one tap: (d-1) skips then the emission
    step = [l_emit + (d - 1) * l_skip + l_stay for d in range(1, par.max_skip + 2)]

    alpha = np.full((N + 1, M + 1), NEG)
    alpha[0, 0] = 0.0
    for j in range(1, min(par.max_skip, M) + 1):  # characters typed before the first tap
        alpha[0, j] = j * l_skip
    for n in range(N):
        row = obs_logp[n]
        nxt = np.full(M + 1, NEG)
        for j in range(M + 1):
            a = alpha[n, j]
            if a <= NEG / 2:
                continue
            v = a + l_ins
            if v > nxt[j]:
                nxt[j] = np.logaddexp(nxt[j], v) if nxt[j] > NEG / 2 else v
            for d in range(1, par.max_skip + 2):
                if j + d > M:
                    break
                v = a + step[d - 1] + row[codes[j + d - 1]]
                nxt[j + d] = np.logaddexp(nxt[j + d], v) if nxt[j + d] > NEG / 2 else v
        alpha[n + 1] = nxt

    beta = np.full((N + 1, M + 1), NEG)
    for j in range(M + 1):  # characters typed after the last tap
        if M - j <= par.max_skip:
            beta[N, j] = (M - j) * l_skip
    for n in range(N - 1, -1, -1):
        row = obs_logp[n]
        cur = np.full(M + 1, NEG)
        for j in range(M + 1):
            terms = [beta[n + 1, j] + l_ins]
            for d in range(1, par.max_skip + 2):
                if j + d > M:
                    break
                terms.append(beta[n + 1, j + d] + step[d - 1] + row[codes[j + d - 1]])
            cur[j] = _lse(np.array(terms))
        beta[n] = cur

    ll = beta[0, 0]
    q = np.zeros((N, NA))
    q_ins = np.zeros(N)
    if ll <= NEG / 2:  # lattice unreachable (e.g. far too few taps); fall back to no evidence
        return float("-inf"), np.full((N, NA), 1.0 / NA), np.ones(N)
    for n in range(N):
        row = obs_logp[n]
        q_ins[n] = math.exp(min(0.0, _lse(alpha[n] + l_ins + beta[n + 1]) - ll))
        for j in range(M + 1):
            a = alpha[n, j]
            if a <= NEG / 2:
                continue
            for d in range(1, par.max_skip + 2):
                if j + d > M:
                    break
                c = codes[j + d - 1]
                v = a + step[d - 1] + row[c] + beta[n + 1, j + d] - ll
                if v > -60:
                    q[n, c] += math.exp(v)
    tot = q.sum(1) + q_ins
    q_ins = q_ins / np.maximum(tot, EPS)
    q = q / np.maximum(q.sum(1, keepdims=True), EPS)
    return float(ll), q, q_ins


def align_all(obs_logp: np.ndarray, segs, par: AlignParams):
    """Run the lattice over every segment. -> (q[N,NA], q_ins[N], per-char loglik, n_chars)."""
    q = np.full((len(obs_logp), NA), 1.0 / NA)
    q_ins = np.ones(len(obs_logp))
    ll = 0.0
    nchar = 0
    for text, idx in segs:
        if len(idx) == 0:
            continue
        l, qq, qi = forward_backward(obs_logp[idx], text, par)
        if not math.isfinite(l):
            continue
        q[idx], q_ins[idx] = qq, qi
        ll += l
        nchar += sum(c in A_INDEX for c in text)
    return q, q_ins, ll / max(1, nchar), nchar


def reestimate(par: AlignParams, q_ins: np.ndarray, segs) -> AlignParams:
    """M-step for the transition model itself: insertion rate from posteriors, skip rate
    from the residual character budget (taps that emitted vs characters that exist)."""
    used = [i for _, idx in segs for i in idx]
    if not used:
        return par
    p_ins = float(np.clip(q_ins[used].mean(), 0.01, 0.5))
    chars = sum(sum(c in A_INDEX for c in t) for t, idx in segs if len(idx))
    return replace(par, **AlignParams.from_counts(len(used), chars, p_ins).__dict__)


# --------------------------------------------------------------------------- soft-label head
class SoftmaxHead:
    """Multinomial logistic regression trained on SOFT targets (that is the whole point:
    LightGBM would need the posteriors hardened, which is exactly the collapse we fear)."""

    def __init__(self, l2: float = 4.0, iters: int = 300):
        self.l2, self.iters = l2, iters
        self.W = None
        self.mu = None
        self.sd = None

    def _z(self, X):
        return (X - self.mu) / self.sd

    def fit(self, X, Q, w=None, init=None):
        from scipy.optimize import minimize

        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        Z = np.hstack([self._z(X), np.ones((len(X), 1))])
        w = np.ones(len(X)) if w is None else np.asarray(w, float)
        w = w / max(w.sum(), EPS) * len(X)
        D = Z.shape[1]
        x0 = np.zeros(D * NA) if init is None else init.ravel().copy()

        def f(v):
            W = v.reshape(D, NA)
            s = Z @ W
            s -= s.max(1, keepdims=True)
            lse = np.log(np.exp(s).sum(1))
            logp = s - lse[:, None]
            loss = -(w[:, None] * Q * logp).sum() / len(Z) + self.l2 * (W[:-1] ** 2).sum() / len(Z)
            g = Z.T @ (w[:, None] * (np.exp(logp) * Q.sum(1, keepdims=True) - Q)) / len(Z)
            g[:-1] += 2 * self.l2 * W[:-1] / len(Z)
            return loss, g.ravel()

        r = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": self.iters})
        self.W = r.x.reshape(D, NA)
        return self

    def proba(self, X):
        s = np.hstack([self._z(X), np.ones((len(X), 1))]) @ self.W
        s -= s.max(1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(1, keepdims=True)


def mix(base: np.ndarray, new: np.ndarray, lam: float) -> np.ndarray:
    p = (1 - lam) * base + lam * new
    return p / p.sum(1, keepdims=True)


# ------------------------------------------------------------------------------ session data
class Target:
    """A session being adapted: taps, pose features, base key distribution, text segments."""

    def __init__(self, sid: str, taps_name: str, control: bool):
        self.sid = sid
        self.path = tp.sess_path(sid)
        self.sess = tp.load_sess(sid)
        self.taps = read_jsonl(self.path / taps_name)
        self.k = self.sess.rows(self.taps)
        self.X = tp.features(self.sess, self.k, "rel", ADAPT_JOINTS, ADAPT_OFFSETS)
        self.Xfull = None
        pos = self.path / "taps_pos.jsonl"
        self.base = np.exp(KeyProbsSpatial().logp(self.path, read_jsonl(pos))) if pos.exists() else None
        idx = {round(t["t"], 6): i for i, t in enumerate(self.taps)}
        segs = kbd_segments(self.path, self.taps) if control else desk_segments(self.path, self.taps)
        self.segs = [(t, np.array([idx[round(x["t"], 6)] for x in s], int)) for t, s in segs]
        self.truth = None
        if control:
            lt, lk = labelled_taps(self.path, taps_name)
            self.truth = np.full(len(self.taps), -1)
            for t, key in zip(lt, lk):
                j = idx.get(round(t["t"], 6))
                if j is not None:
                    self.truth[j] = A_INDEX[key]

    def full_features(self, model):
        if self.Xfull is None:
            self.Xfull = tp.features(self.sess, self.k, model.mode, model.joints, model.offsets)
        return self.Xfull


def load_base_model(path: Path):
    blob = pickle.loads(Path(path).read_bytes())
    return blob["model"]


def base_proba(target: Target, model_path: Path | None) -> np.ndarray:
    if model_path is None:
        if target.base is None:
            raise SystemExit("no taps_pos.jsonl and no --base-model: nothing to initialise from")
        return target.base
    m = load_base_model(model_path)
    return m.proba(target.sess, target.k)


# ------------------------------------------------------------------------------ diagnostics
def diagnostics(q, q_ins, segs, truth=None) -> dict:
    used = np.array([i for _, idx in segs for i in idx], int)
    if len(used) == 0:
        return {}
    qq = q[used]
    ent = -(qq * np.log(np.maximum(qq, EPS))).sum(1)
    d = {
        "ent": float(ent.mean()),
        "conf50": float((qq.max(1) > 0.5).mean()),
        "conf90": float((qq.max(1) > 0.9).mean()),
        "p_ins": float(q_ins[used].mean()),
        "topmass": float(np.bincount(qq.argmax(1), minlength=NA).max() / len(qq)),
    }
    if truth is not None:
        m = truth[used] >= 0
        d["q_acc"] = float((qq[m].argmax(1) == truth[used][m]).mean()) if m.any() else float("nan")
    return d


# ------------------------------------------------------------------------------ the procedure
def adapt(target: Target, segs, base: np.ndarray, iters: int = 6, lam: float = 0.5,
          l2: float = 4.0, mode: str = "em", par: AlignParams | None = None,
          reest: bool = False, verbose: bool = True, truth=None, harden: bool = False):
    """EM / CTC over the alignment lattice. mode 'em' refits the head to convergence each
    round; mode 'ctc' takes a few gradient steps, i.e. plain CTC-style descent."""
    if par is None:
        n = sum(len(i) for _, i in segs)
        par = AlignParams.from_counts(n, sum(sum(c in A_INDEX for c in t) for t, i in segs if len(i)))
    fit_idx = np.unique(np.concatenate([idx for _, idx in segs if len(idx)])) if segs else np.array([], int)
    obs = base.copy()
    head = None
    rows = []
    warm = None
    for it in range(iters + 1):
        q, q_ins, ll, nchar = align_all(np.log(np.maximum(obs, EPS)), segs, par)
        d = diagnostics(q, q_ins, segs, truth)
        d.update(it=it, ll=ll, nchar=nchar, p_ins_par=par.p_ins, p_skip_par=par.p_skip)
        rows.append(d)
        if verbose:
            print(f"  it{it}: ll/char={ll:+.3f} ent={d['ent']:.3f} conf>.5={d['conf50']:.2f} "
                  f"conf>.9={d['conf90']:.2f} p_ins={d['p_ins']:.2f} topmass={d['topmass']:.2f}"
                  + (f" q_acc={d['q_acc']:.3f}" if "q_acc" in d else ""), flush=True)
        if it == iters:
            break
        w = 1.0 - q_ins[fit_idx]
        tgt = q[fit_idx]
        if harden:  # Viterbi-style ablation: does soft posterior mass matter at all?
            tgt = np.zeros_like(tgt)
            tgt[np.arange(len(tgt)), q[fit_idx].argmax(1)] = 1.0
        head = SoftmaxHead(l2=l2, iters=60 if mode == "ctc" else 300)
        head.fit(target.X[fit_idx], tgt, w, init=warm)
        warm = head.W
        obs = mix(base, head.proba(target.X), lam)
        if reest:
            par = reestimate(par, q_ins, segs)
    return head, obs, rows, par


def self_train(target: Target, fit_idx, base, tau=0.6, l2=4.0, lam=0.5):
    """No text used at all: confidence-thresholded pseudo-labels from the base model."""
    m = base[fit_idx].max(1) >= tau
    if m.sum() < 20:
        return None, base
    Q = np.zeros((int(m.sum()), NA))
    Q[np.arange(m.sum()), base[fit_idx][m].argmax(1)] = 1.0
    head = SoftmaxHead(l2=l2).fit(target.X[fit_idx][m], Q)
    return head, mix(base, head.proba(target.X), lam)


def coral_proba(target: Target, model_path: Path, src_sessions, shrink=1.0):
    """Standardise desk features to the keyboard moments, then run the unmodified kbd model.
    Diagonal form: full CORAL on 1266 columns from 591 taps is a rank-deficient covariance."""
    model = load_base_model(model_path)
    Xt = target.full_features(model)
    src = []
    for sid in src_sessions:
        d = tp.dataset(sid)
        src.append(tp.features(d["sess"], d["k"], model.mode, model.joints, model.offsets))
    S = np.vstack(src)
    ms, ss = S.mean(0), S.std(0) + 1e-6
    mt, st = Xt.mean(0), Xt.std(0) + 1e-6
    Z = (Xt - mt) / st * (shrink * ss + (1 - shrink) * st) + (shrink * ms + (1 - shrink) * mt)
    return model.proba_from_features(Z) if hasattr(model, "proba_from_features") else _lgb_proba(model, Z)


def _lgb_proba(model, X):
    p = model.model.predict_proba(X)
    out = np.full((len(X), NA), 1e-6)
    out[:, model.classes] = np.maximum(p, 1e-6)
    return out / out.sum(1, keepdims=True)


# ------------------------------------------------------------------------------ scoring
_LM = {}


def lms(path=LM_PATH):
    if not _LM:
        _LM["c"] = CharLM.load(Path(path))
        _LM["w"] = WordLM.load(Path(path).with_suffix(".words.json"))
    return _LM["c"], _LM["w"]


def cer_on(segs, proba, beam=30):
    char_lm, word_lm = lms()
    err = ref = 0
    for text, idx in segs:
        ref += len(text)
        if len(idx) == 0:
            err += len(text)
            continue
        hyp = beam_decode(np.log(np.maximum(proba[idx], EPS)), char_lm, word_lm, Weights(), beam)
        err += edit_distance(text, hyp)
    return err / max(1, ref), err, ref


def tap_acc(proba, truth, idx=None):
    idx = np.arange(len(proba)) if idx is None else idx
    m = truth[idx] >= 0
    if not m.any():
        return float("nan"), 0
    return float((proba[idx][m].argmax(1) == truth[idx][m]).mean()), int(m.sum())


# ------------------------------------------------------------------------------ CLI: em
def cmd_em(a) -> int:
    t = Target(a.session, a.taps_name, a.control)
    base = base_proba(t, Path(a.base_model) if a.base_model else None)
    print(f"{a.session}: {len(t.taps)} taps, {len(t.segs)} segments, "
          f"{sum(len(x) for x, _ in t.segs)} known characters")
    ref = np.zeros(NA)
    for text, _ in t.segs:
        for c in text:
            if c in A_INDEX:
                ref[A_INDEX[c]] += 1
    ref /= max(ref.sum(), EPS)
    print(f"  reference marginal top: {ALPHABET[ref.argmax()]!r}={ref.max():.3f}")
    print(f"  base marginal top: {ALPHABET[base.mean(0).argmax()]!r}={base.mean(0).max():.3f} "
          f"TV(base,ref)={0.5*np.abs(base.mean(0)-ref).sum():.3f}")
    for mode in a.modes.split(","):
        print(f"\n[{mode}] lam={a.lam} l2={a.l2}")
        _, obs, rows, par = adapt(t, t.segs, base, a.iters, a.lam, a.l2, mode,
                                  reest=a.reest, truth=t.truth)
        m = obs.mean(0)
        print(f"  adapted marginal top: {ALPHABET[m.argmax()]!r}={m.max():.3f} "
              f"TV(adapted,ref)={0.5*np.abs(m-ref).sum():.3f} "
              f"(p_ins={par.p_ins:.2f} p_skip={par.p_skip:.2f})")
        if t.truth is not None:
            print(f"  per-tap top1: base={tap_acc(base, t.truth)[0]:.3f} "
                  f"adapted={tap_acc(obs, t.truth)[0]:.3f}  (FIT DATA, not held out)")
        if a.json:
            Path(a.json).write_text(json.dumps(rows, indent=1))
    return 0


# ------------------------------------------------------------------------------ CLI: cv
def folds(n, k, seed=0):
    r = np.random.RandomState(seed).permutation(n)
    return [np.sort(r[i::k]) for i in range(k)]


def method_proba(name, t: Target, tr, base, a, verbose=False):
    """name may carry a mixing weight, e.g. 'em@1.0' overrides --lam for that method only."""
    lam = a.lam
    if "@" in name:
        name, v = name.split("@")
        lam = float(v)
    if name.startswith("coral+"):  # standardise the features first, then adapt from there
        base = method_proba("coral", t, tr, base, a)
        name = name[6:]
    fit_idx = np.unique(np.concatenate([i for _, i in tr])) if tr else np.array([], int)
    if name == "uniform":
        return np.full_like(base, 1.0 / NA)
    if name == "base":
        return base
    if name in ("em", "ctc"):
        return adapt(t, tr, base, a.iters, lam, a.l2, name, reest=a.reest,
                     verbose=verbose, truth=t.truth)[1]
    if name == "hard":
        return adapt(t, tr, base, a.iters, lam, a.l2, "em", reest=a.reest,
                     verbose=verbose, truth=t.truth, harden=True)[1]
    if name == "selftrain":
        return self_train(t, fit_idx, base, a.tau, a.l2, lam)[1]
    if name == "coral":
        src = [s for s in KBD_TRAIN if s != a.session]
        return coral_proba(t, Path(a.base_model or "models/tap_pos.pkl"), src)
    if name == "sup":  # proxy-only ceiling: the same head trained on the hidden true labels
        m = t.truth[fit_idx] >= 0
        Q = np.zeros((int(m.sum()), NA))
        Q[np.arange(m.sum()), t.truth[fit_idx][m]] = 1.0
        head = SoftmaxHead(l2=a.l2).fit(t.X[fit_idx][m], Q)
        return mix(base, head.proba(t.X), lam)
    raise SystemExit(f"unknown method {name}")


def cmd_cv(a) -> int:
    """Primary metric: fit the adaptation on a subset of phrases, decode the rest."""
    t = Target(a.session, a.taps_name, a.control)
    base = base_proba(t, Path(a.base_model) if a.base_model else None)
    segs = [s for s in t.segs if len(s[1])]
    print(f"{a.session}: {len(t.taps)} taps, {len(segs)} usable segments, beam={a.beam}")
    methods = a.methods.split(",")
    acc = {m: [0, 0] for m in methods}
    per_fold = {m: [] for m in methods}
    dump: dict = {}
    for fi, te in enumerate(folds(len(segs), a.folds, a.seed)):
        tr = [segs[i] for i in range(len(segs)) if i not in set(te.tolist())]
        ev = [segs[i] for i in te]
        line = [f"fold{fi}({len(ev)})"]
        for m in methods:
            p = method_proba(m, t, tr, base, a)
            c, e, r = cer_on(ev, p, a.beam)
            for j in te:
                ec, _, rc = cer_on([segs[j]], p, a.beam)
                dump.setdefault(m, {})[int(j)] = [ec * rc, rc]
            acc[m][0] += e
            acc[m][1] += r
            per_fold[m].append(c)
            line.append(f"{m}={c:.3f}")
        print("  " + "  ".join(line), flush=True)
    print(f"\n{'method':<12}{'pooled CER':>12}{'mean fold':>12}{'sd fold':>10}")
    for m in methods:
        v = np.array(per_fold[m])
        print(f"{m:<12}{acc[m][0]/max(1,acc[m][1]):>12.3f}{v.mean():>12.3f}{v.std(ddof=1):>10.3f}")
    if a.json:
        Path(a.json).write_text(json.dumps(dump))
        print(f"per-phrase edit distances -> {a.json}")
    return 0


# ------------------------------------------------------------------------------ CLI: proxy
def cmd_proxy(a) -> int:
    """Does the whole procedure work where it CAN be checked? Hide a keyboard session's
    keystrokes, keep only the text of each typing burst, adapt, then score the hidden keys."""
    t = Target(a.session, a.taps_name, control=True)
    base = base_proba(t, Path(a.base_model) if a.base_model else None)
    segs = [s for s in t.segs if len(s[1])]
    lab = int((t.truth >= 0).sum())
    print(f"{a.session}: {len(t.taps)} taps ({lab} carry a hidden keystroke), {len(segs)} bursts, "
          f"{sum(len(x) for x, _ in segs)} known characters")
    print(f"base per-tap top1 = {tap_acc(base, t.truth)[0]:.3f}")
    res = {}
    for fi, te in enumerate(folds(len(segs), a.folds, a.seed)):
        tr = [segs[i] for i in range(len(segs)) if i not in set(te.tolist())]
        ev_idx = np.unique(np.concatenate([segs[i][1] for i in te]))
        for m in a.methods.split(","):
            p = method_proba(m, t, tr, base, a, verbose=(fi == 0 and m in ("em", "ctc")))
            acc, n = tap_acc(p, t.truth, ev_idx)
            c, e, r = (0.0, 0, 0) if a.no_cer else cer_on([segs[i] for i in te], p, a.beam)
            res.setdefault(m, []).append((acc, n, e, r))
        print(f"  fold{fi}: " + "  ".join(f"{m}={res[m][-1][0]:.3f}" for m in res), flush=True)
    print(f"\n{'method':<12}{'held-out tap top1':>20}{'n':>7}{'held-out CER':>14}")
    for m, v in res.items():
        n = sum(x[1] for x in v)
        acc = sum(x[0] * x[1] for x in v) / max(1, n)
        print(f"{m:<12}{acc:>20.3f}{n:>7}{sum(x[2] for x in v)/max(1,sum(x[3] for x in v)):>14.3f}")
    return 0


# ------------------------------------------------------------------------------ CLI: perm
def cmd_perm(a) -> int:
    """Label-free signal test: does each tap block align better to its OWN phrase than to the
    other phrases? Under a useless observation model the true text ranks at chance."""
    t = Target(a.session, a.taps_name, a.control)
    base = base_proba(t, Path(a.base_model) if a.base_model else None)
    segs = [s for s in t.segs if len(s[1])]
    obs = np.log(np.maximum(base, EPS))
    uni = np.log(np.full_like(base, 1.0 / NA))
    texts = [x for x, _ in segs]
    ranks, gains = [], []
    par = AlignParams.from_counts(sum(len(i) for _, i in segs),
                                  sum(sum(c in A_INDEX for c in x) for x in texts))
    for i, (text, idx) in enumerate(segs):
        sc = []
        for j, other in enumerate(texts):
            l, _, _ = forward_backward(obs[idx], other, par)
            sc.append(l / max(1, sum(c in A_INDEX for c in other)))
        sc = np.array(sc)
        ranks.append(float((sc > sc[i]).mean()))
        gains.append(sc[i] - np.mean([sc[j] for j in range(len(sc)) if j != i]))
        lu, _, _ = forward_backward(uni[idx], text, par)
        print(f"  phrase{i:>2} taps={len(idx):>3} chars={len(text):>3} "
              f"rank={ranks[-1]:.2f} own-vs-other={gains[-1]:+.3f} "
              f"own-vs-uniform={sc[i]-lu/max(1,sum(c in A_INDEX for c in text)):+.3f}", flush=True)
    r = np.array(ranks)
    print(f"\nmean rank of true text = {r.mean():.3f} (chance 0.500, perfect 0.000); "
          f"top-1 in {(r == 0).mean():.2f} of phrases over n={len(r)}")
    print(f"mean own-minus-other ll/char = {np.mean(gains):+.4f} "
          f"(sd {np.std(gains, ddof=1):.4f}, sem {np.std(gains, ddof=1)/np.sqrt(len(gains)):.4f})")
    return 0


# ------------------------------------------------------------------------------ CLI: probe
def cmd_probe(a) -> int:
    """Sanity floor: feed the lattice a synthetic observation model of known accuracy and
    see what alignment quality it can recover. Tells us whether 0.39 top-1 is even enough."""
    t = Target(a.session, a.taps_name, control=True)
    segs = [s for s in t.segs if len(s[1])]
    rng = np.random.RandomState(0)
    print(f"{'oracle acc':>11}{'q_acc':>9}{'ent':>8}{'conf>.5':>9}{'CER':>8}")
    for acc in [float(x) for x in a.accs.split(",")]:
        o = np.full((len(t.taps), NA), (1 - acc) / (NA - 1))
        for i in range(len(t.taps)):
            c = t.truth[i] if t.truth[i] >= 0 else rng.randint(NA)
            o[i, c] = acc
        o /= o.sum(1, keepdims=True)
        q, q_ins, ll, _ = align_all(np.log(o), segs, AlignParams.from_counts(
            sum(len(i) for _, i in segs), sum(sum(c in A_INDEX for c in t) for t, i in segs)))
        d = diagnostics(q, q_ins, segs, t.truth)
        c, _, _ = cer_on(segs, o, a.beam)
        print(f"{acc:>11.2f}{d['q_acc']:>9.3f}{d['ent']:>8.3f}{d['conf50']:>9.2f}{c:>8.3f}",
              flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.adapt", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("em", cmd_em), ("cv", cmd_cv), ("proxy", cmd_proxy),
                     ("probe", cmd_probe), ("perm", cmd_perm)):
        p = sub.add_parser(name)
        p.add_argument("--session", default=DESK)
        p.add_argument("--taps-name", default="taps.jsonl")
        p.add_argument("--base-model", default=None)
        p.add_argument("--iters", type=int, default=6)
        p.add_argument("--lam", type=float, default=0.5)
        p.add_argument("--l2", type=float, default=4.0)
        p.add_argument("--tau", type=float, default=0.6)
        p.add_argument("--beam", type=int, default=30)
        p.add_argument("--folds", type=int, default=5)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--reest", action="store_true", help="re-estimate p_ins from posteriors")
        p.add_argument("--control", action="store_true")
        p.add_argument("--modes", default="em,ctc")
        p.add_argument("--methods", default="uniform,base,em,ctc,selftrain,coral")
        p.add_argument("--accs", default="0.2,0.3,0.4,0.5,0.65,0.8,1.0")
        p.add_argument("--no-cer", action="store_true", help="proxy: tap accuracy only, skip decoding")
        p.add_argument("--json", default=None)
        p.set_defaults(func=fn)
    a = ap.parse_args(argv)
    t0 = time.time()
    r = a.func(a)
    print(f"\n[{time.time()-t0:.1f}s]", file=sys.stderr)
    return r


if __name__ == "__main__":
    sys.exit(main())
