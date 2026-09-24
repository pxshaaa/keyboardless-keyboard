"""Motor model + word-template scorer shared by P2 (keyboard) and P3 (desk).
- tap_g-style frame: one similarity per session from the median hand configuration over (label-free) tap frames,
  aligned by generalised Procrustes to the training sessions' canonical template (normalize.py recipe, re-implemented).
- per-key 2-D Gaussian (shrunk covariance; unseen keys placed by an affine map of the nominal QWERTZ layout) and
  P(finger | key) over 10 fingertips.
- word scorer: location channel = pair-HMM/DP over taps x letters (match = log N + w_f log P(f|k), tap insertion,
  letter deletion), shape channel = SHARK2 (resample to 16 points, translate+scale normalise, mean L2), unigram prior."""
from __future__ import annotations

import numpy as np

from phase0.analysis.finger_id import FINGERTIP_JOINTS, HAND_NAMES, KEY_LABEL

TEMPLATE_JOINTS = (4, 5, 8, 9, 12, 13, 16, 17, 20)
KW0 = 150.0
LET = "abcdefghijklmnopqrstuvwxyz"
LI = {c: i for i, c in enumerate(LET)}
# German QWERTZ nominal (key pitch units); macOS reports the produced character
_ROWS = ((0.0, 0.0, "qwertzuiop"), (1.0, 0.25, "asdfghjkl"), (2.0, 0.75, "yxcvbnm"))
NOMINAL = np.zeros((26, 2))
for _r, _o, _ks in _ROWS:
    for _j, _c in enumerate(_ks):
        NOMINAL[LI[_c]] = (_o + _j, _r)


def fid(hand: int, tip: int) -> int:
    return int(hand) * 5 + FINGERTIP_JOINTS.index(int(tip))


def gt_fid(key: str) -> tuple[int, int] | None:
    lab = KEY_LABEL.get(key)
    if lab is None or lab[0] == "Either":
        return None
    return HAND_NAMES.index(lab[0]), FINGERTIP_JOINTS[lab[1]]


# ---------------------------------------------------------------- similarity frame
def umeyama(src, dst):
    m = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    src, dst = src[m], dst[m]
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(a))
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / max((a ** 2).sum(1).mean(), 1e-9))
    return s, R, md - s * R @ ms


def apply_sim(xy, T):
    s, R, t = T
    return s * xy @ R.T + t


def template(P, rows):
    return np.nanmedian(P[rows][:, :, TEMPLATE_JOINTS, :2], axis=0)   # [2, 9, 2]


def canonical(templates, iters=10):
    ref = templates[-1].copy()
    for _ in range(iters):
        al = [apply_sim(T.reshape(-1, 2), umeyama(T.reshape(-1, 2), ref.reshape(-1, 2))).reshape(T.shape) for T in templates]
        ref = np.nanmean(al, 0)
        kw = np.nanmean([np.linalg.norm(ref[h, 1] - ref[h, 7]) for h in (0, 1)])
        c = np.nanmean(ref.reshape(-1, 2), 0)
        ref = (ref - c) * (KW0 / kw) + c
    return ref


def to_frame(T, ref):
    return umeyama(T.reshape(-1, 2), ref.reshape(-1, 2))


# ---------------------------------------------------------------- key model
class KeyModel:
    def __init__(self, X, keys, fings, tau=10.0, alpha_f=0.5, mean_prior=None, tau_mean=0.0):
        """X [N,2] canonical positions, keys [N] letter idx, fings [N] finger id 0..9.
        mean_prior: optional [26,2] means (MAP adaptation: posterior mean with tau_mean pseudo-counts)."""
        ok = np.isfinite(X).all(1)
        X, keys, fings = X[ok], keys[ok], fings[ok]
        self.mu = np.full((26, 2), np.nan)
        self.n = np.zeros(26)
        covs = []
        for k in range(26):
            m = keys == k
            self.n[k] = m.sum()
            if m.sum():
                self.mu[k] = X[m].mean(0)
            if m.sum() >= 3:
                covs.append((m.sum(), np.cov(X[m].T)))
        pool = sum(n * c for n, c in covs) / max(sum(n for n, _ in covs), 1) if covs else np.eye(2) * 50.0 ** 2
        # unseen/rare keys: affine map nominal -> canonical fitted on keys with >= 3 samples
        good = self.n >= 3
        if good.sum() >= 3:
            A = np.c_[NOMINAL[good], np.ones(good.sum())]
            Wt = np.sqrt(self.n[good])[:, None]
            coef, *_ = np.linalg.lstsq(A * Wt, self.mu[good] * Wt, rcond=None)
            nom = np.c_[NOMINAL, np.ones(26)] @ coef
        else:
            nom = np.nan_to_num(self.mu)
        w = np.clip(self.n / 3.0, 0, 1)[:, None]
        self.mu = np.where(np.isfinite(self.mu), w * np.nan_to_num(self.mu) + (1 - w) * nom, nom)
        if mean_prior is not None:
            nn = self.n[:, None]
            self.mu = (nn * np.nan_to_num(self.mu) + tau_mean * mean_prior) / np.maximum(nn + tau_mean, 1e-9)
            self.mu = np.where(nn + tau_mean > 0, self.mu, mean_prior)
        self.cov = np.zeros((26, 2, 2))
        for k in range(26):
            m = keys == k
            c = np.cov(X[m].T) if m.sum() >= 3 else pool
            nk = m.sum()
            self.cov[k] = (nk * c + tau * pool) / (nk + tau)
        self.icov = np.linalg.inv(self.cov)
        self.logdet = np.log(np.linalg.det(self.cov))
        F = np.full((26, 10), alpha_f)
        np.add.at(F, (keys, fings), 1.0)
        self.logpf = np.log(F / F.sum(1, keepdims=True))
        lo, hi = self.mu.min(0) - 60, self.mu.max(0) + 60
        self.log_bg = -np.log(np.prod(hi - lo))

    def tap_ll(self, X, fings=None, w_f=1.0):
        """[n,2] (+ finger ids) -> [n,26] log p(tap | key)."""
        d = X[:, None, :] - self.mu[None]
        q = np.einsum("nki,kij,nkj->nk", d, self.icov, d)
        ll = -0.5 * q - 0.5 * self.logdet[None] - np.log(2 * np.pi)
        if fings is not None and w_f:
            ll = ll + w_f * self.logpf[:, fings].T
        return np.where(np.isfinite(ll), ll, -50.0)


# ---------------------------------------------------------------- lexicon templates
class Lexicon:
    def __init__(self, logp: dict, maxlen=16):
        self.words = [w for w in logp if w.isascii() and w.isalpha() and 1 <= len(w) <= maxlen and all(c in LI for c in w)]
        self.prior = np.array([logp[w] for w in self.words])
        self.len = np.array([len(w) for w in self.words])
        self.by_len = {m: np.where(self.len == m)[0] for m in range(1, maxlen + 1)}
        self.codes = {m: np.array([[LI[c] for c in self.words[i]] for i in ix], int).reshape(len(ix), m)
                      for m, ix in self.by_len.items()}
        self.index = {w: i for i, w in enumerate(self.words)}


def resample(pts: np.ndarray, N=16):
    """[B, m, 2] polylines -> [B, N, 2] equidistant by arc length; translation+scale normalised (SHARK2 shape)."""
    B, m, _ = pts.shape
    if m == 1:
        out = np.repeat(pts, N, 1)
    else:
        seg = np.linalg.norm(np.diff(pts, axis=1), axis=2)
        cum = np.concatenate([np.zeros((B, 1)), np.cumsum(seg, 1)], 1)
        tot = np.maximum(cum[:, -1:], 1e-9)
        q = np.linspace(0, 1, N)[None] * tot
        idx = np.clip(np.array([np.searchsorted(cum[b], q[b], side="right") for b in range(B)]) - 1, 0, m - 2)
        c0 = np.take_along_axis(cum, idx, 1)
        sl = np.maximum(np.take_along_axis(seg, idx, 1), 1e-9)
        f = np.clip((q - c0) / sl, 0, 1)[..., None]
        p0 = np.take_along_axis(pts, idx[..., None].repeat(2, 2), 1)
        p1 = np.take_along_axis(pts, (idx + 1)[..., None].repeat(2, 2), 1)
        out = p0 + f * (p1 - p0)
    out = out - out.mean(1, keepdims=True)
    sc = np.sqrt((out ** 2).sum(2).mean(1))[:, None, None]
    return out / np.maximum(sc, 1e-6)


def resample_fast(pts: np.ndarray, N=16):
    """vectorised version for large batches of same length."""
    B, m, _ = pts.shape
    if m == 1:
        return np.zeros((B, N, 2))
    seg = np.linalg.norm(np.diff(pts, axis=1), axis=2)
    cum = np.concatenate([np.zeros((B, 1)), np.cumsum(seg, 1)], 1)
    tot = np.maximum(cum[:, -1:], 1e-9)
    q = np.linspace(0, 1, N)[None] * tot                                      # [B,N]
    idx = np.clip((cum[:, None, :] <= q[:, :, None]).sum(2) - 1, 0, m - 2)    # [B,N]
    c0 = np.take_along_axis(cum, idx, 1)
    sl = np.maximum(np.take_along_axis(seg, idx, 1), 1e-9)
    f = np.clip((q - c0) / sl, 0, 1)[..., None]
    p0 = np.take_along_axis(pts, idx[..., None].repeat(2, 2), 1)
    p1 = np.take_along_axis(pts, (idx + 1)[..., None].repeat(2, 2), 1)
    out = p0 + f * (p1 - p0)
    out = out - out.mean(1, keepdims=True)
    sc = np.sqrt((out ** 2).sum(2).mean(1))[:, None, None]
    return out / np.maximum(sc, 1e-6)


class Templates:
    """per length: resampled normalised key-centroid polylines of every lexicon word under a KeyModel."""

    def __init__(self, lex: Lexicon, km: KeyModel, N=16):
        self.shape = {m: resample_fast(km.mu[c], N) for m, c in lex.codes.items() if len(c)}


def score_words(ll: np.ndarray, X: np.ndarray, lex: Lexicon, tpl: Templates, lens, p_ins=0.05, p_del=0.05, log_bg=-12.0,
                sigma_shape=0.5, N=16):
    """ll [n,26] tap log-lik, X [n,2] tap positions -> dict m -> (loc [N_m], shape [N_m]) for candidate lengths."""
    n = len(ll)
    ins = np.log(p_ins) + log_bg + np.log(0.1)
    dl = np.log(p_del)
    obs_shape = resample_fast(X[None], N)[0] if (tpl is not None and X is not None and n >= 1) else None
    out = {}
    for m in lens:
        if m not in lex.codes or not len(lex.codes[m]):
            continue
        W = lex.codes[m]
        B = len(W)
        prev = np.array([dl * j for j in range(m + 1)], float)[None].repeat(B, 0)   # D[0, j]
        for i in range(1, n + 1):
            cur = np.empty_like(prev)
            cur[:, 0] = prev[:, 0] + ins
            lli = ll[i - 1][W]                                                      # [B, m]
            for j in range(1, m + 1):
                cur[:, j] = np.maximum(np.maximum(prev[:, j - 1] + lli[:, j - 1], prev[:, j] + ins), cur[:, j - 1] + dl)
            prev = cur
        loc = prev[:, m]
        if tpl is not None and n >= 2 and m >= 2:
            d = np.sqrt(((tpl.shape[m] - obs_shape[None]) ** 2).sum(2)).mean(1)
            shp = -0.5 * (d / sigma_shape) ** 2
        else:
            shp = np.zeros(B)
        out[m] = (loc, shp)
    return out


def rank_of(target: int, lex: Lexicon, scores: dict, w_loc=1.0, w_shape=0.0, w_prior=0.0):
    """-> 1-based rank of lexicon index `target` among all candidates in `scores`, and top-5 word list."""
    allix, alls = [], []
    for m, (loc, shp) in scores.items():
        ix = lex.by_len[m]
        allix.append(ix)
        alls.append(w_loc * loc + w_shape * shp + w_prior * lex.prior[ix])
    allix = np.concatenate(allix)
    alls = np.concatenate(alls)
    pos = np.where(allix == target)[0]
    if not len(pos):
        return None, []
    sc = alls[pos[0]]
    rank = int((alls > sc).sum()) + 1
    top = allix[np.argsort(-alls)[:5]]
    return rank, [lex.words[i] for i in top]
