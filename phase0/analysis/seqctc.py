"""Landmark streams -> typed text with CTC, no per-tap classification ("lip-reading for fingers").
Run: python -m phase0.analysis.seqctc {lopo | deskft | decode | curve}"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import torch
import torch.nn as nn
import torch.nn.functional as F

import os  # noqa: E402

torch.set_num_threads(int(os.environ.get("SEQCTC_THREADS", 4)))  # no lightgbm here: its libomp gave random segfaults

SESS = Path("data/sessions")
CACHE = Path(".cache/seqctc")
RES = Path("results/seqctc")
KBD = ("20260910-015217-kbd", "20260910-021315-kbd", "20260910-131629-kbd", "20260911-164237-kbd")
LOSO = ("20260910-021315-kbd", "20260910-131629-kbd", "20260911-164237-kbd")
HELD = "20260910-015948-kbd"
DESK = "20260910-202149-desk"

SYMS = "_abcdefghijklmnopqrstuvwxyz #"  # blank, letters, space, any other non-modifier key
BLANK, SPACE_S, OTHER = 0, 27, 28
NS = len(SYMS)
MODS = {"shift", "ctrl", "alt", "cmd"}
PALM = (0, 5, 9, 13, 17)
FPS = 60.0


# --------------------------------------------------------------------------- streams
@dataclass
class Stream:
    name: str
    t: np.ndarray   # frame times, s
    A: np.ndarray   # [T,2,21,3] anchor-relative, hand-scale units; slot 0 = smaller image y (left)
    M: np.ndarray   # [T,2] hand present
    kt: np.ndarray  # key-down times
    ks: np.ndarray  # key-down symbols (SYMS index)

    def save(self, f: Path):
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez(f, t=self.t, A=self.A, M=self.M, kt=self.kt, ks=self.ks)

    @classmethod
    def load(cls, name: str, f: Path) -> "Stream":
        z = np.load(f)
        return cls(name, z["t"], z["A"], z["M"], z["kt"], z["ks"])


def key_sym(k: str) -> int | None:
    if k in MODS:
        return None  # chorded and held: not a distinct stroke
    if len(k) == 1 and "a" <= k <= "z":
        return SYMS.index(k)
    return SPACE_S if k == "space" else OTHER


def normalise(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raw [T,2,21,3] (x forward, y rightward, z toward camera) -> slotted, anchored, scaled."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pres = ~np.isnan(P[..., :2]).any(axis=(2, 3))
        size = np.linalg.norm(P[:, :, 5, :2] - P[:, :, 17, :2], axis=-1)  # knuckle width: palm length foreshortens top-down
        scale = float(np.nanmedian(size[pres]))
        pc = np.nanmean(P[:, :, PALM, :2], axis=2)
        wy = P[:, :, 0, 1]
        both = pres.all(1)
        two = both & (np.linalg.norm(pc[:, 0] - pc[:, 1], axis=-1) >= 0.6 * scale)  # else a duplicate
        mid = float(np.nanmedian(wy[two].mean(1))) if two.any() else float(np.nanmedian(wy[pres]))
        out = np.full_like(P, np.nan)
        swap = two & (wy[:, 0] > wy[:, 1])
        keep = two & ~swap
        out[keep] = P[keep]
        out[swap] = P[swap][:, ::-1]
        idx = np.where(pres.any(1) & ~two)[0]
        src = np.where(pres[idx, 0], 0, 1)
        hs = P[idx, src]
        out[idx, (hs[:, 0, 1] > mid).astype(int)] = hs
        M = ~np.isnan(out[:, :, 0, 0])
        A = np.zeros_like(out)
        for k in range(2):
            m = M[:, k]
            if not m.any():
                continue
            anchor = np.median(out[m, k][:, PALM, :2].mean(1), 0)
            A[m, k, :, :2] = (out[m, k, :, :2] - anchor) / scale
            A[m, k, :, 2] = np.nan_to_num(out[m, k, :, 2]) / scale
    return A.astype(np.float32), M


LANDMARKS = os.environ.get("CVT_LANDMARKS", "landmarks.parquet")   # e.g. landmarks_hires.parquet -> streams_hires/
STREAMS = "streams" if LANDMARKS == "landmarks.parquet" else "streams_" + Path(LANDMARKS).stem.removeprefix("landmarks_")


def load_session(sid: str) -> Stream:
    f = CACHE / STREAMS / f"{sid}.npz"
    if f.exists():
        return Stream.load(sid, f)
    s = SESS / sid
    fr = [json.loads(line) for line in open(s / "frames.jsonl")]
    fi = np.array([r["i"] for r in fr])
    ft = np.array([r["t"] for r in fr])
    tb = pq.read_table(s / LANDMARKS, columns=["i", "hand", "joint", "x", "y", "z"])
    i = tb.column("i").to_numpy()
    row = np.clip(np.searchsorted(fi, i), 0, len(fi) - 1)
    ok = fi[row] == i
    h = tb.column("hand").to_numpy().astype(int)
    j = tb.column("joint").to_numpy().astype(int)
    P = np.full((len(fi), 2, 21, 3), np.nan, np.float32)
    for c, ch in enumerate("xyz"):
        P[row[ok], h[ok], j[ok], c] = tb.column(ch).to_numpy()[ok]
    A, M = normalise(P)
    kd = []
    if (s / "keys.jsonl").exists():
        for line in open(s / "keys.jsonl"):
            r = json.loads(line)
            if r.get("event") == "down" and (k := key_sym(r["key"])) is not None:
                kd.append((r["t"], k))
    kt = np.array([t for t, _ in kd], np.float64)
    ks = np.array([k for _, k in kd], np.int64)
    st = Stream(sid, ft, A, M, kt, ks)
    st.save(f)
    return st


def sym_text(s) -> str:
    return " ".join("".join(SYMS[k] for k in s if k != OTHER).split())


def eval_windows(st: Stream, gap=1.0, max_len=8.0, pad=0.35, min_syms=2):
    """Typing bursts split at pauses; long bursts split at their widest gap -> (t0, t1, symbols)."""
    kt, ks, out = st.kt, st.ks, []
    if len(kt) == 0:
        return out

    def rec(a, b, lo, hi):
        t0, t1 = max(kt[a] - pad, lo), min(kt[b - 1] + pad, hi)
        if t1 - t0 <= max_len or b - a < 2:
            out.append((t0, t1, a, b))
            return
        m = a + int(np.argmax(np.diff(kt[a:b]))) + 1
        cut = (kt[m - 1] + kt[m]) / 2
        rec(a, m, lo, cut)
        rec(m, b, cut, hi)

    starts = [0] + list(np.where(np.diff(kt) > gap)[0] + 1)
    for a, b in zip(starts, starts[1:] + [len(kt)]):
        rec(a, b, -np.inf, np.inf)
    return [(t0, t1, ks[a:b]) for t0, t1, a, b in out if len(sym_text(ks[a:b]).replace(" ", "")) >= min_syms]


def desk_windows(st: Stream):
    rows = [json.loads(line) for line in open(SESS / st.name / "phrases.jsonl")]
    shown = {r["idx"]: r for r in rows if r["event"] == "shown"}
    done = {r["idx"]: r for r in rows if r["event"] == "done"}
    return [(shown[i]["t"], done[i]["t"], shown[i]["phrase"]) for i in sorted(shown) if i in done]


class Cropper:
    """Random training crops whose edges fall in inter-key gaps, so labels are rarely cut mid-stroke."""

    def __init__(self, st: Stream, dmin=1.5, dmax=8.0, min_gap=0.18, t_range=None):
        self.st, self.dmin, self.dmax = st, dmin, dmax
        lo, hi = t_range if t_range else (st.t[0], st.t[-1])
        kt = st.kt[(st.kt > lo) & (st.kt < hi)]
        g = np.diff(kt)
        cuts = list(((kt[:-1] + kt[1:]) / 2)[g >= min_gap])
        if len(kt):
            cuts += [kt[0] - 0.3, kt[-1] + 0.3]
        for a, b in zip(kt[:-1][g > 5], kt[1:][g > 5]):
            cuts += list(np.arange(a + 0.6, b - 0.6, 4.0))
        self.cuts = np.unique(np.clip(cuts, max(lo, st.t[0]), min(hi, st.t[-1])))
        self.minutes = (min(hi, st.t[-1]) - max(lo, st.t[0])) / 60

    def sample(self, rng: np.random.Generator):
        c = self.cuts
        for _ in range(50):
            i = rng.integers(len(c))
            lo, hi = np.searchsorted(c, c[i] + self.dmin), np.searchsorted(c, c[i] + self.dmax)
            if hi > lo:
                j = rng.integers(lo, hi)
                t0, t1 = c[i], c[j]
                m = (self.st.kt > t0) & (self.st.kt < t1)
                return t0, t1, self.st.ks[m]
        t0 = c[rng.integers(len(c))]
        m = (self.st.kt > t0) & (self.st.kt < t0 + self.dmax)
        return t0, t0 + self.dmax, self.st.ks[m]


def crop(st: Stream, t0: float, t1: float, speed: float = 1.0):
    a, b = np.searchsorted(st.t, t0), max(np.searchsorted(st.t, t1), np.searchsorted(st.t, t0) + 4)
    A, M = st.A[a:b], st.M[a:b]
    if speed != 1.0 and len(A) > 4:
        n = max(8, int(round(len(A) / speed)))
        x = np.linspace(0, len(A) - 1, n)
        lo = np.floor(x).astype(int)
        hi = np.minimum(lo + 1, len(A) - 1)
        w = (x - lo)[:, None, None, None].astype(np.float32)
        A, M = A[lo] * (1 - w) + A[hi] * w, M[lo] & M[hi]
    return A, M


# --------------------------------------------------------------------------- features
@dataclass(frozen=True)
class Aug:
    rot: float = 10.0
    scale: float = 0.10
    trans: float = 0.15
    jitter: float = 0.01
    jdrop: float = 0.05
    hdrop: float = 0.05
    tmask: float = 0.08
    speed: float = 0.2


NOAUG = Aug(0, 0, 0, 0, 0, 0, 0, 0)


def augment(A: torch.Tensor, M: torch.Tensor, p: Aug):
    B, dev = A.shape[0], A.device
    th = (torch.rand(B, device=dev) * 2 - 1) * math.radians(p.rot)
    R = torch.stack([torch.stack([th.cos(), -th.sin()], -1), torch.stack([th.sin(), th.cos()], -1)], -2)
    sc = 1 + (torch.rand(B, 1, 1, 1, 1, device=dev) * 2 - 1) * p.scale
    tr = torch.randn(B, 1, 2, 1, 2, device=dev) * p.trans
    xy = torch.einsum("bij,btkqj->btkqi", R, A[..., :2]) * sc + tr
    xy = xy + torch.randn_like(xy) * p.jitter
    A = torch.cat([xy, A[..., 2:] * sc], -1)
    if p.jdrop > 0:
        A = A * (torch.rand(B, 1, 2, 21, 1, device=dev) > p.jdrop).float()
    if p.hdrop > 0:
        T = A.shape[1]
        drop = torch.zeros(B, T, 2, dtype=torch.bool, device=dev)
        for b in torch.nonzero(torch.rand(B) < p.hdrop).flatten().tolist():
            s = int(torch.randint(0, max(1, T - 10), ()))
            drop[b, s:s + int(torch.randint(5, max(6, T // 4), ())), int(torch.randint(0, 2, ()))] = True
        M = M & ~drop
    return A, M


def featurize(A: torch.Tensor, M: torch.Tensor, zfeat: bool = True) -> torch.Tensor:
    """A[B,T,2,21,3], M[B,T,2] -> [B,T,506]: palm-local pose, anchored pose, both velocities, masks."""
    if not zfeat:
        A = torch.cat([A[..., :2], torch.zeros_like(A[..., 2:])], -1)
    xy, z = A[..., :2], A[..., 2:]
    palm = xy[..., PALM, :].mean(-2, keepdim=True)
    size = (xy[..., 5, :] - xy[..., 17, :]).norm(dim=-1).clamp_min(0.3)[..., None, None]
    m = M[..., None, None].float()
    loc = torch.cat([(xy - palm) / size, z / size], -1) * m
    ab = A * m

    def vel(x):
        v = torch.zeros_like(x)
        v[:, 1:] = (x[:, 1:] - x[:, :-1]) * m[:, 1:] * m[:, :-1] * 10.0
        return v

    f = torch.cat([loc, ab, vel(loc), vel(ab)], -1)
    return torch.cat([f.flatten(2), M.float()], -1)


def time_mask(f: torch.Tensor, frac: float, lens: list[int]) -> torch.Tensor:
    if frac <= 0:
        return f
    f = f.clone()
    for b, n in enumerate(lens):
        for _ in range(2):
            w = int(np.random.randint(0, max(1, int(n * frac))))
            s = int(np.random.randint(0, max(1, n - w)))
            f[b, s:s + w] = 0
    return f


# ------------------------------------------------------------------------------ model
class ConvBlock(nn.Module):
    def __init__(self, d, k=5, drop=0.1, stride=1):
        super().__init__()
        self.dw = nn.Conv1d(d, d, k, stride=stride, padding=k // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(drop)
        self.stride = stride

    def forward(self, x):
        y = self.drop(self.pw(F.gelu(self.bn(self.dw(x)))))
        if self.stride == 1:
            return x + y
        return F.avg_pool1d(x, self.stride, self.stride, ceil_mode=True)[..., :y.shape[-1]] + y


class SeqCTC(nn.Module):
    """Depthwise-separable TCN (stride 2 -> 30 Hz) + BiGRU or transformer -> CTC over SYMS."""

    def __init__(self, din=506, d=192, h=160, nsym=NS, drop=0.2, arch="tfm", layers=3):
        super().__init__()
        self.arch, self.zfeat = arch, False
        self.kw = dict(din=din, d=d, h=h, nsym=nsym, drop=drop, arch=arch, layers=layers)
        self.norm = nn.BatchNorm1d(din)
        self.inp = nn.Conv1d(din, d, 1)
        self.enc = nn.Sequential(ConvBlock(d, 5, drop), ConvBlock(d, 5, drop), ConvBlock(d, 5, drop, 2),
                                 ConvBlock(d, 5, drop), ConvBlock(d, 5, drop))
        self.drop = nn.Dropout(drop)
        if arch == "gru":
            self.gru = nn.GRU(d, h, 2, batch_first=True, bidirectional=True, dropout=drop)
            self.head = nn.Linear(2 * h, nsym)
        else:
            # built one by one: nn.TransformerEncoder deep-copies its layer, which segfaults after MPS use
            self.tfm = nn.ModuleList([nn.TransformerEncoderLayer(d, 4, 2 * d, drop, batch_first=True, norm_first=True)
                                      for _ in range(layers)])
            self.fnorm = nn.LayerNorm(d)
            self.head = nn.Linear(d, nsym)

    @staticmethod
    def out_len(n: int) -> int:
        return (n + 1) // 2

    def forward(self, f: torch.Tensor, lens: list[int]) -> torch.Tensor:
        x = self.enc(F.gelu(self.inp(self.norm(f.transpose(1, 2))))).transpose(1, 2)
        ol = [self.out_len(n) for n in lens]
        if self.arch != "gru":
            pad = (torch.arange(x.shape[1])[None, :] >= torch.tensor(ol)[:, None]).to(x.device)
            x = self.drop(x)
            for layer in self.tfm:
                x = layer(x, src_key_padding_mask=pad)
            return F.log_softmax(self.head(self.fnorm(x)), -1)
        pk = nn.utils.rnn.pack_padded_sequence(self.drop(x), torch.tensor(ol), batch_first=True,
                                               enforce_sorted=False)
        y, _ = self.gru(pk)
        y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True, total_length=x.shape[1])
        return F.log_softmax(self.head(y), -1)


# --------------------------------------------------------------------------- training
@dataclass(frozen=True)
class Cfg:
    steps: int = 2500
    bs: int = 24
    lr: float = 2e-3
    wd: float = 1e-2
    d: int = 192
    h: int = 160
    drop: float = 0.2
    arch: str = "tfm"
    layers: int = 3
    zfeat: bool = False
    aug: Aug = Aug()
    seed: int = 0


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def pad_batch(items, dev):
    lens = [len(a) for a, _ in items]
    T = max(lens)
    A = np.zeros((len(items), T, 2, 21, 3), np.float32)
    M = np.zeros((len(items), T, 2), bool)
    for b, (a, m) in enumerate(items):
        A[b, :len(a)], M[b, :len(m)] = a, m
    return torch.from_numpy(A).to(dev), torch.from_numpy(M).to(dev), lens


def build(cfg: Cfg) -> SeqCTC:
    m = SeqCTC(d=cfg.d, h=cfg.h, drop=cfg.drop, arch=cfg.arch, layers=cfg.layers)
    m.zfeat = cfg.zfeat
    return m


def train(model: SeqCTC, sources, cfg: Cfg, dev=None, log: str = "", eval_fn=None, ckpt: Path | None = None) -> SeqCTC:
    """sources: list of (sampler, weight); a sampler yields (Stream, t0, t1, symbols) via .draw(rng)."""
    dev = dev or device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=cfg.steps, pct_start=0.1)
    w = np.array([s[1] for s in sources], float)
    w /= w.sum()
    t_start, run, first = time.time(), [], 0
    if ckpt is not None and ckpt.exists():
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"])
        model.to(dev)
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        rng.bit_generator.state = c["rng"]
        torch.set_rng_state(c["torch_rng"])
        first = c["step"]
        print(f"{log} resumed at step {first}", flush=True)
    for step in range(first, cfg.steps):
        items, tg, tl = [], [], []
        for k in rng.choice(len(sources), cfg.bs, p=w):
            st, t0, t1, sy = sources[k][0].draw(rng)
            sp = float(np.exp(rng.uniform(-1, 1) * np.log1p(cfg.aug.speed))) if cfg.aug.speed else 1.0
            items.append(crop(st, t0, t1, sp))
            tg.append(torch.as_tensor(sy, dtype=torch.long))
            tl.append(len(sy))
        A, M, lens = pad_batch(items, torch.device("cpu"))
        A, M = augment(A, M, cfg.aug)
        f = time_mask(featurize(A, M, model.zfeat), cfg.aug.tmask, lens).to(dev)
        lp = model(f, lens)
        ol = torch.tensor([model.out_len(n) for n in lens])
        tgt = torch.cat(tg) if sum(tl) else torch.zeros(0, dtype=torch.long)
        loss = F.ctc_loss(lp.transpose(0, 1).float().cpu(), tgt, ol, torch.tensor(tl), blank=BLANK,
                          reduction="mean", zero_infinity=True)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run.append(float(loss.detach()))
        if ckpt is not None and (step + 1) % 250 == 0 and step + 1 < cfg.steps:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "step": step + 1},
                       ckpt.with_suffix(".tmp"))
            os.replace(ckpt.with_suffix(".tmp"), ckpt)
        if log and (step + 1) % 100 == 0:
            msg = f"{log} step {step + 1}/{cfg.steps} loss {np.mean(run[-100:]):.3f} {time.time() - t_start:.0f}s"
            if eval_fn is not None and (step + 1) % 500 == 0:
                msg += " " + eval_fn(model)
                model.train()
            print(msg, flush=True)
    model.eval()
    if ckpt is not None and ckpt.exists():
        ckpt.unlink()
    return model


class KbdSource:
    def __init__(self, st: Stream, t_range=None, dmin=1.5, dmax=8.0):
        self.st, self.c = st, Cropper(st, dmin, dmax, t_range=t_range)

    def draw(self, rng):
        return (self.st, *self.c.sample(rng))


class PhraseSource:
    """Whole labelled windows (desk phrases): random +-0.3 s edge jitter only."""

    def __init__(self, st: Stream, wins):
        self.st, self.wins = st, [(t0, t1, text_syms(txt)) for t0, t1, txt in wins]

    def draw(self, rng):
        t0, t1, sy = self.wins[rng.integers(len(self.wins))]
        return self.st, t0 + rng.uniform(-0.3, 0.3), t1 + rng.uniform(-0.3, 0.3), sy


def text_syms(txt: str) -> np.ndarray:
    return np.array([SYMS.index(c) for c in txt if c in SYMS[1:28]], np.int64)


@torch.no_grad()
def infer(model: SeqCTC, st: Stream, wins, dev=None, bs=16) -> list[np.ndarray]:
    dev = dev or device()
    model.to(dev).eval()
    out = []
    for i in range(0, len(wins), bs):
        items = [crop(st, w[0], w[1]) for w in wins[i:i + bs]]
        A, M, lens = pad_batch(items, torch.device("cpu"))
        lp = model(featurize(A, M, model.zfeat).to(dev), lens).float().cpu().numpy()
        out += [lp[b, :model.out_len(n)] for b, n in enumerate(lens)]
    return out


@torch.no_grad()
def adabn(model: SeqCTC, st: Stream, wins, dev=None) -> SeqCTC:
    """Copy of model whose BatchNorm statistics are re-estimated on unlabelled target windows."""
    dev = dev or device()
    m = SeqCTC(**model.kw)  # copy.deepcopy of the module segfaults here (MPS and CPU alike)
    m.zfeat = model.zfeat
    m.load_state_dict({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    m = m.to(dev).eval()
    bns = [x for x in m.modules() if isinstance(x, nn.BatchNorm1d)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
        bn.train()
    for i in range(0, len(wins), 8):
        A, M, lens = pad_batch([crop(st, w[0], w[1]) for w in wins[i:i + 8]], torch.device("cpu"))
        m(featurize(A, M, m.zfeat).to(dev), lens)
    return m.eval()


# --------------------------------------------------------------------------- decoding
def greedy(lp: np.ndarray) -> str:
    a = lp.argmax(-1)
    keep = a[np.r_[True, a[1:] != a[:-1]]]
    return sym_text([k for k in keep if k != BLANK])


class LMScorer:
    def __init__(self, path="models/charlm.npz"):
        from phase0.analysis.decode import CharLM
        self.lm = CharLM.load(Path(path))

    def __call__(self, prefix: str) -> np.ndarray:
        return self.lm.logprobs((" " + prefix)[-5:])  # indices follow decode.ALPHABET = SYMS[1:28]


def _lae(a, b):
    return np.logaddexp(a, b)


def beam_lm(lp: np.ndarray, lm: LMScorer | None, alpha=0.5, beta=1.0, beam=16, prune=-7.0) -> str:
    """CTC prefix beam search with a char n-gram (shallow fusion); 'other' key mass joins blank."""
    P = lp.astype(np.float64)
    blank = np.logaddexp(P[:, BLANK], P[:, OTHER])
    beams = {"": (0.0, -np.inf)}
    ninf = -np.inf
    for t in range(len(P)):
        row = P[t]
        cand = [c for c in range(1, 28) if row[c] > prune]
        nb: dict[str, list[float]] = defaultdict(lambda: [ninf, ninf])
        for pre, (pb, pnb) in beams.items():
            ptot = _lae(pb, pnb)
            e = nb[pre]
            e[0] = _lae(e[0], ptot + blank[t])
            last = pre[-1] if pre else ""
            lmrow = None
            for c in cand:
                ch = SYMS[c]
                if ch == last:
                    e[1] = _lae(e[1], pnb + row[c])
                    base = pb
                else:
                    base = ptot
                if ch == " " and (not pre or last == " "):
                    continue
                if lm is not None and lmrow is None:
                    lmrow = lm(pre)
                bonus = (alpha * lmrow[c - 1] + beta) if lm is not None else beta
                e2 = nb[pre + ch]
                e2[1] = _lae(e2[1], base + row[c] + bonus)
        beams = dict(sorted(nb.items(), key=lambda kv: -_lae(*kv[1]))[:beam])
    best = max(beams.items(), key=lambda kv: _lae(*kv[1]))[0]
    return " ".join(best.split())


# ------------------------------------------------------------------------------ metrics
def rows_cer(rows) -> float:
    from phase0.analysis.decode import edit_distance
    return sum(edit_distance(r, h) for r, h in rows) / max(1, sum(len(r) for r, _ in rows))


def rows_wer(rows) -> float:
    from phase0.analysis.decode import edit_distance
    return sum(edit_distance(r.split(), h.split()) for r, h in rows) / max(1, sum(len(r.split()) for r, _ in rows))


def boot_ratio(ed: np.ndarray, n: np.ndarray, B: int = 5000, seed: int = 0):
    idx = np.random.default_rng(seed).integers(0, len(ed), (B, len(ed)))
    b = ed[idx].sum(1) / n[idx].sum(1)
    return float(ed.sum() / n.sum()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_delta(ed_a: np.ndarray, ed_b: np.ndarray, n: np.ndarray, B: int = 5000, seed: int = 0):
    """CI on rate(b) - rate(a), same items resampled on both sides."""
    idx = np.random.default_rng(seed).integers(0, len(n), (B, len(n)))
    d = (ed_b[idx].sum(1) - ed_a[idx].sum(1)) / n[idx].sum(1)
    return float((ed_b.sum() - ed_a.sum()) / n.sum()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ------------------------------------------------------------------------ experiments
def run_dir(tag: str, seed: int) -> Path:
    return CACHE / "runs" / tag / f"seed{seed}"


def save_lps(f: Path, lps, refs):
    f.parent.mkdir(parents=True, exist_ok=True)
    off = np.cumsum([0] + [len(x) for x in lps])
    np.savez_compressed(f, lp=np.concatenate(lps).astype(np.float16), off=off, refs=np.array(refs))


def load_lps(f: Path):
    z = np.load(f)
    lp, off = z["lp"].astype(np.float32), z["off"]
    return [lp[off[i]:off[i + 1]] for i in range(len(off) - 1)], [str(r) for r in z["refs"]]


def kbd_sources(sessions, frac: float = 1.0):
    out = []
    for s in sessions:
        st = load_session(s)
        rng_ = None if frac >= 1 else (st.t[0], st.t[0] + frac * (st.t[-1] - st.t[0]))
        src = KbdSource(st, rng_)
        out.append((src, src.c.minutes))
    return out


def eval_sets(fold: str) -> dict:
    """fold '<session stamp>' -> that LOSO session's windows; 'all' -> held-out kbd + desk phrases."""
    if fold != "all":
        st = load_session(next(s for s in LOSO if fold in s))
        w = eval_windows(st)
        return {fold: (st, w, [sym_text(x[2]) for x in w])}
    h, d = load_session(HELD), load_session(DESK)
    hw, dw = eval_windows(h), desk_windows(d)
    return {"held": (h, hw, [sym_text(x[2]) for x in hw]), "desk": (d, dw, [x[2] for x in dw])}


def greedy_cer(model, sets) -> str:
    out = []
    for k, (st, w, refs) in sets.items():
        hyp = [greedy(lp) for lp in infer(model, st, w)]
        out.append(f"{k} CER {rows_cer(list(zip(refs, hyp))):.3f}")
    return " ".join(out)


def load_init(m: SeqCTC, init: str | None, seed: int) -> SeqCTC:
    if init:
        f = CACHE / "pre" / f"{init}.pt"  # explicit checkpoint (e.g. hwt_s0) shared by all fine-tune seeds
        f = f if f.exists() else CACHE / "pre" / f"{init}_s{seed}.pt"
        m.load_state_dict(torch.load(f, map_location="cpu"))
    return m


def cmd_lopo(a) -> int:
    for seed in map(int, a.seeds.split(",")):
        cfg = Cfg(steps=a.steps, lr=a.lr, seed=seed)
        for fold in a.folds.split(","):
            d = run_dir(a.tag, seed)
            if (d / f"{fold}.done").exists():
                continue
            train_s = [s for s in KBD if fold not in s]
            sets = eval_sets(fold)
            m = load_init(build(cfg), a.init, seed)
            t0 = time.time()
            train(m, kbd_sources(train_s, a.frac), cfg, log=f"[{a.tag} s{seed} {fold}]",
                  eval_fn=(lambda mm: greedy_cer(mm, sets)) if a.verbose else None,
                  ckpt=run_dir(a.tag, seed) / f"{fold}.ckpt")
            d.mkdir(parents=True, exist_ok=True)
            for k, (st, w, refs) in sets.items():
                save_lps(d / f"{k}.npz", infer(m, st, w), refs)
                save_lps(d / f"{k}_adabn.npz", infer(adabn(m, st, w), st, w), refs)
            if fold == "all":
                torch.save(m.state_dict(), d / "all.pt")
            (d / f"{fold}.done").write_text(json.dumps({"cfg": asdict(cfg), "frac": a.frac, "init": a.init,
                                                        "secs": time.time() - t0}))
            print(f"[{a.tag} s{seed} {fold}] {greedy_cer(m, sets)} ({time.time() - t0:.0f}s)", flush=True)
    return 0


def cmd_deskft(a) -> int:
    desk = load_session(DESK)
    wins = desk_windows(desk)
    n = len(wins)
    folds = [[j] for j in range(n)] if a.kfold >= n else [list(range(n))[k::a.kfold] for k in range(a.kfold)]
    for seed in map(int, a.seeds.split(",")):
        out = run_dir(a.tag, seed) / f"deskft_{a.name}.npz"
        if out.exists():
            continue
        lps, t0 = [None] * n, time.time()
        for test in folds:
            cfg = Cfg(steps=a.steps, lr=a.lr, seed=seed * 1000 + test[0])
            m = build(cfg)
            if not a.scratch:
                m.load_state_dict(torch.load(run_dir(a.tag, seed) / "all.pt", map_location="cpu"))
            tr = [wins[j] for j in range(n) if j not in test]
            src = [(PhraseSource(desk, tr), a.desk_w)]
            if a.desk_w < 1:
                kb = kbd_sources(KBD)
                tot = sum(w for _, w in kb)
                src += [(s, w / tot * (1 - a.desk_w)) for s, w in kb]
            train(m, src, cfg)
            for j, lp in zip(test, infer(m, desk, [wins[j] for j in test])):
                lps[j] = lp
            print(f"[deskft {a.tag}/{a.name} s{seed}] fold {test} {time.time() - t0:.0f}s", flush=True)
        save_lps(out, lps, [w[2] for w in wins])
    return 0


def cmd_pretrain(a) -> int:
    for seed in map(int, a.seeds.split(",")):
        f = CACHE / "pre" / f"{a.src}_s{seed}.pt"
        if f.exists():
            continue
        cfg = Cfg(steps=a.steps, lr=a.lr, seed=seed)
        src = []
        if "hwt" in a.src:
            from phase0.analysis import howwetype as H
            src += [(KbdSource(st), 1.0) for st in H.streams()]
        if "fsb" in a.src:
            from phase0.analysis.fsboard import ClipSource
            src += [(ClipSource(), max(1.0, len(src)))]
        sets = {s[-10:]: v for s in LOSO for v in eval_sets(s[9:15]).values()}
        m = train(build(cfg), src, cfg, log=f"[pre {a.src} s{seed}]",
                  eval_fn=lambda mm: greedy_cer(mm, sets), ckpt=f.with_suffix(".ckpt"))
        f.parent.mkdir(parents=True, exist_ok=True)
        torch.save(m.state_dict(), f)
    return 0


def _beam_job(args):
    lp, alpha, beta, beam = args
    return beam_lm(lp, _LM[0], alpha, beta, beam)


_LM: list = []


def _pool_init():
    _LM.append(LMScorer())


def decode_many(pool, lps, alpha, beta, beam):
    jobs = [(lp, alpha, beta, beam) for lp in lps]
    if pool is None:
        if not _LM:
            _pool_init()
        return [_beam_job(j) for j in jobs]
    return pool.map(_beam_job, jobs, chunksize=4)


class _NoPool:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def cmd_decode(a) -> int:
    import multiprocessing as mp
    from phase0.analysis.decode import edit_distance

    base = json.loads(Path(".cache/hirecall/cv_all.json").read_text())["per"]
    pipe_ed = np.array([e for e, _ in base], float)
    grid = [(al, be) for al in (0.3, 0.6, 1.0) for be in (0.0, 1.0, 2.0, 3.0)]
    RES.mkdir(parents=True, exist_ok=True)
    report = {}
    ctx = _NoPool() if a.procs <= 1 else mp.get_context("spawn").Pool(a.procs, initializer=_pool_init)
    with ctx as pool:
        for tag in a.tags.split(","):
            seeds = sorted(int(p.name[4:]) for p in (CACHE / "runs" / tag).glob("seed*"))
            per = defaultdict(lambda: defaultdict(list))  # set -> decoder -> [per-seed edits]
            meta = {}
            for seed in seeds:
                d = run_dir(tag, seed)
                sets = {}
                kb = [load_lps(d / f"{s[9:15]}.npz") for s in LOSO if (d / f"{s[9:15]}.npz").exists()]
                if len(kb) == len(LOSO):
                    sets["kbd_loso"] = (sum([x[0] for x in kb], []), sum([x[1] for x in kb], []))
                for k in ["held", "desk", "held_adabn", "desk_adabn"] + [p.stem for p in sorted(d.glob("deskft_*.npz"))]:
                    if (d / f"{k}.npz").exists():
                        sets[k] = load_lps(d / f"{k}.npz")
                best = (a.alpha, a.beta)
                if a.alpha is None and "kbd_loso" in sets:
                    lps, refs = sets["kbd_loso"]
                    scores = {g: rows_cer(list(zip(refs, decode_many(pool, lps, *g, 8)))) for g in grid}
                    best = min(scores, key=scores.get)
                    meta[f"seed{seed}_grid"] = {f"{g[0]},{g[1]}": round(v, 4) for g, v in scores.items()}
                elif a.alpha is None:
                    best = (a.fallback_alpha, a.fallback_beta)  # no LOSO folds for this tag: pilot-fold setting
                meta[f"seed{seed}_lm"] = best
                for k, (lps, refs) in sets.items():
                    hyps = {"greedy": [greedy(lp) for lp in lps]}
                    if best[0] is not None:
                        hyps["lm"] = decode_many(pool, lps, best[0], best[1], a.beam)
                    for dec, hy in hyps.items():
                        per[k][dec].append((
                            [edit_distance(r, h) for r, h in zip(refs, hy)], [len(r) for r in refs],
                            [edit_distance(r.split(), h.split()) for r, h in zip(refs, hy)],
                            [len(r.split()) for r in refs], hy[:3], refs[:3]))
                print(f"[decode {tag}] seed {seed} lm={best}", flush=True)
            out = {"meta": meta}
            for k, decs in per.items():
                for dec, runs in decs.items():
                    ed = np.mean([r[0] for r in runs], 0)
                    n = np.array(runs[0][1], float)
                    wed = np.mean([r[2] for r in runs], 0)
                    wn = np.array(runs[0][3], float)
                    c, lo, hi = boot_ratio(ed, n)
                    row = {"cer": c, "lo": lo, "hi": hi, "wer": float(wed.sum() / wn.sum()),
                           "ed": ed.tolist(), "n": n.tolist(), "wed": wed.tolist(), "wn": wn.tolist(),
                           "seed_cers": [float(np.sum(r[0]) / n.sum()) for r in runs], "n_items": len(n),
                           "examples": list(zip(runs[0][5], runs[0][4]))}
                    if k.startswith("desk") and len(n) == len(pipe_ed):
                        row["delta_vs_pipeline"] = boot_delta(pipe_ed, ed, n)
                    out[f"{k}/{dec}"] = row
                    print(f"  {tag:<22} {k:<18} {dec:<6} CER {c:.3f} [{lo:.3f},{hi:.3f}] WER {row['wer']:.3f}"
                          + (f"  d_pipe {row['delta_vs_pipeline'][0]:+.3f} [{row['delta_vs_pipeline'][1]:+.3f},"
                             f"{row['delta_vs_pipeline'][2]:+.3f}]" if "delta_vs_pipeline" in row else ""),
                          flush=True)
            report[tag] = out
            (RES / f"decode_{tag}{a.suffix}.json").write_text(json.dumps(out, indent=1))
    return 0


def cmd_curve(a) -> int:
    """Desk/held CER vs training keystrokes; log-linear fit per condition, extrapolated to target CERs."""
    sess = [load_session(s) for s in KBD]
    rows, fits = [], {}
    for base in a.bases.split(","):
        for frac in (0.125, 0.25, 0.5, 1.0):
            f = RES / f"decode_{base if frac == 1.0 else f'{base}_f{frac}'}_fixed.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            keys = sum(int((st.kt < st.t[0] + frac * (st.t[-1] - st.t[0])).sum()) for st in sess)
            for k in ("desk/lm", "desk/greedy", "held/lm", "held/greedy"):
                if k in d:
                    rows.append({"base": base, "frac": frac, "keys": keys, "set": k, "cer": d[k]["cer"],
                                 "seed_cers": d[k]["seed_cers"]})
    for base in {r["base"] for r in rows}:
        for k in {r["set"] for r in rows}:
            pts = [(r["keys"], c) for r in rows if r["base"] == base and r["set"] == k for c in r["seed_cers"]]
            if len({x for x, _ in pts}) < 3:
                continue
            x = np.log2([p[0] for p in pts])
            y = np.array([p[1] for p in pts])
            slope, icpt = np.polyfit(x, y, 1)
            need = {str(t): (float(2 ** ((t - icpt) / slope)) if slope < 0 else None) for t in (0.448, 0.30, 0.20)}
            fits[f"{base}|{k}"] = {"cer_per_doubling": float(slope), "intercept": float(icpt), "keys_needed": need}
            print(f"{base:<10} {k:<12} slope {slope:+.3f} CER/doubling; keys for CER 0.448/0.30/0.20: "
                  + " / ".join("n/a" if v is None else f"{v:,.0f}" for v in need.values()))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "curve.json").write_text(json.dumps({"rows": rows, "fits": fits}, indent=1))
    for r in sorted(rows, key=lambda r: (r["base"], r["set"], r["frac"])):
        print(f"  {r['base']:<10} {r['set']:<12} frac {r['frac']:<5} keys {r['keys']:>5} CER {r['cer']:.3f}")
    return 0


def cmd_table(a) -> int:
    """Markdown table of every decode_<tag>.json: mean over seeds, 95% phrase/window bootstrap CI."""
    def cell(d, k):
        if k not in d:
            return "-"
        r = d[k]
        out = f"{r['cer']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"
        if "delta_vs_pipeline" in r:
            out += f" (d {r['delta_vs_pipeline'][0]:+.3f} [{r['delta_vs_pipeline'][1]:+.3f}, {r['delta_vs_pipeline'][2]:+.3f}])"
        return out + f" n={len(r['seed_cers'])}"
    cols = [("kbd LOSO", "kbd_loso"), ("held 015948", "held"), ("desk zero-shot", "desk"),
            ("desk zero-shot AdaBN", "desk_adabn"), ("desk fine-tuned 5-fold", "deskft_kf5"),
            ("desk-only 5-fold", "deskft_deskonly")]
    lines = ["| condition | decoder | " + " | ".join(c for c, _ in cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    main_files = [f for f in sorted(RES.glob("decode_*.json")) if not f.stem.endswith("_fixed") and "_f0." not in f.stem]
    for f in main_files:
        d = json.loads(f.read_text())
        for dec in ("greedy", "lm"):
            lines.append(f"| {f.stem[7:]} | {dec} | " + " | ".join(cell(d, f"{k}/{dec}") for _, k in cols) + " |")
    ref = json.loads((RES / f"decode_{a.ref}.json").read_text()) if (RES / f"decode_{a.ref}.json").exists() else None
    if ref is not None:
        lines += ["", f"Paired deltas vs `{a.ref}` (same items resampled; negative = better):", "",
                  "| condition | set/decoder | delta CER [95% CI] |", "|---|---|---|"]
        for f in main_files:
            if f.stem[7:] == a.ref:
                continue
            d = json.loads(f.read_text())
            for k, r in d.items():
                if k in ref and "ed" in r and "ed" in ref[k] and len(r["ed"]) == len(ref[k]["ed"]):
                    dd = boot_delta(np.array(ref[k]["ed"]), np.array(r["ed"]), np.array(r["n"]))
                    lines.append(f"| {f.stem[7:]} | {k} | {dd[0]:+.3f} [{dd[1]:+.3f}, {dd[2]:+.3f}] |")
    txt = "\n".join(lines)
    (RES / "table.md").write_text(txt + "\n")
    print(txt)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("table")
    s.add_argument("--ref", default="scratch")
    s.set_defaults(fn=cmd_table)
    s = sub.add_parser("curve")
    s.add_argument("--bases", default="scratch,hwt")
    s.set_defaults(fn=cmd_curve)
    s = sub.add_parser("lopo")
    s.add_argument("--tag", required=True)
    s.add_argument("--seeds", default="0,1,2")
    s.add_argument("--folds", default="021315,131629,164237,all")
    s.add_argument("--init", default=None)
    s.add_argument("--frac", type=float, default=1.0)
    s.add_argument("--steps", type=int, default=2500)
    s.add_argument("--lr", type=float, default=2e-3)
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(fn=cmd_lopo)
    s = sub.add_parser("deskft")
    s.add_argument("--tag", required=True)
    s.add_argument("--name", default="lopo")
    s.add_argument("--seeds", default="0,1,2")
    s.add_argument("--kfold", type=int, default=20)
    s.add_argument("--steps", type=int, default=400)
    s.add_argument("--lr", type=float, default=5e-4)
    s.add_argument("--desk-w", type=float, default=0.5)
    s.add_argument("--scratch", action="store_true")
    s.set_defaults(fn=cmd_deskft)
    s = sub.add_parser("pretrain")
    s.add_argument("--src", required=True)
    s.add_argument("--seeds", default="0,1,2")
    s.add_argument("--steps", type=int, default=6000)
    s.add_argument("--lr", type=float, default=2e-3)
    s.set_defaults(fn=cmd_pretrain)
    s = sub.add_parser("decode")
    s.add_argument("--tags", required=True)
    s.add_argument("--alpha", type=float, default=None)
    s.add_argument("--beta", type=float, default=1.0)
    s.add_argument("--beam", type=int, default=16)
    s.add_argument("--procs", type=int, default=6)
    s.add_argument("--suffix", default="")
    s.add_argument("--fallback-alpha", type=float, default=0.3)
    s.add_argument("--fallback-beta", type=float, default=1.0)
    s.set_defaults(fn=cmd_decode)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
