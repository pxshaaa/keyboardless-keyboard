"""Per-key accuracy, the last bottleneck: temporal context, capacity, ensembling, self-training.
Run: python -m phase0.analysis.keymax {keys | fuse | deskpix | table | cv}"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import lightgbm  # noqa: F401  must load before torch or the two libomp copies segfault

from phase0.analysis import masked as mk
from phase0.analysis import tap_pos as tp
from phase0.analysis.appearance import TIP_CROP, TIP_OFFSETS, _prep, _device, _torch
from phase0.analysis.decode import NA

KBD = list(mk.TRAIN_SESSIONS)
HELD = mk.HELDOUT_SESSION
DESK = mk.DESK_SESSION
CACHE = Path(".cache/keymax")
HR_CACHE = Path(".cache/hirecall")
SEEDS = (0, 1, 2, 3, 4)
EPS = 1e-12

# pose windows, in frames at 60 fps: the shipped +-8 (133 ms) against a motor-event window
OFF_BASE = tp.OFFSETS
OFF_MID = (-24, -16, -12, -8, -4, -2, 0, 2, 4, 8, 12, 16, 24)
OFF_WIDE = (-40, -28, -20, -14, -8, -4, -2, 0, 2, 4, 8, 14, 20, 28, 40)
OFF_W60 = (-60, -45, -32, -22, -14, -8, -4, -2, 0, 2, 4, 8, 14, 22, 32, 45, 60)

# pixel channel stacks over the tip bank's offsets (0,2,3,4,5,6 frames after the peak)
REPS = {
    "diff": (("abs", 4), ("diff", 4, 0)),
    "temporal": (("abs", 4), ("diff", 4, 0), ("diff", 6, 4), ("diff", 2, 0)),
    "temporal6": (("abs", 0), ("abs", 2), ("abs", 4), ("abs", 6), ("diff", 4, 0),
                  ("diff", 6, 2)),
}

VARIANTS = {
    # name: (kind, spec)
    "pose": ("lgbm", {"offsets": OFF_BASE}),
    "posemid": ("lgbm", {"offsets": OFF_MID}),
    "posewide": ("lgbm", {"offsets": OFF_WIDE}),
    "pose60": ("lgbm", {"offsets": OFF_W60}),
    "seq": ("seq", {"half": 24, "stride": 2}),
    "seq40": ("seq", {"half": 40, "stride": 3}),
    "pix": ("pix", {"rep": "diff", "mode": "none", "width": 24, "deep": False}),
    "pixT": ("pix", {"rep": "temporal", "mode": "none", "width": 24, "deep": False}),
    "pixW": ("pix", {"rep": "diff", "mode": "none", "width": 40, "deep": True}),
    "pixWT": ("pix", {"rep": "temporal", "mode": "none", "width": 40, "deep": True}),
    "pixWT6": ("pix", {"rep": "temporal6", "mode": "none", "width": 40, "deep": True}),
    "pixWTa": ("pix", {"rep": "temporal", "mode": "none", "width": 40, "deep": True, "tta": 4}),
    "silh": ("pix", {"rep": "diff", "mode": "silh", "width": 24, "deep": False}),
    "pixzL": ("pix", {"rep": "temporal", "mode": "none", "width": 40, "deep": True, "zero": 0}),
    "pixzR": ("pix", {"rep": "temporal", "mode": "none", "width": 40, "deep": True, "zero": 1}),
}


# ------------------------------------------------------------------ data
_DS: dict = {}


def ds(sid: str) -> dict:
    if sid not in _DS:
        _DS[sid] = mk.build_dataset(sid)
    return _DS[sid]


def train_of(held: str) -> list[str]:
    """Held-out session 015948 is never in any training set; LOSO folds drop one of the three."""
    return [s for s in KBD if s != held]


def chans(d: dict, spec, mode: str, rows=None, zero=None, dtype=np.float16) -> np.ndarray:
    """[N,10,C,32,32] tip patches; spec names absolute offsets and frame differences.
    float16 by default: the 6-channel stacks do not fit in RAM twice on a loaded machine."""
    jj = {o: j for j, o in enumerate(TIP_OFFSETS)}
    r = d["bank_rows"] if rows is None else rows
    cache: dict[int, np.ndarray] = {}

    def at(o):
        o = min(TIP_OFFSETS, key=lambda x: abs(x - o))
        if o not in cache:
            a = np.asarray(d["tips"][r, jj[o]], dtype=np.float32) / 255.0
            if mode != "none":
                m = np.asarray(d["masks"][r, jj[o]], dtype=np.float32)
                a = m if mode == "silh" else a * (m if mode == "hand" else 1.0 - m)
            cache[o] = a.reshape(len(a), 10, TIP_CROP, TIP_CROP)
        return cache[o]

    x = np.empty((len(r), 10, len(spec), TIP_CROP, TIP_CROP), dtype)
    for c, s in enumerate(spec):
        x[:, :, c] = at(s[1]) if s[0] == "abs" else at(s[1]) - at(s[2])
    cache.clear()
    if zero is not None:  # blank one hand's five pads: does the model use the other hand?
        x[:, zero * 5:zero * 5 + 5] = 0.0
    return x


# ------------------------------------------------------------------ pose sequence features
def seq_feats(sess, k: np.ndarray, half: int, stride: int) -> np.ndarray:
    """[N,T,F] both hands' 21 joints over a long window: absolute, and relative to the tap frame."""
    offs = np.arange(-half, half + 1, stride)
    idx = np.clip(k[:, None] + offs[None, :], 0, len(sess.P) - 1)
    xy = sess.P[idx][:, :, :, :, :2]                       # [N,T,2,21,2]
    fin = np.isfinite(xy[:, :, :, 0, 0]).astype(np.float32)
    xy = np.nan_to_num(xy, nan=0.0)
    base = xy[:, len(offs) // 2][:, None]
    norm = xy / np.array([tp.W, tp.H], np.float32)
    rel = (xy - base) / sess.span[None, None, :, None, None]
    n, t = xy.shape[:2]
    return np.nan_to_num(np.concatenate(
        [norm.reshape(n, t, -1), rel.reshape(n, t, -1), fin], axis=2).astype(np.float32))


def _gru(f_in: int, n_cls: int, hidden: int = 96):
    torch = _torch()
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(f_in, 128), nn.ReLU(True), nn.Dropout(0.2))
            self.rnn = nn.GRU(128, hidden, num_layers=2, batch_first=True, bidirectional=True,
                              dropout=0.2)
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(4 * hidden, n_cls))

        def forward(self, x):
            h, _ = self.rnn(self.inp(x))
            return self.head(torch.cat([h.mean(1), h.max(1).values], dim=1))

    return Net()


def fit_seq(Xtr, ytr, seed: int, epochs: int = 60, lr: float = 2e-3, bs: int = 64):
    torch = _torch()
    torch.manual_seed(seed)
    dev = _device()
    classes = np.unique(ytr)
    remap = {int(c): i for i, c in enumerate(classes)}
    mu, sd = Xtr.reshape(-1, Xtr.shape[2]).mean(0), Xtr.reshape(-1, Xtr.shape[2]).std(0) + 1e-6
    X = torch.from_numpy(((Xtr - mu) / sd).astype(np.float32))
    y = torch.from_numpy(np.array([remap[int(v)] for v in ytr], np.int64)).to(dev)
    net = _gru(X.shape[2], len(classes)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, len(y) // bs) * epochs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.25)
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    gen = torch.Generator().manual_seed(seed + 1)
    net.train()
    step = 0
    for _ in range(epochs):
        perm = torch.randperm(len(y), generator=gen)
        for i in range(0, len(y) - bs + 1, bs):
            b = perm[i:i + bs]
            xb = (X[b] + 0.02 * torch.randn(X[b].shape, generator=gen)).to(dev)
            loss = lossf(net(xb), y[b.to(dev)])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step < steps:
                sch.step()
    net.eval()
    return {"net": net, "classes": classes, "mu": mu, "sd": sd}


def seq_proba(m, X) -> np.ndarray:
    torch = _torch()
    dev = _device()
    Z = torch.from_numpy(((X - m["mu"]) / m["sd"]).astype(np.float32))
    ps = []
    with torch.no_grad():
        for i in range(0, len(Z), 256):
            ps.append(torch.softmax(m["net"](Z[i:i + 256].to(dev)), 1).cpu().numpy())
    out = np.full((len(X), NA), 1e-6)
    out[:, m["classes"]] = np.maximum(np.vstack(ps), 1e-6)
    return out / out.sum(1, keepdims=True)


# ------------------------------------------------------------------ pixel net
def tip_net(c_in: int, n_cls: int, width: int = 24, deep: bool = False):
    """appearance._tip_net widened, optionally with a fourth stage; joint head over all 10 pads."""
    torch = _torch()
    nn = torch.nn

    def blk(a, b, stride=2):
        return nn.Sequential(nn.Conv2d(a, b, 3, stride, 1), nn.BatchNorm2d(b), nn.ReLU(True))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            w = width
            layers = [blk(c_in, w), blk(w, 2 * w), blk(2 * w, 4 * w)]
            if deep:
                layers += [blk(4 * w, 4 * w, 1), blk(4 * w, 8 * w)]
            self.out_dim = (8 if deep else 4) * w
            self.trunk = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.pos = nn.Parameter(torch.zeros(10, 16))
            self.head = nn.Sequential(nn.Linear(10 * (self.out_dim + 16), 256), nn.ReLU(True),
                                      nn.Dropout(0.3), nn.Linear(256, n_cls))

        def forward(self, x):
            b = x.shape[0]
            e = self.trunk(x.reshape(b * 10, *x.shape[2:])).reshape(b, 10, -1)
            e = torch.cat([e, self.pos.expand(b, -1, -1)], dim=2)
            return self.head(e.reshape(b, -1))

    return Net()


def fit_pix(Xtr, ytr, seed=0, epochs=40, width=24, deep=False, lr=3e-3, bs=64):
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    from collections import Counter
    cnt = Counter(ytr.tolist())
    classes = np.array(sorted(c for c in cnt if cnt[c] >= mk.MIN_N))
    remap = {int(c): i for i, c in enumerate(classes)}
    m = np.array([v in remap for v in ytr])
    dev = _device()
    # batches move to MPS one at a time: a resident training tensor makes the unified-memory
    # allocator thrash once another job is on the machine
    X = torch.from_numpy(np.ascontiguousarray(Xtr[m]))
    y = torch.from_numpy(np.array([remap[int(v)] for v in ytr[m]], np.int64)).to(dev)
    net = tip_net(X.shape[2], len(classes), width, deep).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, len(y) // bs) * epochs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.25)
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    gen = torch.Generator().manual_seed(seed + 1)
    net.train()
    step = 0
    for _ in range(epochs):
        perm = torch.randperm(len(y), generator=gen)
        for i in range(0, len(y) - bs + 1, bs):
            b = perm[i:i + bs]
            xb = X[b].to(dev).float()
            loss = lossf(net(_prep(xb, gen, True)), y[b.to(dev)])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step < steps:
                sch.step()
    net.eval()
    if hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return {"net": net, "classes": classes, "params": sum(p.numel() for p in net.parameters())}


def pix_proba(m, X, tta: int = 0, seed: int = 0) -> np.ndarray:
    torch = _torch()
    dev = _device()
    gen = torch.Generator().manual_seed(seed + 7)
    reps = max(1, tta)
    acc = np.zeros((len(X), len(m["classes"])))
    with torch.no_grad():
        for r in range(reps):
            ps = []
            for i in range(0, len(X), 256):
                xb = torch.from_numpy(np.ascontiguousarray(X[i:i + 256])).to(dev).float()
                ps.append(torch.softmax(m["net"](_prep(xb, gen, tta > 0 and r > 0)), 1)
                          .cpu().numpy())
            acc += np.vstack(ps)
    p = acc / reps
    out = np.full((len(X), NA), 1e-6)
    out[:, m["classes"]] = np.maximum(p, 1e-6)
    return out / out.sum(1, keepdims=True)


# ------------------------------------------------------------------ expert cache
def exp_path(name: str, held: str, seed: int) -> Path:
    return CACHE / f"exp_{name}_{held}_{seed}.npy"


def expert(name: str, held: str, seed: int, force: bool = False) -> np.ndarray:
    p = exp_path(name, held, seed)
    if p.exists() and not force:
        return np.load(p)
    kind, spec = VARIANTS[name]
    tr = [ds(s) for s in train_of(held)]
    te = ds(held)
    t0 = time.time()
    if kind == "lgbm":
        m = tp.KeyClf.fit(tr, "abs", tp.JOINTS_FULL, seed, offsets=spec["offsets"])
        out = m.proba(te["sess"], te["k"])
        extra = ""
    elif kind == "seq":
        Xtr = np.vstack([seq_feats(d["sess"], d["k"], spec["half"], spec["stride"]) for d in tr])
        ytr = np.concatenate([d["y"] for d in tr])
        m = fit_seq(Xtr, ytr, seed)
        out = seq_proba(m, seq_feats(te["sess"], te["k"], spec["half"], spec["stride"]))
        extra = f" T={Xtr.shape[1]} F={Xtr.shape[2]}"
    else:
        rep, mode = REPS[spec["rep"]], spec["mode"]
        z = spec.get("zero")
        Xtr = np.vstack([chans(d, rep, mode, zero=z) for d in tr])
        ytr = np.concatenate([d["y"] for d in tr])
        m = fit_pix(Xtr, ytr, seed, spec.get("epochs", 40), spec["width"], spec["deep"])
        out = pix_proba(m, chans(te, rep, mode, zero=z), spec.get("tta", 0), seed)
        extra = f" params={m['params']:,}"
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(p, out)
    print(f"  {name:<9} held={held[9:15]} seed={seed} ({time.time()-t0:.0f}s){extra}", flush=True)
    return out


# ------------------------------------------------------------------ scoring
def topk(p: np.ndarray, y: np.ndarray, k: int) -> float:
    o = np.argsort(-p, 1)[:, :k]
    return float(np.mean([y[i] in o[i] for i in range(len(y))]))


def loso_held(probs: dict, seeds) -> dict:
    """probs[(held,seed)] -> [N,NA]; LOSO is the tap-weighted mean over the three kbd folds."""
    out = {"loso1": [], "loso5": [], "held1": [], "held5": []}
    for s in seeds:
        a1, a5, w = [], [], []
        for h in KBD:
            y = ds(h)["y"]
            a1.append(topk(probs[(h, s)], y, 1))
            a5.append(topk(probs[(h, s)], y, 5))
            w.append(len(y))
        w = np.array(w, float)
        out["loso1"].append(float(np.average(a1, weights=w)))
        out["loso5"].append(float(np.average(a5, weights=w)))
        y = ds(HELD)["y"]
        out["held1"].append(topk(probs[(HELD, s)], y, 1))
        out["held5"].append(topk(probs[(HELD, s)], y, 5))
    return out


def agg(v) -> str:
    v = np.array(v, float)
    return f"{v.mean():.3f}±{v.std():.3f}" if len(v) > 1 else f"{v.mean():.3f}"


def boot_acc(correct: np.ndarray, n=10000, seed=0) -> tuple[float, float]:
    idx = np.random.default_rng(seed).integers(0, len(correct), (n, len(correct)))
    v = correct[idx].mean(1)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def gmean(ps: list[np.ndarray], w: list[float]) -> np.ndarray:
    q = np.zeros_like(ps[0])
    for p, a in zip(ps, w):
        if a:
            q += a * np.log(np.maximum(p, EPS))
    q = np.exp(q - q.max(1, keepdims=True))
    return q / q.sum(1, keepdims=True)


def ens(name: str, held: str, seeds) -> np.ndarray:
    """Seed ensemble: mean probability, which is what combine.py's pixel expert does."""
    return np.mean([expert(name, held, s) for s in seeds], axis=0)


# ------------------------------------------------------------------ CLI: fit
def use_cpu(threads: int) -> None:
    """MPS segfaults inside .to(device) when the machine is out of memory; CPU is the fallback."""
    global _device
    torch = _torch()
    torch.set_num_threads(max(1, threads))
    _device = lambda: torch.device("cpu")  # noqa: E731


def cmd_fit(a) -> int:
    if a.cpu:
        use_cpu(a.threads)
    for name in a.variants.split(","):
        for held in (a.held.split(",") if a.held else [*KBD, HELD]):
            for s in range(a.seeds):
                expert(name, held, s, a.force)
    return 0


# ------------------------------------------------------------------ CLI: keys
def cmd_keys(a) -> int:
    if a.cpu:
        use_cpu(a.threads)
    seeds = tuple(range(a.seeds))
    names = a.variants.split(",")
    for s in [*KBD, HELD]:
        ds(s)
    rows = []
    for name in names:
        probs = {(h, s): expert(name, h, s, a.force) for h in [*KBD, HELD] for s in seeds}
        r = loso_held(probs, seeds)
        y = ds(HELD)["y"]
        cor = np.mean([(probs[(HELD, s)].argmax(1) == y).astype(float) for s in seeds], axis=0)
        lo, hi = boot_acc(cor)
        e = ens(name, HELD, seeds)
        rows.append({"name": name, **r, "held_ci": [lo, hi],
                     "held_ens1": topk(e, y, 1), "held_ens5": topk(e, y, 5),
                     "loso_ens1": float(np.average(
                         [topk(ens(name, h, seeds), ds(h)["y"], 1) for h in KBD],
                         weights=[len(ds(h)["y"]) for h in KBD])),
                     "loso_ens5": float(np.average(
                         [topk(ens(name, h, seeds), ds(h)["y"], 5) for h in KBD],
                         weights=[len(ds(h)["y"]) for h in KBD]))})
    print(f"\n{'variant':<10}{'LOSO top1':>16}{'LOSO top5':>16}{'held top1':>16}"
          f"{'held top5':>16}{'held ens1':>11}{'held 95% CI':>18}")
    for r in rows:
        ci = f"[{r['held_ci'][0]:.3f}, {r['held_ci'][1]:.3f}]"
        print(f"{r['name']:<10}{agg(r['loso1']):>16}{agg(r['loso5']):>16}{agg(r['held1']):>16}"
              f"{agg(r['held5']):>16}{r['held_ens1']:>11.3f}{ci:>18}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    return 0


# ------------------------------------------------------------------ CLI: fuse
def fuse_grid(names: list[str], step: float = 0.25):
    """Weights on a coarse simplex; log-linear pooling, as combine.py fuses pixels with pose."""
    n = len(names)
    ks = int(round(1 / step))
    out = []

    def rec(i, left, acc):
        if i == n - 1:
            out.append(tuple(acc + [left * step]))
            return
        for v in range(left + 1):
            rec(i + 1, left - v, acc + [v * step])

    rec(0, ks, [])
    return out


def cmd_fuse(a) -> int:
    seeds = tuple(range(a.seeds))
    names = a.variants.split(",")
    for s in [*KBD, HELD]:
        ds(s)
    P = {n: {h: ens(n, h, seeds) for h in [*KBD, HELD]} for n in names}
    best, bacc = None, -1.0
    for w in fuse_grid(names, a.step):
        if sum(w) <= 0:
            continue
        acc = float(np.average(
            [topk(gmean([P[n][h] for n in names], list(w)), ds(h)["y"], 1) for h in KBD],
            weights=[len(ds(h)["y"]) for h in KBD]))
        if acc > bacc:
            best, bacc = w, acc
    y = ds(HELD)["y"]
    ph = gmean([P[n][HELD] for n in names], list(best))
    lo, hi = boot_acc((ph.argmax(1) == y).astype(float))
    print(f"\nfusion over {names}  weights={best}")
    print(f"  LOSO  top1 {bacc:.3f}   top5 "
          f"{float(np.average([topk(gmean([P[n][h] for n in names], list(best)), ds(h)['y'], 5) for h in KBD], weights=[len(ds(h)['y']) for h in KBD])):.3f}")
    print(f"  held  top1 {topk(ph, y, 1):.3f} [{lo:.3f}, {hi:.3f}]   top5 {topk(ph, y, 5):.3f}"
          f"   n={len(y)}")
    if a.json:
        Path(a.json).write_text(json.dumps({"names": names, "weights": list(best),
                                            "loso1": bacc, "held1": topk(ph, y, 1),
                                            "held_ci": [lo, hi]}, indent=1))
    return 0


# ------------------------------------------------------------------ desk: pixels on the bank
def _hr():
    from phase0.analysis import hirecall as hr
    return hr


def bank() -> tuple[np.ndarray, dict]:
    """hirecall's frame-indexed unmasked tip bank; its mask bank is a stub, so mode='none' only."""
    fr = np.array(json.loads((HR_CACHE / f"{DESK}_frames.json").read_text())["frames"], int)
    tips = np.load(HR_CACHE / f"{DESK}_ftips.npy", mmap_mode="r")
    return fr, {"tips": tips, "masks": None, "bank_rows": np.arange(len(tips))}


def desk_pix(variant: str, seeds, force: bool = False) -> np.ndarray:
    out_p = CACHE / f"deskpix_{variant}_{len(seeds)}.npy"
    if out_p.exists() and not force:
        return np.load(out_p)
    _, spec = VARIANTS[variant]
    rep = REPS[spec["rep"]]
    tr = [ds(s) for s in KBD]
    Xtr = np.vstack([chans(d, rep, "none") for d in tr])
    ytr = np.concatenate([d["y"] for d in tr])
    _, b = bank()
    n = len(b["bank_rows"])
    ps = []
    for s in seeds:
        t0 = time.time()
        m = fit_pix(Xtr, ytr, s, spec.get("epochs", 40), spec["width"], spec["deep"])
        # 7k frames x 10 pads x 6 channels does not want to be resident all at once
        ps.append(np.vstack([pix_proba(m, chans(b, rep, "none", rows=np.arange(i, min(i + 1024, n))),
                                       spec.get("tta", 0), s) for i in range(0, n, 1024)]))
        print(f"  desk {variant} seed{s} ({time.time()-t0:.0f}s)", flush=True)
    p = np.mean(ps, axis=0)
    p /= p.sum(1, keepdims=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(out_p, p)
    return p


# ------------------------------------------------------------------ desk: pose with CORAL
_POSE: dict = {}


def desk_pose(variant: str, sess, k: np.ndarray, seeds) -> np.ndarray:
    """Keyboard-fitted key model, desk features standardised to the keyboard moments (CORAL)."""
    import phase0.analysis.pipeline as pl

    if variant == "pose":
        return pl.spatial_proba(sess, k, pl.kbd_fit(), pl.FULL)
    kind, spec = VARIANTS[variant]
    if variant not in _POSE:  # fit once per worker: every tap stream reuses these models
        tr = [tp.dataset(s) for s in KBD]
        if kind == "lgbm":
            off = spec["offsets"]
            S = np.vstack([tp.features(d["sess"], d["k"], "abs", tp.JOINTS_FULL, off) for d in tr])
            _POSE[variant] = ([tp.KeyClf.fit(tr, "abs", tp.JOINTS_FULL, s, offsets=off)
                               for s in seeds], S.mean(0), S.std(0) + 1e-6)
        else:
            S = np.vstack([seq_feats(d["sess"], d["k"], spec["half"], spec["stride"]) for d in tr])
            y = np.concatenate([d["y"] for d in tr])
            F = S.reshape(-1, S.shape[2])
            _POSE[variant] = ([fit_seq(S, y, s) for s in seeds], F.mean(0), F.std(0) + 1e-6)
    models, mu, sd = _POSE[variant]
    ps = []
    if kind == "lgbm":
        X = tp.features(sess, k, "abs", tp.JOINTS_FULL, spec["offsets"])
        X = (X - X.mean(0)) / (X.std(0) + 1e-6) * sd + mu
        for m in models:
            q = m.model.predict_proba(X)
            out = np.full((len(k), NA), 1e-6)
            out[:, m.classes] = np.maximum(q, 1e-6)
            ps.append(out / out.sum(1, keepdims=True))
    else:
        X = seq_feats(sess, k, spec["half"], spec["stride"])
        f = X.reshape(-1, X.shape[2])
        X = ((X - f.mean(0)) / (f.std(0) + 1e-6) * sd + mu)
        ps = [seq_proba(m, X) for m in models]
    p = np.mean(ps, axis=0)
    return p / p.sum(1, keepdims=True)


# ------------------------------------------------------------------ desk: one stack
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
OBS = (2.0, 2.5, 3.0, 4.0)
DELETIONS = (-9.0, -5.0, -3.0)
BAND = (1.2, 3.6)
TAG = "base+rec"


def stack(session, ev: np.ndarray, kf, alpha: float, pose_v: str, pix: np.ndarray, seeds):
    import phase0.analysis.pipeline as pl

    d = pl.frame_probs(session)
    taps = pl.taps_from(d, ev)
    sess = tp.load_sess(str(session))
    k = sess.rows(taps)
    pose = desk_pose(pose_v, sess, k, seeds)
    fr, _ = bank()
    want = d["frames"][ev]
    j = np.searchsorted(fr, want)
    if not (fr[np.minimum(j, len(fr) - 1)] == want).all():
        raise SystemExit("tap frame missing from the hirecall bank")
    p = pose if alpha <= 0 else gmean([pix[j], pose], [alpha, 1.0 - alpha])
    return taps, pl.segments(session, taps), p, sess, k


def _row_job(args):
    import phase0.analysis.combine as CB
    import phase0.analysis.pipeline as pl

    CB._shim()
    sid, tag, pose_v, pix_f, seeds = args
    hr = _hr()
    kf = pl.kbd_fit()
    s = pl.session_path(DESK)
    ev = np.array(hr.load_streams(tag)[sid]["ev"], int)
    pix = np.load(pix_f)
    rows = []
    for a in ALPHAS:
        _, segs, p, _, _ = stack(s, ev, kf, a, pose_v, pix, seeds)
        for w in OBS:
            for dl in DELETIONS:
                c = CB.Cfg(alpha=a, obs=w, deletion=dl, em=False)
                r = CB.decode_rows(p, segs, kf, c)
                rows.append({"sid": sid, "alpha": a, "obs": w, "deletion": dl,
                             "n_taps": int(len(ev)), "per": [(x[2], x[3]) for x in r]})
    return rows


def table_path(pose_v: str, pix_v: str) -> Path:
    return CACHE / f"table_{TAG}_{pose_v}_{pix_v}.json"


def cmd_table(a) -> int:
    import multiprocessing as mp
    import phase0.analysis.combine as CB

    hr = _hr()
    CB.CACHE = HR_CACHE
    seeds = tuple(range(a.seeds))
    pix_f = str(CACHE / f"deskpix_{a.pix}_{len(seeds)}.npy")
    desk_pix(a.pix, seeds)
    nch = hr.nchars()
    sids = [r["sid"] for r in hr.load_streams(TAG)
            if BAND[0] <= len(r["ev"]) / nch <= BAND[1]]
    p = table_path(a.pose, a.pix)
    if p.exists() and not a.force:
        print(f"{p} exists")
        return 0
    print(f"[{a.pose}+{a.pix}] {len(sids)} streams x {len(ALPHAS)} alphas x {len(OBS)} obs x "
          f"{len(DELETIONS)} deletions", flush=True)
    jobs = [(i, TAG, a.pose, pix_f, seeds) for i in sids]
    out, t0 = [], time.time()
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for n, rows in enumerate(pool.imap_unordered(_row_job, jobs, chunksize=1)):
            out += rows
            if (n + 1) % 5 == 0:
                print(f"  {n+1}/{len(jobs)} streams, best "
                      f"{min(CB.pooled(r) for r in out):.3f} ({time.time()-t0:.0f}s)", flush=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    print(f"-> {p}")
    return 0


# ------------------------------------------------------------------ desk: nested CV
def _cv_job(args):
    import phase0.analysis.combine as CB
    import phase0.analysis.pipeline as pl

    CB._shim()
    j, pick, pose_v, pix_f, seeds, em, st_w = args
    hr = _hr()
    kf = pl.kbd_fit()
    s = pl.session_path(DESK)
    ev = np.array(hr.load_streams(TAG)[pick["sid"]]["ev"], int)
    pix = np.load(pix_f)
    c = CB.Cfg(alpha=pick["alpha"], obs=pick["obs"], deletion=pick["deletion"], em=em)
    _, segs, p, sess, k = stack(s, ev, kf, pick["alpha"], pose_v, pix, seeds)
    tr = [x for i, x in enumerate(segs) if i != j and len(x[1])]
    if em:
        p = pl.weakly_supervise(sess, k, p, tr, replace(pl.FULL, deletion=pick["deletion"]))
    if st_w > 0:
        p = self_train(s, ev, segs, tr, p, pick, st_w, seeds)
    return j, CB.decode_rows(p, segs, kf, c, keep={j})[0]


def self_train(session, ev, segs, tr, p, pick, st_w: float, seeds):
    """Fine-tune the pixel expert on the desk itself, labelled by EM's alignment to the other
    19 phrases' text; the scored phrase's text never enters."""
    import phase0.analysis.adapt as ad
    import phase0.analysis.pipeline as pl

    d = pl.frame_probs(session)
    fr, b = bank()
    rows = np.searchsorted(fr, d["frames"][ev])
    par = ad.AlignParams.from_counts(
        sum(len(i) for _, i in tr),
        sum(sum(ch in ad.A_INDEX for ch in t) for t, i in tr if len(i)))
    q, q_ins, _, _ = ad.align_all(np.log(np.maximum(p, EPS)), tr, par)
    fit_idx = np.unique(np.concatenate([i for _, i in tr]))
    conf = (q[fit_idx].max(1) >= 0.5) & (q_ins[fit_idx] < 0.5)
    if int(conf.sum()) < 100:
        return p
    X = chans(b, REPS["temporal"], "none", rows=rows)
    ps = []
    for s in seeds[:1]:  # one seed: 20 folds x a CPU fit is the whole budget
        m = fit_pix(X[fit_idx][conf], q[fit_idx].argmax(1)[conf], s, 40, 24, False)
        ps.append(pix_proba(m, X))
    return gmean([np.mean(ps, axis=0), p], [st_w, 1.0 - st_w])


def cmd_cv(a) -> int:
    import multiprocessing as mp
    import phase0.analysis.combine as CB
    import phase0.analysis.pipeline as pl

    hr = _hr()
    CB.CACHE = HR_CACHE
    seeds = tuple(range(a.seeds))
    pix_f = str(CACHE / f"deskpix_{a.pix}_{len(seeds)}.npy")
    rows = json.loads(table_path(a.pose, a.pix).read_text())
    provenance(a)
    n = len(rows[0]["per"])
    picks = [CB.select(rows, set(range(n)) - {j}) for j in range(n)]
    jobs = [(j, picks[j], a.pose, pix_f, seeds, a.em, a.selftrain) for j in range(n)]
    out: dict = {}
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for j, r in pool.imap_unordered(_cv_job, jobs):
            out[j] = r
            p = picks[j]
            print(f"  phrase{j:>2} sid={p['sid']} taps={p['n_taps']} "
                  f"({p['n_taps']/hr.nchars():.2f}/char) a={p['alpha']} obs={p['obs']} "
                  f"del={p['deletion']:.0f} CER={r[2]/max(1,r[3]):.3f}", flush=True)
    res = [out[j] for j in range(n)]
    name = f"{a.pose} + {a.pix}" + (f" + self-train@{a.selftrain}" if a.selftrain else "")
    print()
    rep = hr.report(name, res)
    base = json.loads(Path(a.baseline).read_text())
    bp = [(None, None, e, nn) for e, nn in base["per"]]
    dd, lo, hi = pl.boot_delta(bp, res)
    print(f"  delta vs {base['name']} ({base['cer']:.3f}): {dd:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print(f"\n-- {min(a.examples, len(res))} decoded phrases --")
    hr.examples(res, a.examples)
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))
    return 0


def md5(p) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def provenance(a) -> None:
    print("pinned inputs:")
    for f in (tp.sess_path(DESK) / "landmarks.parquet", "models/taps_gb.joblib",
              "models/charlm.npz", HR_CACHE / "streams_base+rec.json",
              HR_CACHE / f"{DESK}_ftips.npy", table_path(a.pose, a.pix)):
        print(f"  {f} md5 {md5(f)[:12]}")
    for sid in KBD:
        print(f"  {sid}/{mk.TAPS_NAME} md5 "
              f"{md5(tp.sess_path(sid) / mk.TAPS_NAME)[:12]}  (key-model labels)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.keymax", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("keys", cmd_keys), ("fuse", cmd_fuse), ("table", cmd_table),
                     ("cv", cmd_cv), ("fit", cmd_fit)):
        p = sub.add_parser(name)
        p.add_argument("--variants", default="pose,pix")
        p.add_argument("--seeds", type=int, default=3)
        p.add_argument("--step", type=float, default=0.25)
        p.add_argument("--pose", default="pose")
        p.add_argument("--pix", default="pix")
        p.add_argument("--procs", type=int, default=9)
        p.add_argument("--em", type=int, default=1)
        p.add_argument("--selftrain", type=float, default=0.0)
        p.add_argument("--examples", type=int, default=8)
        p.add_argument("--baseline", default=".cache/hirecall/cv_all.json")
        p.add_argument("--json", default=None)
        p.add_argument("--held", default=None)
        p.add_argument("--cpu", action="store_true")
        p.add_argument("--threads", type=int, default=3)
        p.add_argument("--force", action="store_true")
        p.set_defaults(func=fn)
    a = ap.parse_args(argv)
    t0 = time.time()
    r = a.func(a)
    print(f"\n[{time.time()-t0:.1f}s]", file=sys.stderr)
    return r


if __name__ == "__main__":
    sys.exit(main())
