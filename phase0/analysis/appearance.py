"""Are MediaPipe landmarks an information bottleneck? A CNN on pixel crops at contact.
Run: python -m phase0.analysis.appearance {bank|ablate|eval|apply}"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

import lightgbm  # noqa: F401  must load before torch or the two libomp copies segfault

from phase0.analysis.analyze_drift import read_jsonl
from phase0.analysis.contact import gt_finger
from phase0.analysis.decode import A_INDEX, ALPHABET, NA
from phase0.analysis.finger_id import FINGERTIP_JOINTS, HAND_NAMES
from phase0.analysis.tap_pos import KeyClf, Sess, dataset, load_sess, sess_path

TRAIN_SESSIONS = ("20260910-015217-kbd", "20260910-021315-kbd", "20260910-131629-kbd")
HELDOUT_SESSION = "20260910-015948-kbd"
DESK_SESSION = "20260910-202149-desk"
SEEDS = (0, 1, 2)

BANK = Path(".cache/appearance")
PALM_JOINTS = (0, 5, 9, 13, 17)
CROP = 80                      # px per hand; the two hands are concatenated -> 80x160 input
BOX_SPANS = 2.6                # box side in hand-span units (wrist -> middle MCP)
# contact is 3-5 frames after the detector's peak, so the sweep is centred there and
# the negative offsets exist to form difference images against a pre-contact reference.
BANK_OFFSETS = (-6, -3, 0, 1, 2, 3, 4, 5, 6, 8)
N_FINGER = 10
MIN_N = 12

TIP_CROP = 32                  # px per fingertip patch
TIP_SPANS = 0.9                # patch side in hand-span units
TIP_OFFSETS = (0, 2, 3, 4, 5, 6)


# ------------------------------------------------------------------ crop bank
def _centres(s: Sess, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[N,2,2] box centre per (tap, hand) and [N,2] box side, from landmarks at the tap frame."""
    P = s.P[np.clip(k, 0, len(s.P) - 1)][:, :, :, :2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN hand = hand not detected
        palm = np.nanmean(P[:, :, list(PALM_JOINTS)], axis=2)
        tips = np.nanmean(P[:, :, list(FINGERTIP_JOINTS)], axis=2)
    c = 0.5 * (palm + tips)
    fallback = np.broadcast_to(s.anchor[None], c.shape)
    c = np.where(np.isfinite(c), c, fallback)
    side = np.broadcast_to((BOX_SPANS * s.span)[None], c.shape[:2]).copy()
    return c, side


def bank_paths(sid: str) -> tuple[Path, Path]:
    return BANK / f"{sid}.npy", BANK / f"{sid}.json"


def tip_bank_path(sid: str) -> Path:
    return BANK / f"{sid}_tips.npy"


def build_tip_bank(sid: str, taps_name: str = "taps.jsonl", force: bool = False) -> None:
    """One patch per fingertip: the sharpest form of the hypothesis, since a shared trunk
    then only has to say which of the ten pads is in contact."""
    import cv2

    out_p = tip_bank_path(sid)
    if out_p.exists() and not force:
        return
    BANK.mkdir(parents=True, exist_ok=True)
    p = sess_path(sid)
    s = load_sess(sid)
    taps = read_jsonl(p / taps_name)
    k = s.rows(taps)
    n = len(taps)
    out = np.lib.format.open_memmap(out_p, mode="w+", dtype=np.uint8,
                                    shape=(n, len(TIP_OFFSETS), 2, 5, TIP_CROP, TIP_CROP))
    want: dict[int, list[tuple[int, int]]] = {}
    base = np.asarray([t["i"] for t in taps])
    for j, off in enumerate(TIP_OFFSETS):
        for i, f in enumerate(np.clip(base + off, 0, len(s.frames) - 1)):
            want.setdefault(int(f), []).append((i, j))
    P = s.P[np.clip(k, 0, len(s.P) - 1)][:, :, list(FINGERTIP_JOINTS), :2]
    fb = np.broadcast_to(s.anchor[:, None, :], P.shape[1:])
    P = np.where(np.isfinite(P), P, fb[None])
    side = TIP_SPANS * s.span

    cap = cv2.VideoCapture(str(p / "video.mp4"))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        jobs = want.get(idx)
        if jobs:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for i, j in jobs:
                for h in (0, 1):
                    sc = TIP_CROP / side[h]
                    for f in range(5):
                        cx, cy = P[i, h, f]
                        M = np.array([[sc, 0, -sc * (cx - side[h] / 2)],
                                      [0, sc, -sc * (cy - side[h] / 2)]], np.float32)
                        out[i, j, h, f] = cv2.warpAffine(g, M, (TIP_CROP, TIP_CROP),
                                                         flags=cv2.INTER_AREA,
                                                         borderMode=cv2.BORDER_REPLICATE)
        idx += 1
    cap.release()
    sample = np.asarray(out[:: max(1, n // 200)], dtype=np.uint8)
    lo, hi = np.percentile(sample, [2.0, 99.8])
    hi = max(hi, lo + 1.0)
    for i in range(0, n, 128):
        blk = np.asarray(out[i:i + 128], dtype=np.float32)
        out[i:i + 128] = np.clip((blk - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    out.flush()
    print(f"{sid}: tip bank {out.shape} -> {out_p}", flush=True)


def build_bank(sid: str, taps_name: str = "taps.jsonl", force: bool = False) -> None:
    import cv2

    npy, meta = bank_paths(sid)
    p = sess_path(sid)
    if npy.exists() and meta.exists() and not force:
        return
    BANK.mkdir(parents=True, exist_ok=True)
    s = load_sess(sid)
    taps = read_jsonl(p / taps_name)
    k = s.rows(taps)
    centre, side = _centres(s, k)
    n, no = len(taps), len(BANK_OFFSETS)
    out = np.lib.format.open_memmap(npy, mode="w+", dtype=np.uint8, shape=(n, no, 2, CROP, CROP))

    want: dict[int, list[tuple[int, int]]] = {}
    for j, off in enumerate(BANK_OFFSETS):
        fr = np.clip(np.asarray([t["i"] for t in taps]) + off, 0, len(s.frames) - 1)
        for i, f in enumerate(fr):
            want.setdefault(int(f), []).append((i, j))

    cap = cv2.VideoCapture(str(p / "video.mp4"))
    idx, t0, done = 0, time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        jobs = want.get(idx)
        if jobs:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for i, j in jobs:
                for h in (0, 1):
                    sc = CROP / side[i, h]
                    x0 = centre[i, h, 0] - side[i, h] / 2
                    y0 = centre[i, h, 1] - side[i, h] / 2
                    M = np.array([[sc, 0, -sc * x0], [0, sc, -sc * y0]], np.float32)
                    out[i, j, h] = cv2.warpAffine(g, M, (CROP, CROP), flags=cv2.INTER_AREA,
                                                  borderMode=cv2.BORDER_REPLICATE)
                done += 1
        idx += 1
    cap.release()
    # the scene is dim (mean grey ~29/255); one session-wide stretch keeps difference
    # images valid while giving the 8-bit crops usable contrast.
    sample = np.asarray(out[:: max(1, n // 200)], dtype=np.uint8)
    lo, hi = np.percentile(sample, [2.0, 99.8])
    hi = max(hi, lo + 1.0)
    for i in range(0, n, 128):
        blk = np.asarray(out[i:i + 128], dtype=np.float32)
        out[i:i + 128] = np.clip((blk - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    out.flush()
    meta.write_text(json.dumps({"sid": sid, "n": n, "offsets": list(BANK_OFFSETS), "crop": CROP,
                                "frames_read": idx, "crops": done, "lo": float(lo), "hi": float(hi),
                                "seconds": round(time.time() - t0, 1)}))
    print(f"{sid}: {n} taps x {no} offsets -> {npy} ({idx} frames, {time.time()-t0:.0f}s)", flush=True)


def load_bank(sid: str) -> np.ndarray:
    npy, _ = bank_paths(sid)
    if not npy.exists():
        raise SystemExit(f"missing crop bank for {sid}; run `appearance bank` first")
    return np.load(npy, mmap_mode="r")


# ------------------------------------------------------------------ dataset
def finger_labels(s: Sess, k: np.ndarray, keys: list[str]) -> np.ndarray:
    hand, tip = gt_finger(s, k, keys)
    y = np.full(len(keys), -1)
    ok = (hand >= 0) & (tip >= 0)
    tipi = np.array([FINGERTIP_JOINTS.index(int(v)) if v in FINGERTIP_JOINTS else -1 for v in tip])
    y[ok & (tipi >= 0)] = (hand * 5 + tipi)[ok & (tipi >= 0)]
    return y


def build_dataset(sid: str) -> dict:
    d = dataset(sid)
    bank = load_bank(sid)
    all_taps = read_jsonl(sess_path(sid) / "taps.jsonl")
    pos = {round(t["t"], 6): i for i, t in enumerate(all_taps)}
    rows = np.array([pos[round(t["t"], 6)] for t in d["taps"]])
    d["bank_rows"] = rows
    d["bank"] = bank
    d["yf"] = finger_labels(d["sess"], d["k"], d["keys"])
    return d


def crops(d: dict, offset: int, rep: str) -> np.ndarray:
    """[N,C,80,160] float32 in [0,1]; the two hands are laid side by side, frames are channels."""
    idx = {o: j for j, o in enumerate(BANK_OFFSETS)}

    def at(o: int) -> np.ndarray:
        o = min(BANK_OFFSETS, key=lambda x: abs(x - o))
        a = np.asarray(d["bank"][d["bank_rows"], idx[o]], dtype=np.float32) / 255.0
        return np.concatenate([a[:, 0], a[:, 1]], axis=2)[:, None]     # [N,1,80,160]

    if rep.startswith("tips"):
        return tip_crops(d, offset, rep.split("-", 1)[1] if "-" in rep else "diff")
    base = at(offset)
    if rep == "single":
        return base
    if rep == "stack3":
        return np.concatenate([at(offset - 3), base, at(offset + 3)], axis=1)
    if rep == "diff":
        return np.concatenate([base, base - at(offset - 3)], axis=1)
    if rep == "stack3diff":
        pre = at(offset - 3)
        return np.concatenate([pre, base, at(offset + 3), base - pre], axis=1)
    if rep == "diffonly":
        return base - at(offset - 3)
    raise SystemExit(f"unknown rep {rep}")


def tip_crops(d: dict, offset: int, rep: str = "diff") -> np.ndarray:
    """[N,10,C,32,32] fingertip patches; C=2 adds the patch's own frame difference."""
    if "tips" not in d:
        d["tips"] = np.load(tip_bank_path(d["sid"]), mmap_mode="r")
    jj = {o: j for j, o in enumerate(TIP_OFFSETS)}

    def at(o):
        o = min(TIP_OFFSETS, key=lambda x: abs(x - o))
        a = np.asarray(d["tips"][d["bank_rows"], jj[o]], dtype=np.float32) / 255.0
        return a.reshape(len(a), 10, 1, TIP_CROP, TIP_CROP)

    base = at(offset)
    if rep == "single":
        return base
    return np.concatenate([base, base - at(min(TIP_OFFSETS))], axis=2)


def _tip_net(c_in: int, n_cls: int = N_FINGER):
    """n_cls < 0 forces the joint head (all ten embeddings at once) even for the 10 fingers."""
    torch = _torch()
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Conv2d(c_in, 24, 3, 2, 1), nn.BatchNorm2d(24), nn.ReLU(True),
                nn.Conv2d(24, 48, 3, 2, 1), nn.BatchNorm2d(48), nn.ReLU(True),
                nn.Conv2d(48, 96, 3, 2, 1), nn.BatchNorm2d(96), nn.ReLU(True),
                nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.pos = nn.Parameter(torch.zeros(10, 16))
            self.per_tip = n_cls == N_FINGER
            n_cls_ = abs(n_cls)
            self.head = (nn.Sequential(nn.Linear(96 + 16, 64), nn.ReLU(True), nn.Linear(64, 1))
                         if self.per_tip else
                         nn.Sequential(nn.Linear(10 * (96 + 16), 256), nn.ReLU(True),
                                       nn.Dropout(0.3), nn.Linear(256, n_cls_)))

        def forward(self, x):
            b = x.shape[0]
            e = self.trunk(x.reshape(b * 10, *x.shape[2:])).reshape(b, 10, -1)
            e = torch.cat([e, self.pos.expand(b, -1, -1)], dim=2)
            return self.head(e).squeeze(2) if self.per_tip else self.head(e.reshape(b, -1))

    return Net()


# ------------------------------------------------------------------ model
def _torch():
    import torch
    return torch


def _device():
    torch = _torch()
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class SmallCNN:
    """4 strided conv blocks -> GAP -> linear. Small on purpose: ~3k labelled taps."""

    def __init__(self, model, classes, offset, rep, n_out):
        self.model, self.classes, self.offset, self.rep, self.n_out = model, classes, offset, rep, n_out

    @staticmethod
    def net(c_in: int, n_cls: int, width: int = 32):
        torch = _torch()
        nn = torch.nn

        def blk(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, 2, 1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                                 nn.Conv2d(b, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

        return nn.Sequential(blk(c_in, width), blk(width, width * 2), blk(width * 2, width * 4),
                             blk(width * 4, width * 4), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                             nn.Dropout(0.3), nn.Linear(width * 4, n_cls))

    @staticmethod
    def resnet18(c_in: int, n_cls: int, pretrained: bool):
        torch = _torch()
        import torchvision
        w = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        m = torchvision.models.resnet18(weights=w)
        old = m.conv1.weight.data
        m.conv1 = torch.nn.Conv2d(c_in, 64, 7, 2, 3, bias=False)
        with torch.no_grad():  # replicate the RGB stem across however many frames we stack
            m.conv1.weight.copy_(old.mean(1, keepdim=True).repeat(1, c_in, 1, 1) * (3.0 / c_in))
        m.fc = torch.nn.Linear(512, n_cls)
        return m


def _augment(x, gen):
    """Brightness/contrast + small similarity warp + erasing, all batched on the tensor."""
    torch = _torch()
    dev, b = x.device, x.shape[0]

    def r(*shape):
        return torch.rand(*shape, generator=gen).to(dev)

    x = x * (1 + 0.35 * (r(b, 1, 1, 1) - 0.5) * 2)
    x = x + 0.15 * (r(b, 1, 1, 1) - 0.5) * 2
    ang = (r(b) - 0.5) * 2 * (8 * math.pi / 180)
    sc = 1 + (r(b) - 0.5) * 2 * 0.12
    tx, ty = (r(b) - 0.5) * 2 * 0.10, (r(b) - 0.5) * 2 * 0.10
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    h, w = x.shape[-2:]
    ar = h / w  # the canvas may be 2:1 (two hands side by side); rotation must respect it
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, sin * ar, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = -sin / ar, cos, ty
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
    x = torch.nn.functional.grid_sample(x, grid, padding_mode="border", align_corners=False)
    if float(torch.rand(1, generator=gen)) < 0.5:
        eh, ew = max(2, h // 4), max(2, w // 8)
        h0 = int(torch.randint(0, h - eh, (1,), generator=gen))
        w0 = int(torch.randint(0, w - ew, (1,), generator=gen))
        x[:, :, h0:h0 + eh, w0:w0 + ew] = 0.0
    return x


def _prep(xb, gen, train: bool):
    """Standardise (and optionally augment) per patch; a 5-D batch is [N,10,C,H,W] tips."""
    if xb.dim() == 5:
        b, t = xb.shape[:2]
        f = xb.reshape(b * t, *xb.shape[2:])
        f = _standardize(_augment(f, gen) if train else f)
        return f.reshape(b, t, *xb.shape[2:])
    return _standardize(_augment(xb, gen) if train else xb)


def _standardize(x):
    m = x.mean(dim=(2, 3), keepdim=True)
    s = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-3)
    return (x - m) / s


def fit_cnn(Xtr, ytr, n_out, seed=0, epochs=40, arch="small", width=32, lr=3e-3, bs=64,
            verbose=False):
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(8, __import__("os").cpu_count() or 4)))
    tips = Xtr.ndim == 5
    if tips and n_out == N_FINGER and arch != "tipsjoint":
        classes = np.arange(N_FINGER)
        remap = {i: i for i in range(N_FINGER)}
    else:
        cnt = Counter(ytr.tolist())
        classes = np.array(sorted(c for c in cnt if cnt[c] >= MIN_N))
        remap = {int(c): i for i, c in enumerate(classes)}
    m = np.array([v in remap for v in ytr])
    dev = _device()
    X = torch.from_numpy(np.ascontiguousarray(Xtr[m])).to(dev)
    y = torch.from_numpy(np.array([remap[int(v)] for v in ytr[m]], dtype=np.int64)).to(dev)
    if tips:
        net = _tip_net(X.shape[2], len(classes) if arch != "tipsjoint" else -len(classes))
    elif arch == "small":
        net = SmallCNN.net(X.shape[1], len(classes), width)
    else:
        net = SmallCNN.resnet18(X.shape[1], len(classes), pretrained=(arch == "resnet18"))
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, len(y) // bs) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.25)
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    gen = torch.Generator().manual_seed(seed + 1)
    net.train()
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(len(y), generator=gen).to(dev)
        for i in range(0, len(y) - bs + 1, bs):
            b = perm[i:i + bs]
            xb = _prep(X[b].clone(), gen, True)
            loss = lossf(net(xb), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step < steps:
                sched.step()
        if verbose and ep % 10 == 9:
            print(f"    ep{ep+1} loss={float(loss):.3f}", flush=True)
    net.eval()
    return SmallCNN(net, classes, None, None, n_out)


def cnn_proba(clf: SmallCNN, X: np.ndarray, n_out: int) -> np.ndarray:
    torch = _torch()
    out = np.full((len(X), n_out), 1e-6)
    dev = _device()
    with torch.no_grad():
        ps = []
        for i in range(0, len(X), 256):
            xb = _prep(torch.from_numpy(np.ascontiguousarray(X[i:i + 256])).to(dev), None, False)
            ps.append(torch.softmax(clf.model(xb), 1).cpu().numpy())
    p = np.vstack(ps)
    out[:, clf.classes] = np.maximum(p, 1e-6)
    return out / out.sum(1, keepdims=True)


def n_params(clf: SmallCNN) -> int:
    return sum(p.numel() for p in clf.model.parameters())


# ------------------------------------------------------------------ landmark baseline
class FingerClfLM:
    """The landmark comparator for the finger target: same features tap_pos gives the key model."""

    def __init__(self, model, classes):
        self.model, self.classes = model, classes

    @classmethod
    def fit(cls, ds: list[dict], seed=0):
        from phase0.analysis.tap_pos import JOINTS_FULL, OFFSETS, _lgbm, features
        X = np.vstack([features(d["sess"], d["k"], "abs", JOINTS_FULL, OFFSETS) for d in ds])
        y = np.concatenate([d["yf"] for d in ds])
        m = y >= 0
        cnt = Counter(y[m].tolist())
        keep = m & np.array([cnt.get(int(v), 0) >= MIN_N for v in y])
        classes = np.unique(y[keep])
        remap = {int(c): i for i, c in enumerate(classes)}
        mdl = _lgbm(seed, len(classes))
        mdl.fit(X[keep], np.array([remap[int(v)] for v in y[keep]]))
        return cls(mdl, classes)

    def proba(self, d: dict) -> np.ndarray:
        from phase0.analysis.tap_pos import JOINTS_FULL, OFFSETS, features
        p = self.model.predict_proba(features(d["sess"], d["k"], "abs", JOINTS_FULL, OFFSETS))
        out = np.full((len(d["k"]), N_FINGER), 1e-6)
        out[:, self.classes] = np.maximum(p, 1e-6)
        return out / out.sum(1, keepdims=True)


_LM_CACHE: dict = {}


def lm_proba(target: str, tr: list[dict], te: dict, seed: int) -> np.ndarray:
    """Landmark probabilities do not depend on the crop config, so cache across the ablation."""
    ck = (target, tuple(d["sid"] for d in tr), te["sid"], seed)
    if ck not in _LM_CACHE:
        _LM_CACHE[ck] = (KeyClf.fit(tr, mode="abs", seed=seed).proba(te["sess"], te["k"])
                         if target == "key" else FingerClfLM.fit(tr, seed=seed).proba(te))
    return _LM_CACHE[ck]


def warm_lm(target: str, sids: list[str], seeds, ds: dict, heldout: str | None = None) -> None:
    """LightGBM aborts (OMP #179) once torch has touched MPS, so fit every fold up front."""
    for seed in seeds:
        for held in sids:
            lm_proba(target, [ds[s] for s in sids if s != held], ds[held], seed)
        if heldout:
            lm_proba(target, [ds[s] for s in sids], ds[heldout], seed)


# ------------------------------------------------------------------ scoring
def score(p: np.ndarray, y: np.ndarray) -> dict:
    m = y >= 0
    p, y = p[m], y[m]
    order = np.argsort(-p, axis=1)
    return {"n": int(m.sum()),
            "top1": float((order[:, 0] == y).mean()),
            "top5": float(np.mean([y[i] in order[i, :5] for i in range(len(y))]))}


def _y(d: dict, target: str) -> np.ndarray:
    return d["y"] if target == "key" else d["yf"]


def run_fold(tr: list[dict], te: dict, target: str, offset: int, rep: str, seed: int,
             arch: str, epochs: int, width: int) -> dict:
    n_out = NA if target == "key" else N_FINGER
    Xtr = np.vstack([crops(d, offset, rep) for d in tr])
    ytr = np.concatenate([_y(d, target) for d in tr])
    m = ytr >= 0
    clf = fit_cnn(Xtr[m], ytr[m], n_out, seed=seed, epochs=epochs, arch=arch, width=width)
    pp = cnn_proba(clf, crops(te, offset, rep), n_out)
    pl = lm_proba(target, tr, te, seed)
    return {"pix": pp, "lm": pl, "y": _y(te, target), "params": n_params(clf)}


def fuse(pp: np.ndarray, pl: np.ndarray, a: float) -> np.ndarray:
    q = np.exp(a * np.log(pp) + (1 - a) * np.log(pl))
    return q / q.sum(1, keepdims=True)


def agg(vals: list[float]) -> str:
    v = np.array(vals, float)
    return f"{v.mean():.3f}±{v.std():.3f}"


# ------------------------------------------------------------------ CLI
def cmd_bank(a) -> int:
    for sid in a.sessions:
        build_bank(sid, force=a.force)
        if a.tips:
            build_tip_bank(sid, force=a.force)
    return 0


def _loso(sids, target, offset, rep, seeds, arch, epochs, width, ds):
    """-> per-model list of weighted LOSO top1 (one entry per seed) + fusion sweep."""
    alphas = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
    res = {"pix": [], "lm": [], "params": 0}
    fus = {a_: [] for a_ in alphas}
    top5 = {"pix": [], "lm": []}
    for seed in seeds:
        per, wts = {"pix": [], "lm": []}, []
        per5 = {"pix": [], "lm": []}
        perf = {a_: [] for a_ in alphas}
        for held in sids:
            tr = [ds[s] for s in sids if s != held]
            r = run_fold(tr, ds[held], target, offset, rep, seed, arch, epochs, width)
            res["params"] = r["params"]
            for kk, pv in (("pix", r["pix"]), ("lm", r["lm"])):
                sc = score(pv, r["y"])
                per[kk].append(sc["top1"])
                per5[kk].append(sc["top5"])
            for a_ in alphas:
                perf[a_].append(score(fuse(r["pix"], r["lm"], a_), r["y"])["top1"])
            wts.append(score(r["pix"], r["y"])["n"])
        w = np.array(wts, float)
        for kk in ("pix", "lm"):
            res[kk].append(float(np.average(per[kk], weights=w)))
            top5[kk].append(float(np.average(per5[kk], weights=w)))
        for a_ in alphas:
            fus[a_].append(float(np.average(perf[a_], weights=w)))
    return res, fus, top5


def cmd_ablate(a) -> int:
    sids = list(a.sessions)
    ds = {s: build_dataset(s) for s in sids}
    seeds = tuple(SEEDS[:a.seeds])
    warm_lm(a.target, sids, seeds, ds)
    print(f"target={a.target} arch={a.arch} epochs={a.epochs} seeds={seeds}")
    print(f"{'rep':<12}{'offset':>7}{'LOSO pix':>16}{'LOSO lm':>16}{'best fuse':>22}")
    print("-" * 73)
    for rep in a.reps.split(","):
        for off in [int(v) for v in a.offsets.split(",")]:
            res, fus, _ = _loso(sids, a.target, off, rep, seeds, a.arch, a.epochs, a.width, ds)
            ba = max(fus, key=lambda x: np.mean(fus[x]))
            print(f"{rep:<12}{off:>7}{agg(res['pix']):>16}{agg(res['lm']):>16}"
                  f"{agg(fus[ba]) + f' (a={ba})':>22}", flush=True)
    return 0


def cmd_eval(a) -> int:
    sids = list(a.sessions)
    ds = {s: build_dataset(s) for s in sids + [a.heldout]}
    seeds = tuple(SEEDS[:a.seeds])
    warm_lm(a.target, sids, seeds, ds, a.heldout)
    for s in ds:
        print(f"{s}: {len(ds[s]['keys'])} labelled taps, {int((ds[s]['yf']>=0).sum())} with finger")
    res, fus, top5 = _loso(sids, a.target, a.offset, a.rep, seeds, a.arch, a.epochs, a.width, ds)
    ba = max(fus, key=lambda x: np.mean(fus[x]))
    print(f"\ntarget={a.target} rep={a.rep} offset={a.offset} arch={a.arch} "
          f"params={res['params']:,} seeds={seeds}")
    print(f"LOSO  pixels {agg(res['pix'])} (top5 {agg(top5['pix'])})")
    print(f"LOSO  landmk {agg(res['lm'])} (top5 {agg(top5['lm'])})")
    print(f"LOSO  fusion {agg(fus[ba])} at alpha={ba}  [alpha chosen on LOSO only]")

    held = {"pix": [], "lm": [], "fus": []}
    for seed in seeds:
        r = run_fold([ds[s] for s in sids], ds[a.heldout], a.target, a.offset, a.rep, seed,
                     a.arch, a.epochs, a.width)
        held["pix"].append(score(r["pix"], r["y"])["top1"])
        held["lm"].append(score(r["lm"], r["y"])["top1"])
        held["fus"].append(score(fuse(r["pix"], r["lm"], ba), r["y"])["top1"])
    print(f"HELD  pixels {agg(held['pix'])}")
    print(f"HELD  landmk {agg(held['lm'])}")
    print(f"HELD  fusion {agg(held['fus'])} at alpha={ba}")
    return 0


def cmd_apply(a) -> int:
    """Fit on the training sessions, write key_probs for every tap in `session`."""
    sids = list(a.sessions)
    ds = {s: build_dataset(s) for s in sids}
    tr = [ds[s] for s in sids]
    tgt = _apply_target(a.session, a.offset, a.rep)
    pl = KeyClf.fit(tr, mode="abs", seed=0).proba(tgt["sess"], tgt["k"]) if a.alpha < 1.0 else None
    Xtr = np.vstack([crops(d, a.offset, a.rep) for d in tr])
    ytr = np.concatenate([d["y"] for d in tr])
    probs = []
    for seed in tuple(SEEDS[:a.seeds]):
        clf = fit_cnn(Xtr, ytr, NA, seed=seed, epochs=a.epochs, arch=a.arch, width=a.width)
        probs.append(cnn_proba(clf, tgt["X"], NA))
    p = np.mean(probs, axis=0)
    if pl is not None:
        p = fuse(p, pl, a.alpha)
    if a.temp != 1.0:
        p = p ** (1.0 / a.temp)
        p /= p.sum(1, keepdims=True)
    out = Path(a.out)
    with out.open("w") as fh:
        for n, tp in enumerate(tgt["taps"]):
            order = np.argsort(-p[n])[:a.topk]
            row = dict(tp)
            row["key_probs"] = {ALPHABET[c]: round(float(p[n, c]), 6) for c in order}
            fh.write(json.dumps(row) + "\n")
    print(f"wrote {len(tgt['taps'])} taps with key_probs -> {out}")
    return 0


def _apply_target(sid: str, offset: int, rep: str) -> dict:
    s = load_sess(sid)
    taps = read_jsonl(sess_path(sid) / "taps.jsonl")
    d = {"sid": sid, "sess": s, "taps": taps, "k": s.rows(taps), "bank": load_bank(sid),
         "bank_rows": np.arange(len(taps))}
    d["X"] = crops(d, offset, rep)
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.appearance", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bank")
    b.add_argument("--sessions", nargs="+",
                   default=list(TRAIN_SESSIONS) + [HELDOUT_SESSION, DESK_SESSION])
    b.add_argument("--force", action="store_true")
    b.add_argument("--tips", action="store_true")
    b.set_defaults(fn=cmd_bank, sessions_attr="sessions")

    def common(p):
        p.add_argument("--sessions", nargs="+", default=list(TRAIN_SESSIONS))
        p.add_argument("--target", choices=["key", "finger"], default="key")
        p.add_argument("--arch", default="small",
                       choices=["small", "resnet18", "resnet18scratch", "tipsjoint"])
        p.add_argument("--epochs", type=int, default=40)
        p.add_argument("--width", type=int, default=32)
        p.add_argument("--seeds", type=int, default=3)

    al = sub.add_parser("ablate")
    common(al)
    al.add_argument("--reps", default="single,stack3,diff,stack3diff")
    al.add_argument("--offsets", default="0,3,5")
    al.set_defaults(fn=cmd_ablate)

    ev = sub.add_parser("eval")
    common(ev)
    ev.add_argument("--heldout", default=HELDOUT_SESSION)
    ev.add_argument("--rep", default="stack3diff")
    ev.add_argument("--offset", type=int, default=4)
    ev.set_defaults(fn=cmd_eval)

    ap_ = sub.add_parser("apply")
    common(ap_)
    ap_.add_argument("session")
    ap_.add_argument("--rep", default="stack3diff")
    ap_.add_argument("--offset", type=int, default=4)
    ap_.add_argument("--alpha", type=float, default=1.0)
    ap_.add_argument("--temp", type=float, default=1.0, help="softmax temperature fitted on LOSO")
    ap_.add_argument("--topk", type=int, default=8)
    ap_.add_argument("--out", required=True)
    ap_.set_defaults(fn=cmd_apply)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
