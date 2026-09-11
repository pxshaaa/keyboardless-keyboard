"""Can public keystroke footage fix per-key identification? Pretrain on andrewt28, fine-tune on ours.
Run: python -m phase0.analysis.keypre {check|pubindex|pubpose|pubtips|pose|pix|keys|deskpix|desktable|deskcv}"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

import lightgbm  # noqa: F401  must load before torch or the two libomp copies segfault

from phase0.analysis.decode import A_INDEX, ALPHABET, NA

CACHE = Path(".cache/keypre")
PROBS = CACHE / "probs"
HR_CACHE = Path(".cache/hirecall")
KBD = ("20260910-015217-kbd", "20260910-021315-kbd", "20260910-131629-kbd")
HELD = "20260910-015948-kbd"
DESK = "20260910-202149-desk"
FOLDS = (*KBD, HELD)
FRACS = (0.25, 0.5, 1.0)
SEEDS = (0, 1, 2)
MIN_N = 12
THREADS = 4

# US MacBook legends vs the user's German QWERTZ: y and z sit at each other's physical position
PHYS_SWAP = {"y": "z", "z": "y"}
# our detector tap coincides with keydown (median -2 ms); contact is 4 frames later at 60 fps
CONTACT_S = 4 / 60.0
OFFSET = 4
UPSCALE = 2.0

POSE_CONDS = ("scratch_abs", "scratch_rel", "zeroshot_rel", "finetune_rel", "stack_rel")
PIX_CONDS = ("scratch", "zeroshot", "finetune")
PIX_EPOCHS = 40
PIX_LR = 3e-3
FT_LR = 1e-3
PRE_EPOCHS = 12
ALPHAS = (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0)


def md5(p) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ labels
def pub_class(key: str) -> int | None:
    """Public X11 key name -> our 27-class index by physical key position, or None."""
    if key == "space":
        return A_INDEX[" "]
    if len(key) == 1 and key.isascii() and key.isalpha():
        c = key.lower()
        return A_INDEX[PHYS_SWAP.get(c, c)]
    return None


# ------------------------------------------------------------------ public keypress index
def _clip_index(job):
    from phase0.analysis import pretrain as PT

    ci, clip = job
    t, P = PT.load_public_frames(clip["split"], clip["file_name"])
    fps = float(clip["actual_fps"])
    fr = np.round(t * fps).astype(int)
    two = np.isfinite(P[:, :, :, 0]).all(axis=(1, 2))
    ok = dict(zip(fr.tolist(), two.tolist()))
    n_frames = int(round(clip["duration_sec"] * fps))
    out = []
    for k in clip["keystrokes"]:
        if k.get("event", "down") != "down":
            continue
        c = pub_class(k["key"])
        if c is None:
            continue
        ts = k["timestamp_ms"] / 1000.0
        f0, f4 = int(round(ts * fps)), int(round((ts + CONTACT_S) * fps))
        if ok.get(f0) and f4 < n_frames:
            out.append((ci, f0, f4, ts, c))
    return out


def pub_index(force: bool = False) -> dict:
    """All in-alphabet public keydowns whose tap frame has both hands tracked."""
    p = CACHE / "pub_index.npz"
    if p.exists() and not force:
        d = np.load(p, allow_pickle=True)
        return {k: d[k] for k in d.files}
    import multiprocessing as mp
    from phase0.analysis import pretrain as PT

    clips = PT.all_clips()
    rows = []
    with mp.get_context("spawn").Pool(3) as pool:
        for r in pool.imap(_clip_index, list(enumerate(clips)), chunksize=8):
            rows += r
    a = np.array(rows, float)
    d = {"clip": a[:, 0].astype(int), "f0": a[:, 1].astype(int), "f4": a[:, 2].astype(int),
         "t": a[:, 3], "y": a[:, 4].astype(int),
         "split": np.array([clips[int(i)]["split"] for i in a[:, 0]]),
         "fname": np.array([clips[int(i)]["file_name"] for i in a[:, 0]])}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(p, **d)
    return d


# ------------------------------------------------------------------ hand geometry
def geometry(P: np.ndarray, k: float) -> tuple[np.ndarray, np.ndarray]:
    """-> (anchor [2,2], span [2]) from in-frame joints only: public wrists sit outside the
    frame, so MediaPipe extrapolates them and wrist->MCP spans come out 4x too long."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        anchor = np.nanmedian(P[:, :, 9, :2], axis=0)
        span = k * np.nanmedian(np.linalg.norm(P[:, :, 5, :2] - P[:, :, 17, :2], axis=2), axis=0)
    span = np.where(np.isfinite(span) & (span > 1e-6), span, np.nanmedian(span) if
                    np.isfinite(span).any() else 100.0)
    return np.where(np.isfinite(anchor), anchor, 0.0), span


_K: list = []


def span_k() -> float:
    """wrist->MCP span per palm width on our training sessions, so crop scale matches our banks."""
    if not _K:
        from phase0.analysis import tap_pos as tp
        r = []
        for s in KBD:
            S = tp.load_sess(s)
            r += list(S.span / geometry(S.P, 1.0)[1])
        _K.append(float(np.median(r)))
    return _K[0]


class GeoSess:
    """tap_pos.Sess stand-in: same landmarks, the shared palm-width geometry."""

    def __init__(self, P: np.ndarray, k: float, t: np.ndarray | None = None):
        self.P, self.t = P, t
        self.anchor, self.span = geometry(P, k)

    def rows_at(self, ts: np.ndarray) -> np.ndarray:
        return np.clip(np.round((ts - self.t[0]) * 60.0).astype(int), 0, len(self.t) - 1)


def _clip_pose(job):
    from phase0.analysis import pretrain as PT
    from phase0.analysis import tap_pos as tp

    split, name, ts, k = job
    t, P = PT.load_public_frames(split, name)
    tn, Pn = PT.resample_60(t, P)
    s = GeoSess(Pn, k, tn)
    return tp.features(s, s.rows_at(ts), "rel", tp.JOINTS_FULL, tp.OFFSETS)


def pub_pose(force: bool = False) -> np.ndarray:
    p = CACHE / "pub_pose_rel.npy"
    if p.exists() and not force:
        return np.load(p, mmap_mode="r")
    import multiprocessing as mp

    ix = pub_index()
    clips = np.unique(ix["clip"])
    jobs = [(str(ix["split"][ix["clip"] == c][0]), str(ix["fname"][ix["clip"] == c][0]),
             ix["t"][ix["clip"] == c], span_k()) for c in clips]
    with mp.get_context("spawn").Pool(3) as pool:
        parts = pool.map(_clip_pose, jobs, chunksize=4)
    X = np.zeros((len(ix["y"]), parts[0].shape[1]), np.float32)
    for c, part in zip(clips, parts):
        X[ix["clip"] == c] = part
    np.save(p, X)
    return X


# ------------------------------------------------------------------ public fingertip crops
def _clip_tips(job):
    import cv2
    from phase0.analysis import pretrain as PT
    from phase0.analysis.appearance import TIP_CROP, TIP_SPANS
    from phase0.analysis.finger_id import FINGERTIP_JOINTS

    split, name, fps, f0, f4, k = job
    t, P = PT.load_public_frames(split, name)
    fr = np.round(t * fps).astype(int)
    row = {f: i for i, f in enumerate(fr.tolist())}
    side = TIP_SPANS * geometry(P, k)[1] / UPSCALE
    C = P[[row[f] for f in f0]][:, :, list(FINGERTIP_JOINTS), :2] / UPSCALE
    out = np.zeros((len(f0), 2, 2, 5, TIP_CROP, TIP_CROP), np.uint8)
    want: dict[int, list[tuple[int, int]]] = {}
    for i, (a, b) in enumerate(zip(f0, f4)):
        want.setdefault(int(a), []).append((i, 0))
        want.setdefault(int(b), []).append((i, 1))
    cap = cv2.VideoCapture(str(PT.VIDEO / split / name))
    idx, last = 0, max(want)
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        jobs = want.get(idx, ())
        if jobs:
            g = cv2.cvtColor(cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.COLOR_BGR2GRAY)
        for i, j in jobs:
            for h in (0, 1):
                sc = TIP_CROP / side[h]
                for f in range(5):
                    cx, cy = C[i, h, f]
                    M = np.array([[sc, 0, -sc * (cx - side[h] / 2)],
                                  [0, sc, -sc * (cy - side[h] / 2)]], np.float32)
                    out[i, j, h, f] = cv2.warpAffine(g, M, (TIP_CROP, TIP_CROP),
                                                     flags=cv2.INTER_AREA,
                                                     borderMode=cv2.BORDER_REPLICATE)
        idx += 1
    cap.release()
    lo, hi = np.percentile(out, [2.0, 99.8])
    hi = max(hi, lo + 1.0)
    return np.clip((out.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def pub_tips(force: bool = False) -> np.ndarray:
    """[N, 2 (tap, contact), 2 hands, 5 tips, 32, 32] uint8, row-aligned with pub_index."""
    from phase0.analysis.appearance import TIP_CROP
    from phase0.analysis import pretrain as PT
    import multiprocessing as mp

    p = CACHE / "pub_tips.npy"
    if p.exists() and not force:
        return np.load(p, mmap_mode="r")
    ix = pub_index()
    clips = PT.all_clips()
    ids = np.unique(ix["clip"])
    jobs = [(clips[c]["split"], clips[c]["file_name"], float(clips[c]["actual_fps"]),
             ix["f0"][ix["clip"] == c], ix["f4"][ix["clip"] == c], span_k()) for c in ids]
    tmp = p.with_suffix(".part.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                    shape=(len(ix["y"]), 2, 2, 5, TIP_CROP, TIP_CROP))
    t0 = time.time()
    with mp.get_context("spawn").Pool(3) as pool:
        for n, (c, arr) in enumerate(zip(ids, pool.imap(_clip_tips, jobs, chunksize=2))):
            out[np.where(ix["clip"] == c)[0]] = arr
            if (n + 1) % 100 == 0:
                print(f"  {n+1}/{len(ids)} clips ({time.time()-t0:.0f}s)", flush=True)
    out.flush()
    del out
    tmp.replace(p)
    return np.load(p, mmap_mode="r")


def pub_rep(tips: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """[n,10,2,32,32] float32 in appearance's 'diff' rep: contact frame, contact minus tap frame."""
    a = np.asarray(tips[np.sort(rows)], np.float32) / 255.0
    a = a[np.argsort(np.argsort(rows))]
    n = len(rows)
    base = a[:, 1].reshape(n, 10, 1, *a.shape[-2:])
    ref = a[:, 0].reshape(n, 10, 1, *a.shape[-2:])
    return np.concatenate([base, base - ref], axis=2)


def pub_split(ix: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pretraining rows and a public-only monitoring split; ours never enters either."""
    va = ix["split"] == "validation"
    return np.where(~va)[0], np.where(va)[0]


# ------------------------------------------------------------------ our data
_DS: dict = {}


def ours(sid: str) -> dict:
    if sid not in _DS:
        from phase0.analysis import masked as mk
        d = mk.build_dataset(sid)
        d["time_order"] = np.argsort([t["t"] for t in d["taps"]], kind="stable")
        _DS[sid] = d
    return _DS[sid]


def train_sids(held: str) -> list[str]:
    return [s for s in KBD if s != held]


def subset(n: int, frac: float, seed: int, sid: str) -> np.ndarray:
    """One contiguous window per session: the user simply recorded for less time."""
    if frac >= 1.0:
        return np.arange(n)
    m = max(1, int(round(frac * n)))
    rng = np.random.default_rng([seed, int(md5_str(sid)[:8], 16)])
    off = int(rng.integers(0, n - m + 1))
    return np.arange(off, off + m)


def md5_str(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def fold_rows(held: str, frac: float, seed: int) -> dict[str, np.ndarray]:
    return {s: ours(s)["time_order"][subset(len(ours(s)["y"]), frac, seed, s)]
            for s in train_sids(held)}


def probs_path(family: str, cond: str, frac: float, held: str, seed: int) -> Path:
    return PROBS / f"{family}_{cond}_f{frac}_{held}_s{seed}.npy"


def to_na(p: np.ndarray) -> np.ndarray:
    p = np.maximum(p, 1e-6)
    return p / p.sum(1, keepdims=True)


# ------------------------------------------------------------------ pose models
def lgb_params(seed: int, pre: bool = False) -> dict:
    """tap_pos._lgbm's settings; the pretraining run gets more leaves for 12x the rows."""
    p = dict(objective="multiclass", num_class=NA, learning_rate=0.08, num_leaves=15,
             min_data_in_leaf=8, bagging_fraction=0.9, bagging_freq=1, feature_fraction=0.6,
             verbose=-1, num_threads=THREADS, seed=seed)
    if pre:
        p.update(learning_rate=0.05, num_leaves=31, min_data_in_leaf=20)
    return p


def keep_frequent(y: np.ndarray) -> np.ndarray:
    cnt = Counter(y.tolist())
    return np.array([cnt[v] >= MIN_N for v in y])


def pose_pretrained(seed: int):
    import lightgbm as lgb

    p = CACHE / f"pose_pre_s{seed}.txt"
    if p.exists():
        return lgb.Booster(model_file=str(p))
    ix = pub_index()
    X = np.asarray(pub_pose())
    tr, va = pub_split(ix)
    t0 = time.time()
    b = lgb.train(lgb_params(seed, pre=True), lgb.Dataset(X[tr], ix["y"][tr]), 1000,
                  valid_sets=[lgb.Dataset(X[va], ix["y"][va])],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    acc = float((b.predict(X[va]).argmax(1) == ix["y"][va]).mean())
    print(f"  pose pretrain seed{seed}: {b.best_iteration} rounds, public-val top1 {acc:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    b.save_model(str(p), num_iteration=b.best_iteration)
    (CACHE / f"pose_pre_s{seed}.json").write_text(json.dumps(
        {"rounds": b.best_iteration, "pub_val_top1": acc, "n_train": int(len(tr))}))
    return lgb.Booster(model_file=str(p))


def pose_sess(sid: str, mode: str):
    """'rel' uses the palm-width geometry on both corpora; 'abs' is tap_pos verbatim."""
    s = ours(sid)["sess"]
    return s if mode == "abs" else GeoSess(s.P, span_k())


def pose_fold(cond: str, frac: float, held: str, seed: int) -> np.ndarray:
    import lightgbm as lgb
    from phase0.analysis import tap_pos as tp

    out = probs_path("pose", cond, frac, held, seed)
    if out.exists():
        return np.load(out)
    mode = "abs" if cond.endswith("abs") else "rel"
    rows = fold_rows(held, frac, seed)
    te = ours(held)
    Xte = tp.features(pose_sess(held, mode), te["k"], mode, tp.JOINTS_FULL, tp.OFFSETS)
    if cond == "zeroshot_rel":
        p = pose_pretrained(seed).predict(Xte)
    else:
        X = np.vstack([tp.features(pose_sess(s, mode), ours(s)["k"][r], mode, tp.JOINTS_FULL,
                                   tp.OFFSETS) for s, r in rows.items()])
        y = np.concatenate([ours(s)["y"][r] for s, r in rows.items()])
        m = keep_frequent(y)
        if cond == "stack_rel":
            pre = pose_pretrained(seed)
            X, Xte = np.hstack([X, pre.predict(X)]), np.hstack([Xte, pre.predict(Xte)])
        init = pose_pretrained(seed) if cond == "finetune_rel" else None
        b = lgb.train(lgb_params(seed), lgb.Dataset(X[m], y[m]), 250, init_model=init)
        p = b.predict(Xte)
    p = to_na(p)
    PROBS.mkdir(parents=True, exist_ok=True)
    np.save(out, p)
    return p


# ------------------------------------------------------------------ pixel CNN
_DEV = None


def device():
    import torch
    global _DEV
    if _DEV is None:
        _DEV = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    return _DEV


def load_weights(net, sd):
    """-> net on device with `sd` loaded; a multithreaded CPU copy segfaults with lightgbm's libomp."""
    import torch
    net = net.to(device())
    if device().type == "cpu":
        n = torch.get_num_threads()
        torch.set_num_threads(1)
        net.load_state_dict(sd)
        torch.set_num_threads(n)
    else:
        net.load_state_dict(sd)
    return net


def train_tipnet(get_batch, y: np.ndarray, epochs: int, lr: float, seed: int, init=None,
                 bs: int = 64):
    """appearance.fit_cnn's recipe with an optional warm start; always a 27-way joint head."""
    import torch
    from phase0.analysis.appearance import _prep, _tip_net

    torch.manual_seed(seed)
    np.random.seed(seed)
    net = _tip_net(2, -NA)
    dev = device()
    net = load_weights(net, init) if init is not None else net.to(dev)
    yt = torch.from_numpy(y.astype(np.int64))
    n = len(y)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, n // bs) * epochs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.25)
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    gen = torch.Generator().manual_seed(seed + 1)
    net.train()
    step = 0
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen).numpy()
        for i in range(0, n - bs + 1, bs):
            b = np.sort(perm[i:i + bs])
            xb = torch.from_numpy(get_batch(b)).to(dev)
            loss = lossf(net(_prep(xb, gen, True)), yt[b].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step < steps:
                sch.step()
    net.eval()
    return net


def predict_tipnet(net, get_batch, n: int, bs: int = 256) -> np.ndarray:
    import torch
    from phase0.analysis.appearance import _prep

    ps = []
    with torch.no_grad():
        for i in range(0, n, bs):
            xb = torch.from_numpy(get_batch(np.arange(i, min(i + bs, n)))).to(device())
            ps.append(torch.softmax(net(_prep(xb, None, False)), 1).cpu().numpy())
    return to_na(np.vstack(ps))


def pix_pretrained(seed: int):
    import torch

    p = CACHE / f"pix_pre_s{seed}.pt"
    if p.exists():
        return torch.load(p, map_location="cpu")
    ix = pub_index()
    tips = pub_tips()
    tr, va = pub_split(ix)
    t0 = time.time()
    net = train_tipnet(lambda b: pub_rep(tips, tr[b]), ix["y"][tr], PRE_EPOCHS, PIX_LR, seed)
    pv = predict_tipnet(net, lambda b: pub_rep(tips, va[b]), len(va))
    acc = float((pv.argmax(1) == ix["y"][va]).mean())
    print(f"  pix pretrain seed{seed}: {len(tr)} crops x {PRE_EPOCHS} ep, public-val top1 "
          f"{acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
    sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    torch.save(sd, p)
    (CACHE / f"pix_pre_s{seed}.json").write_text(json.dumps(
        {"epochs": PRE_EPOCHS, "pub_val_top1": acc, "n_train": int(len(tr))}))
    return sd


def our_rep(sid: str, rows: np.ndarray) -> np.ndarray:
    from phase0.analysis import masked as mk
    d = ours(sid)
    return mk.tip_crops(d, OFFSET, "none", "diff", rows=d["bank_rows"][rows])


def pix_fold(cond: str, frac: float, held: str, seed: int) -> np.ndarray:
    out = probs_path("pix", cond, frac, held, seed)
    if out.exists():
        return np.load(out)
    te = our_rep(held, np.arange(len(ours(held)["y"])))
    init = pix_pretrained(seed) if cond in ("zeroshot", "finetune") else None
    if cond == "zeroshot":
        import torch
        from phase0.analysis.appearance import _tip_net
        net = load_weights(_tip_net(2, -NA), init).eval()
    else:
        rows = fold_rows(held, frac, seed)
        X = np.concatenate([our_rep(s, r) for s, r in rows.items()])
        y = np.concatenate([ours(s)["y"][r] for s, r in rows.items()])
        net = train_tipnet(lambda b: X[b], y, PIX_EPOCHS,
                           FT_LR if cond == "finetune" else PIX_LR, seed, init)
    p = predict_tipnet(net, lambda b: te[b], len(te))
    PROBS.mkdir(parents=True, exist_ok=True)
    np.save(out, p)
    return p


# ------------------------------------------------------------------ scoring
def topk_hits(p: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    return (np.argsort(-p, 1)[:, :k] == y[:, None]).any(1).astype(float)


def gmean(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    q = alpha * np.log(np.maximum(a, 1e-12)) + (1 - alpha) * np.log(np.maximum(b, 1e-12))
    q = np.exp(q - q.max(1, keepdims=True))
    return q / q.sum(1, keepdims=True)


def nested_alpha(pix: dict, pose: dict, ys: dict, held: str) -> float:
    """Fusion weight chosen on the other LOSO sessions only, never on `held`."""
    src = [s for s in KBD if s != held]
    best, ba = -1.0, 0.5
    for a in ALPHAS:
        acc = np.concatenate([topk_hits(gmean(pix[s], pose[s], a), ys[s], 1) for s in src]).mean()
        if acc > best + 1e-12:
            best, ba = acc, a
    return ba


def boot(x: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    v = x[idx].mean(1)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def equiv_taps(n_pts, acc_pts, acc: float) -> tuple[float, str]:
    """Labelled taps a from-scratch model would need to reach `acc`, log-linear in taps."""
    x = np.log(np.asarray(n_pts, float))
    a = np.asarray(acc_pts, float)
    o = np.argsort(x)
    x, a = x[o], a[o]
    tag = ""
    if acc > a[-1]:
        i, tag = len(x) - 2, ">"
    elif acc < a[0]:
        i, tag = 0, "<"
    else:
        i = int(np.clip(np.searchsorted(a, acc) - 1, 0, len(x) - 2)) if np.all(np.diff(a) > 0) \
            else int(np.argmin(np.abs(a[:-1] - acc)))
    slope = (a[i + 1] - a[i]) / (x[i + 1] - x[i])
    if slope <= 1e-9:
        return float("nan"), "flat"
    return float(np.exp(x[i] + (acc - a[i]) / slope)), tag


def typing_rate() -> tuple[float, dict]:
    """Labelled taps per minute of recording, from each session's keydown span."""
    from phase0.analysis.analyze_drift import read_jsonl
    from phase0.analysis import tap_pos as tp

    per = {}
    for s in FOLDS:
        kt = [r["t"] for r in read_jsonl(tp.sess_path(s) / "keys.jsonl") if r.get("event") == "down"]
        per[s] = (len(ours(s)["y"]), (max(kt) - min(kt)) / 60.0)
    n = sum(v[0] for k, v in per.items() if k in KBD)
    m = sum(v[1] for k, v in per.items() if k in KBD)
    return n / m, per


# ------------------------------------------------------------------ CLI: data
def cmd_check(a) -> int:
    from phase0.analysis import pretrain as PT
    from phase0.analysis.decode import labelled_taps
    from phase0.analysis.analyze_drift import read_jsonl
    from phase0.analysis import tap_pos as tp

    raw = Counter(k["key"] for c in PT.all_clips() for k in c["keystrokes"])
    mapped = Counter()
    for k, n in raw.items():
        c = pub_class(k)
        if c is not None:
            mapped[ALPHABET[c]] += n
    print(f"public: {len(raw)} key names; {sum(mapped.values())} keydowns map to our alphabet")
    print(f"  layout evidence (US): quotedbl={raw['quotedbl']} apostrophe={raw['apostrophe']} "
          f"(shift+' is US), y={raw['y']+raw['Y']} z={raw['z']+raw['Z']} before the physical swap")
    ix = pub_index()
    cnt = Counter(ALPHABET[v] for v in ix["y"])
    print(f"  usable (both hands tracked at the tap frame): {len(ix['y'])}; per class min "
          f"{min(cnt.values())} ({min(cnt, key=cnt.get)!r}) median {int(np.median(list(cnt.values())))}")
    print("  " + " ".join(f"{k!r}:{cnt[k]}" for k in ALPHABET))
    ours_cnt = Counter()
    for s in KBD:
        ours_cnt.update(ALPHABET[v] for v in ours(s)["y"])
    print(f"ours (training sessions): {sum(ours_cnt.values())} labelled taps")
    print("  " + " ".join(f"{k!r}:{ours_cnt[k]}" for k in ALPHABET))
    for s in FOLDS:
        taps, _ = labelled_taps(tp.sess_path(s), "taps_contact.jsonl")
        kt = np.array([r["t"] for r in read_jsonl(tp.sess_path(s) / "keys.jsonl")
                       if r.get("event") == "down"])
        dt = [1000 * (x["t"] - kt[np.argmin(np.abs(kt - x["t"]))]) for x in taps]
        print(f"  {s}: tap - keydown median {np.median(dt):+.1f} ms")
    rate, per = typing_rate()
    print(f"typing rate: {rate:.0f} labelled taps per recorded minute; "
          + ", ".join(f"{s[9:15]}={v[0]} taps/{v[1]:.1f} min" for s, v in per.items()))
    return 0


def cmd_pubindex(a) -> int:
    ix = pub_index(a.force)
    print(f"{len(ix['y'])} public keypresses, {len(np.unique(ix['clip']))} clips")
    return 0


def cmd_pubpose(a) -> int:
    X = pub_pose(a.force)
    print(f"public rel pose features {X.shape}")
    return 0


def cmd_pubtips(a) -> int:
    T = pub_tips(a.force)
    print(f"public tips {T.shape}, md5 of index {md5(CACHE / 'pub_index.npz')[:12]}")
    return 0


# ------------------------------------------------------------------ CLI: experiments
def cmd_pose(a) -> int:
    for s in FOLDS:
        ours(s)
    for seed in SEEDS[:a.seeds]:
        pose_pretrained(seed)
    for frac in FRACS:
        for seed in SEEDS[:a.seeds]:
            for cond in a.conds.split(","):
                if cond == "zeroshot_rel" and frac != 1.0:
                    continue
                t0 = time.time()
                for held in FOLDS:
                    pose_fold(cond, frac, held, seed)
                print(f"  pose {cond:<13} frac={frac} seed={seed} ({time.time()-t0:.0f}s)",
                      flush=True)
    return 0


def cmd_pix(a) -> int:
    if a.cpu:
        import torch
        global _DEV
        torch.set_num_threads(THREADS)
        _DEV = torch.device("cpu")
    for s in FOLDS:
        ours(s)
    for seed in SEEDS[:a.seeds]:
        pix_pretrained(seed)
    for frac in FRACS:
        for seed in SEEDS[:a.seeds]:
            for cond in a.conds.split(","):
                if cond == "zeroshot" and frac != 1.0:
                    continue
                t0 = time.time()
                for held in FOLDS:
                    pix_fold(cond, frac, held, seed)
                print(f"  pix {cond:<9} frac={frac} seed={seed} ({time.time()-t0:.0f}s)",
                      flush=True)
    return 0


def _load(family, cond, frac, seeds):
    f = 1.0 if cond.startswith("zeroshot") else frac
    return {s: {h: np.load(probs_path(family, cond, f, h, s)) for h in FOLDS} for s in seeds}


def _fuse(pix, pose, ys):
    out = {}
    for s in pix:
        out[s] = {}
        for h in FOLDS:
            out[s][h] = gmean(pix[s][h], pose[s][h], nested_alpha(pix[s], pose[s], ys, h))
    return out


def _metrics(P: dict, ys: dict) -> dict:
    """LOSO pooled over the three kbd sessions, and the held-out session, per seed."""
    r = {"loso1": [], "loso5": [], "held1": [], "held5": []}
    hits_loso, hits_held = [], []
    for s, per in P.items():
        h1 = np.concatenate([topk_hits(per[h], ys[h], 1) for h in KBD])
        r["loso1"].append(h1.mean())
        r["loso5"].append(np.concatenate([topk_hits(per[h], ys[h], 5) for h in KBD]).mean())
        hh = topk_hits(per[HELD], ys[HELD], 1)
        r["held1"].append(hh.mean())
        r["held5"].append(topk_hits(per[HELD], ys[HELD], 5).mean())
        hits_loso.append(h1)
        hits_held.append(hh)
    r["hits_loso"] = np.mean(hits_loso, 0)
    r["hits_held"] = np.mean(hits_held, 0)
    return r


def _cell(v) -> str:
    v = np.asarray(v, float)
    return f"{v.mean():.3f}±{v.std():.3f}"


def cmd_keys(a) -> int:
    seeds = SEEDS[:a.seeds]
    ys = {h: ours(h)["y"] for h in FOLDS}
    rate, _ = typing_rate()
    n_loso = {f: np.mean([sum(len(r) for r in fold_rows(h, f, 0).values()) for h in KBD])
              for f in FRACS}
    n_held = {f: sum(len(r) for r in fold_rows(HELD, f, 0).values()) for f in FRACS}
    out = {"rate_taps_per_min": rate, "n_loso": n_loso, "n_held": n_held, "rows": []}
    fams = [("pose", c) for c in POSE_CONDS] + [("pix", c) for c in PIX_CONDS] + \
        [("fused", "scratch_abs+scratch"), ("fused", "scratch_abs+finetune"),
         ("fused", "finetune_rel+finetune")]
    res: dict = {}
    for frac in FRACS:
        for fam, cond in fams:
            try:
                if fam == "fused":
                    pc, xc = cond.split("+")
                    P = _fuse(_load("pix", xc, frac, seeds), _load("pose", pc, frac, seeds), ys)
                else:
                    P = _load(fam, cond, frac, seeds)
            except FileNotFoundError:
                continue
            res[(fam, cond, frac)] = _metrics(P, ys)
    base_of = {"pose": "scratch_abs", "pix": "scratch", "fused": "scratch_abs+scratch"}
    print(f"per-key accuracy, LOSO over {', '.join(s[9:15] for s in KBD)} (pooled over taps) and "
          f"held-out {HELD[9:15]}; mean±sd over seeds {seeds}")
    print(f"training taps per LOSO fold: " + ", ".join(f"{f:.0%}={n_loso[f]:.0f}" for f in FRACS)
          + f"; held-out fold: " + ", ".join(f"{f:.0%}={n_held[f]}" for f in FRACS))
    hdr = (f"{'family':<6}{'condition':<22}{'frac':>5}{'LOSO top1':>14}{'LOSO top5':>14}"
           f"{'held top1':>14}{'held top5':>14}{'held 95% CI':>16}{'dLOSO vs scratch [CI]':>28}")
    print(hdr)
    print("-" * len(hdr))
    for fam, cond in fams:
        for frac in FRACS:
            r = res.get((fam, cond, frac))
            if r is None or (cond.startswith("zeroshot") and frac != 1.0):
                continue
            lo, hi = boot(r["hits_held"])
            b = res.get((fam, base_of[fam], frac))
            dl = ""
            row = {"family": fam, "cond": cond, "frac": frac,
                   **{k: [float(x) for x in r[k]] for k in ("loso1", "loso5", "held1", "held5")},
                   "held_ci": [lo, hi]}
            if b is not None and cond != base_of[fam]:
                d = r["hits_loso"] - b["hits_loso"]
                dlo, dhi = boot(d)
                dh = r["hits_held"] - b["hits_held"]
                hlo, hhi = boot(dh)
                dl = f"{d.mean():+.3f} [{dlo:+.3f},{dhi:+.3f}]"
                row.update(d_loso=[float(d.mean()), dlo, dhi], d_held=[float(dh.mean()), hlo, hhi])
            print(f"{fam:<6}{cond:<22}{frac:>5.2f}{_cell(r['loso1']):>14}{_cell(r['loso5']):>14}"
                  f"{_cell(r['held1']):>14}{_cell(r['held5']):>14}{f'[{lo:.3f},{hi:.3f}]':>16}"
                  f"{dl:>28}", flush=True)
            out["rows"].append(row)
    print(f"\nwhat the public data is worth, in the user's recording time ({rate:.0f} labelled "
          f"taps/min); from-scratch curve interpolated log-linearly over the three fractions")
    for fam, pre, scr in (("pose", "finetune_rel", "scratch_rel"), ("pose", "finetune_rel", "scratch_abs"),
                          ("pose", "stack_rel", "scratch_rel"), ("pix", "finetune", "scratch"),
                          ("fused", "finetune_rel+finetune", "scratch_abs+scratch"),
                          ("fused", "scratch_abs+finetune", "scratch_abs+scratch")):
        if not all((fam, c, f) in res for c in (pre, scr) for f in FRACS):
            continue
        pts = [np.mean(res[(fam, scr, f)]["loso1"]) for f in FRACS]
        cells = []
        for f in FRACS:
            acc = float(np.mean(res[(fam, pre, f)]["loso1"]))
            ne, tag = equiv_taps([n_loso[x] for x in FRACS], pts, acc)
            worth = (ne - n_loso[f]) / rate if np.isfinite(ne) else float("nan")
            cells.append(f"{f:.0%}: {acc:.3f} ~ {tag}{ne:.0f} taps -> {worth:+.1f} min")
            out.setdefault("worth", []).append({"family": fam, "pre": pre, "scratch": scr,
                                                "frac": f, "equiv_taps": ne, "tag": tag,
                                                "worth_min": worth})
        print(f"  {fam} {pre} vs {scr}: " + " | ".join(cells))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))
    return 0


# ------------------------------------------------------------------ desk downstream
DESK_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DESK_OBS = (2.0, 3.0, 4.0)
DESK_DELS = (-9.0, -5.0, -3.0)
DESK_BAND = (1.0, 3.6)
DESK_STREAMS = 40
TAG = "base+rec"


def desk_frames() -> np.ndarray:
    return np.array(json.loads((HR_CACHE / f"{DESK}_frames.json").read_text())["frames"], int)


def desk_bank():
    return desk_frames(), np.load(HR_CACHE / f"{DESK}_ftips.npy", mmap_mode="r")


def desk_rep(tips, rows: np.ndarray) -> np.ndarray:
    from phase0.analysis.appearance import TIP_OFFSETS
    jj = {o: j for j, o in enumerate(TIP_OFFSETS)}
    a = np.asarray(tips[rows], np.float32) / 255.0
    n = len(rows)
    base = a[:, jj[OFFSET]].reshape(n, 10, 1, *a.shape[-2:])
    ref = a[:, jj[min(TIP_OFFSETS)]].reshape(n, 10, 1, *a.shape[-2:])
    return np.concatenate([base, base - ref], axis=2)


def deskpix_path(cond: str) -> Path:
    return CACHE / f"deskpix_{cond}_{len(SEEDS)}x{PIX_EPOCHS}.npy"


def cmd_deskpix(a) -> int:
    """Pixel expert fitted on all three kbd sessions (plus public for 'finetune'), per desk frame."""
    fr, tips = desk_bank()
    n = len(fr)
    X = np.concatenate([our_rep(s, np.arange(len(ours(s)["y"]))) for s in KBD])
    y = np.concatenate([ours(s)["y"] for s in KBD])
    for cond in a.conds.split(","):
        out = deskpix_path(cond)
        if out.exists() and not a.force:
            print(f"{out} exists")
            continue
        ps = []
        for seed in SEEDS:
            t0 = time.time()
            init = pix_pretrained(seed) if cond == "finetune" else None
            net = train_tipnet(lambda b: X[b], y, PIX_EPOCHS,
                               FT_LR if cond == "finetune" else PIX_LR, seed, init)
            ps.append(predict_tipnet(net, lambda b: desk_rep(tips, b), n, bs=512))
            print(f"  desk {cond} seed{seed} ({time.time()-t0:.0f}s)", flush=True)
        p = np.mean(ps, 0)
        np.save(out, p / p.sum(1, keepdims=True))
        print(f"-> {out} {p.shape} mean max prob {p.max(1).mean():.3f}")
    return 0


def _hr_rows() -> list[dict]:
    """hirecall's existing base+rec tables, re-keyed onto the merged stream file."""
    from phase0.analysis import hirecall as hr
    base, rec = hr.load_streams("base"), hr.load_streams("rec")
    merged = hr.load_streams(TAG)
    assert [r["ev"] for r in merged] == [r["ev"] for r in base + rec], "merged streams drifted"
    return ([dict(r) for r in hr.load_table("base", "none")]
            + [dict(r, sid=r["sid"] + len(base)) for r in hr.load_table("rec", "none")])


def desk_streams() -> list[int]:
    """Streams in the density band, thinned by tap count only, never by CER."""
    from phase0.analysis import hirecall as hr
    nch = hr.nchars()
    have = {}
    for r in _hr_rows():
        have.setdefault(r["sid"], set()).add((r["alpha"], r["obs"], r["deletion"]))
    need = {(al, o, d) for al in DESK_ALPHAS for o in DESK_OBS for d in DESK_DELS}
    st = hr.load_streams(TAG)
    cand = [(r["sid"], np.array(r["ev"])) for r in st
            if DESK_BAND[0] <= len(r["ev"]) / nch <= DESK_BAND[1] and need <= have.get(r["sid"], set())]
    return [sid for sid, _ in hr.subsample(cand, DESK_STREAMS)]


def _grid(rows: list[dict], sids) -> list[dict]:
    s = set(sids)
    return [r for r in rows if r["sid"] in s and r["alpha"] in DESK_ALPHAS
            and r["obs"] in DESK_OBS and r["deletion"] in DESK_DELS]


def _init_worker(pix_file: str | None) -> None:
    from phase0.analysis import combine as CB
    from phase0.analysis import hirecall as hr
    CB._shim()
    hr.use_cache()
    if pix_file:
        CB._PIX["none"] = (desk_frames(), np.load(pix_file))


def _table_job(args):
    from phase0.analysis import hirecall as hr
    sid, alphas = args
    return hr._one((TAG, sid, "none", alphas, DESK_OBS, DESK_DELS))


def desk_table_path(cond: str) -> Path:
    return CACHE / f"table_{TAG}_{cond}.json"


def cmd_desktable(a) -> int:
    import multiprocessing as mp
    from phase0.analysis import hirecall as hr

    sids = desk_streams()
    old = _grid(_hr_rows(), sids)
    print(f"{len(sids)} streams in taps/char {DESK_BAND}, grid {len(DESK_ALPHAS)}x{len(DESK_OBS)}x"
          f"{len(DESK_DELS)}", flush=True)
    if a.verify:
        with mp.get_context("spawn").Pool(1, _init_worker, (None,)) as pool:
            got = pool.map(_table_job, [(sids[len(sids) // 2], (0.75,))])[0]
        ref = {(r["sid"], r["alpha"], r["obs"], r["deletion"]): r["per"] for r in old}
        same = [ref[(g["sid"], g["alpha"], g["obs"], g["deletion"])] == [list(x) for x in g["per"]]
                or ref[(g["sid"], g["alpha"], g["obs"], g["deletion"])] == g["per"] for g in got]
        print(f"verify: recomputed {len(got)} rows with hirecall's own pixels; identical={all(same)}")
        return 0
    out = desk_table_path(a.cond)
    if out.exists() and not a.force:
        print(f"{out} exists")
        return 0
    pix = deskpix_path(a.cond)
    alphas = tuple(x for x in DESK_ALPHAS if x > 0)
    rows = [r for r in old if r["alpha"] == 0.0]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.procs, _init_worker, (str(pix),)) as pool:
        for n, got in enumerate(pool.imap_unordered(_table_job, [(s, alphas) for s in sids])):
            rows += got
            if (n + 1) % 5 == 0:
                print(f"  {n+1}/{len(sids)} streams ({time.time()-t0:.0f}s)", flush=True)
    out.write_text(json.dumps(rows))
    print(f"-> {out}")
    return 0


def _cv_job(args):
    from phase0.analysis import hirecall as hr
    j, pick = args
    return hr._cv_fold((j, TAG, pick, "none", "pipe"))


def desk_cv(rows: list[dict], pix_file: str | None, procs: int) -> list[tuple]:
    """hirecall.nested_cv verbatim, with the pixel expert swapped in each worker."""
    import multiprocessing as mp
    from phase0.analysis import combine as CB

    n = len(rows[0]["per"])
    picks = [CB.select(rows, set(range(n)) - {j}) for j in range(n)]
    out = {}
    with mp.get_context("spawn").Pool(procs, _init_worker, (pix_file,)) as pool:
        for j, r in pool.imap_unordered(_cv_job, list(enumerate(picks))):
            out[j] = r
    return [out[j] for j in range(n)], picks


def cmd_deskcv(a) -> int:
    from phase0.analysis import hirecall as hr
    from phase0.analysis import pipeline as pl

    sids = desk_streams()
    print("pinned inputs:")
    for f in (Path("data/sessions") / DESK / "landmarks.parquet", "models/taps_gb.joblib",
              "models/tap_pos.pkl", "models/charlm.npz", HR_CACHE / f"streams_{TAG}.json",
              HR_CACHE / f"{DESK}_ftips.npy", HR_CACHE / f"pix_{DESK}_none_3x40.npy",
              CACHE / "pub_index.npz", CACHE / "pub_tips.npy", deskpix_path(a.cond),
              desk_table_path(a.cond)):
        print(f"  {f} md5 {md5(f)[:12] if Path(f).exists() else '(not on this host; hash on the MacBook)'}")
    for s in KBD:
        print(f"  {s}/taps_contact.jsonl md5 {md5(Path('data/sessions') / s / 'taps_contact.jsonl')[:12]}")
    for seed in SEEDS:
        f = CACHE / f"pix_pre_s{seed}.pt"
        print(f"  {f} md5 {md5(f)[:12] if f.exists() else '(not on this host; hash on the MacBook)'}")
    base_rows = _grid(_hr_rows(), sids)
    new_rows = json.loads(desk_table_path(a.cond).read_text())
    res = {}
    for name, rows, pix in (("scratch pixels (hirecall's), same grid", base_rows, None),
                            (f"{a.cond} pixels (public-pretrained)", new_rows, str(deskpix_path(a.cond)))):
        t0 = time.time()
        r, picks = desk_cv(rows, pix, a.procs)
        res[name] = r
        hr.report(name, r)
        print(f"{'':<44}   picked: alpha {np.median([p['alpha'] for p in picks]):.2f}, obs "
              f"{np.median([p['obs'] for p in picks]):.1f}, del "
              f"{np.median([p['deletion'] for p in picks]):.0f}, taps/char "
              f"{np.median([p['n_taps'] for p in picks]) / hr.nchars():.2f} ({time.time()-t0:.0f}s)",
              flush=True)
    names = list(res)
    d, lo, hi = pl.boot_delta(res[names[0]], res[names[1]])
    print(f"\npaired delta, pretrained - scratch (same streams, grid, protocol): {d:+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}]")
    off = json.loads((HR_CACHE / "cv_all.json").read_text())
    ob = [(None, None, e, n) for e, n in off["per"]]
    d2, lo2, hi2 = pl.boot_delta(ob, res[names[1]])
    print(f"paired delta vs the official hirecall run ({off['cer']:.3f}): {d2:+.3f} "
          f"[{lo2:+.3f}, {hi2:+.3f}]")
    print("\n-- decoded phrases, pretrained stack --")
    hr.examples(res[names[1]], a.examples)
    if a.json:
        Path(a.json).write_text(json.dumps(
            {n: {"cer": pl.boot_ci(r), "per": [(x[2], x[3]) for x in r],
                 "hyp": [(x[0], x[1]) for x in r]} for n, r in res.items()}
            | {"delta": [d, lo, hi], "delta_official": [d2, lo2, hi2]}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.keypre", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("check", cmd_check), ("pubindex", cmd_pubindex), ("pubpose", cmd_pubpose),
                     ("pubtips", cmd_pubtips), ("pose", cmd_pose), ("pix", cmd_pix),
                     ("keys", cmd_keys), ("deskpix", cmd_deskpix), ("desktable", cmd_desktable),
                     ("deskcv", cmd_deskcv)):
        p = sub.add_parser(name)
        p.add_argument("--force", action="store_true")
        p.add_argument("--seeds", type=int, default=len(SEEDS))
        p.add_argument("--conds", default=",".join(POSE_CONDS if name == "pose" else PIX_CONDS))
        p.add_argument("--cond", default="finetune")
        p.add_argument("--cpu", action="store_true")
        p.add_argument("--verify", action="store_true")
        p.add_argument("--procs", type=int, default=4)
        p.add_argument("--examples", type=int, default=6)
        p.add_argument("--json", default=None)
        p.set_defaults(func=fn)
    a = ap.parse_args(argv)
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = a.func(a)
    print(f"\n[{time.time()-t0:.1f}s]", file=sys.stderr)
    return r


if __name__ == "__main__":
    sys.exit(main())
