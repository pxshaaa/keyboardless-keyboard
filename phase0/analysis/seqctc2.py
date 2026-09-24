"""seqctc follow-up toward desk WER <= 0.2; How We Type data is CC-BY-NC-4.0 (research only). Decoders tuned on kbd only.
Run: python -m phase0.analysis.seqctc2 {geom|diag|audit|pretrain|lopo|deskft|contzs|bigram|tune|decode}"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from phase0.analysis import seqctc as S
from phase0.analysis.seqctc import (BLANK, DESK, HELD, KBD, LOSO, NS, OTHER, PALM, SPACE_S, SYMS, Aug, Cfg,
                                    KbdSource, PhraseSource, Stream, build, desk_windows, device, eval_windows,
                                    featurize, infer, load_session, pad_batch, sym_text)

CACHE = Path(".cache/seqctc2")
RES = Path("results/seqctc2")
SESS = Path("data/sessions")
TJ = (4, 5, 8, 9, 12, 13, 16, 17, 20)   # normalize.py TEMPLATE_JOINTS: tips + MCPs
LEX_BLOCK = Path("models/vocab_blocklist.txt")
SUFFIXES = ("", "s", "es", "ed", "er", "ers", "ing", "in", "y")
NINF = -np.inf


# ============================================================================ label-free session frame
def template(st: Stream, stride: int = 3) -> np.ndarray:
    """[2, J, 2] per-hand median pose over frames with both hands tracked (no taps, no keys), centred per hand."""
    T = np.full((2, len(TJ), 2), np.nan)
    for h in (0, 1):
        X = st.A[st.M[:, h]][::stride][:, h][:, TJ, :2].astype(np.float64)
        if len(X):
            T[h] = np.nanmedian(X, 0)
    return T - np.nanmean(T, 1, keepdims=True)


def fit_linear(T: np.ndarray, ref: np.ndarray, kind: str) -> np.ndarray:
    """2x2 map (row-vector convention xy @ L) taking template T onto ref: 'sim' rotation+scale, 'aff' full linear."""
    X, Y = T.reshape(-1, 2), ref.reshape(-1, 2)
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    X, Y = X[ok], Y[ok]
    if len(X) < 4:
        return np.eye(2)
    if kind == "aff":
        L, *_ = np.linalg.lstsq(X, Y, rcond=None)
        return L
    U, Sv, Vt = np.linalg.svd(Y.T @ X)
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(Sv) @ D) / (X ** 2).sum())
    return (s * R).T


_CANON: dict = {}


def canon() -> np.ndarray:
    """Generalised Procrustes mean of the 30 How We Type participants' templates, knuckle width pinned to 1."""
    if "c" in _CANON:
        return _CANON["c"]
    f = CACHE / "canon.npy"
    if f.exists():
        _CANON["c"] = np.load(f)
        return _CANON["c"]
    from phase0.analysis import howwetype as H
    Ts = [template(st) for st in H.streams()]
    ref = np.nanmean(Ts, 0)
    for _ in range(8):
        ref = np.nanmean([(T.reshape(-1, 2) @ fit_linear(T, ref, "sim")).reshape(T.shape) for T in Ts], 0)
        ref /= np.nanmean([np.linalg.norm(ref[h, 1] - ref[h, 7]) for h in (0, 1)])
    assert np.isfinite(ref).all(), "canonical template has NaN joints"
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(f, ref)
    _CANON["c"] = ref
    return ref


def framed(st: Stream, frame: str) -> Stream:
    if frame == "none":
        return st
    L = fit_linear(template(st), canon(), frame).astype(np.float32)
    A = st.A.copy()
    A[..., :2] = A[..., :2] @ L
    A[..., 2] *= float(np.sqrt(abs(np.linalg.det(L))))
    A[~st.M] = 0
    return Stream(st.name, st.t, A, st.M, st.kt, st.ks)


_ST: dict = {}


def get_stream(sid: str, frame: str) -> Stream:
    k = (sid, frame)
    if k not in _ST:
        _ST[k] = framed(load_session(sid), frame)
    return _ST[k]


def hwt_streams(frame: str) -> list[Stream]:
    from phase0.analysis import howwetype as H
    return [framed(st, frame) for st in H.streams()]


# ============================================================================ augmentation v2
@dataclass(frozen=True)
class Aug2:
    """Camera-geometry + MediaPipe-error augmentation: wider rotation/scale, anisotropic scale and shear (camera tilt
    foreshortens one axis), per-hand size, temporally correlated joint drift, whole-hand glitch frames."""
    rot: float = 20.0
    scale: float = 0.15
    aniso: float = 0.25
    shear: float = 0.10
    hand: float = 0.15
    trans: float = 0.15
    jitter: float = 0.01
    cjit: float = 0.02
    spike: float = 0.02
    jdrop: float = 0.05
    hdrop: float = 0.05
    tmask: float = 0.08
    speed: float = 0.2


_orig_augment = S.augment


def augment2(A: torch.Tensor, M: torch.Tensor, p):
    if not isinstance(p, Aug2):
        return _orig_augment(A, M, p)
    B, T = A.shape[:2]
    xy = A[..., :2]
    if p.hand > 0:
        f = 1 + (torch.rand(B, 1, 2, 1, 1) * 2 - 1) * p.hand
        palm = xy[..., PALM, :].mean(-2, keepdim=True)
        xy = palm + (xy - palm) * f
    th = (torch.rand(B) * 2 - 1) * math.radians(p.rot)
    R = torch.stack([torch.stack([th.cos(), -th.sin()], -1), torch.stack([th.sin(), th.cos()], -1)], -2)
    sc = 1 + (torch.rand(B) * 2 - 1) * p.scale
    an = torch.exp((torch.rand(B, 2) * 2 - 1) * p.aniso)
    K = torch.zeros(B, 2, 2)
    K[:, 0, 0], K[:, 1, 1] = an[:, 0], an[:, 1]
    K[:, 0, 1] = (torch.rand(B) * 2 - 1) * p.shear * an[:, 1]
    G = (R @ K) * sc[:, None, None]
    xy = torch.einsum("bij,btkqj->btkqi", G, xy) + torch.randn(B, 1, 2, 1, 2) * p.trans
    if p.cjit > 0 and T > 2:
        n = max(2, T // 6 + 2)
        low = torch.randn(B, 84, n) * p.cjit
        low = F.interpolate(low, size=T, mode="linear", align_corners=True).reshape(B, 2, 21, 2, T)
        xy = xy + low.permute(0, 4, 1, 2, 3)
    xy = xy + torch.randn_like(xy) * p.jitter
    if p.spike > 0:
        xy = xy + (torch.rand(B, T, 2, 1, 1) < p.spike).float() * torch.randn(B, T, 2, 1, 2) * 0.15
    A = torch.cat([xy, A[..., 2:] * sc[:, None, None, None, None]], -1)
    if p.jdrop > 0:
        A = A * (torch.rand(B, 1, 2, 21, 1) > p.jdrop).float()
    if p.hdrop > 0:
        drop = torch.zeros(B, T, 2, dtype=torch.bool)
        for b in torch.nonzero(torch.rand(B) < p.hdrop).flatten().tolist():
            s = int(torch.randint(0, max(1, T - 10), ()))
            drop[b, s:s + int(torch.randint(5, max(6, T // 4), ())), int(torch.randint(0, 2, ()))] = True
        M = M & ~drop
    return A, M


S.augment = augment2   # seqctc.train resolves `augment` from its module globals at call time


# ============================================================================ runs
def tag_cfg(tag: str) -> dict:
    f = CACHE / "runs" / tag / "cfg.json"
    if f.exists():
        return json.loads(f.read_text())
    return {"frame": "none", "aug": "v1", "d": 192, "layers": 3, "mix": 0.0}   # seqctc baseline tags


def mkcfg(c: dict, seed: int, steps: int, lr: float) -> Cfg:
    return Cfg(steps=steps, lr=lr, seed=seed, d=c["d"], layers=c["layers"], aug=Aug2() if c["aug"] == "v2" else Aug())


def run_dir(tag: str, seed: int) -> Path:
    return CACHE / "runs" / tag / f"seed{seed}"


def find_file(tag: str, seed: int, name: str) -> Path | None:
    for d in (run_dir(tag, seed), S.run_dir(tag, seed)):
        if (d / name).exists():
            return d / name
    return None


def init_path(init: str, seed: int) -> Path:
    for f in (CACHE / "pre" / f"{init}_s{seed}.pt", CACHE / "pre" / f"{init}.pt", S.CACHE / "pre" / f"{init}.pt",
              S.CACHE / "pre" / f"{init}_s{seed}.pt"):
        if f.exists():
            return f
    raise FileNotFoundError(init)


def load_model(c: dict, f: Path):
    m = build(mkcfg(c, 0, 1, 1e-3))
    m.load_state_dict(torch.load(f, map_location="cpu"))
    return m.eval()


def kbd_srcs(frame: str, sessions, mix: float, hwt=None):
    src = []
    for s in sessions:
        k = KbdSource(get_stream(s, frame))
        src.append((k, k.c.minutes))
    if mix > 0:
        tot = sum(w for _, w in src)
        hw = hwt if hwt is not None else hwt_streams(frame)
        src += [(KbdSource(st), mix / (1 - mix) * tot / len(hw)) for st in hw]
    return src


def eval_sets2(fold: str, frame: str) -> dict:
    if fold != "all":
        st = get_stream(next(s for s in LOSO if fold in s), frame)
        w = eval_windows(st)
        return {fold: (st, w, [sym_text(x[2]) for x in w])}
    h, d = get_stream(HELD, frame), get_stream(DESK, frame)
    hw, dw = eval_windows(h), desk_windows(d)
    return {"held": (h, hw, [sym_text(x[2]) for x in hw]), "desk": (d, dw, [x[2] for x in dw])}


@torch.no_grad()
def infer_cont(model, st: Stream, win_s: float = 12.0, hop_s: float = 6.0, bs: int = 8) -> np.ndarray:
    """Whole-stream posteriors [ceil(T/2), NS] from overlapping chunks; each output frame from its most central chunk."""
    dev = device()
    model.to(dev).eval()
    n = len(st.t)
    fps = (n - 1) / (st.t[-1] - st.t[0])
    W, H = max(8, int(win_s * fps) // 2 * 2), max(2, int(hop_s * fps) // 2 * 2)
    starts = list(range(0, max(n - W, 0) + 1, H))
    last = max(0, n - W) // 2 * 2
    if starts[-1] < last:
        starts.append(last)
    G = (n + 1) // 2
    out, dist = np.zeros((G, NS), np.float32), np.full(G, np.inf)
    for i in range(0, len(starts), bs):
        chunk = starts[i:i + bs]
        items = [(st.A[a:(n if a == starts[-1] else min(a + W, n))], st.M[a:(n if a == starts[-1] else min(a + W, n))])
                 for a in chunk]
        A, M, lens = pad_batch(items, torch.device("cpu"))
        lp = model(featurize(A, M, model.zfeat).to(dev), lens).float().cpu().numpy()
        for b, (a, L) in enumerate(zip(chunk, lens)):
            ol = model.out_len(L)
            g = a // 2 + np.arange(ol)
            ok = g < G
            dd = np.abs(np.arange(ol) - (ol - 1) / 2)
            better = ok & (dd < dist[np.minimum(g, G - 1)])
            out[g[better]] = lp[b, :ol][better]
            dist[g[better]] = dd[better]
    return out


def cmd_pretrain(a) -> int:
    c = {"frame": a.frame, "aug": a.aug, "d": a.d, "layers": a.layers}
    for seed in map(int, a.seeds.split(",")):
        f = CACHE / "pre" / f"{a.tag}_s{seed}.pt"
        if f.exists():
            continue
        cfg = mkcfg(c, seed, a.steps, a.lr)
        src = [(KbdSource(st), 1.0) for st in hwt_streams(a.frame)]
        sets = {s[9:15]: v for s in LOSO for v in eval_sets2(s[9:15], a.frame).values()}
        m = build(cfg)
        if a.init:
            m.load_state_dict(torch.load(init_path(a.init, seed), map_location="cpu"))
        m = S.train(m, src, cfg, log=f"[pre {a.tag} s{seed}]", eval_fn=lambda mm: S.greedy_cer(mm, sets),
                    ckpt=f.with_suffix(".ckpt"))
        f.parent.mkdir(parents=True, exist_ok=True)
        torch.save(m.state_dict(), f)
        (CACHE / "pre" / f"{a.tag}.json").write_text(json.dumps({**c, "steps": a.steps, "lr": a.lr}))
    return 0


def cmd_lopo(a) -> int:
    c = {"frame": a.frame, "aug": a.aug, "d": a.d, "layers": a.layers, "mix": a.mix, "init": a.init, "steps": a.steps}
    (CACHE / "runs" / a.tag).mkdir(parents=True, exist_ok=True)
    (CACHE / "runs" / a.tag / "cfg.json").write_text(json.dumps(c))
    hw = hwt_streams(a.frame) if a.mix > 0 else None
    for seed in map(int, a.seeds.split(",")):
        cfg = mkcfg(c, seed, a.steps, a.lr)
        for fold in a.folds.split(","):
            d = run_dir(a.tag, seed)
            if (d / f"{fold}.done").exists():
                continue
            sets = eval_sets2(fold, a.frame)
            m = build(cfg)
            if a.init:
                m.load_state_dict(torch.load(init_path(a.init, seed), map_location="cpu"))
            t0 = time.time()
            S.train(m, kbd_srcs(a.frame, [s for s in KBD if fold not in s], a.mix, hw), cfg,
                    log=f"[{a.tag} s{seed} {fold}]", ckpt=d / f"{fold}.ckpt")
            d.mkdir(parents=True, exist_ok=True)
            for k, (st, w, refs) in sets.items():
                S.save_lps(d / f"{k}.npz", infer(m, st, w), refs)
            if fold == "all":
                torch.save(m.state_dict(), d / "all.pt")
                np.save(d / "desk_cont.npy", infer_cont(m, get_stream(DESK, a.frame)).astype(np.float16))
            (d / f"{fold}.done").write_text(json.dumps({"cfg": asdict(cfg), **c, "secs": time.time() - t0}))
            print(f"[{a.tag} s{seed} {fold}] {S.greedy_cer(m, sets)} ({time.time() - t0:.0f}s)", flush=True)
    return 0


def cmd_contzs(a) -> int:
    """Continuous desk posteriors of an existing 'all' model (e.g. the seqctc baseline tag hwt)."""
    c = tag_cfg(a.tag)
    for seed in map(int, a.seeds.split(",")):
        out = run_dir(a.tag, seed) / "desk_cont.npy"
        if out.exists():
            continue
        m = load_model(c, find_file(a.tag, seed, "all.pt"))
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, infer_cont(m, get_stream(DESK, c["frame"])).astype(np.float16))
    return 0


def cmd_deskft(a) -> int:
    c = tag_cfg(a.tag)
    desk = get_stream(DESK, c["frame"])
    wins = desk_windows(desk)
    n = len(wins)
    folds = [list(range(n))[k::a.kfold] for k in range(a.kfold)]
    hw = hwt_streams(c["frame"]) if a.mix > 0 else None
    for seed in map(int, a.seeds.split(",")):
        out = run_dir(a.tag, seed) / f"deskft_{a.name}.npz"
        if out.exists():
            continue
        lps, conts, t0 = [None] * n, [], time.time()
        for test in folds:
            cfg = replace(mkcfg(c, seed * 1000 + test[0], a.steps, a.lr))
            m = build(cfg)
            m.load_state_dict(torch.load(find_file(a.tag, seed, "all.pt"), map_location="cpu"))
            tr = [wins[j] for j in range(n) if j not in test]
            src = [(PhraseSource(desk, tr), a.desk_w)]
            kb = kbd_srcs(c["frame"], KBD, a.mix, hw)
            tot = sum(w for _, w in kb)
            src += [(s, w / tot * (1 - a.desk_w)) for s, w in kb]
            S.train(m, src, cfg)
            for j, lp in zip(test, infer(m, desk, [wins[j] for j in test])):
                lps[j] = lp
            conts.append(infer_cont(m, desk).astype(np.float16))
            print(f"[deskft {a.tag}/{a.name} s{seed}] fold {test} {time.time() - t0:.0f}s", flush=True)
        np.save(run_dir(a.tag, seed) / f"deskft_{a.name}_cont.npy", np.stack(conts))
        S.save_lps(out, lps, [w[2] for w in wins])
    return 0


# ============================================================================ self-supervised masked landmarks
def hidden(m, f: torch.Tensor, lens: list[int]):
    x = m.enc(F.gelu(m.inp(m.norm(f.transpose(1, 2))))).transpose(1, 2)
    ol = [m.out_len(n) for n in lens]
    pad = (torch.arange(x.shape[1])[None, :] >= torch.tensor(ol)[:, None]).to(x.device)
    x = m.drop(x)
    for layer in m.tfm:
        x = layer(x, src_key_padding_mask=pad)
    return m.fnorm(x), ol


class UnlabSource:
    """Random 3-8 s crops anywhere in a stream; landmarks only, no keys or text are read."""

    def __init__(self, st: Stream):
        self.st = st

    def draw(self, rng):
        d = rng.uniform(3.0, 8.0)
        t0 = rng.uniform(self.st.t[0], max(self.st.t[0], self.st.t[-1] - d))
        return self.st, t0, t0 + d, np.zeros(0, np.int64)


def cmd_ssl(a) -> int:
    """Masked landmark modelling (~40% of time masked in spans, predict the 30 Hz features) on unlabelled streams."""
    c = {"frame": a.frame, "aug": a.aug, "d": a.d, "layers": a.layers}
    f_out = CACHE / "pre" / f"{a.tag}_s0.pt"
    if f_out.exists():
        return 0
    cfg = mkcfg(c, 0, a.steps, a.lr)
    dev = device()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    hw = [UnlabSource(st) for st in hwt_streams(a.frame)]
    user = [UnlabSource(get_stream(s, a.frame)) for s in KBD + (HELD, DESK, "20260910-181947-desk")]
    srcs = [(s, 1.0 / len(hw) * (1 - a.user_w)) for s in hw] + [(s, a.user_w / len(user)) for s in user]
    w = np.array([x[1] for x in srcs])
    w /= w.sum()
    m = build(cfg).to(dev).train()
    rec = torch.nn.Linear(cfg.d, 506).to(dev)
    params = list(m.parameters()) + list(rec.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=cfg.steps, pct_start=0.1)
    scale, run, t_start = None, [], time.time()
    for step in range(cfg.steps):
        items, lens = [], []
        for k in rng.choice(len(srcs), cfg.bs, p=w):
            st, t0, t1, _ = srcs[k][0].draw(rng)
            items.append(S.crop(st, t0, t1))
        A, M, lens = pad_batch(items, torch.device("cpu"))
        A, M = augment2(A, M, cfg.aug)
        f = featurize(A, M, m.zfeat)
        tgt = F.avg_pool1d(f.transpose(1, 2), 2, 2, ceil_mode=True).transpose(1, 2)
        if scale is None:
            scale = tgt.reshape(-1, tgt.shape[-1]).std(0).clamp_min(1e-3)
        mask = torch.zeros(f.shape[:2], dtype=torch.bool)
        for b, n in enumerate(lens):
            while mask[b, :n].float().mean() < 0.4:
                s = int(rng.integers(0, max(1, n - 6)))
                mask[b, s:s + int(rng.integers(6, 21))] = True
        fin = f.masked_fill(mask[..., None], 0.0)
        h, ol = hidden(m, fin.to(dev), lens)
        m2 = F.max_pool1d(mask[:, None].float(), 2, 2, ceil_mode=True)[:, 0].bool()
        valid = torch.arange(m2.shape[1])[None] < torch.tensor(ol)[:, None]
        sel = (m2 & valid).to(dev)
        pred = rec(h)[sel]
        loss = F.smooth_l1_loss(pred, (tgt / scale).to(dev)[sel])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        run.append(float(loss.detach()))
        if (step + 1) % 100 == 0:
            print(f"[ssl {a.tag}] step {step + 1}/{cfg.steps} loss {np.mean(run[-100:]):.4f} {time.time() - t_start:.0f}s", flush=True)
    f_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in m.state_dict().items()}, f_out)
    (CACHE / "pre" / f"{a.tag}.json").write_text(json.dumps({**c, "steps": a.steps, "ssl": True, "user_w": a.user_w}))
    return 0


# ============================================================================ diagnostics
def cmd_geom(a) -> int:
    from phase0.analysis import howwetype as H
    ref = canon()
    rows = []
    items = [(f"hwt/{st.name}", st) for st in H.streams()] + [(s, load_session(s)) for s in KBD + (HELD, DESK)]
    for name, st in items:
        T = template(st)
        r = {"stream": name}
        for kind in ("sim", "aff"):
            L = fit_linear(T, ref, kind)
            U, sv, Vt = np.linalg.svd(L)
            res = np.sqrt((((T.reshape(-1, 2) @ L) - ref.reshape(-1, 2)) ** 2).sum(1).mean())
            r[kind] = {"rot_deg": float(np.degrees(np.arctan2(L[0, 1], L[0, 0]))), "sv": sv.round(3).tolist(),
                       "rms": float(res)}
        r["rms_none"] = float(np.sqrt(((T - ref) ** 2).sum(-1).mean()))
        rows.append(r)
        print(f"{name:<24} none {r['rms_none']:.3f} | sim rot {r['sim']['rot_deg']:+6.1f} s {r['sim']['sv'][0]:.2f} "
              f"rms {r['sim']['rms']:.3f} | aff sv {r['aff']['sv']} rms {r['aff']['rms']:.3f}", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "geom.json").write_text(json.dumps(rows, indent=1))
    return 0


@torch.no_grad()
def ctc_per_char(model, st, wins, refs) -> float:
    lps = infer(model, st, wins)
    tot, nch = 0.0, 0
    for lp, r in zip(lps, refs):
        tg = S.text_syms(r)
        if not len(tg):
            continue
        l = F.ctc_loss(torch.from_numpy(lp)[:, None], torch.from_numpy(tg)[None], [len(lp)], [len(tg)],
                       blank=BLANK, reduction="sum", zero_infinity=True)
        tot, nch = tot + float(l), nch + len(tg)
    return tot / max(1, nch)


def cmd_diag(a) -> int:
    """Domain-gap check without any user labels in training: How-We-Type-only checkpoint read out on the user's
    sessions in each label-free frame. Lower CTC loss per char = closer input distribution."""
    out = {}
    for ck in a.ckpts.split(","):
        tagc, frames = ck.split(":") if ":" in ck else (ck, "none,sim,aff")
        m = load_model({"d": 192, "layers": 3, "aug": "v1", **(json.loads((CACHE / "pre" / f"{tagc.rsplit('_s', 1)[0]}.json").read_text())
                                                                  if (CACHE / "pre" / f"{tagc.rsplit('_s', 1)[0]}.json").exists() else {})},
                       init_path(tagc, 0))
        for frame in frames.split(","):
            res = {}
            for sid in LOSO + (HELD,):
                st = get_stream(sid, frame)
                w = eval_windows(st)
                refs = [sym_text(x[2]) for x in w]
                hyp = [S.greedy(lp) for lp in infer(m, st, w)]
                res[sid[9:15]] = {"cer": S.rows_cer(list(zip(refs, hyp))), "nll": ctc_per_char(m, st, w, refs)}
            st = get_stream(DESK, frame)
            w = desk_windows(st)
            refs = [x[2] for x in w]
            hyp = [S.greedy(lp) for lp in infer(m, st, w)]
            res["desk"] = {"cer": S.rows_cer(list(zip(refs, hyp))), "nll": ctc_per_char(m, st, w, refs)}
            out[f"{tagc}|{frame}"] = res
            print(tagc, frame, " ".join(f"{k} cer {v['cer']:.3f} nll {v['nll']:.2f}" for k, v in res.items()), flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    f = RES / "diag.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    f.write_text(json.dumps({**old, **out}, indent=1))
    return 0


# ============================================================================ lexicon, word LMs
def blocklist() -> set[str]:
    return {l.strip() for l in LEX_BLOCK.read_text().splitlines() if l.strip() and not l.startswith("#")}


def is_blocked(w: str, B: set[str]) -> bool:
    return any(w.endswith(s) and w[:len(w) - len(s)] in B for s in SUFFIXES if len(w) > len(s))


_LEX: dict = {}


class Lexicon:
    """autocorrect lexicon (Tatoeba+Enron+Gutenberg word counts, decontaminated corpora) minus the blocklist;
    prefix -> best unigram log-prob for lookahead."""

    def __init__(self):
        lex = json.loads(Path(".cache/autocorrect/lexicon.json").read_text())
        B = blocklist()
        self.logp = {w: v for w, v in lex.items() if w.isalpha() and w.isascii() and not is_blocked(w, B)}
        self.n_blocked = len(lex) - len(self.logp)
        self.la = {"": max(self.logp.values())}
        for w, lp in self.logp.items():
            for i in range(1, len(w) + 1):
                p = w[:i]
                if lp > self.la.get(p, NINF):
                    self.la[p] = lp


def lexicon() -> Lexicon:
    if "l" not in _LEX:
        _LEX["l"] = Lexicon()
    return _LEX["l"]


class UniScorer:
    def __init__(self):
        self.lex = lexicon()

    def score(self, pairs):
        return [0.0 if w == "\n" else self.lex.logp[w] for _, w in pairs]


class BigramScorer:
    """Interpolated absolute-discount word bigram (D=0.75) over the lexicon; counts from .cache/seqctc2/bigram.npz."""

    def __init__(self, f: Path = CACHE / "bigram.npz", D: float = 0.75):
        z = np.load(f, allow_pickle=True)
        self.vocab = [str(w) for w in z["vocab"]]
        self.ix = {w: i for i, w in enumerate(self.vocab)}
        self.V = len(self.vocab) + 1
        cu = z["uni"].astype(np.float64) + 0.5
        self.puni = cu / cu.sum()
        self.codes, self.cnt = z["codes"], z["cnt"].astype(np.float64)
        self.ctx, self.types = z["ctx"].astype(np.float64), z["types"].astype(np.float64)
        self.D = D
        self.cache: dict = {}

    def score(self, pairs):
        out = []
        for ws, w in pairs:
            key = (ws[-1] if ws else "", w)
            v = self.cache.get(key)
            if v is None:
                if w == "\n":
                    v = 0.0
                else:
                    wi = self.ix[w]
                    pu = self.puni[wi]
                    ci = self.ix.get(ws[-1]) if ws else None
                    if ci is None or self.ctx[ci] == 0:
                        p = pu
                    else:
                        code = ci * self.V + wi
                        k = np.searchsorted(self.codes, code)
                        cnt = self.cnt[k] if k < len(self.codes) and self.codes[k] == code else 0.0
                        p = max(cnt - self.D, 0.0) / self.ctx[ci] + self.D * self.types[ci] / self.ctx[ci] * pu
                    v = float(np.log(p))
                self.cache[key] = v
            out.append(v)
        return out


def cmd_bigram(a) -> int:
    lex = lexicon()
    vocab = sorted(lex.logp)
    ix = {w: i for i, w in enumerate(vocab)}
    V, UNK = len(vocab) + 1, len(vocab)
    uni = np.zeros(V, np.int64)
    parts_c, parts_n = [], []
    for name in ("tatoeba", "enron", "gutenberg"):
        tail = ""
        with (CACHE / "lm" / f"{name}.txt").open() as fh:
            while True:
                chunk = fh.read(8_000_000)
                if not chunk:
                    break
                chunk = tail + chunk
                k = chunk.rfind(" ")
                chunk, tail = chunk[:k], chunk[k:]
                ids = np.array([ix.get(t, UNK) for t in chunk.split()], np.int64)
                uni += np.bincount(ids, minlength=V)
                c = ids[:-1] * V + ids[1:]
                c = c[(ids[:-1] != UNK) & (ids[1:] != UNK)]
                u, n = np.unique(c, return_counts=True)
                parts_c.append(u)
                parts_n.append(n)
        print(name, flush=True)
    u, inv = np.unique(np.concatenate(parts_c), return_inverse=True)
    cnt = np.bincount(inv, weights=np.concatenate(parts_n)).astype(np.int64)
    ctx = np.bincount(u // V, weights=cnt, minlength=V)
    types = np.bincount(u // V, minlength=V)
    np.savez(CACHE / "bigram.npz", vocab=np.array(vocab), uni=uni[:-1], codes=u, cnt=cnt, ctx=ctx[:-1], types=types[:-1])
    print(f"bigram: {len(vocab)} words, {len(u):,} bigram types, {int(cnt.sum()):,} tokens in-vocab pairs")
    return 0


# ============================================================================ decoders
@dataclass(frozen=True)
class WCfg:
    lm: float = 1.0      # word LM log-prob weight (nats) at each word end
    la: float = 0.5      # unigram lookahead weight inside a word
    wb: float = 1.0      # word insertion bonus
    beta: float = 0.0    # per-character bonus
    ac: float = 0.0      # char 6-gram weight inside words
    beam: int = 16
    prune: float = -7.0


def _lae(a, b):
    return np.logaddexp(a, b)


def word_beam(lp: np.ndarray, scorer, c: WCfg, charlm=None) -> str:
    """CTC prefix beam search constrained to the lexicon trie; word LM (bigram / Qwen) applied at word ends with a
    unigram lookahead inside words. 'Other' key mass joins blank as in seqctc.beam_lm."""
    lex = lexicon()
    P = lp.astype(np.float64)
    blank = np.logaddexp(P[:, BLANK], P[:, OTHER])
    beams = {"": (0.0, NINF)}
    info: dict = {}

    def parse(pre):
        r = info.get(pre)
        if r is None:
            k = pre.rfind(" ")
            r = (tuple(pre[:k].split()) if k >= 0 else (), pre[k + 1:])
            info[pre] = r
        return r

    for t in range(len(P)):
        row = P[t]
        cand = [ch for ch in range(1, 28) if row[ch] > c.prune]
        lmv = {}
        if SPACE_S in cand:
            need = [parse(pre) for pre in beams]
            need = [x for x in need if x[1] and x[1] in lex.logp]
            if need:
                lmv = dict(zip(need, scorer.score(need)))
        nb: dict = defaultdict(lambda: [NINF, NINF])
        for pre, (pb, pnb) in beams.items():
            ptot = _lae(pb, pnb)
            e = nb[pre]
            e[0] = _lae(e[0], ptot + blank[t])
            last = pre[-1] if pre else ""
            ws, part = parse(pre)
            for ch in cand:
                s = SYMS[ch]
                if s == last:
                    e[1] = _lae(e[1], pnb + row[ch])
                    base = pb
                else:
                    base = ptot
                if base == NINF:
                    continue
                if s == " ":
                    if not part or part not in lex.logp:
                        continue
                    bonus = c.lm * lmv[(ws, part)] + c.wb - c.la * lex.la[part] + c.beta
                else:
                    la = lex.la.get(part + s)
                    if la is None:
                        continue
                    bonus = c.la * (la - lex.la[part]) + c.beta
                    if c.ac and charlm is not None:
                        bonus += c.ac * charlm(pre)[ch - 1]
                e2 = nb[pre + s]
                e2[1] = _lae(e2[1], base + row[ch] + bonus)
        beams = dict(sorted(nb.items(), key=lambda kv: -_lae(*kv[1]))[:c.beam])
    fin: dict = {}
    need = [parse(pre) for pre in beams]
    need = [x for x in need if x[1] and x[1] in lex.logp]
    lmv = dict(zip(need, scorer.score(need))) if need else {}
    for pre, (pb, pnb) in beams.items():
        ws, part = parse(pre)
        sc = _lae(pb, pnb)
        if part:
            if part not in lex.logp:
                continue
            sc += c.lm * lmv[(ws, part)] + c.wb - c.la * lex.la[part]
            ws = ws + (part,)
        if sc > fin.get(ws, NINF):
            fin[ws] = sc
    if not fin:
        return ""
    cands = list(fin.items())
    eos = scorer.score([(ws, "\n") for ws, _ in cands])
    best = max(zip(cands, eos), key=lambda x: x[0][1] + c.lm * x[1])
    return " ".join(best[0][0])


_SC: dict = {}


def get_scorer(kind: str):
    if kind not in _SC:
        if kind == "uni":
            _SC[kind] = UniScorer()
        elif kind == "bigram":
            _SC[kind] = BigramScorer()
        elif kind == "qwen":
            from phase0.analysis import autocorrect as AC
            p = CACHE / "qwen2.5-0.5b"
            _SC[kind] = AC.NLM(str(p) if p.exists() else "Qwen/Qwen2.5-0.5B")
    return _SC[kind]


_CLM: list = []


def charlm():
    if not _CLM:
        _CLM.append(S.LMScorer())
    return _CLM[0]


def run_decoder(dec: dict, lp: np.ndarray) -> str:
    if dec["kind"] == "greedy":
        return S.greedy(lp)
    if dec["kind"] == "char":
        return S.beam_lm(lp, charlm(), dec["alpha"], dec["beta"], dec.get("beam", 16))
    c = WCfg(**dec["cfg"])
    return word_beam(lp, get_scorer(dec["kind"]), c, charlm() if c.ac else None)


# ============================================================================ keyboard tuning set (final text)
def key_names(sid: str) -> list[str]:
    out = []
    for line in open(SESS / sid / "keys.jsonl"):
        r = json.loads(line)
        if r.get("event") == "down" and S.key_sym(r["key"]) is not None:
            out.append(r["key"])
    return out


def final_text(names) -> str | None:
    buf = []
    for k in names:
        if len(k) == 1 and "a" <= k <= "z":
            buf.append(k)
        elif k in ("space", "enter", "tab"):
            buf.append(" ")
        elif k == "backspace":
            if buf:
                buf.pop()
        else:
            buf.append("#")
    return " ".join("".join(buf).split())


def kbd_tune_items(tag: str, seeds, n: int, minlex: float = 0.0) -> list[tuple[int, str, int, str]]:
    """(seed, fold, window, intended text) for out-of-fold keyboard windows (LOSO folds + held-out 015948) with >=2
    words and no other keys; OOV words kept (decoder weights see the same misses). Never desk."""
    items = []
    for s in LOSO + (HELD,):
        st = load_session(s)
        names = np.array(key_names(s), dtype=object)
        assert len(names) == len(st.kt), s
        for i, (t0, t1, sy) in enumerate(eval_windows(st)):
            idx = np.where((st.kt >= t0) & (st.kt <= t1))[0]
            assert len(idx) == len(sy)
            ft = final_text(names[idx])
            if "#" in ft or len(ft.split()) < 2:
                continue
            if minlex > 0 and np.mean([w in lexicon().logp for w in ft.split()]) < minlex:
                continue
            items.append(("held" if s == HELD else s[9:15], i, ft))
    rng = np.random.default_rng(0)
    items = [items[j] for j in rng.permutation(len(items))][:n]
    return [(seed, f, i, ft) for seed in seeds for f, i, ft in items]


def load_item_lps(tag: str, items):
    cache = {}
    out = []
    for seed, fold, i, ft in items:
        k = (seed, fold)
        if k not in cache:
            cache[k] = S.load_lps(find_file(tag, seed, f"{fold}.npz"))[0]
        out.append((cache[k][i], ft))
    return out


def cer_of(pairs) -> tuple[float, float]:
    return S.rows_cer(pairs), S.rows_wer(pairs)


def cmd_tune(a) -> int:
    """Coordinate search of word-decoder weights on keyboard LOSO windows (final text), objective CER."""
    items = kbd_tune_items(a.tag, [int(x) for x in a.seeds.split(",")], a.n, a.minlex)
    data = load_item_lps(a.tag, items)
    print(f"tune {a.kind}{'@' + a.name if a.name else ''}: {len(data)} keyboard windows (tag {a.tag}, seeds {a.seeds}, "
          f"minlex {a.minlex}, objective {a.objective})", flush=True)
    f = CACHE / f"tune_{a.kind}{'_' + a.name if a.name else ''}.json"
    obj = 0 if a.objective == "cer" else 1
    res = json.loads(f.read_text()) if f.exists() else {"trials": {}}
    ref_char = [(ft, S.beam_lm(lp, charlm(), 0.6, 2.0, 16)) for lp, ft in data]
    res["char_ref"] = dict(zip(("cer", "wer"), cer_of(ref_char)))
    print(f"  char 6-gram beam (0.6, 2.0) on the same windows: CER {res['char_ref']['cer']:.3f} WER {res['char_ref']['wer']:.3f}", flush=True)
    cfg = WCfg(**json.loads(a.start)) if a.start else WCfg()
    grids = {"lm": (0.3, 0.6, 1.0, 1.6), "la": (0.0, 0.3, 0.7), "wb": (-2.0, 0.0, 2.0, 4.0), "beta": (-1.0, 0.0, 1.0),
             "ac": (0.0, 0.3)}
    best = None
    for rnd in range(a.rounds):
        for field in a.fields.split(","):
            scores = {}
            for v in grids[field]:
                cc = replace(cfg, **{field: v})
                key = json.dumps(asdict(cc), sort_keys=True)
                if key not in res["trials"]:
                    t0 = time.time()
                    pairs = [(ft, word_beam(lp, get_scorer(a.kind), cc, charlm() if cc.ac else None)) for lp, ft in data]
                    c_, w_ = cer_of(pairs)
                    res["trials"][key] = {"cer": c_, "wer": w_, "secs": time.time() - t0}
                    f.write_text(json.dumps(res, indent=1))
                scores[v] = res["trials"][key]["cer"] if obj == 0 else res["trials"][key]["wer"] + 1e-3 * res["trials"][key]["cer"]
                print(f"  r{rnd} {field}={v:g} CER {res['trials'][key]['cer']:.3f} WER {res['trials'][key]['wer']:.3f} "
                      f"({res['trials'][key]['secs']:.0f}s)", flush=True)
            cfg = replace(cfg, **{field: min(scores, key=scores.get)})
            best = min(scores.values())
    res["best"] = asdict(cfg)
    res["best_objective"] = {"objective": a.objective, "value": best,
                             **res["trials"][json.dumps(asdict(cfg), sort_keys=True)]}
    res["n_windows"] = len(data)
    f.write_text(json.dumps(res, indent=1))
    print(f"best {a.kind}: {cfg} {a.objective} objective {best:.3f} (CER {res['best_objective']['cer']:.3f}, "
          f"WER {res['best_objective']['wer']:.3f})", flush=True)
    return 0


# ============================================================================ evaluation
def char_params(tag: str, seed: int) -> tuple[float, float]:
    """Char 6-gram alpha/beta tuned per seed on that tag's keyboard LOSO keystroke windows (as seqctc.cmd_decode);
    the baseline seqctc tags reuse their published choice."""
    f = S.RES / f"decode_{tag}.json"
    if f.exists():
        m = json.loads(f.read_text())["meta"].get(f"seed{seed}_lm")
        if m and m[0] is not None:
            return tuple(m)
    cf = CACHE / f"charlm_{tag}.json"
    got = json.loads(cf.read_text()) if cf.exists() else {}
    if str(seed) in got:
        return tuple(got[str(seed)])
    kb = [S.load_lps(p) for s in LOSO if (p := find_file(tag, seed, f"{s[9:15]}.npz")) is not None]
    if len(kb) < len(LOSO):
        return (0.6, 2.0)
    lps, refs = sum([x[0] for x in kb], []), sum([x[1] for x in kb], [])
    grid = [(al, be) for al in (0.3, 0.6, 1.0) for be in (0.0, 1.0, 2.0, 3.0)]
    sc = {g: S.rows_cer(list(zip(refs, [S.beam_lm(lp, charlm(), g[0], g[1], 8) for lp in lps]))) for g in grid}
    got[str(seed)] = min(sc, key=sc.get)
    cf.write_text(json.dumps(got))
    return tuple(got[str(seed)])


def decoders_for(tag: str, seed: int, names) -> dict:
    out = {}
    for nm in names:
        if nm == "greedy":
            out[nm] = {"kind": "greedy"}
        elif nm == "char":
            al, be = char_params(tag, seed)
            out[nm] = {"kind": "char", "alpha": al, "beta": be, "beam": 16}
        else:
            kind, _, var = nm.partition("@")
            t = json.loads((CACHE / f"tune_{kind}{'_' + var if var else ''}.json").read_text())
            out[nm] = {"kind": kind, "cfg": t["best"]}
    return out


def hyp_cache(tag, seed, setname, dec) -> Path:
    h = hashlib.md5(json.dumps(dec, sort_keys=True).encode()).hexdigest()[:10]
    return CACHE / "hyps" / tag / f"seed{seed}" / f"{setname}_{dec['kind']}_{h}.json"


def decode_set(tag, seed, setname, dec, lps):
    f = hyp_cache(tag, seed, setname, dec)
    if f.exists():
        return json.loads(f.read_text())
    hy = [run_decoder(dec, lp) for lp in lps]
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(hy))
    return hy


def aligned_edits(ref_units, hyp_units, owner, n_owner) -> np.ndarray:
    """Levenshtein alignment; each edit charged to the reference unit's phrase (insertions to the next ref unit)."""
    n, m = len(ref_units), len(hyp_units)
    D = np.zeros((n + 1, m + 1), np.int32)
    D[0] = np.arange(m + 1)
    ar = np.arange(m + 1)
    hy = np.array(hyp_units, dtype=object)
    for i in range(1, n + 1):
        cost = (hy != ref_units[i - 1]).astype(np.int32) if m else np.zeros(0, np.int32)
        tmp = np.empty(m + 1, np.int32)
        tmp[0] = i
        if m:
            tmp[1:] = np.minimum(D[i - 1, 1:] + 1, D[i - 1, :-1] + cost)
        D[i] = np.minimum.accumulate(tmp - ar) + ar
    ed = np.zeros(n_owner)
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (hyp_units[j - 1] != ref_units[i - 1]):
            ed[owner[i - 1]] += hyp_units[j - 1] != ref_units[i - 1]
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            ed[owner[i - 1]] += 1
            i -= 1
        else:
            ed[owner[min(i, n - 1)]] += 1
            j -= 1
    return ed


def cont_lp(tag: str, seed: int, setname: str) -> np.ndarray | None:
    desk = load_session(DESK)
    if setname == "desk":
        f = find_file(tag, seed, "desk_cont.npy")
        return None if f is None else np.load(f).astype(np.float32)
    f = find_file(tag, seed, f"{setname}_cont.npy")
    if f is None:
        return None
    L = np.load(f).astype(np.float32)          # [folds, G, NS]
    wins = desk_windows(desk)
    tg = desk.t[::2][:L.shape[1]]
    ph = np.clip(np.searchsorted([w[0] for w in wins], tg, side="right") - 1, 0, len(wins) - 1)
    k = L.shape[0]
    return L[ph % k, np.arange(L.shape[1])]


def segments(lp: np.ndarray, times: np.ndarray, gap: float, pad: float = 0.5, thr: float = 0.5):
    pnb = 1 - np.exp(np.logaddexp(lp[:, BLANK], lp[:, OTHER]))
    em = np.where(pnb > thr)[0]
    if not len(em):
        return []
    groups = np.split(em, np.where(np.diff(times[em]) > gap)[0] + 1)
    return [(int(np.searchsorted(times, times[g[0]] - pad)), int(np.searchsorted(times, times[g[-1]] + pad, side="right")))
            for g in groups]


def stats(ed_runs, n, wed_runs, wn, ref=None):
    ed, wed = np.mean(ed_runs, 0), np.mean(wed_runs, 0)
    c, lo, hi = S.boot_ratio(ed, n)
    w, wlo, whi = S.boot_ratio(wed, wn)
    row = {"cer": c, "lo": lo, "hi": hi, "wer": w, "wer_lo": wlo, "wer_hi": whi, "words_correct_approx": 1 - w,
           "seed_cers": [float(np.sum(r) / n.sum()) for r in ed_runs],
           "seed_wers": [float(np.sum(r) / wn.sum()) for r in wed_runs],
           "ed": ed.tolist(), "wed": wed.tolist(), "n": n.tolist(), "wn": wn.tolist()}
    if ref is not None:
        row["d_cer_vs_ref"] = S.boot_delta(np.array(ref["ed"]), ed, n)
        row["d_wer_vs_ref"] = S.boot_delta(np.array(ref["wed"]), wed, wn)
    return row


def cmd_decode(a) -> int:
    from phase0.analysis.decode import edit_distance
    base = json.loads((S.RES / "decode_hwt.json").read_text())
    refs_pub = {"desk": base["desk/lm"], "deskft": base["deskft_kf5/lm"]}
    desk = load_session(DESK)
    wins = desk_windows(desk)
    phrases = [w[2] for w in wins]
    RES.mkdir(parents=True, exist_ok=True)
    for tag in a.tags.split(","):
        outf = RES / f"decode_{tag}.json"
        out = json.loads(outf.read_text()) if outf.exists() else {}
        seeds = [int(x) for x in a.seeds.split(",")]
        for setname in a.sets.split(","):
            for dname in a.decoders.split(","):
                ed_r, wed_r, cont_r = [], [], defaultdict(lambda: ([], []))
                meta = {}
                for seed in seeds:
                    f = find_file(tag, seed, f"{setname}.npz")
                    if f is None:
                        break
                    lps, refs = S.load_lps(f)
                    dec = decoders_for(tag, seed, [dname])[dname]
                    meta[f"seed{seed}"] = dec
                    hy = decode_set(tag, seed, setname, dec, lps)
                    ed_r.append([edit_distance(r, h) for r, h in zip(refs, hy)])
                    wed_r.append([edit_distance(r.split(), h.split()) for r, h in zip(refs, hy)])
                    if a.cont:
                        L = cont_lp(tag, seed, setname)
                        if L is None:
                            continue
                        times = desk.t[::2][:len(L)]
                        ref_all = " ".join(phrases)
                        owner = sum([[j] * (len(p) + (1 if j < len(phrases) - 1 else 0)) for j, p in enumerate(phrases)], [])
                        wowner = sum([[j] * len(p.split()) for j, p in enumerate(phrases)], [])
                        for gap in (1.0, 2.0):
                            segs = segments(L, times, gap)
                            hs = decode_set(tag, seed, f"{setname}_cont_g{gap}", dec, [L[s0:s1] for s0, s1 in segs])
                            hyp_all = " ".join(h for h in hs if h)
                            ce = aligned_edits(list(ref_all), list(hyp_all), owner, len(phrases))
                            we = aligned_edits(ref_all.split(), hyp_all.split(), wowner, len(phrases))
                            cont_r[gap][0].append(ce)
                            cont_r[gap][1].append(we)
                            meta[f"seed{seed}_cont_g{gap}"] = {"n_segments": len(segs), "hyp_head": hyp_all[:160]}
                if len(ed_r) < len(seeds):
                    print(f"[{tag}] {setname} missing seeds, skipped", flush=True)
                    continue
                n = np.array([len(r) for r in phrases], float)
                wn = np.array([len(r.split()) for r in phrases], float)
                refp = refs_pub["desk" if setname == "desk" else "deskft"] if setname in ("desk", "deskft_kf5", "deskft_kf5r") else None
                row = stats(ed_r, n, wed_r, wn, refp)
                row["meta"] = meta
                row["examples"] = list(zip(phrases[:4], decode_set(tag, seeds[0], setname, decoders_for(tag, seeds[0], [dname])[dname],
                                                                    S.load_lps(find_file(tag, seeds[0], f"{setname}.npz"))[0])[:4]))
                out[f"{setname}/{dname}"] = row
                msg = f"[{tag}] {setname:<14} {dname:<7} CER {row['cer']:.3f} [{row['lo']:.3f},{row['hi']:.3f}] WER {row['wer']:.3f} [{row['wer_lo']:.3f},{row['wer_hi']:.3f}]"
                if "d_cer_vs_ref" in row:
                    msg += f" dCER {row['d_cer_vs_ref'][0]:+.3f} [{row['d_cer_vs_ref'][1]:+.3f},{row['d_cer_vs_ref'][2]:+.3f}]"
                    msg += f" dWER {row['d_wer_vs_ref'][0]:+.3f} [{row['d_wer_vs_ref'][1]:+.3f},{row['d_wer_vs_ref'][2]:+.3f}]"
                print(msg, flush=True)
                for gap, (cer_runs, wer_runs) in cont_r.items():
                    if len(cer_runs) < len(seeds):
                        continue
                    nc = np.array([len(p) + (1 if j < len(phrases) - 1 else 0) for j, p in enumerate(phrases)], float)
                    rc = stats(cer_runs, nc, wer_runs, wn)
                    rc["d_cer_vs_windows"] = S.boot_delta(np.mean(ed_r, 0), np.mean(cer_runs, 0), nc)
                    out[f"{setname}_cont_g{gap}/{dname}"] = rc
                    print(f"[{tag}] {setname} continuous gap {gap}s {dname:<7} CER {rc['cer']:.3f} [{rc['lo']:.3f},{rc['hi']:.3f}] "
                          f"WER {rc['wer']:.3f} [{rc['wer_lo']:.3f},{rc['wer_hi']:.3f}]", flush=True)
                outf.write_text(json.dumps(out, indent=1))
        # keyboard LOSO / held (keystroke refs) with the char decoder for model comparisons
        if a.kbd:
            for dname in ("greedy", "char"):
                for setname in ("kbd_loso", "held"):
                    ed_r, n = [], None
                    for seed in seeds:
                        names = [f"{s[9:15]}.npz" for s in LOSO] if setname == "kbd_loso" else ["held.npz"]
                        fs = [find_file(tag, seed, nm) for nm in names]
                        if any(x is None for x in fs):
                            break
                        lps, refs = [], []
                        for x in fs:
                            l_, r_ = S.load_lps(x)
                            lps += l_
                            refs += r_
                        dec = decoders_for(tag, seed, [dname])[dname]
                        hy = decode_set(tag, seed, setname, dec, lps)
                        ed_r.append([edit_distance(r, h) for r, h in zip(refs, hy)])
                        n = np.array([len(r) for r in refs], float)
                    if len(ed_r) < len(seeds):
                        continue
                    ed = np.mean(ed_r, 0)
                    c, lo, hi = S.boot_ratio(ed, n)
                    row = {"cer": c, "lo": lo, "hi": hi, "ed": ed.tolist(), "n": n.tolist(),
                           "seed_cers": [float(np.sum(r) / n.sum()) for r in ed_r]}
                    k = f"{setname}/{'lm' if dname == 'char' else 'greedy'}"
                    if k in base and len(base[k]["ed"]) == len(ed):
                        row["d_cer_vs_hwt"] = S.boot_delta(np.array(base[k]["ed"]), ed, n)
                    out[f"{setname}/{dname}"] = row
                    print(f"[{tag}] {setname:<14} {dname:<7} CER {c:.3f} [{lo:.3f},{hi:.3f}]"
                          + (f" d_vs_hwt {row['d_cer_vs_hwt'][0]:+.3f} [{row['d_cer_vs_hwt'][1]:+.3f},{row['d_cer_vs_hwt'][2]:+.3f}]"
                             if "d_cer_vs_hwt" in row else ""), flush=True)
            outf.write_text(json.dumps(out, indent=1))
    return 0


def cmd_table(a) -> int:
    """Print every results/seqctc2/decode_<tag>.json row: CER/WER [95% phrase bootstrap], paired deltas vs published."""
    def ci(r, k="cer"):
        lo, hi = ("lo", "hi") if k == "cer" else ("wer_lo", "wer_hi")
        return f"{r[k]:.3f} [{r[lo]:.3f}, {r[hi]:.3f}]" if lo in r else f"{r[k]:.3f}"

    def dl(r, k):
        return f"{r[k][0]:+.3f} [{r[k][1]:+.3f}, {r[k][2]:+.3f}]" if k in r else "-"

    print("| model | set | decoder | CER | WER | dCER vs published | dWER vs published | seeds CER |")
    print("|---|---|---|---|---|---|---|---|")
    for f in sorted(RES.glob("decode_*.json")):
        d = json.loads(f.read_text())
        for k, r in d.items():
            st, dec = k.split("/")
            if "wer" not in r and not a.kbd:
                continue
            print(f"| {f.stem[7:]} | {st} | {dec} | {ci(r)} | {ci(r, 'wer') if 'wer' in r else '-'} | "
                  f"{dl(r, 'd_cer_vs_ref') if 'd_cer_vs_ref' in r else dl(r, 'd_cer_vs_hwt')} | {dl(r, 'd_wer_vs_ref')} | "
                  f"{', '.join(f'{x:.3f}' for x in r.get('seed_cers', []))} |")
    return 0


# ============================================================================ leakage audit
def cmd_audit(a) -> int:
    desk = load_session(DESK)
    phrases = [w[2] for w in desk_windows(desk)]
    rep: dict = {"desk_phrases": len(phrases), "distinct": len(set(phrases))}

    def ngrams(ws, k):
        return {tuple(ws[i:i + k]) for i in range(len(ws) - k + 1)}

    # 1. keyboard training text (keystrokes and intended text) vs desk phrases
    kb_final, kb_keys = [], []
    for s in KBD + (HELD,):
        st = load_session(s)
        kb_keys.append(sym_text(st.ks))
        kb_final.append(final_text([k if k in ("space", "backspace") or (len(k) == 1 and "a" <= k <= "z") else "space"
                                    for k in key_names(s)]) or "")
    kbtxt = " ".join(kb_final + kb_keys)
    kw = kbtxt.split()
    rows = []
    for p in phrases:
        ws = p.split()
        longest = 0
        for k in range(len(ws), 0, -1):
            if ngrams(ws, k) & ngrams(kw, k):
                longest = k
                break
        rows.append({"phrase_idx": phrases.index(p), "exact_in_kbd": p in kbtxt, "longest_shared_word_ngram_kbd": longest,
                     "shared_trigrams": [" ".join(g) for g in ngrams(ws, 3) & ngrams(kw, 3)]})
    rep["kbd_vs_desk"] = rows
    rep["kbd_phrases_exact"] = sum(r["exact_in_kbd"] for r in rows)
    rep["kbd_phrases_sharing_trigram"] = sum(bool(r["shared_trigrams"]) for r in rows)
    # 2. How We Type typed text (Finnish) vs desk phrases
    from phase0.analysis import howwetype as H
    hw = " ".join(sym_text(st.ks) for st in H.streams())
    hww = hw.split()
    rep["hwt_exact"] = sum(p in hw for p in phrases)
    rep["hwt_shared_trigram_phrases"] = sum(bool(ngrams(p.split(), 3) & ngrams(hww, 3)) for p in phrases)
    # 3. LM corpora (char 6-gram, lexicon, bigram, which all come from these) vs desk phrases
    lm_rows = []
    texts = {nm: (CACHE / "lm" / f"{nm}.txt").read_text() for nm in ("tatoeba", "enron", "gutenberg")}
    for j, p in enumerate(phrases):
        ws = p.split()
        r = {"phrase_idx": j, "exact": sum(t.count(" " + p + " ") for t in texts.values())}
        longest = 0
        for k in range(len(ws), 2, -1):
            if any(any((" " + " ".join(g) + " ") in t for t in texts.values()) for g in ngrams(ws, k)):
                longest = k
                break
        r["longest_word_ngram_in_corpora"], r["phrase_words"] = longest, len(ws)
        lm_rows.append(r)
    rep["lm_corpora"] = lm_rows
    rep["lm_exact_total"] = sum(r["exact"] for r in lm_rows)
    # 4. desk folds: phrase-disjoint; text shared between phrases of different folds
    folds = [list(range(20))[k::5] for k in range(5)]
    cross = []
    for i in range(20):
        for j in range(i + 1, 20):
            if i % 5 != j % 5:
                sh = ngrams(phrases[i].split(), 3) & ngrams(phrases[j].split(), 3)
                if sh:
                    cross.append({"i": i, "j": j, "trigrams": [" ".join(g) for g in sh]})
    rep["desk_folds"] = folds
    rep["cross_fold_shared_trigrams"] = cross
    # 5. other desk sessions (never used): their prompts vs the test phrases
    other = {}
    for s in ("20260910-181947-desk", "20260910-201610-desk", "20260910-201720-desk"):
        pr = [json.loads(l)["phrase"] for l in open(SESS / s / "phrases.jsonl") if '"shown"' in l]
        other[s] = {"n": len(pr), "sharing_trigram_with_test": sum(bool(ngrams(q.split(), 3) & ngrams(p.split(), 3))
                                                                  for q in pr for p in phrases)}
    rep["other_desk_sessions_not_used"] = other
    rep["lexicon"] = {"words": len(lexicon().logp), "blocked_removed": lexicon().n_blocked,
                      "desk_oov": sorted({w for p in phrases for w in p.split() if w not in lexicon().logp})}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "audit.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("kbd_vs_desk", "lm_corpora")}, indent=1))
    print("kbd shared trigrams:", [(r["phrase_idx"], r["shared_trigrams"]) for r in rows if r["shared_trigrams"]])
    print("LM longest n-gram per phrase:", [(r["phrase_idx"], r["longest_word_ngram_in_corpora"], r["phrase_words"]) for r in lm_rows])
    return 0


# ============================================================================ CLI
def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("geom")
    s.set_defaults(fn=cmd_geom)
    s = sub.add_parser("diag")
    s.add_argument("--ckpts", default="hwt_s0")
    s.set_defaults(fn=cmd_diag)
    s = sub.add_parser("audit")
    s.set_defaults(fn=cmd_audit)
    for nm, fn in (("pretrain", cmd_pretrain), ("lopo", cmd_lopo)):
        s = sub.add_parser(nm)
        s.add_argument("--tag", required=True)
        s.add_argument("--seeds", default="0,1,2")
        s.add_argument("--frame", default="none", choices=["none", "sim", "aff"])
        s.add_argument("--aug", default="v1", choices=["v1", "v2"])
        s.add_argument("--d", type=int, default=192)
        s.add_argument("--layers", type=int, default=3)
        s.add_argument("--init", default=None)
        s.add_argument("--steps", type=int, default=1500 if nm == "lopo" else 16000)
        s.add_argument("--lr", type=float, default=2e-3)
        s.add_argument("--mix", type=float, default=0.0)
        s.add_argument("--folds", default="021315,131629,164237,all")
        s.set_defaults(fn=fn)
    s = sub.add_parser("table")
    s.add_argument("--kbd", action="store_true")
    s.set_defaults(fn=cmd_table)
    s = sub.add_parser("ssl")
    s.add_argument("--tag", required=True)
    s.add_argument("--frame", default="none", choices=["none", "sim", "aff"])
    s.add_argument("--aug", default="v2", choices=["v1", "v2"])
    s.add_argument("--d", type=int, default=192)
    s.add_argument("--layers", type=int, default=3)
    s.add_argument("--steps", type=int, default=6000)
    s.add_argument("--lr", type=float, default=2e-3)
    s.add_argument("--user-w", type=float, default=0.3)
    s.set_defaults(fn=cmd_ssl)
    s = sub.add_parser("contzs")
    s.add_argument("--tag", required=True)
    s.add_argument("--seeds", default="0,1,2")
    s.set_defaults(fn=cmd_contzs)
    s = sub.add_parser("deskft")
    s.add_argument("--tag", required=True)
    s.add_argument("--name", default="kf5")
    s.add_argument("--seeds", default="0,1,2")
    s.add_argument("--kfold", type=int, default=5)
    s.add_argument("--steps", type=int, default=300)
    s.add_argument("--lr", type=float, default=5e-4)
    s.add_argument("--desk-w", type=float, default=0.5)
    s.add_argument("--mix", type=float, default=0.0)
    s.set_defaults(fn=cmd_deskft)
    s = sub.add_parser("bigram")
    s.set_defaults(fn=cmd_bigram)
    s = sub.add_parser("tune")
    s.add_argument("--kind", required=True, choices=["uni", "bigram", "qwen"])
    s.add_argument("--tag", default="hwt")
    s.add_argument("--seeds", default="0")
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--rounds", type=int, default=1)
    s.add_argument("--fields", default="lm,la,wb,beta,ac")
    s.add_argument("--start", default="")
    s.add_argument("--minlex", type=float, default=0.0)
    s.add_argument("--objective", default="cer", choices=["cer", "wer"])
    s.add_argument("--name", default="")
    s.set_defaults(fn=cmd_tune)
    s = sub.add_parser("decode")
    s.add_argument("--tags", required=True)
    s.add_argument("--seeds", default="0,1,2")
    s.add_argument("--sets", default="desk,deskft_kf5")
    s.add_argument("--decoders", default="greedy,char")
    s.add_argument("--cont", action="store_true")
    s.add_argument("--kbd", action="store_true")
    s.set_defaults(fn=cmd_decode)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
