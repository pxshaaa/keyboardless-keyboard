"""ctc_v3 desk CTC: ensembles/TTA, desk-supervision curve, synthetic personal-vocab motion, noisy student.
Run: PYTHONPATH=. python -m phase0.analysis.ctcv3 <cmd>; CTCV3_CPU=1 forces CPU (mini GPU is reserved)."""

from __future__ import annotations

import functools
import os

import torch

if os.environ.get("CTCV3_CPU"):
    torch.backends.mps.is_available = functools.lru_cache()(lambda: False)

import argparse
import hashlib
import json
import math
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from phase0.analysis import seqctc as S
from phase0.analysis import seqctc2 as S2
from phase0.analysis.seqctc import BLANK, DESK, HELD, KBD, LOSO, NS, OTHER, SPACE_S, Stream

ROOT = Path(".cache/ctc_v3")
SRC = ROOT / "src"
SESS = Path("data/sessions")
BLIND1 = "20260912-174542-desk"
FREE = "20260910-181947-desk"
CORPUS = Path(".cache/personal/corpus_all_projected.txt")
TTA = [(0, 1.0, 1.0), (5, 1.0, 1.0), (-5, 1.0, 1.0), (0, 1.06, 1.0), (0, 0.94, 1.0), (0, 1.0, 0.9), (0, 1.0, 1.1)]


def rdir(tag: str, seed: int) -> Path:
    return ROOT / "runs" / tag / f"seed{seed}"


def stream(sid: str) -> Stream:
    return S2.get_stream(sid, "none")


def recipe(tag: str, a=None) -> dict:
    f = ROOT / "runs" / tag / "recipe.json"
    if f.exists():
        return json.loads(f.read_text())
    r = {"init": a.init, "mix": a.mix, "syn": a.syn, "synsrc": a.synsrc, "steps": a.steps, "lr": a.lr, "aug": a.aug,
         "d": a.d, "layers": a.layers}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(r))
    return r


def mkcfg(r: dict, seed: int, steps: int, lr: float):
    return S2.mkcfg({"aug": r.get("aug", "v1"), "d": r.get("d", 192), "layers": r.get("layers", 3)}, seed, steps, lr)


def load(r: dict, f: Path):
    m = S.build(mkcfg(r, 0, 1, 1e-3))
    m.load_state_dict(torch.load(f, map_location="cpu"))
    return m.eval()


# ============================================================================ labelled desk windows
def blind1_windows():
    """Blind session 1: the decipher segments (hwtmix posteriors, no text) map 1:1 onto the revealed lines."""
    from phase0.analysis.decipher import norm
    st = stream(BLIND1)
    dj = json.loads((SESS / BLIND1 / "decipher.json").read_text())
    lines = [norm(l) for l in (SESS / BLIND1 / "truth.txt").read_text().splitlines() if norm(l)]
    s2l = json.loads((SESS / BLIND1 / "decipher_score.json").read_text())["variants"]["zs/qwen"]["segment_to_lines"]
    assert len(s2l) == len(lines) and all(len(x) == 1 for x in s2l)
    return [(st.t[0] + r["t0"], st.t[0] + r["t1"], lines[x[0]]) for r, x in zip(dj["segments"], s2l)]


def free_windows():
    """181947: 12 s shown->done windows of the first 10 prompts (11-12 are cut short by keyboard typing)."""
    return S.desk_windows(stream(FREE))[:10]


def pseudo_windows(name: str):
    d = json.loads((ROOT / "pseudo" / f"{name}.json").read_text())
    return d["sid"], [(r["t0"], r["t1"], r["text"]) for r in d["keep"]]


# ============================================================================ synthetic motion (technique 3)
class SynthLib:
    """Per-key landmark snippets from streams with key-down times: key i spans mid(prev key, i) .. mid(i, next key);
    indexed by (previous symbol, symbol) so the approach movement matches the bigram."""

    def __init__(self, items, max_iti: float = 0.8):
        self.streams, self.big, self.uni = [], defaultdict(list), defaultdict(list)
        for st, kt, ks in items:
            si = len(self.streams)
            self.streams.append(st)
            for i in range(1, len(kt) - 1):
                c = int(ks[i])
                if c == OTHER or kt[i] - kt[i - 1] > max_iti or kt[i + 1] - kt[i] > max_iti:
                    continue
                a = int(np.searchsorted(st.t, (kt[i - 1] + kt[i]) / 2))
                b = int(np.searchsorted(st.t, (kt[i] + kt[i + 1]) / 2))
                if b - a < 3 or st.M[a:b].all(1).mean() < 0.5:
                    continue
                self.big[(int(ks[i - 1]), c)].append((si, a, b))
                self.uni[c].append((si, a, b))
        self.n = sum(len(v) for v in self.uni.values())

    def make(self, syms, rng, margin: int = 4, drift: float = 0.05):
        segs = []
        prev = SPACE_S
        for c in syms:
            L = self.big.get((prev, int(c)))
            if not L or rng.random() < 0.15:
                L = self.uni.get(int(c))
            prev = int(c)
            if not L:
                continue
            segs.append(L[rng.integers(len(L))])
        if not segs:
            return None
        parts = []
        for k, (si, a, b) in enumerate(segs):
            st = self.streams[si]
            lo = a - margin - (int(rng.uniform(12, 30)) if k == 0 else 0)
            hi = b + margin + (int(rng.uniform(12, 30)) if k == len(segs) - 1 else 0)
            lo, hi = max(0, lo), min(len(st.t), hi)
            parts.append((st.A[lo:hi].astype(np.float32), st.M[lo:hi]))
        A, M = parts[0]
        ov = 2 * margin
        for A2, M2 in parts[1:]:
            o = min(ov, len(A), len(A2))
            w = np.linspace(0, 1, o, dtype=np.float32)[:, None]
            m1, m2 = M[-o:].astype(np.float32) * (1 - w), M2[:o].astype(np.float32) * w
            den = np.maximum(m1 + m2, 1e-6)
            blend = (A[-o:] * m1[..., None, None] + A2[:o] * m2[..., None, None]) / den[..., None, None]
            A = np.concatenate([A[:-o], blend, A2[o:]])
            M = np.concatenate([M[:-o], M[-o:] | M2[:o], M2[o:]])
        n = len(A)
        if drift > 0 and n > 4:
            knots = max(2, int(n / 60) + 2)
            walk = np.cumsum(rng.normal(0, drift, (knots, 2, 2)), 0)
            x = np.linspace(0, knots - 1, n)
            dr = np.stack([np.stack([np.interp(x, np.arange(knots), walk[:, h, d]) for d in range(2)], -1)
                           for h in range(2)], 1).astype(np.float32)
            A = A.copy()
            A[..., :2] += dr[:, :, None, :] * M[..., None, None]
        return Stream("syn", np.arange(n) / 60.0, A, M, np.zeros(0), np.zeros(0, np.int64))


_CORP: list = []


def corpus() -> list[str]:
    if not _CORP:
        ok = set("abcdefghijklmnopqrstuvwxyz ")
        for l in CORPUS.read_text().splitlines():
            l = " ".join("".join(c for c in l if c in ok).split())
            if len(l) >= 8:
                _CORP.append(l)
    return _CORP


class SynthSource:
    def __init__(self, lib: SynthLib):
        self.lib, self.lines = lib, corpus()

    def draw(self, rng):
        for _ in range(20):
            ws = self.lines[rng.integers(len(self.lines))].split()
            s = int(rng.integers(len(ws)))
            target, out = rng.uniform(8, 40), []
            for w in ws[s:]:
                out.append(w)
                if len(" ".join(out)) >= target:
                    break
            sy = S.text_syms(" ".join(out))
            st = self.lib.make(sy, rng)
            if st is not None and len(st.t) > 20:
                return st, st.t[0], st.t[-1] + 1e-3, sy
        raise RuntimeError("synth failed")


def kbd_lib(sessions) -> SynthLib:
    items = []
    for s in sessions:
        st = stream(s)
        items.append((st, st.kt, st.ks))
    return SynthLib(items)


def ctc_align(lp: np.ndarray, sy) -> np.ndarray | None:
    """Viterbi CTC forced alignment ('other' joins blank) -> 30 Hz frame of each label's peak posterior."""
    T, L = len(lp), len(sy)
    if L == 0:
        return None
    ext = np.full(2 * L + 1, BLANK)
    ext[1::2] = sy
    Sn = len(ext)
    em = lp[:, ext].astype(np.float64)
    em[:, ext == BLANK] = np.logaddexp(lp[:, BLANK], lp[:, OTHER]).astype(np.float64)[:, None]
    skip = np.zeros(Sn, bool)
    skip[2:] = (ext[2:] != BLANK) & (ext[2:] != ext[:-2])
    D = np.full(Sn, -np.inf)
    D[0], D[1] = em[0, 0], em[0, 1]
    bp = np.zeros((T, Sn), np.int8)
    ar = np.arange(Sn)
    for t in range(1, T):
        c = np.stack([D, np.r_[-np.inf, D[:-1]], np.where(skip, np.r_[-np.inf, -np.inf, D[:-2]], -np.inf)])
        k = c.argmax(0)
        D = c[k, ar] + em[t]
        bp[t] = k
    end = Sn - 1 if D[Sn - 1] >= D[Sn - 2] else Sn - 2
    if not np.isfinite(D[end]):
        return None
    path, s = np.empty(T, int), end
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= bp[t, s]
    frames = np.zeros(L, int)
    for k in range(L):
        f = np.where(path == 2 * k + 1)[0]
        if not len(f):
            return None
        frames[k] = f[np.argmax(lp[f, sy[k]])]
    return frames


def desk_lib(model, groups) -> SynthLib:
    """Desk-domain snippet library: key times of labelled desk windows from forced alignment with `model`."""
    items = []
    for st, wins in groups:
        kt, ks = [], []
        for (t0, t1, txt), lp in zip(wins, infer_win(model, st, wins)):
            sy = S.text_syms(txt)
            fr = ctc_align(lp, sy)
            if fr is None:
                continue
            idx = np.minimum(int(np.searchsorted(st.t, t0)) + 2 * fr + 1, len(st.t) - 1)
            kt += st.t[idx].tolist()
            ks += list(sy)
        o = np.argsort(kt, kind="stable")
        items.append((st, np.array(kt)[o], np.array(ks, np.int64)[o]))
    return SynthLib(items)


# ============================================================================ training sources
_HW: list = []


def hwt():
    if not _HW:
        _HW.append(S2.hwt_streams("none"))
    return _HW[0]


def kbd_sources(r: dict, sessions, extra_syn=None):
    src = S2.kbd_srcs("none", sessions, r["mix"], hwt() if r["mix"] > 0 else None)
    if r.get("syn", 0) > 0:
        lib = kbd_lib(sessions) if r.get("synsrc", "kbd") == "kbd" else extra_syn
        tot = sum(w for _, w in src)
        src.append((SynthSource(lib), r["syn"] / (1 - r["syn"]) * tot))
        print(f"  synth library: {lib.n} snippets, {len(lib.big)} bigrams", flush=True)
    return src


# ============================================================================ inference (+ TTA)
def tf_arrays(A, M, rot, sc, speed):
    if speed != 1.0 and len(A) > 4:
        n = max(8, int(round(len(A) / speed)))
        x = np.linspace(0, len(A) - 1, n)
        lo = np.floor(x).astype(int)
        hi = np.minimum(lo + 1, len(A) - 1)
        w = (x - lo)[:, None, None, None].astype(np.float32)
        A, M = A[lo] * (1 - w) + A[hi] * w, M[lo] & M[hi]
    if rot or sc != 1.0:
        th = math.radians(rot)
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], np.float32)
        A = A.copy()
        A[..., :2] = A[..., :2] @ R.T * sc
        A[..., 2] *= sc
    return A.astype(np.float32), M


def resample_lp(lp: np.ndarray, G: int) -> np.ndarray:
    if len(lp) == G:
        return lp
    x = np.linspace(0, len(lp) - 1, G)
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, len(lp) - 1)
    w = (x - lo)[:, None]
    p = np.exp(lp[lo].astype(np.float64)) * (1 - w) + np.exp(lp[hi].astype(np.float64)) * w
    return np.log(p / p.sum(1, keepdims=True) + 1e-12).astype(np.float32)


def logmean(L) -> np.ndarray:
    lp = np.mean([x.astype(np.float64) for x in L], 0)
    return (lp - np.logaddexp.reduce(lp, axis=-1, keepdims=True)).astype(np.float32)


@torch.no_grad()
def infer_win(model, st, wins, tta=((0, 1.0, 1.0),), bs=16):
    dev = S.device()
    model.to(dev).eval()
    per = []
    for rot, sc, sp in tta:
        out = []
        for i in range(0, len(wins), bs):
            raw = [S.crop(st, w[0], w[1]) for w in wins[i:i + bs]]
            items = [tf_arrays(A, M, rot, sc, sp) for A, M in raw]
            A, M, lens = S.pad_batch(items, torch.device("cpu"))
            lp = model(S.featurize(A, M, model.zfeat).to(dev), lens).float().cpu().numpy()
            out += [resample_lp(lp[b, :model.out_len(n)], model.out_len(len(raw[b][0]))) for b, n in enumerate(lens)]
        per.append(out)
    return [logmean([p[j] for p in per]) if len(per) > 1 else per[0][j] for j in range(len(wins))]


@torch.no_grad()
def infer_cont(model, st, tta=((0, 1.0, 1.0),)):
    G = (len(st.t) + 1) // 2
    L = []
    for rot, sc, sp in tta:
        A, M = tf_arrays(st.A, st.M, rot, sc, sp)
        t = np.linspace(st.t[0], st.t[-1], len(A))
        L.append(resample_lp(S2.infer_cont(model, Stream(st.name, t, A, M, st.kt, st.ks)), G))
    return logmean(L) if len(L) > 1 else L[0]


def save_zs_products(m, d: Path, suffix: str = "", tta=((0, 1.0, 1.0),)):
    desk = stream(DESK)
    if not (d / f"zs_desk_win{suffix}.npz").exists():
        w = S.desk_windows(desk)
        S.save_lps(d / f"zs_desk_win{suffix}.npz", infer_win(m, desk, w, tta), [x[2] for x in w])
    for key, sid in (("desk", DESK), ("blind1", BLIND1), ("free", FREE)):
        f = d / f"zs_{key}_cont{suffix}.npy"
        if not f.exists():
            np.save(f, infer_cont(m, stream(sid), tta).astype(np.float16))
    if not (d / f"held{suffix}.npz").exists():
        h = stream(HELD)
        w = S.eval_windows(h)
        S.save_lps(d / f"held{suffix}.npz", infer_win(m, h, w, tta), [S.sym_text(x[2]) for x in w])


# ============================================================================ commands: training
def cmd_zs(a) -> int:
    r = recipe(a.tag, a)
    for seed in map(int, a.seeds.split(",")):
        d = rdir(a.tag, seed)
        d.mkdir(parents=True, exist_ok=True)
        f = d / "all.pt"
        if not f.exists():
            src_ck = SRC / "hwtmix" / f"seed{seed}" / "all.pt"
            if a.tag == "hwtmix" and src_ck.exists():
                shutil.copy(src_ck, f)
            else:
                cfg = mkcfg(r, seed, r["steps"], r["lr"])
                m = S.build(cfg)
                if r["init"]:
                    m.load_state_dict(torch.load(S2.init_path(r["init"], seed), map_location="cpu"))
                t0 = time.time()
                S.train(m, kbd_sources(r, KBD), cfg, log=f"[zs {a.tag} s{seed}]", ckpt=d / "all.ckpt")
                torch.save(m.state_dict(), f)
                (d / "all.json").write_text(json.dumps({**r, "seed": seed, "secs": time.time() - t0}))
        save_zs_products(load(r, f), d)
        print(f"[zs {a.tag} s{seed}] done", flush=True)
    return 0


def cmd_loso(a) -> int:
    r = recipe(a.tag)
    for seed in map(int, a.seeds.split(",")):
        d = rdir(a.tag, seed)
        for fold in a.folds.split(","):
            out = d / f"loso_{fold}.npz"
            if out.exists():
                continue
            sid = next(s for s in LOSO if fold in s)
            cfg = mkcfg(r, seed, r["steps"], r["lr"])
            m = S.build(cfg)
            if r["init"]:
                m.load_state_dict(torch.load(S2.init_path(r["init"], seed), map_location="cpu"))
            t0 = time.time()
            S.train(m, kbd_sources(r, [s for s in KBD if fold not in s]), cfg, log=f"[loso {a.tag} s{seed} {fold}]",
                    ckpt=d / f"loso_{fold}.ckpt")
            st = stream(sid)
            w = S.eval_windows(st)
            d.mkdir(parents=True, exist_ok=True)
            S.save_lps(out, infer_win(m, st, w), [S.sym_text(x[2]) for x in w])
            torch.save(m.state_dict(), d / f"loso_{fold}.pt")
            print(f"[loso {a.tag} s{seed} {fold}] {time.time() - t0:.0f}s", flush=True)
    return 0


def desk_sources(r: dict, a, train_wins, model=None, held_wins=None):
    """Desk-domain phrase windows (old session train folds + extras) at weight desk_w, keyboard (+HWT mix) the rest."""
    groups = [(stream(DESK), train_wins)] if train_wins else []
    ex = [e for e in a.extra.split(",") if e]
    if "blind1" in ex:
        groups.append((stream(BLIND1), blind1_windows()))
    if "prompt" in ex:
        groups.append((stream(FREE), free_windows()))
    for e in ex:
        if e.startswith("pseudo:"):
            sid, w = pseudo_windows(e[7:])
            groups.append((stream(sid), w))
        elif e.startswith("pseudoheld:"):  # only pseudo windows over this fold's held-out phrases
            sid, w = pseudo_windows(e[11:])
            if held_wins is not None:
                w = [x for x in w if any(min(x[1], h[1]) - max(x[0], h[0]) > 0.5 * (x[1] - x[0]) for h in held_wins)]
            groups.append((stream(sid), w))
    ntot = sum(len(w) for _, w in groups)
    dsyn = a.ftsyn if a.ftsynsrc == "desk" else 0.0
    src = [(S.PhraseSource(st, w), a.desk_w * (1 - dsyn) * len(w) / ntot) for st, w in groups if w]
    if dsyn > 0:
        lib = desk_lib(model, [(st, w) for st, w in groups if w])
        src.append((SynthSource(lib), a.desk_w * dsyn))
        print(f"  desk synth library: {lib.n} snippets, {len(lib.big)} bigrams", flush=True)
    fr = dict(r, mix=a.ftmix, syn=a.ftsyn - dsyn, synsrc="kbd")
    kb = kbd_sources(fr, KBD)
    tot = sum(w for _, w in kb)
    return src + [(s, w / tot * (1 - a.desk_w)) for s, w in kb]


def cmd_ft(a) -> int:
    """5-fold phrase CV on the old desk session, starting from <tag>/all.pt."""
    r = recipe(a.tag)
    desk = stream(DESK)
    wins = S.desk_windows(desk)
    n = len(wins)
    folds = [list(range(n))[k::5] for k in range(5)]
    for seed in map(int, a.seeds.split(",")):
        d = rdir(a.tag, seed) / f"ft_{a.name}"
        if (d / "desk_win.npz").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        lps, conts, t0 = [None] * n, [], time.time()
        for k, test in enumerate(folds):
            fm = d / f"fold{k}.pt"
            cfg = mkcfg(r, seed * 1000 + test[0], a.steps, a.lr)
            if fm.exists():
                m = load(r, fm)
            else:
                m = S.build(cfg)
                m.load_state_dict(torch.load(rdir(a.tag, seed) / "all.pt", map_location="cpu"))
                tr = [j for j in range(n) if j not in test]
                if a.ntrain < len(tr):
                    tr = sorted(np.random.default_rng(seed * 100 + k).permutation(tr)[:a.ntrain].tolist())
                S.train(m, desk_sources(r, a, [wins[j] for j in tr], m, [wins[j] for j in test]), cfg)
                torch.save(m.state_dict(), fm)
            for j, lp in zip(test, infer_win(m, desk, [wins[j] for j in test])):
                lps[j] = lp
            conts.append(infer_cont(m, desk).astype(np.float16))
            print(f"[ft {a.tag}/{a.name} s{seed}] fold {k} {time.time() - t0:.0f}s", flush=True)
        np.save(d / "desk_cont.npy", np.stack(conts))
        S.save_lps(d / "desk_win.npz", lps, [w[2] for w in wins])
        (d / "meta.json").write_text(json.dumps({**vars(a), "fn": None, "secs": time.time() - t0}, default=str))
    return 0


def cmd_final(a) -> int:
    """Deployment fine-tune on all 20 old desk phrases (+extras); blind-1/free posteriors when not trained on them."""
    r = recipe(a.tag)
    desk = stream(DESK)
    wins = S.desk_windows(desk)
    for seed in map(int, a.seeds.split(",")):
        d = rdir(a.tag, seed) / f"final_{a.name}"
        d.mkdir(parents=True, exist_ok=True)
        f = d / "model.pt"
        if not f.exists():
            cfg = mkcfg(r, seed * 1000, a.steps, a.lr)
            m = S.build(cfg)
            m.load_state_dict(torch.load(rdir(a.tag, seed) / "all.pt", map_location="cpu"))
            t0 = time.time()
            S.train(m, desk_sources(r, a, wins if a.ntrain > 0 else [], m), cfg, log=f"[final {a.tag}/{a.name} s{seed}]")
            torch.save(m.state_dict(), f)
            (d / "meta.json").write_text(json.dumps({**vars(a), "fn": None, "secs": time.time() - t0}, default=str))
        m = load(r, f)
        for key, sid in (("blind1", BLIND1), ("free", FREE), ("desk", DESK)):
            if not (d / f"{key}_cont.npy").exists():
                np.save(d / f"{key}_cont.npy", infer_cont(m, stream(sid)).astype(np.float16))
        if not (d / "desk_win.npz").exists():  # in-sample unless ntrain == 0 (meta.json says which)
            S.save_lps(d / "desk_win.npz", infer_win(m, desk, wins), [w[2] for w in wins])
        for sid in filter(None, a.infer_sids.split(",")):
            if not (d / f"{sid}_cont.npy").exists():
                np.save(d / f"{sid}_cont.npy", infer_cont(m, stream(sid)).astype(np.float16))
        print(f"[final {a.tag}/{a.name} s{seed}] done", flush=True)
    return 0


def cmd_tta(a) -> int:
    """TTA posteriors (7 geometric/time transforms, log-mean) for zs models and ft fold models."""
    r = recipe(a.tag)
    desk = stream(DESK)
    wins = S.desk_windows(desk)
    for seed in map(int, a.seeds.split(",")):
        d = rdir(a.tag, seed)
        for kind in a.kinds.split(","):
            if kind == "zs":
                save_zs_products(load(r, d / "all.pt"), d, "_tta", TTA)
            elif kind.startswith("ft_"):
                fd = d / kind
                if (fd / "desk_win_tta.npz").exists():
                    continue
                lps, conts = [None] * 20, []
                for k in range(5):
                    m = load(r, fd / f"fold{k}.pt")
                    test = list(range(20))[k::5]
                    for j, lp in zip(test, infer_win(m, desk, [wins[j] for j in test], TTA)):
                        lps[j] = lp
                    conts.append(infer_cont(m, desk, TTA).astype(np.float16))
                np.save(fd / "desk_cont_tta.npy", np.stack(conts))
                S.save_lps(fd / "desk_win_tta.npz", lps, [w[2] for w in wins])
            elif kind.startswith("final_"):
                fd = d / kind
                m = load(r, fd / "model.pt")
                for key, sid in (("blind1", BLIND1), ("free", FREE)):
                    if not (fd / f"{key}_cont_tta.npy").exists():
                        np.save(fd / f"{key}_cont_tta.npy", infer_cont(m, stream(sid), TTA).astype(np.float16))
            print(f"[tta {a.tag} s{seed} {kind}] done", flush=True)
    return 0


# ============================================================================ CLI
def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("zs")
    s.add_argument("--tag", required=True)
    s.add_argument("--seeds", default="0,1,2")
    s.add_argument("--init", default="hwt_s0")
    s.add_argument("--mix", type=float, default=0.3)
    s.add_argument("--syn", type=float, default=0.0)
    s.add_argument("--synsrc", default="kbd")
    s.add_argument("--steps", type=int, default=1500)
    s.add_argument("--lr", type=float, default=2e-3)
    s.add_argument("--aug", default="v1")
    s.add_argument("--d", type=int, default=192)
    s.add_argument("--layers", type=int, default=3)
    s.set_defaults(fn=cmd_zs)
    s = sub.add_parser("loso")
    s.add_argument("--tag", required=True)
    s.add_argument("--seeds", default="0")
    s.add_argument("--folds", default="021315,131629,164237")
    s.set_defaults(fn=cmd_loso)
    for nm, fn in (("ft", cmd_ft), ("final", cmd_final)):
        s = sub.add_parser(nm)
        s.add_argument("--tag", required=True)
        s.add_argument("--name", required=True)
        s.add_argument("--seeds", default="0,1,2")
        s.add_argument("--steps", type=int, default=300)
        s.add_argument("--lr", type=float, default=5e-4)
        s.add_argument("--desk-w", type=float, default=0.5)
        s.add_argument("--ftmix", type=float, default=0.3)
        s.add_argument("--ftsyn", type=float, default=0.0)
        s.add_argument("--ftsynsrc", default="kbd", choices=["kbd", "desk"])
        s.add_argument("--ntrain", type=int, default=99)
        s.add_argument("--extra", default="")
        s.add_argument("--infer-sids", default="")
        s.set_defaults(fn=fn)
    s = sub.add_parser("tta")
    s.add_argument("--tag", required=True)
    s.add_argument("--seeds", default="0,1,2")
    s.add_argument("--kinds", default="zs")
    s.set_defaults(fn=cmd_tta)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
