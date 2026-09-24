"""Idea 4: taps_seq TCN fine-tuned with a desk tap-count loss (5 phrase folds).
Run: python -m phase0.analysis.taps_seq_count run --seed S [--lam L]"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace

import numpy as np

import torch  # before lightgbm here: this process trains no LightGBM, and lightgbm-first segfaults torch's 4-thread ops on the mini
import lightgbm  # noqa: F401
import torch.nn.functional as F

from phase0.analysis import taps_seq as ts
from phase0.analysis import taps_gb as gb
from phase0.analysis import touch_common as tc

N_FOLDS = 5
torch.set_num_threads(4)


_S: dict = {}


def sessions(cfg: ts.Cfg):
    key = cfg.groups
    if key not in _S:
        kb = [ts.Sess(tc.spath(s), cfg.groups) for s in tc.KBD_TRAIN4]
        te = ts.Sess(tc.spath(tc.KBD_TEST), cfg.groups)
        fr, t, P = gb.load_frames(tc.spath(tc.DESK))
        g = gb.build_groups(P)
        X = gb.assemble(g, cfg.groups)
        d = type("S", (), {})()
        d.finite = np.isfinite(X).all(1).astype(np.float32)
        d.X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        d.t, d.frames, d.P, d.kt = t, fr, P, np.array([])
        _S[key] = (kb, te, d)
    return _S[key]


def train(kb, desk, phrases, cfg: ts.Cfg, seed: int, lam: float, steps: int, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    norm = ts.Norm(kb)
    Xn = [ts.model_input(norm, s) for s in kb]
    Y = [ts.gauss_target(s.t, s.kt, cfg.sigma_ms / 1000.0) for s in kb]
    Yg = [gb.labels(s.t, s.kt, gb.TYPING_HALF_WIDTH_S).astype(np.float32) for s in kb]
    d_hand = kb[0].X.shape[1] // 2
    model = ts.TCN(Xn[0].shape[1], cfg, ts.n_channels(cfg))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=steps)
    w = np.array([len(x) for x in Xn], float); w /= w.sum()
    Xd = ts.model_input(norm, desk)
    W = tc.windows()
    win = [(np.searchsorted(desk.t, W[j]["t0"]), np.searchsorted(desk.t, W[j]["t1"]), W[j]["n"])
           for j in phrases]
    mass = None
    model.train()
    for step in range(steps):
        xb, yb, gbt = [], [], []
        for _ in range(cfg.batch):
            k = int(rng.choice(len(kb), p=w))
            lo, hi = ts.sample_crop(rng, Xn[k], cfg.crop)
            xb.append(torch.from_numpy(ts._augment(rng, Xn[k][lo:hi].T.copy(), cfg, d_hand)))
            yb.append(torch.from_numpy(Y[k][lo:hi].copy()))
            gbt.append(torch.from_numpy(Yg[k][lo:hi].copy()))
        raw = model(torch.stack(xb))
        yy = torch.stack(yb).unsqueeze(1)
        loss = ts.focal_heatmap(raw[:, :1], yy)
        loss = loss + cfg.gate_w * F.binary_cross_entropy_with_logits(raw[:, 1], torch.stack(gbt))
        with torch.no_grad():  # heatmap mass per true tap on keyboard crops
            npk = float((yy >= 0.999).sum().clamp(min=1))
            m_now = float(torch.sigmoid(raw[:, :1]).sum()) / npk
            mass = m_now if mass is None else 0.95 * mass + 0.05 * m_now
        if lam > 0 and win and step >= steps // 4:
            pad = 60
            cl = []
            for i in rng.choice(len(win), size=min(4, len(win)), replace=False):
                a, b, n = win[i]
                x = torch.from_numpy(Xd[max(0, a - pad):b + pad].T.copy()).unsqueeze(0)
                r = model(x)
                p = torch.sigmoid(r[0, 0, (a - max(0, a - pad)):(a - max(0, a - pad)) + (b - a)])
                cnt = p.sum() / max(mass, 1e-3)
                cl.append(((cnt - n) / n) ** 2)
            loss = loss + lam * torch.stack(cl).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if verbose and (step + 1) % 300 == 0:
            print(f"      step {step+1} loss={loss.item():.4f} mass={mass:.2f}", flush=True)
    model.eval()
    return {"model": model, "norm": norm, "cfg": cfg, "mass": mass}


def kbd_cfg(kb, cfg, seed, steps):
    """event params from leave-one-session-out OOF scores of the plain TCN."""
    tt, kk, pp, gg, off = [], [], [], [], 0.0
    for te in range(len(kb)):
        tr = [s for i, s in enumerate(kb) if i != te]
        bundle = train(tr, _dummy_desk(kb[0]), [], cfg, seed, 0.0, steps)
        p, g = ts.frame_scores(bundle, kb[te])
        t = kb[te].t
        tt.append(t - t[0] + off); kk.append(kb[te].kt - t[0] + off); pp.append(p); gg.append(g)
        off = tt[-1][-1] + 100.0
    T, K, Pp, Gg = map(np.concatenate, (tt, kk, pp, gg))
    return gb.tune_events(Pp, T, K, np.ones(len(T), bool), Gg)


def _dummy_desk(s):
    d = type("S", (), {})()
    d.X, d.finite, d.t = s.X[:10], s.finite[:10], s.t[:10]
    return d


def run(seed: int, lam: float, steps: int = 900) -> dict:
    cfg = replace(ts.Cfg(), steps=steps)
    kb, te, desk = sessions(cfg)
    arm = f"tcn_count{lam:g}" if lam > 0 else "tcn"
    ckf = tc.CACHE / f"tcn_kbdcfg_s{seed}.json"
    if ckf.exists():
        cfg_ev = json.loads(ckf.read_text())
    else:
        t0 = time.time()
        cfg_ev, f1 = kbd_cfg(kb, cfg, seed, steps)
        ckf.write_text(json.dumps(cfg_ev))
        print(f"  seed{seed} TCN OOF cfg {cfg_ev} F1={100*f1:.1f} ({time.time()-t0:.0f}s)", flush=True)
    W = tc.windows()
    p_desk = np.zeros(len(desk.t)); g_desk = np.zeros(len(desk.t))
    kres = None
    folds = range(N_FOLDS) if lam > 0 else [None]
    for f in folds:
        t0 = time.time()
        tr = [j for j in range(len(W)) if f is None or j % N_FOLDS != f]
        b = train(kb, desk, tr if lam > 0 else [], cfg, seed, lam, steps, verbose=False)
        p, g = ts.frame_scores(b, desk)
        if f is None:
            p_desk, g_desk = p, g
        else:
            for j in range(len(W)):
                if j % N_FOLDS == f:
                    m = (desk.t >= W[j]["t0"]) & (desk.t < W[j]["t1"])
                    p_desk[m], g_desk[m] = p[m], g[m]
            if f == 0:  # outside-window frames and the keyboard check use fold 0's model
                out = np.ones(len(desk.t), bool)
                for w_ in W:
                    out &= ~((desk.t >= w_["t0"]) & (desk.t < w_["t1"]))
                p_desk[out], g_desk[out] = p[out], g[out]
        if kres is None:
            pk, gk = ts.frame_scores(b, te)
            kres = tc.score_events(tc.KBD_TEST, pk, gk, cfg_ev)
        print(f"    [{arm} s{seed}] fold {f} ({time.time()-t0:.0f}s) mass={b['mass']:.2f}", flush=True)
    kres["cfg"] = cfg_ev
    (tc.CACHE / f"kbd_{arm}_s{seed}.json").write_text(json.dumps(kres, default=float))
    np.savez(tc.CACHE / f"desk_probs_{arm}_s{seed}.npz", p=p_desk, gate=g_desk)
    return {"arm": arm, "kbd": kres}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=900)
    a = ap.parse_args(argv)
    r = run(a.seed, a.lam, a.steps)
    print(json.dumps(r, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
