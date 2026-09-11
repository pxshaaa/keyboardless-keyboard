"""Sequence keypress detector: a small TCN over landmark features with heatmap / CTC / cls heads.
Run: python -m phase0.analysis.taps_seq {train --sessions A B C | predict SESSION | experiment}"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase0.analysis.detect_taps import FINGERTIP_JOINTS
from phase0.analysis.eval_taps import keydown_times, score
from phase0.analysis.taps_gb import (
    TYPING_HALF_WIDTH_S,
    apply_events,
    assemble,
    build_groups,
    flex_velocity,
    labels,
    load_frames,
    tune_events,
)

DEFAULT_MODEL = Path("models/taps_seq.pt")
FPS = 60.0
# deriv/ctx are hand-built temporal context; the TCN is supposed to learn that itself
DEFAULT_GROUPS = ("pose", "depth", "wflex", "palm", "rest", "still")
HEADS = ("heatmap", "ctc", "cls")


@dataclass
class Cfg:
    groups: tuple[str, ...] = DEFAULT_GROUPS
    head: str = "heatmap"
    width: int = 32
    blocks: int = 4
    kernel: int = 5
    dropout: float = 0.15
    crop: int = 512
    batch: int = 16
    steps: int = 900
    lr: float = 3e-3
    wd: float = 1e-4
    sigma_ms: float = 30.0
    noise: float = 0.05
    hand_drop: float = 0.05
    gate: bool = True
    gate_w: float = 0.5


# ---------------------------------------------------------------- data
class Sess:
    """One session's per-frame features, times and keydowns."""

    def __init__(self, path: Path, groups: tuple[str, ...]):
        self.path = Path(path)
        frames, t, P = load_frames(self.path)
        g = build_groups(P)
        X = assemble(g, groups)
        self.finite = np.isfinite(X).all(1).astype(np.float32)
        self.X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self.t = t
        self.frames = frames
        self.P = P
        self.kt = keydown_times(self.path)


def load_sessions(paths, groups) -> list[Sess]:
    return [Sess(p, groups) for p in paths]


class Norm:
    def __init__(self, sessions: list[Sess]):
        A = np.vstack([s.X for s in sessions])
        self.mu = A.mean(0)
        self.sd = A.std(0)
        self.sd[self.sd < 1e-6] = 1.0

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.mu) / self.sd, -8.0, 8.0).astype(np.float32)


def model_input(norm: "Norm", s) -> np.ndarray:
    """-> [F,D+1]; the trailing flag tells the net a frame's landmarks were missing, not zero."""
    return np.hstack([norm(s.X), getattr(s, "finite", np.ones(len(s.X), np.float32))[:, None]])


def gauss_target(t: np.ndarray, kt: np.ndarray, sigma_s: float) -> np.ndarray:
    if len(kt) == 0:
        return np.zeros(len(t), np.float32)
    d = np.abs(t[:, None] - kt[None, :]).min(1)
    return np.exp(-0.5 * (d / sigma_s) ** 2).astype(np.float32)


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, c: int, k: int, dil: int, p: float):
        super().__init__()
        self.conv = nn.Conv1d(c, c, k, padding=dil * (k - 1) // 2, dilation=dil)
        self.norm = nn.GroupNorm(1, c)
        self.pw = nn.Conv1d(c, c, 1)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h = self.drop(F.gelu(self.norm(self.conv(x))))
        return x + self.pw(h)


class TCN(nn.Module):
    """Dilated temporal CNN; n_out=1 for heatmap/cls scores, 2 for CTC (blank + key)."""

    def __init__(self, d_in: int, cfg: Cfg, n_out: int):
        super().__init__()
        c = cfg.width
        self.inp = nn.Conv1d(d_in, c, 1)
        self.blocks = nn.ModuleList(
            [Block(c, cfg.kernel, 2**i, cfg.dropout) for i in range(cfg.blocks)]
        )
        self.out = nn.Conv1d(c, n_out, 1)
        if n_out == 1:
            nn.init.constant_(self.out.bias, -4.0)  # rare-positive prior

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.out(h)


def receptive_field(cfg: Cfg) -> int:
    return 1 + sum((cfg.kernel - 1) * 2**i for i in range(cfg.blocks))


# ---------------------------------------------------------------- losses
def focal_heatmap(logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """CenterNet penalty-reduced focal loss on a Gaussian target."""
    p = torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)
    pos = y >= 0.999
    lpos = -((1 - p) ** 2) * torch.log(p) * pos
    lneg = -((1 - y) ** 4) * (p**2) * torch.log(1 - p) * (~pos)
    n = pos.sum().clamp(min=1.0)
    return (lpos.sum() + lneg.sum()) / n


def bce_cls(logit: torch.Tensor, y: torch.Tensor, pos_weight: float) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logit, y, pos_weight=torch.tensor(pos_weight, device=logit.device)
    )


# ---------------------------------------------------------------- crops
def sample_crop(rng, Xn: np.ndarray, L: int) -> tuple[int, int]:
    lo = int(rng.integers(0, max(1, len(Xn) - L)))
    return lo, min(lo + L, len(Xn))


def _augment(rng, x: np.ndarray, cfg: Cfg, d_hand: int) -> np.ndarray:
    x = x + rng.normal(0, cfg.noise, x.shape).astype(np.float32)
    if rng.random() < cfg.hand_drop:  # a hand vanishing is common at inference
        h = int(rng.integers(0, 2))
        x[h * d_hand:(h + 1) * d_hand] = 0.0
    return x


def train_model(sessions: list[Sess], cfg: Cfg, seed: int = 0, verbose: bool = False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    norm = Norm(sessions)
    Xn = [model_input(norm, s) for s in sessions]
    sig = cfg.sigma_ms / 1000.0
    Y = [gauss_target(s.t, s.kt, sig) for s in sessions]
    Yc = [labels(s.t, s.kt, 0.033).astype(np.float32) for s in sessions]
    d_in = Xn[0].shape[1]
    d_hand = sessions[0].X.shape[1] // 2
    Yg = [labels(s.t, s.kt, TYPING_HALF_WIDTH_S).astype(np.float32) for s in sessions]
    model = TCN(d_in, cfg, n_channels(cfg))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=cfg.steps)
    w = np.array([len(x) for x in Xn], float)
    w = w / w.sum()
    pos_w = float((1 - np.mean([y.mean() for y in Yc])) / max(np.mean([y.mean() for y in Yc]), 1e-6))
    model.train()
    for step in range(cfg.steps):
        xb, yb, gb, tgt, tlen = [], [], [], [], []
        for _ in range(cfg.batch):
            k = int(rng.choice(len(sessions), p=w))
            s = sessions[k]
            lo, hi = sample_crop(rng, Xn[k], cfg.crop)
            x = _augment(rng, Xn[k][lo:hi].T.copy(), cfg, d_hand)
            xb.append(torch.from_numpy(x))
            if cfg.head == "ctc":
                t0, t1 = s.t[lo], s.t[hi - 1]
                n = int(((s.kt > t0 + 0.02) & (s.kt < t1 - 0.02)).sum())
                tgt.append(torch.ones(n, dtype=torch.long))
                tlen.append(n)
            else:
                src = Y[k] if cfg.head == "heatmap" else Yc[k]
                yb.append(torch.from_numpy(src[lo:hi].copy()))
            gb.append(torch.from_numpy(Yg[k][lo:hi].copy()))
        X = torch.stack(xb)
        raw = model(X)
        logit = raw[:, :n_main(cfg)]
        if cfg.head == "ctc":
            lp = F.log_softmax(logit, dim=1).permute(2, 0, 1)  # [T,N,C]
            T = lp.shape[0]
            loss = F.ctc_loss(
                lp,
                torch.cat(tgt) if sum(tlen) else torch.zeros(0, dtype=torch.long),
                torch.full((cfg.batch,), T, dtype=torch.long),
                torch.tensor(tlen, dtype=torch.long),
                blank=0,
                zero_infinity=True,
            )
        else:
            yy = torch.stack(yb).unsqueeze(1)
            loss = (
                focal_heatmap(logit, yy)
                if cfg.head == "heatmap"
                else bce_cls(logit, yy, pos_w)
            )
        if cfg.gate:
            loss = loss + cfg.gate_w * F.binary_cross_entropy_with_logits(
                raw[:, n_main(cfg)], torch.stack(gb)
            )
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if verbose and (step + 1) % 200 == 0:
            print(f"    step {step+1}/{cfg.steps} loss={loss.item():.4f}", flush=True)
    model.eval()
    return {"model": model, "norm": norm, "cfg": cfg}


# ---------------------------------------------------------------- inference
def n_main(cfg: Cfg) -> int:
    return 2 if cfg.head == "ctc" else 1


def n_channels(cfg: Cfg) -> int:
    return n_main(cfg) + (1 if cfg.gate else 0)


@torch.no_grad()
def frame_scores(bundle, s: Sess) -> tuple[np.ndarray, np.ndarray | None]:
    """-> (event score [F], typing-gate score [F] or None). Score is P(key) for CTC."""
    m, cfg = bundle["model"], bundle["cfg"]
    x = torch.from_numpy(model_input(bundle["norm"], s).T).unsqueeze(0)
    raw = m(x)
    main = raw[:, : n_main(cfg)]
    p = (F.softmax(main, dim=1)[0, 1] if cfg.head == "ctc" else torch.sigmoid(main)[0, 0])
    g = torch.sigmoid(raw[0, n_main(cfg)]).numpy().astype(np.float64) if cfg.gate else None
    return p.numpy().astype(np.float64), g


def ctc_collapse(p: np.ndarray, thr: float, min_run: int = 1) -> np.ndarray:
    """Greedy CTC decode: each contiguous non-blank run emits one event at its peak frame."""
    on = p >= thr
    ev, i = [], 0
    while i < len(on):
        if not on[i]:
            i += 1
            continue
        j = i
        while j < len(on) and on[j]:
            j += 1
        if j - i >= min_run:
            ev.append(i + int(np.argmax(p[i:j])))
        i = j
    return np.array(ev, dtype=int)


CTC_GRID = {"thr": tuple(np.round(np.arange(0.05, 0.96, 0.025), 4)), "min_run": (1, 2, 3)}


def tune_ctc(p: np.ndarray, t: np.ndarray, kt: np.ndarray, mask: np.ndarray,
             gate: np.ndarray | None = None) -> tuple[dict, float]:
    from phase0.analysis.taps_gb import EVENT_GRID, prep_mask, score_prep

    prep = prep_mask(kt, t, mask)
    best, best_f1 = {"thr": 0.5, "min_run": 1, "gate_thr": 0.0, "mode": "ctc"}, -1.0
    for gt in ((0.0,) if gate is None else EVENT_GRID["gate"]):
        pg = p if gt <= 0 else np.where(gate >= gt, p, 0.0)
        for mr in CTC_GRID["min_run"]:
            for thr in CTC_GRID["thr"]:
                f1 = score_prep(prep, t, ctc_collapse(pg, thr, mr))["f1"]
                if f1 > best_f1:
                    best_f1 = f1
                    best = {"thr": float(thr), "min_run": mr, "gate_thr": gt, "mode": "ctc"}
    return best, best_f1


def decode(p: np.ndarray, cfg: dict, gate: np.ndarray | None = None) -> np.ndarray:
    if cfg.get("mode") == "ctc":
        pg = p if gate is None or cfg.get("gate_thr", 0.0) <= 0 else np.where(gate >= cfg["gate_thr"], p, 0.0)
        return ctc_collapse(pg, cfg["thr"], cfg["min_run"])
    return apply_events(p, cfg, gate)


def tune_decode(head: str, p, t, kt, mask, gate=None):
    if head == "ctc":
        return tune_ctc(p, t, kt, mask, gate)
    return tune_events(p, t, kt, mask, gate)


def write_taps(out: Path, s: Sess, ev: np.ndarray, p: np.ndarray) -> None:
    fv = flex_velocity(s.P)
    with open(out, "w") as fh:
        for k in ev:
            m = np.abs(np.nan_to_num(fv[k], nan=-1.0))
            side, f = np.unravel_index(int(np.argmax(m)), m.shape)
            tip = FINGERTIP_JOINTS[f]
            fh.write(
                json.dumps(
                    {
                        "t": float(s.t[k]),
                        "hand": int(side),
                        "finger": int(tip),
                        "x": float(np.nan_to_num(s.P[k, side, tip, 0])),
                        "y": float(np.nan_to_num(s.P[k, side, tip, 1])),
                        "conf": float(np.nan_to_num(s.P[k, side, tip, 2])),
                        "i": int(s.frames[k]),
                        "score": float(p[k]),
                    }
                )
                + "\n"
            )


# ---------------------------------------------------------------- protocol
def clip_taps(kt: np.ndarray, tt: np.ndarray) -> np.ndarray:
    """Same span clipping as eval_taps --clip: hands arriving/leaving are not the detector's fault."""
    return tt[(tt >= kt.min() - 0.1) & (tt <= kt.max() + 0.1)] if len(kt) else tt


def score_clipped(kt: np.ndarray, tt: np.ndarray) -> dict:
    return score(kt, clip_taps(kt, tt))


def loso(sessions: list[Sess], cfg: Cfg, seed: int, verbose: bool = False) -> dict:
    """Leave-one-session-out: -> {per-session F1, pooled scores/decode cfgs}."""
    out = {"f1": {}, "scores": [], "cfgs": []}
    for i, s in enumerate(sessions):
        tr = [x for j, x in enumerate(sessions) if j != i]
        b = train_model(tr, cfg, seed=seed, verbose=verbose)
        p, g = frame_scores(b, s)
        # decoding params come from the training sessions' own scores, never from s
        ins = [frame_scores(b, x) for x in tr]
        inner = np.concatenate([a for a, _ in ins])
        ing = np.concatenate([b_ for _, b_ in ins]) if cfg.gate else None
        it = np.concatenate([x.t for x in tr])
        ikt = np.sort(np.concatenate([x.kt for x in tr]))
        dcfg, _ = tune_decode(cfg.head, inner, it, ikt, np.ones(len(it), bool), ing)
        ev = decode(p, dcfg, g)
        sc = score_clipped(s.kt, s.t[ev])
        out["f1"][s.path.name] = sc
        out["scores"].append(p)
        out["gates"] = out.get("gates", []) + [g]
        out["cfgs"].append(dcfg)
        if verbose:
            print(f"  LOSO {s.path.name}: F1={100*sc['f1']:.1f}% cfg={dcfg}", flush=True)
    return out


def rollover_recall(kt: np.ndarray, tt: np.ndarray, gap: float = 0.080) -> dict:
    """Recall restricted to keydowns arriving <=gap after their predecessor."""
    from phase0.analysis.analyze_drift import pair_events

    if len(kt) == 0 or len(tt) == 0:
        return {"n": 0, "recall": 0.0}
    pairs = pair_events(np.sort(kt).tolist(), np.sort(tt).tolist(), 0.080)
    hit = np.zeros(len(kt), bool)
    for i, _ in pairs:
        hit[i] = True
    fast = np.zeros(len(kt), bool)
    fast[1:] = np.diff(np.sort(kt)) <= gap
    return {
        "n": int(fast.sum()),
        "frac": float(fast.mean()),
        "recall": float(hit[fast].mean()) if fast.any() else 0.0,
        "recall_slow": float(hit[~fast].mean()) if (~fast).any() else 0.0,
    }


def oof_decode(sessions: list[Sess], cfg: Cfg, seed: int, verbose: bool = False) -> tuple[dict, dict]:
    """LOSO once to get out-of-fold scores, and tune the decoder on those only."""
    r = loso(sessions, cfg, seed=seed, verbose=verbose)
    p = np.concatenate(r["scores"])
    g = np.concatenate(r["gates"]) if cfg.gate else None
    t = np.concatenate([s.t for s in sessions])
    kt = np.sort(np.concatenate([s.kt for s in sessions]))
    dcfg, f1 = tune_decode(cfg.head, p, t, kt, np.ones(len(t), bool), g)
    return dcfg, {"oof_f1": f1, "loso": r["f1"], "p": p, "g": g, "t": t, "kt": kt}


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(out)


def iqr(v) -> tuple[float, float]:
    return float(np.percentile(v, 25)), float(np.percentile(v, 75))


# ---------------------------------------------------------------- CLI
def cmd_experiment(a) -> int:
    cfg = Cfg(head=a.head, steps=a.steps, gate=not a.no_gate)
    S = load_sessions(a.sessions, cfg.groups)
    test = Sess(a.test, cfg.groups) if a.test else None
    rows = []
    for seed in range(a.seeds):
        t0 = time.time()
        dcfg, info = oof_decode(S, cfg, seed=seed, verbose=a.verbose)
        row = {"seed": seed, "loso": {k: v["f1"] for k, v in info["loso"].items()},
               "oof_f1": info["oof_f1"], "decode": dcfg}
        if test is not None:
            b = train_model(S, cfg, seed=seed)
            p, g = frame_scores(b, test)
            tt = test.t[decode(p, dcfg, g)]
            sc = score_clipped(test.kt, tt)
            row["test"] = sc
            row["rollover"] = rollover_recall(test.kt, clip_taps(test.kt, tt))
            row["ece"] = ece(labels(test.t, test.kt, 0.033).astype(float), p)
        rows.append(row)
        print(f"seed {seed}: LOSO={ {k: round(100*v,1) for k, v in row['loso'].items()} } "
              f"oofF1={100*row['oof_f1']:.1f}% "
              + (f"TEST F1={100*row['test']['f1']:.1f}% R={100*row['test']['recall']:.1f}% "
                 f"P={100*row['test']['precision']:.1f}% stillFP={row['test']['still_fp']} "
                 f"err={row['test']['median_abs_err_ms']:.0f}ms "
                 f"rollR={100*row['rollover']['recall']:.1f}% ECE={row['ece']:.3f}" if test else "")
              + f" [{time.time()-t0:.0f}s]", flush=True)
    summarise(rows, a.head)
    if a.json:
        print(json.dumps([{k: v for k, v in r.items()} for r in rows], default=float))
    return 0


def summarise(rows: list[dict], head: str) -> None:
    def stat(v):
        lo, hi = iqr(v)
        return f"mean {100*np.mean(v):.1f}% median {100*np.median(v):.1f}% IQR [{100*lo:.1f}, {100*hi:.1f}]"

    print(f"\n== {head}, {len(rows)} seeds ==")
    for name in rows[0]["loso"]:
        print(f"  LOSO {name:<22} {stat([r['loso'][name] for r in rows])}")
    if "test" in rows[0]:
        for k in ("f1", "recall", "precision"):
            print(f"  TEST {k:<22} {stat([r['test'][k] for r in rows])}")
        print(f"  TEST stillFP              {[r['test']['still_fp'] for r in rows]}")
        print(f"  TEST timing err ms        {[round(r['test']['median_abs_err_ms']) for r in rows]}")
        print(f"  TEST rollover recall      {stat([r['rollover']['recall'] for r in rows])}")
        print(f"  TEST non-rollover recall  {stat([r['rollover']['recall_slow'] for r in rows])}")
        print(f"  TEST ECE                  {np.mean([r['ece'] for r in rows]):.4f} "
              f"{[round(r['ece'], 3) for r in rows]}")


def cmd_train(a) -> int:
    cfg = Cfg(head=a.head, steps=a.steps, gate=not a.no_gate)
    S = load_sessions(a.sessions, cfg.groups)
    print(f"sessions={len(S)} frames={sum(len(s.t) for s in S)} keys={sum(len(s.kt) for s in S)} "
          f"d_in={S[0].X.shape[1] + 1} RF={receptive_field(cfg)} frames")
    dcfg, info = oof_decode(S, cfg, seed=a.seed, verbose=True)
    print(f"LOSO F1={ {k: round(100 * v['f1'], 1) for k, v in info['loso'].items()} } "
          f"OOF-tuned decode F1={100*info['oof_f1']:.1f}% cfg={dcfg}")
    b = train_model(S, cfg, seed=a.seed, verbose=True)
    n = sum(p.numel() for p in b["model"].parameters())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": b["model"].state_dict(), "mu": b["norm"].mu, "sd": b["norm"].sd,
                "cfg": asdict(cfg), "decode": dcfg}, a.out)
    print(f"saved {a.out} ({n} params)")
    return 0


def load_bundle(path: Path):
    d = torch.load(path, weights_only=False)
    cfg = Cfg(**{**d["cfg"], "groups": tuple(d["cfg"]["groups"])})
    # d_in is implied by the saved input projection
    d_in = d["state"]["inp.weight"].shape[1]
    m = TCN(d_in, cfg, n_channels(cfg))
    m.load_state_dict(d["state"])
    m.eval()
    norm = Norm.__new__(Norm)
    norm.mu, norm.sd = d["mu"], d["sd"]
    return {"model": m, "norm": norm, "cfg": cfg}, d["decode"]


def cmd_predict(a) -> int:
    b, dcfg = load_bundle(a.model)
    s = Sess(a.session, b["cfg"].groups)
    p, g = frame_scores(b, s)
    ev = decode(p, dcfg, g)
    out = a.out or a.session / "taps_seq.jsonl"
    write_taps(out, s, ev, p)
    print(f"wrote {len(ev)} events -> {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.taps_seq", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--sessions", type=Path, nargs="+", required=True)
    tr.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    tr.add_argument("--head", default="heatmap", choices=HEADS)
    tr.add_argument("--steps", type=int, default=Cfg.steps)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--no-gate", action="store_true")
    pr = sub.add_parser("predict")
    pr.add_argument("session", type=Path)
    pr.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    pr.add_argument("--out", type=Path, default=None)
    ex = sub.add_parser("experiment")
    ex.add_argument("--sessions", type=Path, nargs="+", required=True)
    ex.add_argument("--test", type=Path, default=None)
    ex.add_argument("--head", default="heatmap", choices=HEADS)
    ex.add_argument("--steps", type=int, default=Cfg.steps)
    ex.add_argument("--seeds", type=int, default=5)
    ex.add_argument("--no-gate", action="store_true")
    ex.add_argument("--verbose", action="store_true")
    ex.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    return {"train": cmd_train, "predict": cmd_predict, "experiment": cmd_experiment}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
