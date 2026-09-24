"""Idea 2b: fingertip-crop CNN (contact appearance) trained on kbd, used to rescore desk taps.
Run: python -m phase0.analysis.touch_pix {bank | train --seeds 3}"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import lightgbm  # noqa: F401

from phase0.analysis import touch_common as tc

PIXBANK = Path(".cache/contact_ego/pixbank")
KBD5 = tc.KBD_TRAIN4 + (tc.KBD_TEST,)
DESK_BANK = Path(".cache/hirecall/20260910-202149-desk_ftips.npy")
DESK_FRAMES = Path(".cache/hirecall/20260910-202149-desk_frames.json")
ALPHA = 0.5   # fixed a priori: p_new = p^(1-ALPHA) * r^ALPHA


def cmd_bank(a) -> int:
    from phase0.analysis import combine as CB
    from phase0.analysis import hirecall as hr
    CB.CACHE = PIXBANK
    hr._stub_masks()
    rng = np.random.default_rng(0)
    for sid in KBD5:
        fr, t, _ = tc.frames(sid)
        kt = tc.kt(sid)
        rk = np.unique(np.clip(np.searchsorted(t, kt), 0, len(t) - 1))
        d = np.abs(np.arange(len(t))[:, None] - rk[None, :]).min(1) if len(t) < 60000 else None
        typing = np.zeros(len(t), bool)
        for r in rk:
            typing[max(0, r - 120):r + 120] = True
        near = np.where((d >= 6) & (d <= 15))[0]
        far = np.where(typing & (d > 15))[0]
        n = len(rk)
        neg = np.concatenate([rng.choice(near, min(len(near), int(0.75 * n)), replace=False),
                              rng.choice(far, min(len(far), int(0.75 * n)), replace=False)])
        rows = np.unique(np.concatenate([rk, neg]))
        y = np.isin(rows, rk)
        vf = fr[rows]
        order = np.argsort(vf)
        vf, rows, y = vf[order], rows[order], y[order]
        t0 = time.time()
        CB.build_frame_banks(sid, vf, force=True)
        (PIXBANK / f"{sid}_labels.json").write_text(json.dumps(
            {"frames": vf.tolist(), "rows": rows.tolist(), "y": y.astype(int).tolist()}))
        (PIXBANK / f"{sid}_fmask.npy").unlink(missing_ok=True)   # stub masks: never read
        print(f"{sid}: {len(vf)} crops ({int(y.sum())} keydown) {time.time()-t0:.0f}s", flush=True)
    return 0


def _load(sid):
    X = np.load(PIXBANK / f"{sid}_ftips.npy", mmap_mode="r")
    L = json.loads((PIXBANK / f"{sid}_labels.json").read_text())
    return X, np.array(L["y"], np.float32)


def _net():
    import torch.nn as nn

    class Tip(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv2d(6, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(48, 1))

        def forward(self, x):                       # x [B,6,2,5,32,32]
            import torch
            B = x.shape[0]
            z = x.permute(0, 2, 3, 1, 4, 5).reshape(B * 10, 6, 32, 32)
            z = z - z[:, :1]                         # change relative to the anchor frame...
            z = torch.cat([x.permute(0, 2, 3, 1, 4, 5).reshape(B * 10, 6, 32, 32)[:, :1], z[:, 1:]], 1)
            return torch.logsumexp(self.f(z).view(B, 10), 1)   # ...and the anchor itself

    return Tip()


def _tensor(X, idx, dev):
    import torch
    return torch.from_numpy(np.asarray(X[idx], np.float32) / 255.0).to(dev)


def fit(Xs, ys, seed, dev, epochs=12):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = np.concatenate([np.asarray(x) for x in Xs])
    y = np.concatenate(ys)
    net = _net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(len(y))
        for b in range(0, len(y), 128):
            i = perm[b:b + 128]
            xb = _tensor(X, i, dev)
            if rng.random() < 0.5:                  # mirror augmentation across the patch
                xb = torch.flip(xb, dims=[-1])
            loss = lossf(net(xb), torch.from_numpy(y[i]).to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


def predict(net, X, dev):
    import torch
    out = []
    with torch.no_grad():
        for b in range(0, len(X), 512):
            out.append(torch.sigmoid(net(_tensor(X, np.arange(b, min(b + 512, len(X))), dev))).cpu().numpy())
    return np.concatenate(out)


def rescore(p: np.ndarray, r_bank: np.ndarray) -> tuple[np.ndarray, float]:
    fr = tc.frames(tc.DESK)[0]
    bf = np.array(json.loads(DESK_FRAMES.read_text())["frames"], int)
    j = np.clip(np.searchsorted(bf, fr), 1, len(bf) - 1)
    jj = np.where(np.abs(fr - bf[j - 1]) <= np.abs(bf[j] - fr), j - 1, j)
    ok = np.abs(bf[jj] - fr) <= 2
    r = np.where(ok, r_bank[jj], np.median(r_bank))
    return p ** (1 - ALPHA) * r ** ALPHA, float(ok.mean())


def cmd_train(a) -> int:
    import torch
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(4)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tr = [_load(s) for s in tc.KBD_TRAIN4]
    Xte, yte = _load(tc.KBD_TEST)
    D = np.load(DESK_BANK, mmap_mode="r")
    out = {"auc": [], "coverage": None}
    for seed in range(a.seeds):
        t0 = time.time()
        net = fit([x for x, _ in tr], [y for _, y in tr], seed, dev)
        auc = roc_auc_score(yte, predict(net, Xte, dev))
        r = predict(net, D, dev)
        out["auc"].append(float(auc))
        z = np.load(tc.CACHE / f"desk_probs_base_s{seed}.npz")
        p2, cov = rescore(z["p"], r)
        out["coverage"] = cov
        np.savez(tc.CACHE / f"desk_probs_pix_s{seed}.npz", p=p2, gate=z["gate"])
        np.save(tc.CACHE / f"pix_r_s{seed}.npy", r)
        bk = json.loads((tc.CACHE / f"kbd_base_s{seed}.json").read_text())
        bk["note"] = "kbd F1 not re-measured (crop bank is keydown-anchored, not dense); see pix auc"
        (tc.CACHE / f"kbd_pix_s{seed}.json").write_text(json.dumps(bk, default=float))
        print(f"seed{seed}: held-out 015948 keydown-vs-other crop AUC={auc:.3f}  desk r mean={r.mean():.3f} "
              f"coverage={cov:.2f} ({time.time()-t0:.0f}s)", flush=True)
    Path("results/contact").mkdir(parents=True, exist_ok=True)
    Path("results/contact/pix_contact.json").write_text(json.dumps(out, indent=1))
    return 0


def cmd_kbdeval(a) -> int:
    """Missed/extra taps on held-out kbd at matched density, base probs vs pixel-rescored probs."""
    import torch
    from phase0.analysis.touch_missextra import at_density
    torch.set_num_threads(4)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tr = [_load(s) for s in tc.KBD_TRAIN4]
    cand = Path(".cache/contact_ego/pixbank_cand")
    C = np.load(cand / f"{tc.KBD_TEST}_ftips.npy", mmap_mode="r")
    bf = np.array(json.loads((cand / f"{tc.KBD_TEST}_frames.json").read_text())["frames"], int)
    fr, t, _ = tc.frames(tc.KBD_TEST)
    kt = tc.kt(tc.KBD_TEST)
    j = np.clip(np.searchsorted(bf, fr), 1, len(bf) - 1)
    jj = np.where(np.abs(fr - bf[j - 1]) <= np.abs(bf[j] - fr), j - 1, j)
    ok = np.abs(bf[jj] - fr) <= 2
    out = {"base": {}, "pix": {}}
    for seed in range(a.seeds):
        net = fit([x for x, _ in tr], [y for _, y in tr], seed, dev)
        r_b = predict(net, C, dev)
        z = np.load(tc.CACHE / f"kbdtest_probs_base_s{seed}.npz")
        r = np.where(ok, r_b[jj], np.median(r_b))
        p2 = z["p"] ** (1 - ALPHA) * r ** ALPHA
        for d in (1.0, 1.3):
            out["base"].setdefault(str(d), []).append(at_density(z["p"], z["gate"], t, kt, d))
            out["pix"].setdefault(str(d), []).append(at_density(p2, z["gate"], t, kt, d))
        print(f"seed{seed}: " + " ".join(f"@{d} base miss={out['base'][str(d)][-1]['miss_rate']:.3f} "
              f"extra={out['base'][str(d)][-1]['extra_per_char']:.3f} | pix miss={out['pix'][str(d)][-1]['miss_rate']:.3f} "
              f"extra={out['pix'][str(d)][-1]['extra_per_char']:.3f}" for d in (1.0, 1.3)), flush=True)
    summ = {arm: {d: {k: float(np.mean([x[k] for x in v])) for k in v[0]} for d, v in dd.items()} for arm, dd in out.items()}
    Path("results/contact/pix_kbd_missextra.json").write_text(json.dumps({"mean": summ, "per_seed": out}, indent=1))
    print(json.dumps(summ, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("bank", "train", "kbdeval"))
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args(argv)
    return {"bank": cmd_bank, "train": cmd_train, "kbdeval": cmd_kbdeval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
