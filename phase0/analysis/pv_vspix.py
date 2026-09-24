"""PressureVision++ vs our own fingertip-crop CNN on identical frames; the CNN is retrained without the probe session.
Usage: python -m phase0.analysis.pv_vspix <session_id> [--seeds 3]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phase0.analysis import pv_common as pv
from phase0.analysis import pv_probe as pp
from phase0.analysis import taps_gb as gb
from phase0.analysis import touch_pix as tp
from phase0.analysis import touch_common as tc

OUT = Path("results/pressure")


def main(argv=None) -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("sid")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    train_sids = [s for s in tc.KBD_TRAIN4 + (tc.KBD_TEST,) if s != a.sid]
    tr = [tp._load(s) for s in train_sids]
    Xte, yte = tp._load(a.sid)
    L = json.loads((tp.PIXBANK / f"{a.sid}_labels.json").read_text())
    bf = np.array(L["frames"], int)

    z = np.load(pv.PV_ROOT / "feats" / f"{a.sid}.npz")
    zi = z["i"]
    j = np.searchsorted(zi, bf)
    ok = (j < len(zi)) & (zi[np.clip(j, 0, len(zi) - 1)] == bf)
    cp = z["cp_max"][np.clip(j, 0, len(zi) - 1)].reshape(len(bf), 10)
    pr = z["pr_max"][np.clip(j, 0, len(zi) - 1)].reshape(len(bf), 10)
    t = z["t"][np.clip(j, 0, len(zi) - 1)]

    s = Path("data/sessions") / a.sid
    kfr, ft, P = gb.load_frames(s)
    from phase0.analysis import touch_feats as tf
    _, pool0 = tf.hand_feats(P[:, 0])
    _, pool1 = tf.hand_feats(P[:, 1])
    pool = np.hstack([pool0[:, :, 0], pool1[:, :, 0]])
    k2 = np.searchsorted(kfr, bf)
    ok &= (k2 < len(kfr)) & (kfr[np.clip(k2, 0, len(kfr) - 1)] == bf)
    kin = np.nanmax(pool[np.clip(k2, 0, len(kfr) - 1)], 1)

    m = ok & np.isfinite(cp).any(1)
    y = yte.astype(bool)
    blk = pp.block_ids(t)
    res = {"session": a.sid, "train_sessions": train_sids, "bank_rows": int(len(bf)),
           "rows_used": int(m.sum()), "positives": int(y[m].sum()),
           "note": "bank rows are keydown-anchored plus sampled near/far negatives, so absolute AUCs "
                   "are not comparable with the dense-frame table; the ranking is."}

    sc = {"pv_contact_max_raw": np.nanmax(cp, 1), "pv_pressure_max_raw": np.nanmax(pr, 1),
          "kin_stopmotion_raw": kin}
    aucs = {}
    for seed in range(a.seeds):
        net = tp.fit([x for x, _ in tr], [yy for _, yy in tr], seed, dev)
        r = tp.predict(net, Xte, dev)
        sc[f"ourcnn_seed{seed}"] = r
        aucs[f"ourcnn_seed{seed}"] = pp.auc(y[m], r[m])
        print(f"seed{seed}: our contact CNN AUC={aucs[f'ourcnn_seed{seed}']:.4f}", flush=True)
    sc["ourcnn_mean"] = np.mean([sc[f"ourcnn_seed{i}"] for i in range(a.seeds)], 0)

    tbl = {}
    for k, v in sc.items():
        if k.startswith("ourcnn_seed"):
            continue
        tbl[k] = {"auc": pp.auc(y[m], v[m]), "ci95": pp.boot_auc(y[m], v[m], blk[m])}
        print(f"{k:24s} AUC={tbl[k]['auc']:.4f} 95% CI {tbl[k]['ci95'][0]:.4f}-{tbl[k]['ci95'][1]:.4f}", flush=True)
    res["auc"] = tbl
    res["per_seed_ourcnn"] = aucs
    res["deltas"] = {}
    for A, B in (("pv_contact_max_raw", "ourcnn_mean"), ("pv_contact_max_raw", "kin_stopmotion_raw"),
                 ("ourcnn_mean", "kin_stopmotion_raw")):
        d, ci = pp.boot_delta(y[m], sc[A][m], sc[B][m], blk[m])
        res["deltas"][f"{A}__minus__{B}"] = {"delta": d, "ci95": ci}
        print(f"delta {A} - {B}: {d:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]", flush=True)

    (OUT / f"step1_vspix_{a.sid}.json").write_text(json.dumps(res, indent=1, default=float))
    print("wrote", OUT / f"step1_vspix_{a.sid}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
