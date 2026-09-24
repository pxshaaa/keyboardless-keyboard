"""Desk extraction for tasks 2-4: per truth word (same word list/order as swipe_p3, so its CTC cache is reused)
truth-aligned letter times, label-free CTC peak times, span, classifier features at anchors, kinematic count features.
  PYTHONPATH=. .venv/bin/python -m phase0.analysis.finger_desk [--sets old,b1,b2]"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from phase0.analysis import finger_cls as FC
from phase0.analysis import finger_id as FI
from phase0.analysis import finger_truth as FT
from phase0.analysis import swipe_common as C

CACHE = Path(".cache/finger")
PADF = 3
OFFS = (0, -6, -3, 3, 6)   # 60 Hz rows added to CTC anchor times


def kin_series(P):
    """per-frame kinematic tap-evidence series from P [F,2,21,7]: [F, k] (label-free)."""
    xy = np.stack([P[:, :, j, :2] for j in FT.TIPS], 2)                      # F,2,5,2
    wr = P[:, :, 0, :2]
    span = np.nanmedian(np.linalg.norm(P[:, :, 9, :2] - wr, axis=2))
    rel = (xy - wr[:, :, None]) / span
    sp = np.linalg.norm(np.gradient(np.nan_to_num(xy), axis=0), axis=-1) / span   # F,2,5
    rsp = np.linalg.norm(np.gradient(np.nan_to_num(rel), axis=0), axis=-1)
    fl = np.stack([np.linalg.norm(P[:, :, j, :2] - P[:, :, FT.MCP[j], :2], axis=2) for j in FT.TIPS], 2) / span
    fv = np.abs(np.gradient(np.nan_to_num(fl), axis=0))
    z = np.stack([P[:, :, j, 3] - P[:, :, 0, 3] for j in FT.TIPS], 2)
    zv = np.abs(np.gradient(np.nan_to_num(z), axis=0))
    return {"rsp_max": np.nanmax(rsp.reshape(len(P), -1), 1), "rsp_sum": np.nansum(rsp.reshape(len(P), -1), 1),
            "fv_max": np.nanmax(fv.reshape(len(P), -1), 1), "zv_max": np.nanmax(zv.reshape(len(P), -1), 1),
            "sp_sum": np.nansum(sp.reshape(len(P), -1), 1)}


def count_feats(series, S_t, t0, t1):
    a, b = np.searchsorted(S_t, [t0, t1])
    b = max(b, a + 2)
    out = {"dur": float(t1 - t0)}
    for k, v in series.items():
        seg = np.nan_to_num(v[a:b])
        thr = np.median(v) if len(v) else 0
        pk = np.sum((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:]) & (seg[1:-1] > thr))
        out[f"{k}_npk"] = float(pk)
        out[f"{k}_mean"] = float(seg.mean())
        out[f"{k}_npk_rate"] = float(pk / max(t1 - t0, 1e-3))
        from scipy.signal import find_peaks
        for prom in (0.5, 1.0):
            out[f"{k}_fp{prom}"] = float(len(find_peaks(seg, prominence=prom * np.std(v), distance=5)[0]))
    return out


def extract(name):
    from phase0.analysis import tap_pos as tp
    import __main__
    __main__.Sess = tp.Sess
    setname, sdir, src = C.SETS[name]
    segs, lines = C.load(name)
    dj = json.loads(Path(f"data/sessions/{sdir}/decipher.json").read_text())
    times = np.load(f".cache/decipher/out/{sdir}_lp.npz")["times"]
    S = tp.load_sess(src)
    series = kin_series(S.P)
    hands = FI.prep(Path("data/sessions") / src)
    words = []
    for si, (s, row) in enumerate(zip(segs, dj["segments"])):
        s0, s1 = row["frames30"]
        M = C.ens28(s["lps"])
        if not s["truth"]:
            continue
        lf = C.letter_frames(M, s["truth"])
        k = 0
        for w in s["truth"].split():
            ch = lf[k:k + len(w)]
            k += len(w) + 1
            peaks = [c[2] for c in ch]
            if any(p < 0 for p in peaks):
                continue
            f0, f1 = min(c[0] for c in ch), max(c[1] for c in ch)
            a, b = max(0, f0 - PADF), min(len(M), f1 + PADF + 1)
            let = 1 - np.exp(np.logaddexp(M[a:b, 0], M[a:b, 27]))
            pk = [a + i for i in range(len(let)) if let[i] > 0.3 and (i == 0 or let[i] >= let[i - 1]) and (i == len(let) - 1 or let[i] > let[i + 1])]
            rec = {"set": name, "seg": si, "id": s["id"], "wi": len(words), "w": w,
                   "t_aligned": times[s0 + np.array(peaks)], "t_peaks": times[s0 + np.array(pk, int)] if pk else np.zeros(0),
                   "span_t": (float(times[s0 + a]), float(times[s0 + b - 1])), "span": (a, b),
                   "let_mass": float(let.sum()), "n_peaks": len(pk)}
            rec["count_feats"] = count_feats(series, S.t, *rec["span_t"])
            rec["count_feats"].update({"ctc_let_mass": rec["let_mass"], "ctc_npeaks": float(len(pk))})
            for o in OFFS:
                rec[f"X_aligned{o}"], rec[f"ok_aligned{o}"], _ = FC.feats_at(hands, rec["t_aligned"] + o / 60.0)
                if pk:
                    rec[f"X_peaks{o}"], rec[f"ok_peaks{o}"], _ = FC.feats_at(hands, rec["t_peaks"] + o / 60.0)
            rows = np.clip(np.searchsorted(S.t, rec["t_aligned"]), 0, len(S.t) - 1)
            rec["tips_aligned_raw"] = FT.tips_xy(S.P, rows)
            prow = np.clip(np.searchsorted(S.t, rec["t_peaks"]), 0, len(S.t) - 1) if pk else np.zeros(0, int)
            rec["tips_peaks_raw"] = FT.tips_xy(S.P, prow) if pk else np.zeros((0, 10, 2))
            words.append(rec)
    (CACHE / f"desk_{name}.pkl").write_bytes(pickle.dumps(words, protocol=4))
    print(name, "words", len(words), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="old,b1,b2")
    a = ap.parse_args()
    for n in a.sets.split(","):
        extract(n)


if __name__ == "__main__":
    main()
