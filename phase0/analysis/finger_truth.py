"""Task 1: motion/position-derived ground truth of WHICH fingertip pressed each keyboard letter keystroke.
Per session: key location (image px, camera fixed within a session) and pressing finger are estimated jointly by EM
(E: tip nearest the key centroid at the keydown frame, M: per-key median of the assigned tips). Two inits (touch-typing
finger; label-free argmax flexion velocity) must converge to the same answer. Independent check: location-free motion
signals (tip depth drop / flexion / stop) ranked among the 10 tips. Consistency of the typist's finger-per-key mapping.
  PYTHONPATH=. .venv/bin/python -m phase0.analysis.finger_truth {truth|crops}"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from phase0.analysis import swipe_motor as SM

KBD = ["20260910-015217-kbd", "20260910-021315-kbd", "20260910-131629-kbd", "20260911-164237-kbd", "20260910-015948-kbd"]
TIPS = (4, 8, 12, 16, 20)
MCP = {4: 2, 8: 5, 12: 9, 16: 13, 20: 17}
FK = np.array([SM.fid(*SM.gt_fid(c)) for c in SM.LET])
OUT = Path("results/finger")
CACHE = Path(".cache/finger")
FN = ["L-th", "L-ix", "L-mi", "L-ri", "L-pi", "R-th", "R-ix", "R-mi", "R-ri", "R-pi"]


def sess(sid):
    from phase0.analysis import tap_pos as tp
    import __main__
    __main__.Sess = tp.Sess
    return tp.load_sess(sid)


def letter_downs(sid):
    """-> list of (t, letter idx, index in all downs) for a-z keydowns."""
    downs = [e for e in map(json.loads, (Path("data/sessions") / sid / "keys.jsonl").read_text().splitlines())
             if e["event"] == "down"]
    return downs, [(e["t"], SM.LI[e["key"]], k) for k, e in enumerate(downs) if len(e["key"]) == 1 and "a" <= e["key"] <= "z"]


def tips_xy(P, rows):
    return np.stack([P[rows][:, h, j, :2] for h in (0, 1) for j in TIPS], 1)   # [n,10,2]


def motion_scores(P, rows):
    """location-free per-tip press evidence at rows: [n,10] for several signals (higher = more press-like)."""
    span = np.linalg.norm(P[:, :, 9, :2] - P[:, :, 0, :2], axis=2)            # [F,2]
    span = np.where(span > 1e-6, span, np.nan)
    fl = np.stack([np.linalg.norm(P[:, :, j, :2] - P[:, :, MCP[j], :2], axis=2) for j in TIPS], 2) / span[:, :, None]
    z = np.stack([P[:, :, j, 3] - P[:, :, 0, 3] for j in TIPS], 2)             # MediaPipe relative depth (px-ish)
    xy = np.stack([P[:, :, j, :2] for j in TIPS], 2)                          # [F,2,5,2]
    F = len(P)

    def at(off):
        return np.clip(rows + off, 0, F - 1)
    out = {}
    # flexion change over the approach (tip curls toward MCP when pressing, seen from above)
    out["flex_drop"] = (fl[at(-9)] - fl[at(0)]).reshape(len(rows), 10)
    # depth: tip moves away from camera relative to wrist (z increases = farther in MediaPipe convention)
    out["z_drop"] = (z[at(0)] - z[at(-9)]).reshape(len(rows), 10)
    out["negz_drop"] = -out["z_drop"]
    # image-plane travel into the key then stop: speed over [-9,-2] minus speed over [0,+3]
    sp = np.linalg.norm(np.diff(xy, axis=0), axis=-1)                         # [F-1,2,5]
    sp = np.concatenate([sp, sp[-1:]], 0)
    pre = np.nanmean([sp[at(o)] for o in range(-9, -1)], 0)
    post = np.nanmean([sp[at(o)] for o in range(0, 4)], 0)
    out["travel_stop"] = (pre - post).reshape(len(rows), 10) / np.nanmedian(span)
    out["travel"] = pre.reshape(len(rows), 10) / np.nanmedian(span)
    # relative to the hand: tip displacement minus the wrist's
    wr = P[:, :, 0, :2]
    rel = xy - wr[:, :, None, :]
    out["rel_travel"] = np.linalg.norm(rel[at(0)] - rel[at(-9)], axis=-1).reshape(len(rows), 10) / np.nanmedian(span)
    return out


def em(T, keys, f0, iters=10):
    ok = np.isfinite(T).all((1, 2))
    f = f0.copy()
    for _ in range(iters):
        mu = np.full((26, 2), np.nan)
        for k in range(26):
            m = (keys == k) & ok
            if m.sum():
                mu[k] = np.nanmedian(T[m, f[m]], 0)
        d = np.linalg.norm(T - mu[keys][:, None], axis=2)
        d = np.where(np.isfinite(d), d, 1e9)
        fn = d.argmin(1)
        if (fn[ok] == f[ok]).all():
            break
        f = np.where(ok, fn, f)
    ds = np.sort(d, 1)
    return f, mu, ds[:, 0], ds[:, 1], ok


def cmd_truth():
    from phase0.analysis import tap_pos as tp
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"sessions": {}}
    allk, allf, allsid = [], [], []
    mot_rank = {}
    for sid in KBD:
        S = sess(sid)
        downs, L = letter_downs(sid)
        t = np.array([x[0] for x in L])
        keys = np.array([x[1] for x in L])
        rows = np.clip(np.searchsorted(S.t, t), 0, len(S.t) - 1)
        rows = np.where((rows > 0) & (np.abs(S.t[rows - 1] - t) < np.abs(S.t[rows] - t)), rows - 1, rows)
        T = tips_xy(S.P, rows)
        f_tt, mu, d1, d2, ok = em(T, keys, FK[keys])
        # label-free init: argmax |flexion velocity| over 10 tips at the keydown frame (taps_gb rule)
        fs = tp._flex_score(S)[rows].reshape(len(rows), 10)
        f_lf, mu2, *_ = em(T, keys, fs.argmax(1))
        ratio = d2 / np.maximum(d1, 1.0)
        pitch = float(np.nanmedian([np.linalg.norm(mu[SM.LI[a]] - mu[SM.LI[b]]) for a, b in
                                    (("a", "s"), ("s", "d"), ("d", "f"), ("j", "k"), ("k", "l"), ("e", "r"), ("r", "t"),
                                     ("u", "i"), ("i", "o"))]))
        M = motion_scores(S.P, rows)
        conf = ok & (ratio > 2.5)
        for nm, sc in M.items():
            sc = np.where(np.isfinite(sc), sc, -1e9)
            mot_rank.setdefault(nm, []).append(np.mean(sc[conf].argmax(1) == f_tt[conf]))
        rep["sessions"][sid] = {
            "letters": int(len(L)), "tracked": float(ok.mean()), "key_pitch_px": pitch,
            "inits_agree": float((f_tt == f_lf)[ok].mean()),
            "match_touch_typing": float((f_tt == FK[keys])[ok].mean()),
            "nearest_px_median": float(np.median(d1[ok])), "second_px_median": float(np.median(d2[ok])),
            "frac_ratio_gt2.5": float(conf.sum() / ok.sum()),
            "frac_nearest_gt_half_pitch": float(np.mean(d1[ok] > 0.5 * pitch)),
            "motion_top1_vs_truth(conf)": {nm: float(v[-1]) for nm, v in mot_rank.items()},
        }
        np.savez(CACHE / f"truth_{sid}.npz", t=t, keys=keys, rows=rows, frame_i=S.frames[rows], finger=f_tt,
                 finger_lfinit=f_lf, ok=ok, d1=d1, d2=d2, mu=mu, pitch=pitch, down_index=np.array([x[2] for x in L]))
        print(sid, json.dumps(rep["sessions"][sid]), flush=True)
        allk.append(keys[ok]); allf.append(f_tt[ok]); allsid += [sid] * int(ok.sum())
    K, Fg = np.concatenate(allk), np.concatenate(allf)
    sids = np.array(allsid)
    cnt = np.zeros((26, 10), int)
    np.add.at(cnt, (K, Fg), 1)
    maj = cnt.argmax(1)
    per_key = {}
    for k in range(26):
        n = cnt[k].sum()
        if not n:
            continue
        p = cnt[k] / n
        per_key[SM.LET[k]] = {"n": int(n), "touch": FN[FK[k]], "majority": FN[maj[k]], "maj_share": float(p.max()),
                              "dist": {FN[j]: int(cnt[k, j]) for j in range(10) if cnt[k, j]}}
    # cross-session stability: majority map from other sessions predicts this session's finger
    loso = {}
    for sid in KBD:
        c = np.zeros((26, 10))
        np.add.at(c, (K[sids != sid], Fg[sids != sid]), 1)
        m = sids == sid
        loso[sid] = float((c.argmax(1)[K[m]] == Fg[m]).mean())
    rep["mapping"] = {
        "n": int(len(K)), "match_touch_typing": float((FK[K] == Fg).mean()),
        "majority_map_acc_insample": float((maj[K] == Fg).mean()),
        "majority_map_acc_loso": loso,
        "keys_with_maj_share_lt_0.8": [k for k, v in per_key.items() if v["maj_share"] < 0.8 and v["n"] >= 10],
        "keys_majority_differs_from_touch": [k for k, v in per_key.items() if v["majority"] != v["touch"]],
        "per_key": per_key,
        "motion_signal_top1_vs_position_truth_mean": {nm: float(np.mean(v)) for nm, v in mot_rank.items()},
    }
    np.save(CACHE / "finger_per_key_counts.npy", cnt)
    (OUT / "t1_truth.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep["mapping"].items() if k != "per_key"}, indent=1))
    for k, v in per_key.items():
        print(k, v)


def cmd_crops():
    """montages of keydown frames: green = EM finger, red = touch-typing finger if different. Random 30 + 30 non-touch."""
    import cv2
    rng = np.random.default_rng(0)
    OUT.joinpath("crops").mkdir(parents=True, exist_ok=True)
    picks = []
    for sid in KBD:
        z = np.load(CACHE / f"truth_{sid}.npz")
        ok = z["ok"]
        for i in np.where(ok)[0]:
            picks.append((sid, i, bool(z["finger"][i] != FK[z["keys"][i]])))
    rnd = [picks[i] for i in rng.choice(len(picks), 30, replace=False)]
    non = [p for p in picks if p[2]]
    non = [non[i] for i in rng.choice(len(non), min(30, len(non)), replace=False)]
    for tag, sel in (("random", rnd), ("nontouch", non)):
        tiles, meta = [], []
        for sid, i, _ in sorted(sel):
            z = np.load(CACHE / f"truth_{sid}.npz")
            S = sess(sid)
            cap = cv2.VideoCapture(f"data/sessions/{sid}/video.mp4")
            fi = int(z["frame_i"][i])
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            okr, img = cap.read()
            if not okr:
                continue
            r = int(z["rows"][i])
            T = tips_xy(S.P, np.array([r]))[0]
            f = int(z["finger"][i])
            g = int(FK[z["keys"][i]])
            c = T[f]
            for j in range(10):
                if np.isfinite(T[j]).all():
                    cv2.circle(img, tuple(int(v) for v in T[j]), 4, (255, 255, 0), 1)
            if g != f and np.isfinite(T[g]).all():
                cv2.circle(img, tuple(int(v) for v in T[g]), 12, (0, 0, 255), 2)
            cv2.circle(img, tuple(int(v) for v in c), 9, (0, 255, 0), 2)
            x0, y0 = int(np.clip(c[0] - 110, 0, img.shape[1] - 220)), int(np.clip(c[1] - 110, 0, img.shape[0] - 220))
            tile = img[y0:y0 + 220, x0:x0 + 220].copy()
            lab = f"{SM.LET[z['keys'][i]]} {FN[f]}" + (f" tt:{FN[g]}" if g != f else "")
            cv2.putText(tile, lab, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            tiles.append(tile)
            meta.append({"sid": sid, "i": int(i), "frame": fi, "key": SM.LET[z["keys"][i]], "em": FN[f], "touch": FN[g]})
        for p in range(0, len(tiles), 15):
            chunk = tiles[p:p + 15] + [np.zeros_like(tiles[0])] * (15 - len(tiles[p:p + 15]))
            grid = np.vstack([np.hstack(chunk[r * 5:(r + 1) * 5]) for r in range(3)])
            cv2.imwrite(str(OUT / "crops" / f"{tag}_{p // 15}.jpg"), grid)
        (OUT / "crops" / f"{tag}.json").write_text(json.dumps(meta, indent=1))
        print(tag, len(tiles))


if __name__ == "__main__":
    {"truth": cmd_truth, "crops": cmd_crops}[sys.argv[1]]()
