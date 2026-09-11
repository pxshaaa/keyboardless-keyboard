"""Contact appearance or key-legend reading? Masks appearance.py's tip crops to the hand.
Run: python -m phase0.analysis.masked {mask|peek|loso|cer|desk}"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import lightgbm  # noqa: F401  must load before torch or the two libomp copies segfault

from phase0.analysis.analyze_drift import read_jsonl
from phase0.analysis.appearance import (
    MIN_N,
    N_FINGER,
    TIP_CROP,
    TIP_OFFSETS,
    TIP_SPANS,
    SmallCNN,
    agg,
    cnn_proba,
    finger_labels,
    fit_cnn,
    fuse,
    lm_proba,
    n_params,
    score,
    tip_bank_path,
    warm_lm,
)
from phase0.analysis.decode import A_INDEX, ALPHABET, NA
from phase0.analysis.finger_id import FINGERTIP_JOINTS
from phase0.analysis.tap_pos import H as FRAME_H
from phase0.analysis.tap_pos import W as FRAME_W
from phase0.analysis.tap_pos import Sess, labelled_taps, load_sess, sess_path

TRAIN_SESSIONS = ("20260910-015217-kbd", "20260910-021315-kbd", "20260910-131629-kbd")
HELDOUT_SESSION = "20260910-015948-kbd"
TUNE_SESSION = "20260910-131629-kbd"
DESK_SESSION = "20260910-202149-desk"
SEEDS = (0, 1, 2)

# appearance.py's crop banks were built from taps_contact.jsonl; taps.jsonl has since been
# repointed at a different detector, so pin the bank's own tap file to stay row-aligned.
TAPS_NAME = "taps_contact.jsonl"
MASKS = Path(".cache/masked")

# MediaPipe hand skeleton; 0-17 closes the palm loop.
EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
         (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
         (18, 19), (19, 20), (0, 17))
PALM_POLY = (0, 1, 2, 5, 9, 13, 17)
BONE_W = 0.34      # bone stroke width in hand-span units
JOINT_R = 0.20     # joint disc radius in hand-span units
DILATE = 0.12      # extra silhouette margin in hand-span units


# ------------------------------------------------------------------ mask bank
def mask_bank_path(sid: str) -> Path:
    return MASKS / f"{sid}_tipmask.npy"


def _frame_mask(cv2, P: np.ndarray, span: np.ndarray, shape) -> np.ndarray:
    """[H,W] uint8 0/1 hand silhouette for one video frame, union over both hands."""
    m = np.zeros(shape, np.uint8)
    for h in (0, 1):
        q = P[h]
        if not np.isfinite(q).all():
            continue
        pts = np.round(q).astype(np.int32)
        w = max(2, int(round(BONE_W * span[h])))
        for a, b in EDGES:
            cv2.line(m, tuple(pts[a]), tuple(pts[b]), 1, w, cv2.LINE_8)
        for j in range(21):
            cv2.circle(m, tuple(pts[j]), max(2, int(round(JOINT_R * span[h]))), 1, -1)
        cv2.fillConvexPoly(m, cv2.convexHull(pts[list(PALM_POLY)]), 1)
    k = max(3, int(round(DILATE * float(np.nanmean(span)))) * 2 + 1)
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def _tip_geometry(s: Sess, taps: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicates appearance.build_tip_bank: patch centres frozen at the TAP frame."""
    k = s.rows(taps)
    P = s.P[np.clip(k, 0, len(s.P) - 1)][:, :, list(FINGERTIP_JOINTS), :2]
    fb = np.broadcast_to(s.anchor[:, None, :], P.shape[1:])
    P = np.where(np.isfinite(P), P, fb[None])
    return P, TIP_SPANS * s.span, np.asarray([t["i"] for t in taps])


def build_mask_bank(sid: str, force: bool = False) -> None:
    """Same [n,offsets,2,5,32,32] layout as the tip bank, but 0/1 hand membership."""
    import cv2

    out_p = mask_bank_path(sid)
    if out_p.exists() and not force:
        return
    MASKS.mkdir(parents=True, exist_ok=True)
    s = load_sess(sid)
    taps = read_jsonl(sess_path(sid) / TAPS_NAME)
    P, side, base = _tip_geometry(s, taps)
    n = len(taps)
    shape = (int(FRAME_H), int(FRAME_W))
    out = np.lib.format.open_memmap(out_p, mode="w+", dtype=np.uint8,
                                    shape=(n, len(TIP_OFFSETS), 2, 5, TIP_CROP, TIP_CROP))
    t0 = time.time()
    for j, off in enumerate(TIP_OFFSETS):
        fr = np.clip(base + off, 0, len(s.frames) - 1)
        rows = np.clip(np.searchsorted(s.frames, fr), 0, len(s.P) - 1)
        cache: dict[int, np.ndarray] = {}
        for i in range(n):
            r = int(rows[i])
            fm = cache.get(r)
            if fm is None:
                # the mask must come from the frame actually sampled, not the tap frame
                fm = _frame_mask(cv2, s.P[r, :, :, :2], s.span, shape)
                cache[r] = fm
            for h in (0, 1):
                sc = TIP_CROP / side[h]
                for f in range(5):
                    cx, cy = P[i, h, f]
                    M = np.array([[sc, 0, -sc * (cx - side[h] / 2)],
                                  [0, sc, -sc * (cy - side[h] / 2)]], np.float32)
                    out[i, j, h, f] = cv2.warpAffine(fm, M, (TIP_CROP, TIP_CROP),
                                                     flags=cv2.INTER_NEAREST,
                                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out.flush()
    print(f"{sid}: mask bank {out.shape} hand-frac={float(np.asarray(out[::7]).mean()):.3f} "
          f"-> {out_p} ({time.time()-t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ dataset
def build_dataset(sid: str, labelled: bool = True) -> dict:
    s = load_sess(sid)
    all_taps = read_jsonl(sess_path(sid) / TAPS_NAME)
    tips = np.load(tip_bank_path(sid), mmap_mode="r")
    if len(tips) != len(all_taps):
        raise SystemExit(f"{sid}: tip bank has {len(tips)} rows but {TAPS_NAME} has "
                         f"{len(all_taps)}; rebuild the bank")
    d = {"sid": sid, "sess": s, "all_taps": all_taps, "tips": tips,
         "masks": np.load(mask_bank_path(sid), mmap_mode="r")}
    if labelled:
        taps, keys = labelled_taps(sess_path(sid), TAPS_NAME)
        pos = {round(t["t"], 6): i for i, t in enumerate(all_taps)}
        d["taps"], d["keys"] = taps, keys
        d["bank_rows"] = np.array([pos[round(t["t"], 6)] for t in taps])
        d["k"] = s.rows(taps)
        d["y"] = np.array([A_INDEX[c] for c in keys])
        d["yf"] = finger_labels(s, d["k"], keys)
    else:
        d["taps"] = all_taps
        d["bank_rows"] = np.arange(len(all_taps))
        d["k"] = s.rows(all_taps)
    return d


def tip_crops(d: dict, offset: int, mode: str, rep: str = "diff",
              rows: np.ndarray | None = None) -> np.ndarray:
    """mode: none = appearance.py verbatim; hand = background blanked; bg = hand blanked (the
    direct legend-reading probe); silh = the mask alone, no texture at all."""
    jj = {o: j for j, o in enumerate(TIP_OFFSETS)}
    r = d["bank_rows"] if rows is None else rows

    def at(o):
        o = min(TIP_OFFSETS, key=lambda x: abs(x - o))
        a = np.asarray(d["tips"][r, jj[o]], dtype=np.float32) / 255.0
        if mode != "none":
            m = np.asarray(d["masks"][r, jj[o]], dtype=np.float32)
            a = m if mode == "silh" else a * (m if mode == "hand" else 1.0 - m)
        return a.reshape(len(a), 10, 1, TIP_CROP, TIP_CROP)

    base = at(offset)
    if rep == "single":
        return base
    return np.concatenate([base, base - at(min(TIP_OFFSETS))], axis=2)


def _y(d: dict, target: str) -> np.ndarray:
    return d["y"] if target == "key" else d["yf"]


# ------------------------------------------------------------------ LOSO / held-out
def run_fold(tr: list[dict], te: dict, target: str, offset: int, mode: str, seed: int,
             epochs: int, rep: str) -> dict:
    n_out = NA if target == "key" else N_FINGER
    Xtr = np.vstack([tip_crops(d, offset, mode, rep) for d in tr])
    ytr = np.concatenate([_y(d, target) for d in tr])
    m = ytr >= 0
    clf = fit_cnn(Xtr[m], ytr[m], n_out, seed=seed, epochs=epochs, arch="small")
    pp = cnn_proba(clf, tip_crops(te, offset, mode, rep), n_out)
    return {"pix": pp, "lm": lm_proba(target, tr, te, seed), "y": _y(te, target),
            "params": n_params(clf), "clf": clf}


def boot_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    r = np.random.RandomState(seed)
    idx = r.randint(0, len(correct), size=(n_boot, len(correct)))
    v = correct[idx].mean(1)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def _correct(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = y >= 0
    return (p[m].argmax(1) == y[m]).astype(float)


def cmd_loso(a) -> int:
    sids = list(TRAIN_SESSIONS)
    ds = {s: build_dataset(s) for s in sids + [HELDOUT_SESSION]}
    seeds = tuple(SEEDS[:a.seeds])
    for tgt in a.targets.split(","):
        warm_lm(tgt, sids, seeds, ds, HELDOUT_SESSION)
    alphas = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
    out = {}
    for tgt in a.targets.split(","):
        print(f"\n=== target={tgt} offset={a.offset} rep={a.rep} seeds={seeds} ===")
        print(f"{'mask':<6}{'LOSO pix':>15}{'LOSO lm':>15}{'LOSO fuse':>20}"
              f"{'HELD pix':>15}{'HELD fuse':>15}")
        for mode in a.modes.split(","):
            res = {"pix": [], "lm": []}
            fus = {x: [] for x in alphas}
            for seed in seeds:
                per, wts, perf = {"pix": [], "lm": []}, [], {x: [] for x in alphas}
                for held in sids:
                    tr = [ds[s] for s in sids if s != held]
                    r = run_fold(tr, ds[held], tgt, a.offset, mode, seed, a.epochs, a.rep)
                    for kk in ("pix", "lm"):
                        per[kk].append(score(r[kk], r["y"])["top1"])
                    for x in alphas:
                        perf[x].append(score(fuse(r["pix"], r["lm"], x), r["y"])["top1"])
                    wts.append(score(r["pix"], r["y"])["n"])
                w = np.array(wts, float)
                for kk in ("pix", "lm"):
                    res[kk].append(float(np.average(per[kk], weights=w)))
                for x in alphas:
                    fus[x].append(float(np.average(perf[x], weights=w)))
            ba = max(fus, key=lambda x: np.mean(fus[x]))
            held = {"pix": [], "lm": [], "fus": []}
            cor = {"pix": [], "fus": [], "lm": []}
            for seed in seeds:
                r = run_fold([ds[s] for s in sids], ds[HELDOUT_SESSION], tgt, a.offset, mode,
                             seed, a.epochs, a.rep)
                held["pix"].append(score(r["pix"], r["y"])["top1"])
                held["lm"].append(score(r["lm"], r["y"])["top1"])
                held["fus"].append(score(fuse(r["pix"], r["lm"], ba), r["y"])["top1"])
                cor["pix"].append(_correct(r["pix"], r["y"]))
                cor["lm"].append(_correct(r["lm"], r["y"]))
                cor["fus"].append(_correct(fuse(r["pix"], r["lm"], ba), r["y"]))
            ci = {kk: boot_ci(np.mean(cor[kk], axis=0)) for kk in cor}
            print(f"{mode:<6}{agg(res['pix']):>15}{agg(res['lm']):>15}"
                  f"{agg(fus[ba]) + f' a={ba}':>20}{agg(held['pix']):>15}{agg(held['fus']):>15}")
            print(f"      held CI  pix [{ci['pix'][0]:.3f},{ci['pix'][1]:.3f}]  "
                  f"lm [{ci['lm'][0]:.3f},{ci['lm'][1]:.3f}]  "
                  f"fuse [{ci['fus'][0]:.3f},{ci['fus'][1]:.3f}]  n={len(cor['pix'][0])}",
                  flush=True)
            out[f"{tgt}/{mode}"] = {"loso_pix": res["pix"], "loso_lm": res["lm"],
                                    "loso_fuse": fus[ba], "alpha": ba, "held_pix": held["pix"],
                                    "held_lm": held["lm"], "held_fus": held["fus"],
                                    "held_ci": ci}
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


# ------------------------------------------------------------------ CER on a held-out session
def _write_probs(taps: list[dict], p: np.ndarray, out: Path, topk: int = 8) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for n, tp in enumerate(taps):
            order = np.argsort(-p[n])[:topk]
            row = dict(tp)
            row["key_probs"] = {ALPHABET[c]: round(float(p[n, c]), 6) for c in order}
            fh.write(json.dumps(row) + "\n")


def _decode_cer(sid: str, probs_file: Path, obs: float, control: bool) -> float:
    from phase0.analysis.decode import main as decode_main
    import contextlib
    import io

    buf = io.StringIO()
    argv = ["decode", str(sess_path(sid)), "--taps", str(probs_file), "--spatial", "keyprobs",
            "--obs-weight", str(obs)] + (["--control"] if control else [])
    with contextlib.redirect_stdout(buf):
        decode_main(argv)
    line = [x for x in buf.getvalue().splitlines() if x.startswith("OVERALL")][-1]
    return float(line.split("CER=")[1].split()[0])


def _fit_predict(tr: list[dict], tgt: dict, mode: str, a, alpha: float) -> np.ndarray:
    """Mean-over-seeds pixel probs for every tap of `tgt`, optionally fused with pose."""
    pl = lm_proba("key", tr, tgt, 0) if alpha < 1.0 else None
    Xtr = np.vstack([tip_crops(d, a.offset, mode, a.rep) for d in tr])
    ytr = np.concatenate([d["y"] for d in tr])
    Xte = tip_crops(tgt, a.offset, mode, a.rep)
    ps = []
    for seed in tuple(SEEDS[:a.seeds]):
        clf = fit_cnn(Xtr, ytr, NA, seed=seed, epochs=a.epochs, arch="small")
        ps.append(cnn_proba(clf, Xte, NA))
    p = np.mean(ps, axis=0)
    return fuse(p, pl, alpha) if pl is not None else p


def cmd_cer(a) -> int:
    """obs-weight is tuned out-of-fold on 131629 only, exactly as appearance.py did."""
    sids = list(TRAIN_SESSIONS)
    ev = a.eval
    control = ev.endswith("kbd")
    lab = {s: build_dataset(s) for s in sids}
    full = {s: build_dataset(s, labelled=False) for s in (TUNE_SESSION, ev)}
    oof = [lab[s] for s in sids if s != TUNE_SESSION]
    all_tr = [lab[s] for s in sids]
    lm_proba("key", oof, full[TUNE_SESSION], 0)  # LightGBM must run before torch touches MPS
    lm_proba("key", all_tr, full[ev], 0)
    rows = []
    for mode in a.modes.split(","):
        for alpha in [float(x) for x in a.alphas.split(",")]:
            p = _fit_predict(oof, full[TUNE_SESSION], mode, a, alpha)
            f = MASKS / f"probs_{TUNE_SESSION}_{mode}_{alpha}.jsonl"
            _write_probs(full[TUNE_SESSION]["all_taps"], p, f)
            sweep = {w: _decode_cer(TUNE_SESSION, f, w, True) for w in
                     [float(x) for x in a.obs.split(",")]}
            bw = min(sweep, key=lambda w: sweep[w])
            p = _fit_predict(all_tr, full[ev], mode, a, alpha)
            f2 = MASKS / f"probs_{ev}_{mode}_{alpha}.jsonl"
            _write_probs(full[ev]["all_taps"], p, f2)
            cer = _decode_cer(ev, f2, bw, control)
            rows.append((mode, alpha, bw, sweep[bw], cer))
            print(f"mask={mode:<5} alpha={alpha:<5} obs*={bw} (tune CER {sweep[bw]:.3f}) "
                  f"-> {ev} CER {cer:.3f}   sweep={ {k: round(v,3) for k,v in sweep.items()} }",
                  flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    return 0


# ------------------------------------------------------------------ desk-internal
def cmd_desk(a) -> int:
    """Train and test the pixel model entirely inside the bare-desk session, held out by
    phrase. Labels come from adapt.py's EM alignment to the known phrase text."""
    from phase0.analysis import adapt as ad
    from phase0.analysis.decode import desk_segments

    sids = list(TRAIN_SESSIONS)
    lab = {s: build_dataset(s) for s in sids}
    d = build_dataset(DESK_SESSION, labelled=False)
    t = ad.Target(DESK_SESSION, TAPS_NAME, control=False)
    base = ad.coral_proba(t, Path("models/tap_pos.pkl"), sids)
    segs = [s for s in t.segs if len(s[1])]
    print(f"{DESK_SESSION}: {len(t.taps)} taps, {len(segs)} phrases, "
          f"{sum(len(x) for x, _ in segs)} characters")
    seeds = tuple(SEEDS[:a.seeds])
    Xall = {m: tip_crops(d, a.offset, m, a.rep) for m in a.modes.split(",")}
    res: dict = {}
    for fi, te in enumerate(ad.folds(len(segs), a.folds, a.seed)):
        tr = [segs[i] for i in range(len(segs)) if i not in set(te.tolist())]
        ev = [segs[i] for i in te]
        _, obs, _, par = ad.adapt(t, tr, base, a.iters, a.lam, a.l2, "em", verbose=False)
        q, q_ins, _, _ = ad.align_all(np.log(np.maximum(obs, ad.EPS)), tr, par)
        fit_idx = np.unique(np.concatenate([i for _, i in tr]))
        conf = (q[fit_idx].max(1) >= a.thr) & (q_ins[fit_idx] < 0.5)
        ytr = q[fit_idx].argmax(1)[conf]
        line = [f"fold{fi}(n={int(conf.sum())}/{len(fit_idx)})"]
        for name, p in (("base", base), ("em", obs)):
            res.setdefault(name, []).append(ad.cer_on(ev, p, a.beam)[0])
            line.append(f"{name}={res[name][-1]:.3f}")
        for mode in a.modes.split(","):
            ps = []
            for seed in seeds:
                clf = fit_cnn(Xall[mode][fit_idx][conf], ytr, NA, seed=seed, epochs=a.epochs,
                              arch="small")
                ps.append(cnn_proba(clf, Xall[mode], NA))
            pix = np.mean(ps, axis=0)
            for nm, pv in ((f"pix-{mode}", pix), (f"fus-{mode}", fuse(pix, obs, a.alpha))):
                res.setdefault(nm, []).append(ad.cer_on(ev, pv, a.beam)[0])
                line.append(f"{nm}={res[nm][-1]:.3f}")
        print("  " + "  ".join(line), flush=True)
    print(f"\n{'method':<14}{'mean fold CER':>15}{'sd':>8}{'95% CI':>18}")
    for m, v in res.items():
        v = np.array(v)
        lo, hi = boot_ci(v, seed=1)
        print(f"{m:<14}{v.mean():>15.3f}{v.std(ddof=1):>8.3f}   [{lo:.3f},{hi:.3f}]")
    if a.json:
        Path(a.json).write_text(json.dumps({k: list(map(float, v)) for k, v in res.items()},
                                           indent=1))
    return 0


# ------------------------------------------------------------------ visual check
def cmd_peek(a) -> int:
    import cv2

    d = build_dataset(a.session)
    r = d["bank_rows"][: a.n]
    rowimgs = []
    for mode in ("none", "hand", "bg"):
        x = tip_crops(d, a.offset, mode, "single", rows=r)[:, :, 0]
        rowimgs.append(np.concatenate([np.concatenate(list(x[i]), axis=1) for i in range(len(r))],
                                      axis=0))
    img = np.concatenate(rowimgs, axis=1)
    out = MASKS / f"peek_{a.session}.png"
    cv2.imwrite(str(out), (255 * np.clip(img, 0, 1)).astype(np.uint8))
    print(f"wrote {out} (columns: unmasked | hand-only | background-only)")
    return 0


def cmd_mask(a) -> int:
    for sid in a.sessions:
        build_mask_bank(sid, force=a.force)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.masked", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mask")
    m.add_argument("--sessions", nargs="+",
                   default=list(TRAIN_SESSIONS) + [HELDOUT_SESSION, DESK_SESSION])
    m.add_argument("--force", action="store_true")
    m.set_defaults(fn=cmd_mask)

    def common(p):
        p.add_argument("--modes", default="none,hand,bg")
        p.add_argument("--offset", type=int, default=4)
        p.add_argument("--rep", default="diff")
        p.add_argument("--epochs", type=int, default=40)
        p.add_argument("--seeds", type=int, default=3)
        p.add_argument("--json", default=None)

    l = sub.add_parser("loso")
    common(l)
    l.add_argument("--targets", default="key,finger")
    l.set_defaults(fn=cmd_loso)

    c = sub.add_parser("cer")
    common(c)
    c.add_argument("--alphas", default="0.75,1.0")
    c.add_argument("--obs", default="1.0,1.5,2.0,2.5,3.0")
    c.add_argument("--eval", default=HELDOUT_SESSION)
    c.set_defaults(fn=cmd_cer)

    k = sub.add_parser("desk")
    common(k)
    k.add_argument("--folds", type=int, default=5)
    k.add_argument("--seed", type=int, default=0)
    k.add_argument("--iters", type=int, default=6)
    k.add_argument("--lam", type=float, default=0.5)
    k.add_argument("--l2", type=float, default=4.0)
    k.add_argument("--thr", type=float, default=0.5)
    k.add_argument("--alpha", type=float, default=0.5)
    k.add_argument("--beam", type=int, default=30)
    k.set_defaults(fn=cmd_desk)

    p = sub.add_parser("peek")
    p.add_argument("--session", default=TUNE_SESSION)
    p.add_argument("--offset", type=int, default=4)
    p.add_argument("--n", type=int, default=6)
    p.set_defaults(fn=cmd_peek)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
