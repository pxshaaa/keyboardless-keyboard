"""How We Type mocap re-projected through a virtual camera at any elevation, as seqctc Streams.

Gives the project a camera-angle knob for the HWT pretraining corpus (today's streams are effectively
top-down orthographic) and measures, on REAL ten-finger typing, what tilting the camera costs and buys:
keystroke-vs-hover separability, key-identification accuracy and fingertip occlusion.

How We Type (Feit et al., CHI'16) is CC-BY-NC-4.0: research measurement only, never shippable.
Run: python -m phase0.analysis.hwt45 {build | measure | pretrain | evalcer}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# NB: contact_angle imports lightgbm at module level, and lightgbm's OpenMP runtime segfaults when it is
# loaded in the same process as torch/MPS on this machine, so it is imported lazily inside cmd_measure only.

CACHE = Path(".cache/hwt")
RES = Path("results/angle45")
TIPS = (4, 8, 12, 16, 20)
ELS = (90, 60, 45, 30)
WORK_M = 0.227                     # camera-to-palm distance of the real rig
F_PX = 538.3086                    # 136 px knuckle at 227 mm, 1280x720 (results/angle rig)
IMW, IMH = 1280, 720
PIX_JITTER = 0.8                   # px, the landmark-noise model used throughout the angle study (contact_angle.PIX_JITTER)
UP = np.array([0.0, 0.0, -1.0])    # world z is depth-below-desk, so "up" is -z


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def cam_at(T: np.ndarray, el_deg: float, work: float = WORK_M, f_px: float = F_PX) -> dict:
    """Front-azimuth camera at `el_deg` above the desk, `work` from the hand centre T.
    Image axes chosen so el=90 reproduces How We Type's own convention: u = forward, v = rightward."""
    e = np.radians(el_deg)
    d = unit(np.cos(e) * np.array([1.0, 0.0, 0.0]) + np.sin(e) * UP)   # target -> camera
    C = T + work * d
    zc = -d
    xc = unit(np.cross(np.array([0.0, 1.0, 0.0]), zc))
    yc = np.cross(zc, xc)
    return {"C": C, "R": np.stack([xc, yc, zc]), "f": f_px, "cx": IMW / 2.0, "cy": IMH / 2.0, "el": el_deg}


def project(P: np.ndarray, cam: dict) -> tuple[np.ndarray, np.ndarray]:
    Pc = (P - cam["C"]) @ cam["R"].T
    z = Pc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam["f"] * Pc[..., 0] / z + cam["cx"]
        v = cam["f"] * Pc[..., 1] / z + cam["cy"]
    uv = np.stack([u, v], -1)
    uv[z <= 1e-6] = np.nan
    return uv, Pc


def hand_target(J: np.ndarray) -> np.ndarray:
    ok = np.isfinite(J[:, :, 0, 0])
    pc = [np.nanmedian(J[ok[:, h], h][:, [0, 5, 9, 13, 17]].mean(1), 0) for h in (0, 1) if ok[:, h].any()]
    return np.mean(pc, 0)


def stream_at(st, el: float, rng, work: float = WORK_M) -> tuple:
    """-> (A, M) in seqctc's normalised convention, for the HWT stream `st` seen from elevation `el`."""
    from phase0.analysis import angle_hwt as AH
    from phase0.analysis.seqctc import normalise
    J, info = AH.world_joints(st)
    T = hand_target(J)
    cam = cam_at(T, el, work)
    raw = np.full(J.shape, np.nan, np.float64)
    for h in (0, 1):
        uv, Pc = project(J[:, h], cam)
        uv = uv + rng.normal(0, PIX_JITTER, uv.shape)
        raw[:, h, :, 0] = uv[..., 0]
        raw[:, h, :, 1] = uv[..., 1]
        raw[:, h, :, 2] = (Pc[..., 2] - Pc[:, :1, 2]) * 1000.0      # wrist-relative depth, mm (recipe has zfeat=False)
    A, M = normalise(raw.astype(np.float32))
    return A, M, cam, T, info


def cmd_build(a) -> int:
    from phase0.analysis import howwetype as HWT
    from phase0.analysis.seqctc import Stream
    sts = HWT.streams()
    for el in [float(x) for x in a.els.split(",")]:
        out = CACHE / f"streams_el{int(el):02d}"
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(a.seed)
        t0 = time.time()
        for st in sts:
            f = out / f"{st.name}.npz"
            if f.exists() and not a.force:
                continue
            A, M, cam, T, info = stream_at(st, el, rng)
            Stream(st.name, st.t, A, M, st.kt, st.ks).save(f)
        print(f"el{int(el):02d}: {len(sts)} streams -> {out} ({time.time()-t0:.0f}s)", flush=True)
    return 0


# ------------------------------------------------------------------ measurements on real typing
def press_events(J: np.ndarray, t: np.ndarray, kt: np.ndarray, ks: np.ndarray):
    """Key-down frames with the pressing fingertip = lowest tip at that frame -> (rows, hand, finger, key)."""
    H = -J[:, :, list(TIPS), 2] * 1000.0
    rows = np.clip(np.searchsorted(t, kt), 0, len(t) - 1)
    flat = H[rows].reshape(len(rows), -1)
    ok = np.isfinite(flat).any(1)
    low = np.nanargmin(np.where(np.isfinite(flat), flat, np.inf)[ok], 1)
    return rows[ok], low // 5, low % 5, ks[ok], np.where(ok)[0]


def hover_rows(t: np.ndarray, kt: np.ndarray, rng, n: int, clear: float = 0.15) -> np.ndarray:
    """Frames at least `clear` from any key-down (the negatives for keystroke-vs-hover)."""
    d = np.abs(t[:, None] - kt[None, :]).min(1) if len(kt) else np.full(len(t), np.inf)
    cand = np.where(d > clear)[0]
    return rng.choice(cand, min(n, len(cand)), replace=False) if len(cand) else cand


def cmd_measure(a) -> int:
    """Per camera elevation, on real ten-finger typing: key-ID accuracy, keystroke-vs-hover AUC, occlusion."""
    import lightgbm  # noqa: F401
    from sklearn.metrics import roc_auc_score
    from phase0.analysis import contact_angle as CA
    from phase0.analysis import angle_hwt as AH, howwetype as HWT, taps_gb as gb
    from phase0.analysis.seqctc import SYMS
    RES.mkdir(parents=True, exist_ok=True)
    els = [float(x) for x in a.els.split(",")]
    sts = HWT.streams()[: a.sessions]
    rng = np.random.default_rng(0)
    data = {el: {"Xk": [], "yk": [], "pk": [], "Xc": [], "yc": [], "pc": [], "occ": [], "occn": [],
                 "occ_press": [], "occ_press_n": [], "loc": []} for el in els}
    t0 = time.time()
    for n, st in enumerate(sts):
        J, info = AH.world_joints(st)
        T = hand_target(J)
        rows, hand, fing, keys, _ = press_events(J, st.t, st.kt, st.ks)
        letters = np.array([SYMS[k] if 0 <= k < len(SYMS) else "?" for k in keys])
        keep = np.isin(letters, list(SYMS[1:28]))
        rows, hand, fing, keys = rows[keep], hand[keep], fing[keep], keys[keep]
        hov = hover_rows(st.t, st.kt, rng, n=len(rows))
        for el in els:
            cam = cam_at(T, el)
            raw = np.full(J.shape, np.nan, np.float64)
            P7 = np.full((len(J), 2, 21, 7), np.nan)
            occ = np.zeros((len(J), 2, 5), bool)
            for h in (0, 1):
                uv, Pc = project(J[:, h], cam)
                uvn = uv + rng.normal(0, PIX_JITTER, uv.shape)
                raw[:, h, :, :2] = uvn
                raw[:, h, :, 2] = (Pc[..., 2] - Pc[:, :1, 2]) * 1000.0
                P7[:, h, :, :2] = uvn
                P7[:, h, :, 2] = np.where(np.isfinite(uvn[..., 0]), 0.99, np.nan)
                _, oo = CA.occluded(J[:, h], cam)
                occ[:, h] = oo | AH.occluded_cross(J[:, h], J[:, 1 - h], cam)
                xy = CA.unproject_to_plane(uvn[:, list(TIPS)], cam)
                data[el]["loc"].append(1000.0 * np.linalg.norm(xy - J[:, h, list(TIPS), :2], axis=-1)[rows, fing][hand == h]
                                       if (hand == h).any() else np.zeros(0))
            from phase0.analysis.seqctc import normalise
            A, M = normalise(raw.astype(np.float32))
            flat = np.concatenate([A[..., :2].reshape(len(A), -1), M.astype(np.float32)], 1)
            data[el]["Xk"].append(flat[rows]); data[el]["yk"].append(keys); data[el]["pk"].append(np.full(len(rows), n))
            X = gb.assemble(gb.build_groups(P7), gb.GROUPS).astype(np.float32)
            idx = np.concatenate([rows, hov])
            data[el]["Xc"].append(X[idx])
            data[el]["yc"].append(np.concatenate([np.ones(len(rows), bool), np.zeros(len(hov), bool)]))
            data[el]["pc"].append(np.full(len(idx), n))
            fin = np.isfinite(J[:, :, list(TIPS), 0])
            data[el]["occ"].append(int(occ[fin].sum())); data[el]["occn"].append(int(fin.sum()))
            data[el]["occ_press"].append(int(occ[rows, hand, fing].sum())); data[el]["occ_press_n"].append(len(rows))
        print(f"  [{n+1}/{len(sts)}] {st.name} keys={len(rows)} {time.time()-t0:.0f}s", flush=True)

    for el in els:                      # cache features so the scoring can be redone without re-projecting
        d = data[el]
        np.savez_compressed(Path(".cache/angle45") / f"feat_el{int(el):02d}.npz",
                            Xk=np.vstack(d["Xk"]), yk=np.concatenate(d["yk"]), pk=np.concatenate(d["pk"]),
                            Xc=np.vstack(d["Xc"]), yc=np.concatenate(d["yc"]), pc=np.concatenate(d["pc"]),
                            occ=np.array(d["occ"]), occn=np.array(d["occn"]),
                            occ_press=np.array(d["occ_press"]), occ_press_n=np.array(d["occ_press_n"]),
                            loc=np.concatenate([x for x in d["loc"] if len(x)]))
    out = {"sessions": len(sts), "els": els, "pix_jitter": PIX_JITTER, "work_mm": 1000 * WORK_M,
           "f_px": F_PX, "arms": {}}
    for el in els:
        d = data[el]
        Xk, yk, pk = np.vstack(d["Xk"]), np.concatenate(d["yk"]), np.concatenate(d["pk"])
        Xc, yc, pc = np.vstack(d["Xc"]), np.concatenate(d["yc"]), np.concatenate(d["pc"])
        u = np.unique(pk)
        folds = [np.isin(pk, u[i::a.folds]) for i in range(a.folds)]
        pred = np.zeros(len(yk), np.int64)
        for te in folds:
            m = gb.make_model("lgbm", random_state=0, n_jobs=4, n_estimators=a.trees, objective="multiclass",
                              num_class=int(yk.max()) + 1, min_child_samples=20)
            m.fit(Xk[~te], yk[~te])
            pred[te] = m.predict(Xk[te])
        acc = float((pred == yk).mean())
        uc = np.unique(pc)
        fc = [np.isin(pc, uc[i::a.folds]) for i in range(a.folds)]
        sc = np.zeros(len(yc))
        for te in fc:
            m = gb.make_model("lgbm", random_state=0, n_jobs=4, n_estimators=a.trees, min_child_samples=20)
            m.fit(Xc[~te], yc[~te])
            sc[te] = m.predict_proba(Xc[te])[:, 1]
        auc = float(roc_auc_score(yc, sc))
        # per-participant spread, so the CI is over people not frames
        accs = [float((pred[pk == p] == yk[pk == p]).mean()) for p in u]
        aucs = [float(roc_auc_score(yc[pc == p], sc[pc == p])) for p in uc if len(np.unique(yc[pc == p])) == 2]
        loc = np.concatenate([x for x in d["loc"] if len(x)])
        out["arms"][f"el{int(el):02d}"] = {
            "el": el, "n_keys": int(len(yk)), "n_classes": int(len(np.unique(yk))),
            "key_id_top1": acc, "key_id_per_participant_mean": float(np.mean(accs)),
            "key_id_per_participant_sd": float(np.std(accs, ddof=1)),
            "keystroke_vs_hover_auc": auc, "kvh_per_participant_mean": float(np.mean(aucs)),
            "kvh_per_participant_sd": float(np.std(aucs, ddof=1)),
            "occlusion_all_tips": sum(d["occ"]) / max(sum(d["occn"]), 1),
            "occlusion_pressing_tip": sum(d["occ_press"]) / max(sum(d["occ_press_n"]), 1),
            "loc_err_mm_median": float(np.nanmedian(loc)), "loc_err_mm_p90": float(np.nanpercentile(loc, 90)),
            "key_id_per_participant": accs, "kvh_per_participant": aucs,
            "participants": [int(x) for x in u]}
        r = out["arms"][f"el{int(el):02d}"]
        print(f"el{int(el):02d}: key-ID {acc:.3f} (per-person {r['key_id_per_participant_mean']:.3f}+-{r['key_id_per_participant_sd']:.3f}) "
              f"kvh AUC {auc:.3f} (+-{r['kvh_per_participant_sd']:.3f}) occ all {r['occlusion_all_tips']:.3f} press {r['occlusion_pressing_tip']:.3f} "
              f"loc {r['loc_err_mm_median']:.1f}/{r['loc_err_mm_p90']:.1f} mm", flush=True)
    base = out["arms"].get("el90")
    if base:
        out["paired_vs_el90"] = {}
        for k, r in out["arms"].items():
            if k == "el90":
                continue
            dk = np.array(r["key_id_per_participant"]) - np.array(base["key_id_per_participant"])
            da = np.array(r["kvh_per_participant"]) - np.array(base["kvh_per_participant"])
            out["paired_vs_el90"][k] = {
                "d_key_id_mean": float(dk.mean()), "d_key_id_sd": float(dk.std(ddof=1)),
                "d_key_id_se": float(dk.std(ddof=1) / np.sqrt(len(dk))),
                "d_key_id_participants_better": int((dk > 0).sum()), "n_participants": int(len(dk)),
                "d_kvh_mean": float(da.mean()), "d_kvh_se": float(da.std(ddof=1) / np.sqrt(len(da))),
                "d_kvh_participants_better": int((da > 0).sum())}
            q = out["paired_vs_el90"][k]
            print(f"paired {k} - el90: key-ID {q['d_key_id_mean']:+.3f} +- {q['d_key_id_se']:.3f} SE "
                  f"({q['d_key_id_participants_better']}/{q['n_participants']} people better) | "
                  f"kvh {q['d_kvh_mean']:+.3f} +- {q['d_kvh_se']:.3f} SE", flush=True)
    (RES / "hwt_angle_metrics.json").write_text(json.dumps(out, indent=1))
    return 0


# ------------------------------------------------------------------ pretraining at an angle
def hwt_streams_at(el: int):
    from phase0.analysis.seqctc import Stream
    d = CACHE / f"streams_el{int(el):02d}"
    return [Stream.load(p.stem, p) for p in sorted(d.glob("*.npz"))]


def split(sts, holdout: int):
    te = [s for i, s in enumerate(sts) if i % holdout == 0]
    tr = [s for i, s in enumerate(sts) if i % holdout != 0]
    return tr, te


def cmd_pretrain(a) -> int:
    """Train the shipped seqctc recipe on HWT at one elevation; evaluate CER on held-out participants."""
    import torch  # noqa: F401
    from phase0.analysis.seqctc import Cfg, KbdSource, build, train, eval_windows, sym_text, greedy_cer
    RES.mkdir(parents=True, exist_ok=True)
    out = CACHE / "pre45"
    out.mkdir(parents=True, exist_ok=True)
    el = int(a.el)
    sts = hwt_streams_at(el)
    tr, te = split(sts, a.holdout)
    sets = {}
    for s in te[: a.eval_sessions]:
        w = eval_windows(s)[: a.eval_windows]
        sets[s.name] = (s, w, [sym_text(x[2]) for x in w])
    for seed in [int(x) for x in a.seeds.split(",")]:
        f = out / f"hwt_el{el:02d}_s{seed}.pt"
        if f.exists() and not a.force:
            print(f"skip {f}", flush=True)
            continue
        cfg = Cfg(steps=a.steps, lr=a.lr, seed=seed)
        m = train(build(cfg), [(KbdSource(s), 1.0) for s in tr], cfg, log=f"[hwt el{el:02d} s{seed}]",
                  eval_fn=lambda mm: greedy_cer(mm, sets), ckpt=f.with_suffix(".ckpt"))
        torch.save({"cfg": cfg.__dict__, "sd": m.state_dict()}, f)
        print(f"wrote {f}", flush=True)
    return 0


def cmd_evalcer(a) -> int:
    """CER of each trained angle model on its own held-out participants (same events, same windows)."""
    import torch
    from phase0.analysis.seqctc import (Cfg, build, eval_windows, sym_text, infer, greedy, rows_cer)
    RES.mkdir(parents=True, exist_ok=True)
    res = {"holdout": a.holdout, "arms": {}}
    for el in [int(x) for x in a.els.split(",")]:
        sts = hwt_streams_at(el)
        tr, te = split(sts, a.holdout)
        for seed in [int(x) for x in a.seeds.split(",")]:
            f = CACHE / "pre45" / f"hwt_el{el:02d}_s{seed}.pt"
            if not f.exists():
                continue
            ck = torch.load(f, map_location="cpu", weights_only=False)
            m = build(Cfg(**{k: v for k, v in ck["cfg"].items() if k != "aug"}))
            m.load_state_dict(ck["sd"])
            per = {}
            for s in te[: a.eval_sessions]:
                w = eval_windows(s)[: a.eval_windows]
                refs = [sym_text(x[2]) for x in w]
                hyp = [greedy(lp) for lp in infer(m, s, w)]
                per[s.name] = float(rows_cer(list(zip(refs, hyp))))
            key = f"el{el:02d}_s{seed}"
            res["arms"][key] = {"per_session_cer": per, "mean_cer": float(np.mean(list(per.values()))),
                                "n_sessions": len(per)}
            print(f"{key}: mean CER {res['arms'][key]['mean_cer']:.3f} over {len(per)} held-out participants", flush=True)
    (RES / "hwt_pretrain_cer.json").write_text(json.dumps(res, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.hwt45", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build"); s.add_argument("--els", default="90,60,45,30"); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--force", action="store_true"); s.set_defaults(fn=cmd_build)
    s = sub.add_parser("measure"); s.add_argument("--els", default="90,60,45,30"); s.add_argument("--sessions", type=int, default=30)
    s.add_argument("--folds", type=int, default=5); s.add_argument("--trees", type=int, default=200); s.set_defaults(fn=cmd_measure)
    s = sub.add_parser("pretrain"); s.add_argument("--el", type=int, required=True); s.add_argument("--seeds", default="0")
    s.add_argument("--steps", type=int, default=1500); s.add_argument("--lr", type=float, default=2e-3)
    s.add_argument("--holdout", type=int, default=5); s.add_argument("--eval-sessions", type=int, default=3)
    s.add_argument("--eval-windows", type=int, default=40); s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_pretrain)
    s = sub.add_parser("evalcer"); s.add_argument("--els", default="90,45"); s.add_argument("--seeds", default="0")
    s.add_argument("--holdout", type=int, default=5); s.add_argument("--eval-sessions", type=int, default=6)
    s.add_argument("--eval-windows", type=int, default=60); s.set_defaults(fn=cmd_evalcer)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
