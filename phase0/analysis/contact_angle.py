"""EgoPressure MANO joints re-projected through a virtual camera at any elevation/azimuth,
scored with contact_ego.cmd_press_hover's protocol. EgoPressure is CC-BY-NC: research only.
Run: python -m phase0.analysis.contact_angle {events | sweep | shadow}"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import lightgbm  # noqa: F401

from phase0.analysis import taps_gb as gb
from phase0.analysis import contact_ego as ce

OUT = Path(".cache/contact_angle")
RES = Path("results/angle")
TIPS = ce.TIPS                       # (4, 8, 12, 16, 20)
PATTERNS = ("*index_press_high*", "*index_press_low*", "*index_press_no-contact*")

# f is set so a median EgoPressure hand subtends our rig's 136 px across knuckles (joints
# 5..17) at its 227 mm working distance, matching the virtual camera's angular resolution.
IMW, IMH = 1280, 720
WORK_MM = 227.0
KNUCKLE_PX = 136.0
PIX_JITTER = 0.8                      # px, MediaPipe-like landmark noise (as in contact_ego)
OCC_RADIUS_MM = 9.0                   # finger half-thickness for the capsule occlusion test
PALM_H_MM = 39.7                      # measured median palm-centroid height above the pad

ELEVATIONS = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90)
SWEEP_ELEV = {"front": ELEVATIONS, "side": (0, 20, 40, 60, 90)}
AZIMUTHS = {"front": 0.0, "side": 90.0}   # 0 deg = beyond the fingertips, looking back at the hand

# MANO bones (+ palm cross-links) for the occlusion test
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10),
         (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18),
         (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]


def cam_name(el: float, az: str) -> str:
    return f"el{int(el):02d}_{az}"


def hand_forward(J: np.ndarray) -> np.ndarray:
    """Unit horizontal direction the fingers point in world (pad) coordinates."""
    ok = np.isfinite(J[:, 0, 0])
    f = np.nanmedian(J[ok, 9, :2] - J[ok, 0, :2], 0)
    n = np.linalg.norm(f)
    return f / n if n > 1e-9 else np.array([0.0, -1.0])


def virtual_cam(J: np.ndarray, el_deg: float, az_deg: float, f_px: float) -> dict:
    """Camera WORK_MM from the median palm centroid. World z=0 is the desk, height is -z.
    elevation 0 = along the desk, 90 = straight down; azimuth 0 = facing the fingertips."""
    ok = np.isfinite(J[:, 0, 0])
    target = np.nanmedian(J[np.ix_(ok, [0, 5, 9, 13, 17])].mean(1), 0)     # m
    up = np.array([0.0, 0.0, -1.0])                                        # world "up"
    fwd = hand_forward(J)
    f3 = np.array([fwd[0], fwd[1], 0.0])
    side = np.cross(up, f3)
    side /= np.linalg.norm(side)
    e, a = np.radians(el_deg), np.radians(az_deg)
    d = np.cos(e) * (np.cos(a) * f3 + np.sin(a) * side) + np.sin(e) * up   # target -> camera
    d /= np.linalg.norm(d)
    C = target + (WORK_MM / 1000.0) * d
    zc = -d                                                                # optical axis
    xc = np.cross(up, zc)
    if np.linalg.norm(xc) < 1e-6:                    # straight down: pick a stable roll
        xc = np.cross(f3, zc)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.stack([xc, yc, zc])                       # world -> camera rotation
    return {"R": R, "C": C, "f": f_px, "cx": IMW / 2.0, "cy": IMH / 2.0,
            "el": el_deg, "az": az_deg}


def vproject(J: np.ndarray, cam: dict) -> tuple[np.ndarray, np.ndarray]:
    """world metres -> (uv px [...,2], camera-frame mm [...,3])."""
    Pc = (J - cam["C"]) @ cam["R"].T * 1000.0
    z = np.where(np.abs(Pc[..., 2]) < 1e-6, np.nan, Pc[..., 2])
    u = cam["f"] * Pc[..., 0] / z + cam["cx"]
    v = cam["f"] * Pc[..., 1] / z + cam["cy"]
    return np.stack([u, v], -1), Pc


def unproject_to_plane(uv: np.ndarray, cam: dict) -> np.ndarray:
    """pixel -> (x, y) where its ray meets the desk plane z = 0, in metres."""
    d = np.stack([(uv[..., 0] - cam["cx"]) / cam["f"], (uv[..., 1] - cam["cy"]) / cam["f"],
                  np.ones(uv.shape[:-1])], -1)
    dw = d @ cam["R"]                                    # camera -> world
    t = -cam["C"][2] / np.where(np.abs(dw[..., 2]) < 1e-9, np.nan, dw[..., 2])
    return cam["C"][:2] + t[..., None] * dw[..., :2]


def occluded(J: np.ndarray, cam: dict, r_mm: float = OCC_RADIUS_MM) -> tuple[np.ndarray, np.ndarray]:
    """-> ([F,5] any-part, [F,5] other-finger-or-palm only) tip hidden behind a bone of the
    same hand (capsule, r_mm). No forearm / other hand / desk edge, so it is a lower bound."""
    P = (J - cam["C"]) * 1000.0                          # camera-centred world mm
    F = len(J)
    out = np.zeros((F, 5), bool)
    oth = np.zeros((F, 5), bool)
    for k, tip in enumerate(TIPS):
        chain = set(range(tip - 3, tip + 1)) | {0}
        T = P[:, tip]
        dT = np.linalg.norm(T, axis=1)
        u = T / np.where(dT[:, None] < 1e-9, np.nan, dT[:, None])
        for (i, j) in BONES:
            if i == tip or j == tip:
                continue
            A, B = P[:, i], P[:, j]
            own = i in chain and j in chain
            for w in np.linspace(0.0, 1.0, 7):           # 7 samples is enough at r_mm = 9
                Q = A + w * (B - A)
                s = (Q * u).sum(1)                       # depth along the line of sight
                perp = np.linalg.norm(Q - s[:, None] * u, axis=1)
                hit = (perp < r_mm) & (s > 0) & (s < dT - r_mm)
                out[:, k] |= hit
                if not own:
                    oth[:, k] |= hit
    bad = ~np.isfinite(J[:, 0, 0])
    out[bad] = False
    oth[bad] = False
    return out, oth


def ours_params(uv: np.ndarray, side: str) -> dict:
    """to_ours' similarity transform, derived ONCE from the top-down view and reused at every
    angle so foreshortening is preserved instead of being renormalised away per camera."""
    o = ce.OURS[side]
    mir = ce._parity(uv, side) != ce.OUR_PARITY[side]
    if mir:
        uv = uv * np.array([1.0, -1.0])
    f = np.nanmedian(uv[:, 9] - uv[:, 0], 0)
    ang = np.arctan2(o["dir"][1], o["dir"][0]) - np.arctan2(f[1], f[0])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    span = np.nanmedian(np.linalg.norm(uv[:, 9] - uv[:, 0], axis=1))
    return {"mir": mir, "R": R, "s": o["span"] / span, "w0": np.nanmedian(uv[:, 0], 0),
            "wrist": o["wrist"]}


def apply_ours(uv: np.ndarray, prm: dict) -> np.ndarray:
    """-> P-like [F,21,7] (x, y, conf only) under a fixed similarity transform."""
    if prm["mir"]:
        uv = uv * np.array([1.0, -1.0])
    out = np.full(uv.shape[:-1] + (7,), np.nan, np.float32)
    out[..., :2] = (uv - prm["w0"]) @ prm["R"].T * prm["s"] + prm["wrist"]
    out[..., 2] = np.where(np.isfinite(out[..., 0]), 0.99, np.nan)
    return out


# ---- event extraction --------------------------------------------------------------------
def _descents(H: np.ndarray, C: np.ndarray, prom_mm: float = 8.0):
    """Same rule as contact_ego.descent_events, per sequence, but also returns the finger."""
    from scipy.signal import find_peaks
    rows, fing, lab, hmm = [], [], [], []
    for k in range(5):
        h = H[:, k]
        ok = np.isfinite(h)
        if ok.sum() < 30:
            continue
        hh = np.where(ok, h, np.nanmax(h))
        pk, _ = find_peaks(-hh, prominence=prom_mm, distance=20)
        for i in pk:
            if not (0 <= hh[i] <= 60):
                continue
            rows.append(i)
            fing.append(k)
            lab.append(C[max(0, i - 4):i + 5, k].max() >= ce.CONTACT_THR)
            hmm.append(hh[i])
    return np.array(rows, int), np.array(fing, int), np.array(lab, bool), np.array(hmm, float)


def seq_list() -> list[Path]:
    ds = []
    for p in PATTERNS:
        ds += ce.seq_dirs(p)
    return sorted(set(ds))


def knuckle_mm() -> float:
    f = OUT / "knuckle.json"
    if f.exists():
        return json.loads(f.read_text())["knuckle_mm"]
    vals = []
    for d in seq_list():
        s = ce.load_seq(d)
        if s is None:
            continue
        J = s["J"]
        ok = np.isfinite(J[:, 0, 0])
        if ok.sum():
            vals.append(1000.0 * np.nanmedian(np.linalg.norm(J[ok, 5] - J[ok, 17], axis=1)))
    OUT.mkdir(parents=True, exist_ok=True)
    v = float(np.median(vals))
    f.write_text(json.dumps({"knuckle_mm": v, "n_seq": len(vals),
                             "per_seq_p10_p90": list(np.percentile(vals, [10, 90]))}, indent=1))
    return v


def cmd_events(a) -> int:
    """One pass over the index_press sequences: every virtual camera, descent-bottom rows
    only, plus the geometry the occlusion/localisation/shadow parts need."""
    OUT.mkdir(parents=True, exist_ok=True)
    kn = knuckle_mm()
    f_px = KNUCKLE_PX * WORK_MM / kn
    print(f"median EgoPressure knuckle (5..17) = {kn:.1f} mm -> virtual f = {f_px:.1f} px "
          f"({IMW}x{IMH}, hfov {2*np.degrees(np.arctan(IMW/2/f_px)):.0f} deg)", flush=True)
    cams = [(cam_name(el, az), el, AZIMUTHS[az]) for az in AZIMUTHS for el in ELEVATIONS]
    rng = np.random.default_rng(0)
    store = {n: [] for n, _, _ in cams}
    store["ref_cam4"] = []
    store["el90_front_postjit"] = []
    meta = {"lab": [], "hmm": [], "part": [], "gest": [], "fing": [], "seq": []}
    geo = {"tip_xyz": [], "occ": {n: [] for n, _, _ in cams},
           "occo": {n: [] for n, _, _ in cams},
           "loc_err": {n: [] for n, _, _ in cams}, "cams": {}}
    t0 = time.time()
    for n, d in enumerate(seq_list()):
        s = ce.load_seq(d)
        if s is None:
            continue
        t30 = (s["frame"] - s["frame"][0]) / 30.0
        ff = ce.finger_force(s["J"], s["force"])
        t60, J60 = ce.resample60(t30, s["J"])
        _, ff60 = ce.resample60(t30, ff)
        H = -J60[:, list(TIPS), 2] * 1000.0                     # tip height above pad, mm
        C = np.nan_to_num(ff60[:, :5])
        rows, fing, lab, hmm = _descents(H, C)
        if len(rows) == 0:
            continue
        slot = 0 if s["side"] == "left" else 1
        meta["lab"].append(lab); meta["hmm"].append(hmm); meta["fing"].append(fing)
        meta["part"].append(np.full(len(rows), int(s["participant"][2:])))
        meta["gest"].append(np.array([s["name"].split("_", 2)[2]] * len(rows)))
        meta["seq"].append(np.full(len(rows), n))
        geo["tip_xyz"].append(J60[rows[:, None], np.array(TIPS)[fing][:, None]][:, 0])

        prm = ours_params(vproject(s["J"], virtual_cam(s["J"], 90, 0.0, f_px))[0], s["side"])

        def feats(P1) -> np.ndarray:
            _, P60 = ce.resample60(t30, P1)
            P = np.full((len(t60), 2, 21, 7), np.nan)
            P[:, slot] = P60
            return gb.assemble(gb.build_groups(P), gb.GROUPS).astype(np.float32)[rows]

        for name, el, az in cams:
            cam = virtual_cam(s["J"], el, az, f_px)
            uv, Pc = vproject(s["J"], cam)
            store[name].append(feats(apply_ours(uv + rng.normal(0, PIX_JITTER, uv.shape), prm)))
            geo["cams"].setdefault(name, {"el": el, "az": az})
            # occlusion + planar localisation, measured at the descent bottoms
            oc, oco = occluded(s["J"], cam)
            _, oc60 = ce.resample60(t30, np.stack([oc, oco], 1).astype(np.float32))
            geo["occ"][name].append(oc60[rows, 0, fing] > 0.5)
            geo["occo"][name].append(oc60[rows, 1, fing] > 0.5)
            uvn = uv + rng.normal(0, PIX_JITTER, uv.shape)
            xy = unproject_to_plane(uvn, cam)
            _, xy60 = ce.resample60(t30, xy)
            tipxy = J60[rows[:, None], np.array(TIPS)[fing][:, None]][:, 0, :2]
            est = xy60[rows[:, None], np.array(TIPS)[fing][:, None]][:, 0]
            geo["loc_err"][name].append(1000.0 * np.linalg.norm(est - tipxy, axis=1))
        # reproduction check: the real EgoPressure overhead camera, exactly as contact_ego
        uv4, Pc4 = ce.project(s["J"], s["cal"])
        store["ref_cam4"].append(feats(ce.to_ours(uv4, Pc4, s["side"], "2d", rng)))
        uv9 = vproject(s["J"], virtual_cam(s["J"], 90, 0.0, f_px))[0]
        store["el90_front_postjit"].append(
            feats(ce.to_ours(uv9, Pc4, s["side"], "2d", rng)))
        if n % 10 == 0:
            print(f"  [{n}/{len(seq_list())}] {s['name']} rows={len(rows)} "
                  f"{time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(
        OUT / "events.npz",
        **{f"X_{k}": np.vstack(v) for k, v in store.items() if v},
        lab=np.concatenate(meta["lab"]), hmm=np.concatenate(meta["hmm"]),
        part=np.concatenate(meta["part"]), gest=np.concatenate(meta["gest"]).astype(str),
        fing=np.concatenate(meta["fing"]), seq=np.concatenate(meta["seq"]),
        tip_xyz=np.vstack(geo["tip_xyz"]),
        **{f"occ_{k}": np.concatenate(v) for k, v in geo["occ"].items()},
        **{f"occo_{k}": np.concatenate(v) for k, v in geo["occo"].items()},
        **{f"loc_{k}": np.concatenate(v) for k, v in geo["loc_err"].items()},
        cams=json.dumps(geo["cams"]),
        rig=json.dumps({"knuckle_mm": kn, "f_px": f_px, "imw": IMW, "imh": IMH,
                        "work_mm": WORK_MM, "knuckle_px": KNUCKLE_PX,
                        "jitter_px": PIX_JITTER}),
    )
    n = len(np.concatenate(meta["lab"]))
    print(f"events={n} contact={int(np.concatenate(meta['lab']).sum())} "
          f"in {time.time()-t0:.0f}s -> {OUT/'events.npz'}", flush=True)
    return 0


# ---- discriminability --------------------------------------------------------------------
def _oof(X: np.ndarray, y: np.ndarray, part: np.ndarray, seeds: int, k: int = 7) -> np.ndarray:
    """-> [seeds, n] out-of-fold scores, participant-grouped folds (contact_ego protocol)."""
    u = np.unique(part)
    folds = [np.isin(part, u[i::k]) for i in range(k)]
    out = np.zeros((seeds, len(y)))
    for s in range(seeds):
        for te in folds:
            m = gb.make_model("lgbm", random_state=s, n_jobs=4, n_estimators=300,
                              scale_pos_weight=1.0, min_child_samples=20)
            m.fit(X[~te], y[~te])
            out[s, te] = m.predict_proba(X[te])[:, 1]
    return out


def auc_ci(y: np.ndarray, sc: np.ndarray, part: np.ndarray, n_boot: int = 2000,
           seed: int = 0) -> dict:
    """Mean-over-seeds ROC AUC (contact_ego's statistic) + 95% participant-cluster bootstrap CI.
    sc is [n] or [seeds, n]; the ensemble AUC of the seed-averaged score is reported too."""
    from sklearn.metrics import roc_auc_score
    S = np.atleast_2d(sc)
    rng = np.random.default_rng(seed)
    u = np.unique(part)
    idx = {p: np.where(part == p)[0] for p in u}
    boots = []
    for _ in range(n_boot):
        r = np.concatenate([idx[p] for p in rng.choice(u, len(u), replace=True)])
        if len(np.unique(y[r])) < 2:
            continue
        boots.append(np.mean([roc_auc_score(y[r], v[r]) for v in S]))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    per = [float(roc_auc_score(y, v)) for v in S]
    return {"auc": float(np.mean(per)), "ci": [float(lo), float(hi)], "n_boot": len(boots),
            "auc_seeds": per, "auc_ensemble": float(roc_auc_score(y, S.mean(0)))}


def cmd_sweep(a) -> int:
    RES.mkdir(parents=True, exist_ok=True)
    Z = np.load(OUT / "events.npz", allow_pickle=True)
    lab, part, hmm = Z["lab"], Z["part"], Z["hmm"]
    rig = json.loads(str(Z["rig"]))
    cams = json.loads(str(Z["cams"]))
    names = [k[2:] for k in Z.files if k.startswith("X_")]
    print(f"events={len(lab)} contact={int(lab.sum())} hover={int((~lab).sum())} "
          f"participants={len(np.unique(part))}", flush=True)
    oofs: dict[str, np.ndarray] = {}
    res = {"rig": rig, "cams": cams, "n": int(len(lab)), "pos": int(lab.sum()),
           "true_height_auc": auc_ci(lab, -hmm, part), "arms": {}}
    order = ["ref_cam4", "el90_front_postjit"] + \
        [cam_name(e, az) for az in AZIMUTHS for e in SWEEP_ELEV[az]]
    for name in [n for n in order if n in names]:
        X = Z[f"X_{name}"].astype(np.float32)
        S = _oof(X, lab, part, a.seeds)
        oofs[name] = S.astype(np.float32)
        r = auc_ci(lab, S, part)
        if name in cams:
            r.update({"el": cams[name]["el"], "az": cams[name]["az"]})
        if f"occ_{name}" in Z.files:
            occ = Z[f"occ_{name}"]
            loc = Z[f"loc_{name}"]
            r["occlusion_rate"] = float(occ.mean())
            r["occlusion_rate_contact"] = float(occ[lab].mean())
            r["occlusion_rate_other"] = float(Z[f"occo_{name}"].mean())
            r["loc_err_mm_median"] = float(np.nanmedian(loc))
            r["loc_err_mm_p90"] = float(np.nanpercentile(loc, 90))
            r["loc_err_mm_median_contact"] = float(np.nanmedian(loc[lab]))
            r["loc_err_mm_p90_contact"] = float(np.nanpercentile(loc[lab], 90))
        res["arms"][name] = r
        print(f"  {name:<14} AUC={r['auc']:.3f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}]"
              + (f"  occ={r['occlusion_rate']:.3f}/{r['occlusion_rate_other']:.3f}"
               f" loc(contact)={r['loc_err_mm_median_contact']:.1f}mm"
                 if "occlusion_rate" in r else ""), flush=True)
    # two-view combinations (mirror)
    for a_, b_ in a.combine:
        if f"X_{a_}" in Z.files and f"X_{b_}" in Z.files:
            X = np.hstack([Z[f"X_{a_}"], Z[f"X_{b_}"]]).astype(np.float32)
            S = _oof(X, lab, part, a.seeds)
            oofs[f"mirror:{a_}+{b_}"] = S.astype(np.float32)
            res["arms"][f"mirror:{a_}+{b_}"] = auc_ci(lab, S, part)
            print(f"  mirror:{a_}+{b_} AUC={res['arms'][f'mirror:{a_}+{b_}']['auc']:.3f}",
                  flush=True)
    np.savez_compressed(OUT / "oof.npz", lab=lab, part=part, hmm=hmm,
                        **{k.replace(":", "_").replace("+", "_"): v for k, v in oofs.items()})
    (RES / "angle_sweep.json").write_text(json.dumps(res, indent=1))
    return 0


# ---- raking light ------------------------------------------------------------------------
def cmd_shadow(a) -> int:
    """Top-down camera + a low raking light: image-plane gap between a fingertip and its cast
    shadow on the desk.  Pure geometry: gap_mm = height / tan(light elevation)."""
    RES.mkdir(parents=True, exist_ok=True)
    Z = np.load(OUT / "events.npz", allow_pickle=True)
    lab, part, hmm = Z["lab"], Z["part"], Z["hmm"]
    rig = json.loads(str(Z["rig"]))
    f_px = rig["f_px"]
    desk_mm = WORK_MM + PALM_H_MM          # camera-to-desk distance for the top-down rig
    mm_to_px = f_px / desk_mm
    out = {"rig": rig, "desk_mm": desk_mm, "mm_per_px": 1.0 / mm_to_px, "n": int(len(lab)), "pos": int(lab.sum()),
           "true_height_auc": auc_ci(lab, -hmm, part), "lights": {}}
    rng = np.random.default_rng(1)
    for el in a.light:
        gap_mm = hmm / np.tan(np.radians(el))
        gap_px = gap_mm * mm_to_px
        for sig, tag in ((0.0, "noiseless"), (1.7, "sigma1.7px"), (3.0, "sigma3.0px")):
            meas = gap_px + rng.normal(0, sig, len(gap_px))
            for thr, ttag in ((0.0, ""), (2.0, "+2px_floor")):
                m = np.where(meas < thr, 0.0, meas)
                out["lights"][f"{el}deg_{tag}{ttag}"] = {
                    "light_elev_deg": el, "noise_px": sig, "floor_px": thr,
                    **auc_ci(lab, -m, part),
                    "gap_px_median_hover": float(np.median(gap_px[~lab])),
                    "gap_px_median_contact": float(np.median(gap_px[lab])),
                    "gap_mm_median_hover": float(np.median(gap_mm[~lab])),
                }
                r = out["lights"][f"{el}deg_{tag}{ttag}"]
                print(f"  light {el:>2} deg {tag}{ttag:<11} AUC={r['auc']:.3f} "
                      f"[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]  gap px hover "
                      f"{r['gap_px_median_hover']:.1f} contact {r['gap_px_median_contact']:.1f}",
                      flush=True)
    # degradation: a fraction of shadows unmeasurable (merged with a neighbour, off the clean
    # desk zone, penumbra); and a saturation cap when the shadow leaves that zone.
    base = hmm / np.tan(np.radians(20.0)) * mm_to_px + rng.normal(0, 1.7, len(hmm))
    for miss in (0.3, 0.5):
        m = base.copy()
        bad = rng.random(len(m)) < miss
        m[bad] = np.median(base)
        out["lights"][f"20deg_miss{int(miss*100)}pct"] = {"light_elev_deg": 20, "miss": miss,
                                                          **auc_ci(lab, -m, part)}
    for cap in (60.0, 30.0):
        m = np.minimum(base, cap)
        out["lights"][f"20deg_cap{int(cap)}px"] = {"light_elev_deg": 20, "cap_px": cap,
                                                   **auc_ci(lab, -m, part)}
    for k in [k for k in out["lights"] if "miss" in k or "cap" in k]:
        r = out["lights"][k]
        print(f"  {k:<22} AUC={r['auc']:.3f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}]", flush=True)
    (RES / "shadow.json").write_text(json.dumps(out, indent=1))
    return 0


def cmd_jitter(a) -> int:
    """Re-extract a few cameras under several landmark-noise draws: the 0.8 px jitter alone
    moves this AUC by several points, so the curve needs that variance quantified."""
    RES.mkdir(parents=True, exist_ok=True)
    f_px = KNUCKLE_PX * WORK_MM / knuckle_mm()
    keys = [(n, int(n[2:4]), AZIMUTHS[n.split("_")[1]], j) for n in a.cams
            for j in range(a.jitter_seeds)]
    store = {f"{n}_j{j}": [] for n, _, _, j in keys}
    labs, parts = [], []
    for d in seq_list():
        s_ = ce.load_seq(d)
        if s_ is None:
            continue
        t30 = (s_["frame"] - s_["frame"][0]) / 30.0
        ff = ce.finger_force(s_["J"], s_["force"])
        t60, J60 = ce.resample60(t30, s_["J"])
        _, ff60 = ce.resample60(t30, ff)
        rows, fing, lab, _ = _descents(-J60[:, list(TIPS), 2] * 1000.0,
                                       np.nan_to_num(ff60[:, :5]))
        if len(rows) == 0:
            continue
        labs.append(lab)
        parts.append(np.full(len(rows), int(s_["participant"][2:])))
        slot = 0 if s_["side"] == "left" else 1
        prm = ours_params(vproject(s_["J"], virtual_cam(s_["J"], 90, 0.0, f_px))[0], s_["side"])
        for name, el, az, j in keys:
            rng = np.random.default_rng(1000 + j)
            uv = vproject(s_["J"], virtual_cam(s_["J"], el, az, f_px))[0]
            P1 = apply_ours(uv + rng.normal(0, PIX_JITTER, uv.shape), prm)
            _, P60 = ce.resample60(t30, P1)
            P = np.full((len(t60), 2, 21, 7), np.nan)
            P[:, slot] = P60
            store[f"{name}_j{j}"].append(
                gb.assemble(gb.build_groups(P), gb.GROUPS).astype(np.float32)[rows])
    lab, part = np.concatenate(labs), np.concatenate(parts)
    out = {"jitter_px": PIX_JITTER, "seeds": a.seeds, "draws": a.jitter_seeds, "arms": {}}
    for k, v in store.items():
        S = _oof(np.vstack(v), lab, part, a.seeds)
        out["arms"][k] = auc_ci(lab, S, part)
        print(f"  {k:<16} AUC={out['arms'][k]['auc']:.3f} "
              f"[{out['arms'][k]['ci'][0]:.3f},{out['arms'][k]['ci'][1]:.3f}]", flush=True)
    for n in a.cams:
        v = [out["arms"][f"{n}_j{j}"]["auc"] for j in range(a.jitter_seeds)]
        out[f"{n}_draw_spread"] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                                   "min": float(min(v)), "max": float(max(v))}
        print(f"  {n} across draws: mean {np.mean(v):.3f} sd {np.std(v, ddof=1):.3f} "
              f"range [{min(v):.3f},{max(v):.3f}]", flush=True)
    (RES / "jitter_draws.json").write_text(json.dumps(out, indent=1))
    return 0


TYPING_PATTERNS = ("*type_ipad*", "*press_fingers_high*", "*press_fingers_low*",
                   "*press_flat_onebyone_high*", "*press_flat_onebyone_low*")


def cmd_occ(a) -> int:
    """Occlusion / planar localisation on typing-like gestures, where all fingers are
    deployed over the desk (the index_press subset curls four of them into a fist)."""
    RES.mkdir(parents=True, exist_ok=True)
    f_px = KNUCKLE_PX * WORK_MM / knuckle_mm()
    cams = [(cam_name(el, az), el, AZIMUTHS[az]) for az in AZIMUTHS for el in ELEVATIONS]
    acc = {n: {"occ": [], "occo": [], "loc": []} for n, _, _ in cams}
    fings, labs = [], []
    rng = np.random.default_rng(2)
    ds = sorted({d for p in TYPING_PATTERNS for d in ce.seq_dirs(p)})
    t0 = time.time()
    for n, d in enumerate(ds):
        s = ce.load_seq(d)
        if s is None:
            continue
        t30 = (s["frame"] - s["frame"][0]) / 30.0
        ff = ce.finger_force(s["J"], s["force"])
        t60, J60 = ce.resample60(t30, s["J"])
        _, ff60 = ce.resample60(t30, ff)
        H = -J60[:, list(TIPS), 2] * 1000.0
        rows, fing, lab, _ = _descents(H, np.nan_to_num(ff60[:, :5]))
        if len(rows) == 0:
            continue
        fings.append(fing); labs.append(lab)
        tipxy = J60[rows[:, None], np.array(TIPS)[fing][:, None]][:, 0, :2]
        for name, el, az in cams:
            cam = virtual_cam(s["J"], el, az, f_px)
            oc, oco = occluded(s["J"], cam)
            _, o60 = ce.resample60(t30, np.stack([oc, oco], 1).astype(np.float32))
            acc[name]["occ"].append(o60[rows, 0, fing] > 0.5)
            acc[name]["occo"].append(o60[rows, 1, fing] > 0.5)
            uv, _ = vproject(s["J"], cam)
            xy = unproject_to_plane(uv + rng.normal(0, PIX_JITTER, uv.shape), cam)
            _, xy60 = ce.resample60(t30, xy)
            est = xy60[rows[:, None], np.array(TIPS)[fing][:, None]][:, 0]
            acc[name]["loc"].append(1000.0 * np.linalg.norm(est - tipxy, axis=1))
        if n % 20 == 0:
            print(f"  [{n}/{len(ds)}] {s['name']} {time.time()-t0:.0f}s", flush=True)
    fing, lab = np.concatenate(fings), np.concatenate(labs)
    out = {"n": int(len(fing)), "contact": int(lab.sum()), "n_seq": len(ds),
           "patterns": list(TYPING_PATTERNS), "per_finger_n": np.bincount(fing, minlength=5).tolist(),
           "arms": {}}
    for name, el, az in cams:
        o = np.concatenate(acc[name]["occ"]); oo = np.concatenate(acc[name]["occo"])
        L = np.concatenate(acc[name]["loc"])
        out["arms"][name] = {"el": el, "az": az, "occlusion_rate": float(o.mean()),
                             "occlusion_rate_other": float(oo.mean()),
                             "occlusion_rate_contact": float(o[lab].mean()),
                             "loc_err_mm_median": float(np.nanmedian(L)),
                             "loc_err_mm_p90": float(np.nanpercentile(L, 90)),
                             "loc_err_mm_median_contact": float(np.nanmedian(L[lab])),
                             "loc_err_mm_p90_contact": float(np.nanpercentile(L[lab], 90))}
        r = out["arms"][name]
        print(f"  {name:<13} occ={r['occlusion_rate']:.3f}/{r['occlusion_rate_other']:.3f} "
              f"loc={r['loc_err_mm_median']:.1f}mm locC={r['loc_err_mm_median_contact']:.1f}mm",
              flush=True)
    (RES / "occlusion_typing.json").write_text(json.dumps(out, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.contact_angle",
                                 description=__doc__)
    ap.add_argument("cmd", choices=("events", "sweep", "shadow", "occ", "jitter"))
    ap.add_argument("--jitter-seeds", type=int, default=5)
    ap.add_argument("--cams", nargs="*", default=["el90_front", "el20_front"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--light", type=float, nargs="*", default=[10, 20, 30])
    ap.add_argument("--combine", nargs="*", default=["el90_front,el10_front",
                                                     "el90_front,el20_side"])
    a = ap.parse_args(argv)
    a.combine = [tuple(c.split(",")) for c in a.combine]
    return globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    sys.exit(main())
