"""Ten-finger typing through the same virtual cameras as contact_angle, using How We Type mocap
(Feit et al. CHI'16, CC-BY-NC-4.0: research measurement only). 30 participants, both hands, all
fingers deployed, keylogger truth. Complements the EgoPressure index-press sweep, whose hands have
four fingers curled into a fist and so cannot say anything about ten-finger occlusion.
Run: python -m phase0.analysis.angle_hwt {events | sweep}"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from phase0.analysis import contact_angle as CA
from phase0.analysis import contact_ego as ce
from phase0.analysis import howwetype as HWT
from phase0.analysis import taps_gb as gb

OUT = Path(".cache/contact_angle")
RES = Path("results/angle")
TIPS = CA.TIPS
HAND_SEP_M = 0.14          # assumed lateral distance between the two palm centres (normalise() drops it)
KEYDOWN_HALF_S = 0.05      # a descent bottom within this of a key-down, by the lowest tip, is a contact
HOVER_CLEAR_S = 0.10       # a bottom with no key-down within this is a hover; in between = ambiguous, dropped
SIDES = ("left", "right")  # slot 0 / slot 1 after normalise()


def world_joints(st) -> tuple[np.ndarray, dict]:
    """HWT normalised stream -> [T,2,21,3] world metres in contact_angle's convention (z=0 desk, height=-z).
    Planar: knuckle-width units * S_MM, hands placed HAND_SEP_M apart. Height: HWT z is wrist-relative, so
    each hand's wrist height above the keys is estimated as the median height (over its key-downs) of the
    lowest fingertip at the key-down instant, i.e. 'the pressing tip is on the key'."""
    S = CA.knuckle_mm() / 1000.0                       # m per knuckle-width unit (EgoPressure median)
    A, M = st.A.astype(np.float64), st.M
    J = np.full(A.shape, np.nan)
    J[..., 0] = A[..., 0] * S
    J[..., 1] = A[..., 1] * S + np.array([-HAND_SEP_M / 2, HAND_SEP_M / 2])[None, :, None]
    hrel = -A[..., 2] * S                             # m above the wrist (+ = higher)
    J[~M] = np.nan
    # wrist height per hand from key-downs
    idx = np.clip(np.searchsorted(st.t, st.kt), 0, len(st.t) - 1)
    tips = hrel[idx][:, :, list(TIPS)]                 # [K,2,5]
    flat = tips.reshape(len(idx), -1)
    ok = np.isfinite(flat).any(1)
    low = np.nanargmin(np.where(np.isfinite(flat), flat, np.inf)[ok], 1)
    hand, tip = low // 5, low % 5
    hw = {}
    for h in (0, 1):
        v = tips[ok][hand == h, h, :][np.arange((hand == h).sum()), tip[hand == h]]
        hw[h] = float(-np.median(v)) if len(v) else np.nan
    habs = hrel + np.array([hw[0], hw[1]])[None, :, None]     # m above the key plane
    J[..., 2] = -habs
    return J, {"wrist_height_m": hw, "n_keydowns_used": int(ok.sum()),
               "presser_hand_frac_right": float((hand == 1).mean())}


def virtual_cam2(J: np.ndarray, el: float, az: float, f_px: float) -> dict:
    """contact_angle.virtual_cam aimed at the midpoint of both palms, forward = mean finger direction."""
    ok = np.isfinite(J[:, :, 0, 0])
    pc = np.stack([np.nanmedian(J[ok[:, h], h][:, [0, 5, 9, 13, 17]].mean(1), 0) for h in (0, 1)])
    target = pc.mean(0)
    fw = np.nanmedian(np.concatenate([J[ok[:, h], h, 9, :2] - J[ok[:, h], h, 0, :2] for h in (0, 1)]), 0)
    fw = fw / np.linalg.norm(fw)
    up = np.array([0.0, 0.0, -1.0])
    f3 = np.array([fw[0], fw[1], 0.0])
    side = np.cross(up, f3); side /= np.linalg.norm(side)
    e, a = np.radians(el), np.radians(az)
    d = np.cos(e) * (np.cos(a) * f3 + np.sin(a) * side) + np.sin(e) * up
    d /= np.linalg.norm(d)
    C = target + (CA.WORK_MM / 1000.0) * d
    zc = -d
    xc = np.cross(up, zc)
    if np.linalg.norm(xc) < 1e-6:
        xc = np.cross(f3, zc)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    return {"R": np.stack([xc, yc, zc]), "C": C, "f": f_px, "cx": CA.IMW / 2.0, "cy": CA.IMH / 2.0,
            "el": el, "az": az}


def occluded_cross(Jh: np.ndarray, Jo: np.ndarray, cam: dict, r_mm: float = CA.OCC_RADIUS_MM) -> np.ndarray:
    """[F,5] tip of hand Jh hidden behind any bone of the OTHER hand Jo (capsule test)."""
    P = (Jh - cam["C"]) * 1000.0
    Q = (Jo - cam["C"]) * 1000.0
    out = np.zeros((len(Jh), 5), bool)
    for k, tip in enumerate(TIPS):
        T = P[:, tip]
        dT = np.linalg.norm(T, axis=1)
        u = T / np.where(dT[:, None] < 1e-9, np.nan, dT[:, None])
        for (i, j) in CA.BONES:
            A, B = Q[:, i], Q[:, j]
            for w in np.linspace(0.0, 1.0, 7):
                X = A + w * (B - A)
                s = (X * u).sum(1)
                perp = np.linalg.norm(X - s[:, None] * u, axis=1)
                out[:, k] |= (perp < r_mm) & (s > 0) & (s < dT - r_mm)
    out[~np.isfinite(Jh[:, 0, 0]) | ~np.isfinite(Jo[:, 0, 0])] = False
    return out


def descents(J: np.ndarray, t: np.ndarray, kt: np.ndarray, prom_mm: float = 8.0):
    """-> rows, hand, finger, label (1 contact / 0 hover / -1 ambiguous), height mm at the bottom."""
    from scipy.signal import find_peaks
    H = -J[:, :, list(TIPS), 2] * 1000.0               # [T,2,5] mm above the key plane
    kt = np.sort(kt)
    rows, hands, fings, labs, hmm = [], [], [], [], []
    for h in (0, 1):
        for k in range(5):
            x = H[:, h, k]
            ok = np.isfinite(x)
            if ok.sum() < 60:
                continue
            xx = np.where(ok, x, np.nanmax(x))
            pk, _ = find_peaks(-xx, prominence=prom_mm, distance=20)
            for i in pk:
                if not ok[i] or not (-10.0 <= xx[i] <= 60.0):
                    continue
                j = np.searchsorted(kt, t[i])
                dt = min(abs(t[i] - kt[j - 1]) if j > 0 else np.inf, abs(kt[j] - t[i]) if j < len(kt) else np.inf)
                allt = H[i].ravel()
                lowest = np.isfinite(allt).any() and np.nanargmin(allt) == h * 5 + k
                lab = 1 if (dt <= KEYDOWN_HALF_S and lowest) else (0 if dt > HOVER_CLEAR_S else -1)
                rows.append(i); hands.append(h); fings.append(k); labs.append(lab); hmm.append(xx[i])
    return (np.array(rows, int), np.array(hands, int), np.array(fings, int), np.array(labs, int),
            np.array(hmm, float))


def cmd_events(a) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    kn = CA.knuckle_mm()
    f_px = CA.KNUCKLE_PX * CA.WORK_MM / kn
    cams = [(CA.cam_name(el, az), el, CA.AZIMUTHS[az]) for az in CA.AZIMUTHS for el in CA.ELEVATIONS]
    rng = np.random.default_rng(0)
    store = {n: [] for n, _, _ in cams}
    meta = {"lab": [], "hmm": [], "part": [], "hand": [], "fing": []}
    geo = {"occ": {n: [] for n, _, _ in cams}, "occx": {n: [] for n, _, _ in cams},
           "occo": {n: [] for n, _, _ in cams}, "occ6": {n: [] for n, _, _ in cams},
           "loc": {n: [] for n, _, _ in cams},
           "frame_occ": {n: [] for n, _, _ in cams}, "frame_occx": {n: [] for n, _, _ in cams},
           "frame_typing": [], "wrist": {}}
    t0 = time.time()
    sts = HWT.streams()
    for n, st in enumerate(sts):
        J, info = world_joints(st)
        geo["wrist"][st.name] = info
        rows, hand, fing, lab, hmm = descents(J, st.t, st.kt)
        keep = lab >= 0
        if a.max_hover_ratio and keep.sum():
            # cap hovers at max_hover_ratio x contacts per session (keeps LightGBM tractable)
            c = np.where(keep & (lab == 1))[0]; hv = np.where(keep & (lab == 0))[0]
            if len(hv) > a.max_hover_ratio * len(c):
                drop = rng.choice(hv, len(hv) - int(a.max_hover_ratio * len(c)), replace=False)
                keep[drop] = False
        rows, hand, fing, lab, hmm = rows[keep], hand[keep], fing[keep], lab[keep], hmm[keep]
        if len(rows) == 0:
            continue
        meta["lab"].append(lab == 1); meta["hmm"].append(hmm); meta["hand"].append(hand); meta["fing"].append(fing)
        meta["part"].append(np.full(len(rows), n))
        # typing frames = within 300 ms of a key-down, for the frame-level occlusion rate
        j = np.clip(np.searchsorted(np.sort(st.kt), st.t), 1, len(st.kt) - 1)
        skt = np.sort(st.kt)
        typing = np.minimum(np.abs(st.t - skt[j - 1]), np.abs(skt[j] - st.t)) <= 0.3
        geo["frame_typing"].append(int(typing.sum()))
        cam90 = virtual_cam2(J, 90, 0.0, f_px)
        prm = [CA.ours_params(CA.vproject(J[:, h], cam90)[0], SIDES[h]) for h in (0, 1)]
        tipxy = J[rows, hand, np.array(TIPS)[fing], :2]
        for name, el, az in cams:
            cam = virtual_cam2(J, el, az, f_px)
            P = np.full((len(st.t), 2, 21, 7), np.nan)
            occ = np.zeros((len(st.t), 2, 5), bool); occx = np.zeros((len(st.t), 2, 5), bool)
            occo = np.zeros((len(st.t), 2, 5), bool); occ6 = np.zeros((len(st.t), 2, 5), bool)
            loc = np.full((len(st.t), 2, 5), np.nan)
            for h in (0, 1):
                uv, _ = CA.vproject(J[:, h], cam)
                uvn = uv + rng.normal(0, CA.PIX_JITTER, uv.shape)
                P[:, h] = CA.apply_ours(uvn, prm[h])
                occ[:, h], occo[:, h] = CA.occluded(J[:, h], cam)
                occx[:, h] = occluded_cross(J[:, h], J[:, 1 - h], cam)
                _, o6 = CA.occluded(J[:, h], cam, r_mm=6.0)
                occ6[:, h] = o6 | occluded_cross(J[:, h], J[:, 1 - h], cam, r_mm=6.0)
                xy = CA.unproject_to_plane(uvn[:, list(TIPS)], cam)
                loc[:, h] = 1000.0 * np.linalg.norm(xy - J[:, h, list(TIPS), :2], axis=-1)
            X = gb.assemble(gb.build_groups(P), gb.GROUPS).astype(np.float32)[rows]
            store[name].append(X)
            geo["occ"][name].append(occ[rows, hand, fing])
            geo["occx"][name].append(occx[rows, hand, fing])
            geo["occo"][name].append(occo[rows, hand, fing])
            geo["occ6"][name].append(occ6[rows, hand, fing])
            geo["loc"][name].append(loc[rows, hand, fing])
            ok = np.isfinite(J[:, :, list(TIPS), 0]) & typing[:, None, None]
            geo["frame_occ"][name].append(int(occo[ok].sum())); geo["frame_occx"][name].append(int((occo | occx)[ok].sum()))
            geo.setdefault("frame_occ6", {}).setdefault(name, []).append(int(occ6[ok].sum()))
            geo.setdefault("frame_n", {}).setdefault(name, []).append(int(ok.sum()))
        print(f"  [{n}/{len(sts)}] {st.name} events={len(rows)} contact={int((lab == 1).sum())} "
              f"wrist {info['wrist_height_m'][0]*1000:.0f}/{info['wrist_height_m'][1]*1000:.0f} mm "
              f"{time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(
        OUT / "events_hwt.npz",
        **{f"X_{k}": np.vstack(v) for k, v in store.items()},
        lab=np.concatenate(meta["lab"]), hmm=np.concatenate(meta["hmm"]), part=np.concatenate(meta["part"]),
        hand=np.concatenate(meta["hand"]), fing=np.concatenate(meta["fing"]),
        **{f"occ_{k}": np.concatenate(v) for k, v in geo["occ"].items()},
        **{f"occx_{k}": np.concatenate(v) for k, v in geo["occx"].items()},
        **{f"occo_{k}": np.concatenate(v) for k, v in geo["occo"].items()},
        **{f"occ6_{k}": np.concatenate(v) for k, v in geo["occ6"].items()},
        **{f"loc_{k}": np.concatenate(v) for k, v in geo["loc"].items()},
        frame_occ=json.dumps({k: [sum(geo["frame_occ"][k]), sum(geo["frame_occx"][k]), sum(geo["frame_n"][k]),
                                  sum(geo["frame_occ6"][k])] for k in store}),
        cams=json.dumps({n: {"el": el, "az": az} for n, el, az in cams}),
        rig=json.dumps({"knuckle_mm": kn, "f_px": f_px, "imw": CA.IMW, "imh": CA.IMH, "work_mm": CA.WORK_MM,
                        "knuckle_px": CA.KNUCKLE_PX, "jitter_px": CA.PIX_JITTER, "hand_sep_m": HAND_SEP_M,
                        "wrist": geo["wrist"]}),
    )
    lab = np.concatenate(meta["lab"])
    print(f"events={len(lab)} contact={int(lab.sum())} hover={int((~lab).sum())} in {time.time()-t0:.0f}s", flush=True)
    return 0


def cmd_sweep(a) -> int:
    RES.mkdir(parents=True, exist_ok=True)
    Z = np.load(OUT / "events_hwt.npz", allow_pickle=True)
    lab, part, hmm, fing = Z["lab"], Z["part"], Z["hmm"], Z["fing"]
    cams = json.loads(str(Z["cams"]))
    fr = json.loads(str(Z["frame_occ"]))
    print(f"events={len(lab)} contact={int(lab.sum())} hover={int((~lab).sum())} participants={len(np.unique(part))} "
          f"hover height mm median={np.median(hmm[~lab]):.1f} contact median={np.median(hmm[lab]):.1f}", flush=True)
    res = {"rig": json.loads(str(Z["rig"])), "cams": cams, "n": int(len(lab)), "pos": int(lab.sum()),
           "hover_height_mm": {"median": float(np.median(hmm[~lab])), "p10": float(np.percentile(hmm[~lab], 10)),
                               "p90": float(np.percentile(hmm[~lab], 90))},
           "contact_height_mm": {"median": float(np.median(hmm[lab])), "p10": float(np.percentile(hmm[lab], 10)),
                                 "p90": float(np.percentile(hmm[lab], 90))},
           "per_finger_n": np.bincount(fing, minlength=5).tolist(),
           "true_height_auc": CA.auc_ci(lab, -hmm, part), "arms": {}}
    print(f"  true-height AUC={res['true_height_auc']['auc']:.3f}", flush=True)
    oofs = {}
    for name in cams:
        r = {"el": cams[name]["el"], "az": cams[name]["az"]}
        fit_this = (not a.geometry_only) and (not a.cams or name in a.cams)
        occ, occx, loc = Z[f"occo_{name}"], Z[f"occx_{name}"], Z[f"loc_{name}"]
        r["occlusion_rate_samehand"] = float(occ.mean())            # other finger / palm of the same hand, r 9 mm
        r["occlusion_rate_samehand_contact"] = float(occ[lab].mean())
        r["occlusion_rate_incl_otherhand"] = float((occ | occx).mean())
        r["occlusion_rate_incl_otherhand_r6mm"] = float(Z[f"occ6_{name}"].mean())
        r["occlusion_rate_own_chain_any"] = float(Z[f"occ_{name}"].mean())
        r["frame_occlusion_samehand"] = fr[name][0] / max(fr[name][2], 1)
        r["frame_occlusion_incl_otherhand"] = fr[name][1] / max(fr[name][2], 1)
        r["frame_occlusion_incl_otherhand_r6mm"] = fr[name][3] / max(fr[name][2], 1)
        r["occlusion_per_finger_incl_otherhand"] = [float((occ | occx)[fing == k].mean()) for k in range(5)]
        r["loc_err_mm_median"] = float(np.nanmedian(loc)); r["loc_err_mm_p90"] = float(np.nanpercentile(loc, 90))
        r["loc_err_mm_median_contact"] = float(np.nanmedian(loc[lab]))
        r["loc_err_mm_p90_contact"] = float(np.nanpercentile(loc[lab], 90))
        if fit_this:
            X = Z[f"X_{name}"].astype(np.float32)
            S = CA._oof(X, lab, part, a.seeds, k=a.folds)
            oofs[name] = S.astype(np.float32)
            r.update(CA.auc_ci(lab, S, part))
            # the same score restricted to unoccluded tips (what a real tracker could see)
            vis = ~(occ | occx)
            if vis.sum() > 100 and len(np.unique(lab[vis])) == 2:
                r["auc_visible_only"] = CA.auc_ci(lab[vis], S[:, vis], part[vis])["auc"]
        res["arms"][name] = r
        print(f"  {name:<12}" + (f" AUC={r['auc']:.3f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}]" if "auc" in r else "")
              + f"  occ same/incl-other {r['occlusion_rate_samehand']:.3f}/{r['occlusion_rate_incl_otherhand']:.3f}"
              f"  frame-occ {r['frame_occlusion_samehand']:.3f}/{r['frame_occlusion_incl_otherhand']:.3f}"
              f"  loc(contact) {r['loc_err_mm_median_contact']:.1f}/{r['loc_err_mm_p90_contact']:.1f} mm", flush=True)
    if oofs:
        np.savez_compressed(OUT / "oof_hwt.npz", lab=lab, part=part, hmm=hmm, **oofs)
    (RES / a.out).write_text(json.dumps(res, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.angle_hwt", description=__doc__)
    ap.add_argument("cmd", choices=("events", "sweep"))
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-hover-ratio", type=float, default=2.0)
    ap.add_argument("--geometry-only", action="store_true")
    ap.add_argument("--cams", nargs="*", default=[], help="fit the contact model only for these cameras")
    ap.add_argument("--out", default="angle_sweep_hwt.json")
    a = ap.parse_args(argv)
    return globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    sys.exit(main())
