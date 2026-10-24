"""A MIRROR as a second virtual viewpoint for the top-down phone camera: reflect the real camera through the mirror
plane, triangulate fingertip height from the direct + mirrored 2D projections, and cost it (occlusion, resolution,
frame budget). EgoPressure descent-bottom events (CC BY-NC, research only) give height error and press-vs-hover
AUC; How We Type mocap (CC-BY-NC) gives both-view visibility, mirrored-hand resolution and frame footprint for
real ten-finger typing. Run: python -m phase0.analysis.mirror_sim {ego | hwt | report}"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from phase0.analysis import contact_angle as CA

OUT = Path(".cache/mirror")
RES = Path("results/mirror")
TIPS = CA.TIPS
F_PX_227 = 538.3086                    # results/angle rig: 136 px knuckle at 227 mm
IMW, IMH = CA.IMW, CA.IMH
PALM_H = 0.040                         # m, palm centroid above the desk (results/angle: 39.7 mm)
HAND_REGION = (0.110, 0.090)           # m half-extents (x sideways, y forward) the mirror must show around the hand centre
PLACEMENTS = {                         # name: (azimuth of the mirror from the hand centre, distance m, centre height m)
    "rear_160_h60": ("rear", 0.16, 0.06), "rear_240_h60": ("rear", 0.24, 0.06), "rear_240_h120": ("rear", 0.24, 0.12),
    "side_160_h60": ("side", 0.16, 0.06), "side_240_h60": ("side", 0.24, 0.06), "side_240_h120": ("side", 0.24, 0.12),
    "diag45_200_h60": ("diag", 0.20, 0.06), "diag45_200_h120": ("diag", 0.20, 0.12),
}
WORK_DISTANCES = (0.227, 0.320, 0.420)  # m from camera to the palm plane


# ---------------------------------------------------------------- geometry
def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def real_cam(T: np.ndarray, work: float) -> dict:
    """Top-down camera `work` above the hand target T (world: desk z=0, height = -z). Image x axis along
    world -y (the typist's forward direction = the sensor's long 1280 axis, as on the real rig), image y along x."""
    C = T + np.array([0.0, 0.0, -work])
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # rows: xc, yc, zc (optical axis = +z = down)
    return {"C": C, "R": R, "f": F_PX_227, "cx": IMW / 2.0, "cy": IMH / 2.0}


def project(P: np.ndarray, cam: dict) -> tuple[np.ndarray, np.ndarray]:
    Pc = (P - cam["C"]) @ cam["R"].T
    z = Pc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam["f"] * Pc[..., 0] / z + cam["cx"]
        v = cam["f"] * Pc[..., 1] / z + cam["cy"]
    uv = np.stack([u, v], -1)
    uv[z <= 1e-6] = np.nan
    return uv, Pc


def mirror_plane(T: np.ndarray, C: np.ndarray, az: str, dist: float, hc: float, forward=np.array([0.0, -1.0, 0.0]),
                 sideways=np.array([1.0, 0.0, 0.0])) -> dict:
    """Mirror centre `dist` from the hand centre along the azimuth, standing with its centre `hc` above the desk,
    tilted so that the camera's ray off the mirror centre lands on the hand centre."""
    d = {"rear": forward, "side": sideways, "diag": unit(forward + sideways)}[az]
    M = np.array([T[0], T[1], 0.0]) + dist * d + np.array([0.0, 0.0, -hc])
    n = unit(unit(C - M) + unit(T - M))
    return {"M": M, "n": n, "az": az, "dist": dist, "hc": hc, "horiz": d}


def reflect_pts(P: np.ndarray, pl: dict) -> np.ndarray:
    s = ((P - pl["M"]) @ pl["n"])[..., None]
    return P - 2.0 * s * pl["n"]


def reflect_dir(D: np.ndarray, pl: dict) -> np.ndarray:
    return D - 2.0 * (D @ pl["n"])[..., None] * pl["n"]


def virtual_cam(cam: dict, pl: dict) -> dict:
    """The camera seen through the mirror: reflected centre and axes (handedness flips, which is fine for geometry)."""
    Cv = reflect_pts(cam["C"][None], pl)[0]
    Rv = np.stack([reflect_dir(r, pl) for r in cam["R"]])
    return {"C": Cv, "R": Rv, "f": cam["f"], "cx": cam["cx"], "cy": cam["cy"]}


def stereo_geometry(cam: dict, pl: dict, T: np.ndarray) -> dict:
    vc = virtual_cam(cam, pl)
    d1, d2 = unit(T - cam["C"]), unit(T - vc["C"])
    verg = np.degrees(np.arccos(np.clip(d1 @ d2, -1, 1)))
    # elevation as in results/angle: 90 = straight down. World +z is down, so d2[2] > 0 means the virtual view looks down.
    el = np.degrees(np.arcsin(np.clip(d2[2], -1, 1)))
    az = abs(np.degrees(np.arctan2(-(vc["C"] - T)[0], -(vc["C"] - T)[1])))  # 0 = beyond the fingertips (front), 90 = side
    path = np.linalg.norm(vc["C"] - T)
    return {"baseline_mm": 1000 * np.linalg.norm(vc["C"] - cam["C"]), "vergence_deg": float(verg),
            "virtual_elevation_deg": float(el), "virtual_azimuth_deg": float(az), "virtual_path_mm": 1000 * path,
            "direct_path_mm": 1000 * np.linalg.norm(T - cam["C"]),
            "mirror_tilt_from_vertical_deg": float(np.degrees(np.arcsin(abs(pl["n"][2])))),   # lean-back of the glass
            "virtual_cam_below_desk": bool(vc["C"][2] > 0)}


def pixel_ray(uv: np.ndarray, cam: dict) -> np.ndarray:
    d = np.stack([(uv[..., 0] - cam["cx"]) / cam["f"], (uv[..., 1] - cam["cy"]) / cam["f"], np.ones(uv.shape[:-1])], -1)
    return unit(d @ cam["R"])


def triangulate(uv1: np.ndarray, cam: dict, uv2: np.ndarray, pl: dict) -> np.ndarray:
    """Direct ray from the real camera + the ray of the mirrored observation unfolded through the plane
    (origin = reflected camera centre, direction = reflected pixel ray). Least-squares closest point."""
    d1 = pixel_ray(uv1, cam)
    d2 = reflect_dir(pixel_ray(uv2, cam), pl)
    C1 = np.broadcast_to(cam["C"], d1.shape)
    C2 = np.broadcast_to(reflect_pts(cam["C"][None], pl)[0], d2.shape)
    # solve for s,t minimising |C1 + s d1 - C2 - t d2|
    r = C2 - C1
    a, b, c = (d1 * d1).sum(-1), (d1 * d2).sum(-1), (d2 * d2).sum(-1)
    d, e = (d1 * r).sum(-1), (d2 * r).sum(-1)
    den = a * c - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (d * c - b * e) / den
        t = (b * d - a * e) / den
    P1 = C1 + s[..., None] * d1
    P2 = C2 + t[..., None] * d2
    return 0.5 * (P1 + P2)


def mirror_hit(P: np.ndarray, cam: dict, pl: dict) -> np.ndarray:
    """Where the camera->reflected-point ray meets the mirror plane, expressed in the mirror's own 2D frame (m)."""
    Pr = reflect_pts(P, pl)
    D = Pr - cam["C"]
    denom = D @ pl["n"]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((pl["M"] - cam["C"]) @ pl["n"]) / denom
    H = cam["C"] + t[..., None] * D
    ex = unit(np.cross(np.array([0, 0, 1.0]), pl["n"]))            # horizontal axis in the mirror plane
    ey = np.cross(pl["n"], ex)                                      # "up" axis in the mirror plane
    rel = H - pl["M"]
    return np.stack([rel @ ex, rel @ ey], -1)


def required_mirror(T: np.ndarray, cam: dict, pl: dict) -> dict:
    """Mirror extent needed to reflect a HAND_REGION box around T (at heights 0..60 mm) into the camera."""
    hx, hy = HAND_REGION
    pts = np.array([[T[0] + sx * hx, T[1] + sy * hy, -h] for sx in (-1, 1) for sy in (-1, 1) for h in (0.0, 0.06)])
    mh = mirror_hit(pts, cam, pl)
    w = mh[:, 0].max() - mh[:, 0].min(); h = mh[:, 1].max() - mh[:, 1].min()
    corners_m = np.array([[mh[:, 0].min(), mh[:, 1].min()], [mh[:, 0].max(), mh[:, 1].min()],
                          [mh[:, 0].max(), mh[:, 1].max()], [mh[:, 0].min(), mh[:, 1].max()]])
    ex = unit(np.cross(np.array([0, 0, 1.0]), pl["n"])); ey = np.cross(pl["n"], ex)
    corners = pl["M"] + corners_m[:, :1] * ex + corners_m[:, 1:] * ey
    uv, _ = project(corners, cam)
    return {"width_cm": 100 * w, "height_cm": 100 * h, "corners_world": corners, "corners_px": uv,
            "bottom_edge_height_cm": 100 * (-corners[:, 2].max()), "top_edge_height_cm": 100 * (-corners[:, 2].min())}


def frame_budget(T: np.ndarray, cam: dict, pl: dict, req: dict) -> dict:
    hx, hy = HAND_REGION
    hand = np.array([[T[0] + sx * 0.125, T[1] + sy * hy, -h] for sx in (-1, 1) for sy in (-1, 1) for h in (0.0, 0.05)])   # both hands: 250 mm across
    uvh, _ = project(hand, cam)
    uvm = req["corners_px"]
    inside = lambda uv: bool(np.all((uv[:, 0] >= 0) & (uv[:, 0] < IMW) & (uv[:, 1] >= 0) & (uv[:, 1] < IMH)))
    bh = [uvh[:, 0].min(), uvh[:, 0].max(), uvh[:, 1].min(), uvh[:, 1].max()]
    bm = [uvm[:, 0].min(), uvm[:, 0].max(), uvm[:, 1].min(), uvm[:, 1].max()]
    overlap = not (bh[1] < bm[0] or bm[1] < bh[0] or bh[3] < bm[2] or bm[3] < bh[2])
    area = float(np.clip((bm[1] - bm[0]), 0, IMW) * np.clip((bm[3] - bm[2]), 0, IMH) / (IMW * IMH))
    return {"hands_in_frame": inside(uvh), "mirror_in_frame": inside(uvm), "overlap": overlap,
            "mirror_frame_fraction": area, "hand_bbox_px": [float(x) for x in bh], "mirror_bbox_px": [float(x) for x in bm]}


# ---------------------------------------------------------------- EgoPressure events
def cmd_ego(a) -> int:
    RES.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    Z = np.load(".cache/contact_angle/events.npz", allow_pickle=True)
    lab, part, hmm, seq = Z["lab"], Z["part"], Z["hmm"], Z["seq"]
    tip = Z["tip_xyz"].astype(float)
    X90 = Z["X_el90_front"].astype(np.float32)
    # nominal per-sequence hand target: median tip, 60 mm back toward the wrist (+y in the pad frame), palm height
    T = np.zeros_like(tip)
    for q in np.unique(seq):
        m = seq == q
        T[m] = np.array([np.median(tip[m, 0]), np.median(tip[m, 1]) + 0.06, -PALM_H])
    rng = np.random.default_rng(0)
    res = {"n": int(len(lab)), "pos": int(lab.sum()), "true_height_auc": CA.auc_ci(lab, -hmm, part),
           "topdown_2d_lgbm_auc": None, "placements": {}}
    S90 = CA._oof(X90, lab, part, a.seeds)
    res["topdown_2d_lgbm_auc"] = CA.auc_ci(lab, S90, part)
    print(f"events={len(lab)} true-height {res['true_height_auc']['auc']:.3f} top-down 2D {res['topdown_2d_lgbm_auc']['auc']:.3f}", flush=True)
    vis = json.loads((RES / "hwt_visibility.json").read_text())["arms"] if (RES / "hwt_visibility.json").exists() else {}
    for work in WORK_DISTANCES:
        for name, (az, dist, hc) in PLACEMENTS.items():
            key = f"{name}@{int(work*1000)}"
            cam = real_cam(T[0], work)          # intrinsics only; per-event cameras below
            out = {"work_mm": 1000 * work, "placement": name}
            h_tri = {"fixed": np.full(len(lab), np.nan), "scaled": np.full(len(lab), np.nan)}
            sg = None; pxmm = []
            for q in np.unique(seq):
                m = np.where(seq == q)[0]
                camq = real_cam(T[m[0]], work)
                pl = mirror_plane(T[m[0]], camq["C"], az, dist, hc)
                if sg is None:
                    sg = stereo_geometry(camq, pl, T[m[0]])
                    req = required_mirror(T[m[0]], camq, pl)
                    fb = frame_budget(T[m[0]], camq, pl, req)
                uv1, Pc1 = project(tip[m], camq)
                uv2, Pc2 = project(reflect_pts(tip[m], pl), camq)
                s1 = camq["f"] / Pc1[:, 2]; s2 = camq["f"] / Pc2[:, 2]      # px per m at the tip, direct / via mirror
                pxmm.append(np.median(s2 / s1))
                for tag in ("fixed", "scaled"):
                    n1 = rng.normal(0, CA.PIX_JITTER, uv1.shape)
                    sig2 = CA.PIX_JITTER * (np.ones(len(m)) if tag == "fixed" else np.maximum(1.0, s1 / s2))[:, None]
                    n2 = rng.normal(0, 1.0, uv2.shape) * sig2
                    P = triangulate(uv1 + n1, camq, uv2 + n2, pl)
                    h_tri[tag][m] = -P[:, 2] * 1000.0
            out["geometry"] = sg
            out["mirror_required"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in req.items()}
            out["frame"] = fb
            out["mirror_px_per_direct_px"] = float(np.median(pxmm))
            for tag in ("fixed", "scaled"):
                h = h_tri[tag]; err = h - hmm
                ok = np.isfinite(h)
                r = {"height_err_mm_median_abs": float(np.median(np.abs(err[ok]))), "height_err_mm_p90_abs": float(np.percentile(np.abs(err[ok]), 90)),
                     "height_err_mm_bias": float(np.median(err[ok])), "n_ok": int(ok.sum())}
                hh = np.where(ok, h, np.nanmedian(h))
                r["auc_triangulated_height_raw"] = CA.auc_ci(lab, -hh, part)
                Xc = np.hstack([X90, hh[:, None].astype(np.float32)])
                r["auc_topdown2d_plus_height_lgbm"] = CA.auc_ci(lab, CA._oof(Xc, lab, part, a.seeds), part)
                # fallback: tips not visible in both views (HWT index-finger rate for this placement) lose the height
                pv = vis.get(key, {}).get("both_visible_index", None)
                if pv is not None:
                    keep = rng.random(len(lab)) < pv
                    hv = np.where(keep, hh, np.nan)
                    Xf = np.hstack([X90, hv[:, None].astype(np.float32)])
                    r["fallback_both_visible_rate_index"] = pv
                    r["auc_fallback_lgbm"] = CA.auc_ci(lab, CA._oof(Xf, lab, part, a.seeds), part)
                    r["auc_fallback_raw_height_else_median"] = CA.auc_ci(lab, -np.where(keep, hh, np.nanmedian(hh)), part)
                out[f"noise_{tag}"] = r
            res["placements"][key] = out
            g = out["geometry"]; rf = out["noise_fixed"]; rs = out["noise_scaled"]
            print(f"  {key:<22} base {g['baseline_mm']:.0f}mm verg {g['vergence_deg']:.0f}deg vel {g['virtual_elevation_deg']:.0f} "
                  f"pathx {g['virtual_path_mm']/g['direct_path_mm']:.2f} mirror {out['mirror_required']['width_cm']:.0f}x{out['mirror_required']['height_cm']:.0f}cm "
                  f"fit={fb['mirror_in_frame']}/{fb['hands_in_frame']}/ovl={fb['overlap']} | h-err {rf['height_err_mm_median_abs']:.1f}/{rf['height_err_mm_p90_abs']:.1f} "
                  f"AUC raw {rf['auc_triangulated_height_raw']['auc']:.3f} +2D {rf['auc_topdown2d_plus_height_lgbm']['auc']:.3f} | scaled-noise h-err {rs['height_err_mm_median_abs']:.1f} AUC {rs['auc_triangulated_height_raw']['auc']:.3f}"
                  + (f" | fallback {rf['auc_fallback_lgbm']['auc']:.3f} (vis {rf['fallback_both_visible_rate_index']:.2f})" if "auc_fallback_lgbm" in rf else ""), flush=True)
    (RES / "mirror_ego.json").write_text(json.dumps(res, indent=1, default=float))
    return 0


# ---------------------------------------------------------------- How We Type: visibility, resolution, footprint
def cmd_hwt(a) -> int:
    from phase0.analysis import angle_hwt as AH, howwetype as HWT
    RES.mkdir(parents=True, exist_ok=True)
    sts = HWT.streams()[: a.sessions]
    acc = {}
    t0 = time.time()
    for n, st in enumerate(sts):
        J, info = AH.world_joints(st)
        ok = np.isfinite(J[:, :, 0, 0])
        pc = np.stack([np.nanmedian(J[ok[:, h], h][:, [0, 5, 9, 13, 17]].mean(1), 0) for h in (0, 1)])
        T = pc.mean(0); T[2] = -PALM_H
        # HWT frame: x forward, y rightward. Mirror azimuths: rear = beyond the fingertips = +x, side = +y
        rows, hand, fing, lab, hmm = AH.descents(J, st.t, st.kt)
        keep = lab >= 0; rows, hand, fing, lab = rows[keep], hand[keep], fing[keep], lab[keep]
        skt = np.sort(st.kt); j = np.clip(np.searchsorted(skt, st.t), 1, len(skt) - 1)
        typing = np.minimum(np.abs(st.t - skt[j - 1]), np.abs(skt[j] - st.t)) <= 0.3
        for work in WORK_DISTANCES:
            cam = real_cam(T, work)
            # the real camera's image axes: keep the long axis along forward (x here)
            cam["R"] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            occ_d = np.zeros((len(st.t), 2, 5), bool)
            for h in (0, 1):
                _, o = CA.occluded(J[:, h], cam); occ_d[:, h] = o | AH.occluded_cross(J[:, h], J[:, 1 - h], cam)
            for name, (az, dist, hc) in PLACEMENTS.items():
                key = f"{name}@{int(work*1000)}"
                pl = mirror_plane(T, cam["C"], az, dist, hc, forward=np.array([1.0, 0.0, 0.0]), sideways=np.array([0.0, 1.0, 0.0]))
                vc = virtual_cam(cam, pl)
                occ_m = np.zeros((len(st.t), 2, 5), bool)
                for h in (0, 1):
                    _, o = CA.occluded(J[:, h], vc); occ_m[:, h] = o | AH.occluded_cross(J[:, h], J[:, 1 - h], vc)
                both = ~occ_d & ~occ_m
                fin = np.isfinite(J[:, :, list(TIPS), 0]) & typing[:, None, None]
                a_ = acc.setdefault(key, {"both_frames": 0, "n_frames": 0, "mirror_vis_frames": 0, "direct_vis_frames": 0,
                                          "both_bottoms": 0, "n_bottoms": 0, "both_index_bottoms": 0, "n_index_bottoms": 0,
                                          "knuckle_px_direct": [], "knuckle_px_mirror": [], "per_finger_both": np.zeros(5), "per_finger_n": np.zeros(5)})
                a_["both_frames"] += int(both[fin].sum()); a_["n_frames"] += int(fin.sum())
                a_["mirror_vis_frames"] += int((~occ_m)[fin].sum()); a_["direct_vis_frames"] += int((~occ_d)[fin].sum())
                bb = both[rows, hand, fing]
                a_["both_bottoms"] += int(bb.sum()); a_["n_bottoms"] += len(bb)
                ix = fing == 1
                a_["both_index_bottoms"] += int(bb[ix].sum()); a_["n_index_bottoms"] += int(ix.sum())
                for k in range(5):
                    a_["per_finger_both"][k] += bb[fing == k].sum(); a_["per_finger_n"][k] += (fing == k).sum()
                for h in (0, 1):
                    uvd, _ = project(J[:, h, [5, 17]], cam)
                    uvm, _ = project(reflect_pts(J[:, h, [5, 17]], pl), cam)
                    a_["knuckle_px_direct"].append(np.nanmedian(np.linalg.norm(uvd[:, 0] - uvd[:, 1], axis=1)))
                    a_["knuckle_px_mirror"].append(np.nanmedian(np.linalg.norm(uvm[:, 0] - uvm[:, 1], axis=1)))
        print(f"  [{n}/{len(sts)}] {st.name} {time.time()-t0:.0f}s", flush=True)
    out = {}
    for key, a_ in acc.items():
        out[key] = {"both_visible_frames": a_["both_frames"] / max(a_["n_frames"], 1),
                    "direct_visible_frames": a_["direct_vis_frames"] / max(a_["n_frames"], 1),
                    "mirror_visible_frames": a_["mirror_vis_frames"] / max(a_["n_frames"], 1),
                    "both_visible_bottoms": a_["both_bottoms"] / max(a_["n_bottoms"], 1),
                    "both_visible_index": a_["both_index_bottoms"] / max(a_["n_index_bottoms"], 1),
                    "both_visible_per_finger": (a_["per_finger_both"] / np.maximum(a_["per_finger_n"], 1)).tolist(),
                    "knuckle_px_direct": float(np.nanmedian(a_["knuckle_px_direct"])),
                    "knuckle_px_mirror": float(np.nanmedian(a_["knuckle_px_mirror"]))}
        o = out[key]
        print(f"  {key:<22} both-visible frames {o['both_visible_frames']:.2f} (direct {o['direct_visible_frames']:.2f}, mirror {o['mirror_visible_frames']:.2f}) "
              f"bottoms {o['both_visible_bottoms']:.2f} index {o['both_visible_index']:.2f} | knuckle px direct {o['knuckle_px_direct']:.0f} mirror {o['knuckle_px_mirror']:.0f}", flush=True)
    (RES / "hwt_visibility.json").write_text(json.dumps({"sessions": len(sts), "capsule_mm": CA.OCC_RADIUS_MM, "arms": out}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("ego", "hwt"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sessions", type=int, default=12)
    a = ap.parse_args(argv)
    return {"ego": cmd_ego, "hwt": cmd_hwt}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
