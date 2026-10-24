"""REAL-FOOTAGE camera-angle A/B on PressureVisionDB (MIT licence — usable commercially, unlike HWT/EgoPressure).

The same hands, the same instants, filmed simultaneously from a near-normal view (~90 deg above the surface)
and two oblique views (~39-40 deg), with a per-frame pressure image as ground truth. Runs MediaPipe on each
view and asks, with OUR feature stack and OUR probe protocol: does the oblique view make contact more visible,
and does the tracker survive the angle?

Usage: python -m phase0.analysis.pvdb45 {landmarks | probe}
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PV = Path(".cache/pvdb")
OUT = Path(".cache/angle45")
RES = Path("results/angle45")
MODEL = Path("models/hand_landmarker.task")
SEQ = "type_sentence_5x_both"
# elevation above the pressure surface, from each sequence's own camera calibration (camera_2 ~ surface normal)
CAM_EL = {"camera_2": 90, "camera_1": 51, "camera_3": 40, "camera_0": 39}
CONTACT_FRAC = 0.05          # frame counts as contact if total force > this fraction of the session max


def participants() -> list[str]:
    return sorted(p.name for p in PV.iterdir() if (p / SEQ).is_dir())


def frames_of(p: str, cam: str) -> list[Path]:
    return sorted((PV / p / SEQ / cam).glob("*.jpg"))


def forces_of(p: str) -> np.ndarray:
    fs = sorted((PV / p / SEQ / "force").glob("*.pkl"))
    return np.array([float(pickle.load(open(f, "rb")).sum()) for f in fs])


def cmd_landmarks(a) -> int:
    """MediaPipe over every frame of every requested view -> P [F,2,21,7] per (participant, camera)."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    OUT.mkdir(parents=True, exist_ok=True)
    cams = a.cams.split(",")
    lm = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3))
    stats = {}
    t0 = time.time()
    for p in participants()[: a.participants]:
        for cam in cams:
            fs = frames_of(p, cam)
            if not fs:
                continue
            f_out = OUT / f"lm_{p}_{cam}.npz"
            if f_out.exists() and not a.force:
                continue
            P = np.full((len(fs), 2, 21, 7), np.nan, np.float32)
            score = np.zeros((len(fs), 2), np.float32)
            for i, f in enumerate(fs):
                im = cv2.imread(str(f))
                H, W = im.shape[:2]
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                r = lm.detect(img)              # IMAGE mode: the published small set is ~3 fps, no tracking continuity
                for h, (lms, hd) in enumerate(zip(r.hand_landmarks, r.handedness)):
                    slot = 0 if hd[0].category_name == "Left" else 1
                    if np.isfinite(P[i, slot, 0, 0]):
                        slot = 1 - slot
                    for j, q in enumerate(lms):
                        P[i, slot, j] = (q.x * W, q.y * H, hd[0].score, q.z * W, np.nan, np.nan, np.nan)
                    score[i, slot] = hd[0].score
            det = np.isfinite(P[:, :, 0, 0])
            np.savez_compressed(f_out, P=P, score=score, files=[f.name for f in fs])
            stats[f"{p}/{cam}"] = {"frames": len(fs), "hand_detect_rate": float(det.mean()),
                                   "both_hands_rate": float(det.all(1).mean()),
                                   "mean_score": float(np.nanmean(score[score > 0])) if (score > 0).any() else None}
            s = stats[f"{p}/{cam}"]
            print(f"  {p} {cam} ({CAM_EL.get(cam, '?')} deg): {len(fs)} frames, hand-detect {s['hand_detect_rate']:.3f}, "
                  f"both {s['both_hands_rate']:.3f}, score {s['mean_score']:.3f} [{time.time()-t0:.0f}s]", flush=True)
    lm.close()
    RES.mkdir(parents=True, exist_ok=True)
    f = RES / "pvdb_landmark_yield.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    old.update(stats)
    f.write_text(json.dumps(old, indent=1))
    return 0


# ---- contact probe, per view -------------------------------------------------------------
STATIC_GROUPS = ("pose", "wflex", "palm")    # the 3 fps sampling makes velocity/dwell features meaningless


def cmd_probe(a) -> int:
    """Press-vs-hover (frame in contact?) from each view's landmarks, leave-one-participant-out."""
    from sklearn.metrics import roc_auc_score
    from phase0.analysis import taps_gb as gb
    RES.mkdir(parents=True, exist_ok=True)
    cams = a.cams.split(",")
    res = {"groups": list(STATIC_GROUPS), "contact_frac": CONTACT_FRAC, "arms": {}}
    store = {c: {"X": [], "y": [], "p": [], "det": []} for c in cams}
    for pi, p in enumerate(participants()[: a.participants]):
        tot = forces_of(p)
        y = tot > CONTACT_FRAC * tot.max()
        for cam in cams:
            f = OUT / f"lm_{p}_{cam}.npz"
            if not f.exists():
                continue
            P = np.load(f)["P"]
            n = min(len(P), len(y))
            X = gb.assemble(gb.build_groups(P[:n]), STATIC_GROUPS).astype(np.float32)
            store[cam]["X"].append(X); store[cam]["y"].append(y[:n])
            store[cam]["p"].append(np.full(n, pi)); store[cam]["det"].append(np.isfinite(P[:n, :, 0, 0]).any(1))
    for cam in cams:
        d = store[cam]
        if not d["X"]:
            continue
        X, y, pp, det = np.vstack(d["X"]), np.concatenate(d["y"]), np.concatenate(d["p"]), np.concatenate(d["det"])
        oof = np.zeros(len(y))
        for q in np.unique(pp):
            te = pp == q
            m = gb.make_model("lgbm", random_state=0, n_jobs=4, n_estimators=200, min_child_samples=10)
            m.fit(X[~te], y[~te])
            oof[te] = m.predict_proba(X[te])[:, 1]
        auc = float(roc_auc_score(y, oof))
        rng = np.random.default_rng(0)
        shuf = float(roc_auc_score(y, np.roll(oof, len(oof) // 3)))
        per = {int(q): float(roc_auc_score(y[pp == q], oof[pp == q])) for q in np.unique(pp)
               if len(np.unique(y[pp == q])) == 2}
        res["arms"][cam] = {"elevation_deg": CAM_EL.get(cam), "n": int(len(y)), "pos": int(y.sum()),
                            "landmark_rate": float(det.mean()), "auc": auc,
                            "auc_per_participant": per, "auc_timeshuffled": shuf,
                            "auc_sd_over_participants": float(np.std(list(per.values()), ddof=1)) if len(per) > 1 else None}
        r = res["arms"][cam]
        print(f"  {cam} ({r['elevation_deg']} deg): AUC {auc:.3f} (per-participant {list(per.values())}, "
              f"shuffled {shuf:.3f}), landmarks {r['landmark_rate']:.3f}, n={r['n']} pos={r['pos']}", flush=True)
    (RES / "pvdb_contact_probe.json").write_text(json.dumps(res, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m phase0.analysis.pvdb45", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("landmarks"); s.add_argument("--cams", default="camera_2,camera_0,camera_3,camera_1")
    s.add_argument("--participants", type=int, default=3); s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_landmarks)
    s = sub.add_parser("probe"); s.add_argument("--cams", default="camera_2,camera_0,camera_3,camera_1")
    s.add_argument("--participants", type=int, default=3); s.set_defaults(fn=cmd_probe)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
